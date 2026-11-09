# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Qwen3.8-Flash-Next (qwen4_exp) text backbone.

The text tower is a hybrid stack of GatedDeltaNet and Qwen Sparse Attention
(QSA) layers.  Every decoder layer threads a four-stream Hyper-Connection
residual bundle; one configured layer inserts an n-gram PLE side path before
its attention mixer.  The FFN is a 512-expert top-10 sparse MoE with a gated
shared expert.

This module intentionally implements the text path only.  The checkpoint's
vision tower and one-layer MTP head live under ``model.visual.*`` / ``mtp.*``
and are skipped by the text export.
"""

import itertools
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...config import LAYER_GDN, ModelConfig, PleConfig
from ..default.modeling_default import OnnxSpec
from ..linear import TPMode, make_linear
from ..ops import KV_PAGE_SIZE, qsa_attention_plugin
from ..qwen3_5.modeling_qwen3_5_text import (GdnMixer, Qwen3_5RMSNorm,
                                             _make_flat_wrapper_hybrid_ragged)
from ..qwen3_moe.modeling_qwen3_moe import Qwen3SparseMoeBlock

__all__ = [
    "GroupedRMSNorm",
    "Qwen4ExpHyperConnection",
    "Qwen4ExpPleLayer",
    "Qwen4ExpQsaAttention",
    "Qwen4ExpDecoderLayer",
    "Qwen4ExpBackbone",
    "Qwen4ExpCausalLM",
]

_BATCH_SIZE = 2
_SEQ_LEN = 2
_MAX_POS = 8


# ---------------------------------------------------------------------------
# Grouped RMSNorm (Gemma delta-weight convention)
# ---------------------------------------------------------------------------


class GroupedRMSNorm(nn.Module):
    """RMS-normalize independent groups of the last dimension.

    Qwen4-Exp stores every Hyper-Connection / PLE norm weight as a delta
    around zero and applies ``(1 + w)`` (Gemma convention).  The FP32
    accumulation mirrors the reference implementation; the output is cast
    back to the input dtype.
    """

    def __init__(self,
                 normalized_shape: int,
                 group_size: Optional[int] = None,
                 eps: float = 1e-6) -> None:
        super().__init__()
        if group_size is not None and normalized_shape % group_size != 0:
            raise ValueError(
                f"normalized_shape ({normalized_shape}) must be divisible by "
                f"group_size ({group_size})")
        self.normalized_shape = normalized_shape
        self.group_size = group_size
        self.variance_epsilon = eps
        self.weight = nn.Parameter(torch.zeros(normalized_shape,
                                               dtype=torch.float32))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        x = hidden_states.float()
        if self.group_size is None:
            groups = x
        else:
            groups = x.unflatten(-1, (self.normalized_shape // self.group_size,
                                      self.group_size))
        variance = groups.pow(2).mean(dim=-1, keepdim=True)
        normed = groups * torch.rsqrt(variance + self.variance_epsilon)
        if self.group_size is not None:
            normed = normed.flatten(-2)
        weight = (1.0 + self.weight.float()).to(normed.dtype)
        return (normed * weight).to(input_dtype)


# ---------------------------------------------------------------------------
# Hyper-Connection
# ---------------------------------------------------------------------------


class Qwen4ExpHyperConnection(nn.Module):
    """Gated residual bundle used by every Qwen4-Exp decoder block.

    ``mix`` collapses the ``hc_count`` streams to one block input.  The
    per-layer instances also keep the pre-norm bundle and the learned
    per-stream injection logits so the block output can be added back with
    ``combine`` (or fused with the next mix in ``combine_and_mix``).
    """

    def __init__(self,
                 hc_count: int,
                 hidden_size: int,
                 hc_lowrank: int,
                 eps: float = 1e-6,
                 use_mix: bool = True,
                 use_combine: bool = True) -> None:
        super().__init__()
        if not (use_mix or use_combine):
            raise ValueError("Qwen4ExpHyperConnection needs mix or combine")
        self.hc_count = hc_count
        self.hidden_size = hidden_size
        self.hc_lowrank = hc_lowrank
        self.use_mix = use_mix
        self.use_combine = use_combine
        hc_dim = hc_count * hidden_size
        self.hc_norm = GroupedRMSNorm(hc_dim,
                                      group_size=hidden_size,
                                      eps=eps)
        self.input_mix_weight_down = nn.Linear(hc_dim,
                                               hc_lowrank,
                                               bias=False,
                                               dtype=torch.float16)
        self.input_mix_weight_up = nn.Linear(hc_lowrank,
                                             hc_dim,
                                             bias=False,
                                             dtype=torch.float16)
        if use_combine:
            self.block_inject_weight = nn.Linear(hc_dim,
                                                 hc_count,
                                                 bias=False,
                                                 dtype=torch.float16)

    def _normed(self, hyper_input: torch.Tensor) -> torch.Tensor:
        return self.hc_norm(hyper_input)

    def _mix_normed(
        self, hyper_input: torch.Tensor, normed: torch.Tensor
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, Optional[torch.Tensor]]]:
        hc, hs = self.hc_count, self.hidden_size
        down = self.input_mix_weight_down(normed)
        gate = F.silu(down / hc)
        gate = self.input_mix_weight_up(gate)
        gate = torch.sigmoid(gate).unflatten(-1, (hc, hs))
        mixed = (gate * normed.unflatten(-1, (hc, hs))).mean(dim=-2)
        injection_logits = (self.block_inject_weight(normed)
                            if self.use_combine else None)
        return mixed, (hyper_input, injection_logits)

    def mix(
        self, hyper_input: torch.Tensor
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, Optional[torch.Tensor]]]:
        normed = self._normed(hyper_input)
        return self._mix_normed(hyper_input, normed)

    def combine(
        self,
        block_output: torch.Tensor,
        residual: Tuple[torch.Tensor, Optional[torch.Tensor]],
    ) -> torch.Tensor:
        hyper_input, injection_logits = residual
        if injection_logits is None:
            raise RuntimeError("Hyper-Connection combine is missing "
                               "injection logits")
        hc, hs = self.hc_count, self.hidden_size
        streams = hyper_input.unflatten(-1, (hc, hs))
        inject_gate = 2.0 * torch.sigmoid(injection_logits / hc)
        return (streams + block_output.unsqueeze(-2)
                * inject_gate.unsqueeze(-1)).flatten(-2)

    def combine_and_mix(
        self,
        block_output: torch.Tensor,
        residual: Tuple[torch.Tensor, Optional[torch.Tensor]],
    ) -> Tuple[torch.Tensor, torch.Tensor,
               Tuple[torch.Tensor, Optional[torch.Tensor]]]:
        hidden_states = self.combine(block_output, residual)
        mixed, new_residual = self.mix(hidden_states)
        return hidden_states, mixed, new_residual


# ---------------------------------------------------------------------------
# n-gram PLE side path
# ---------------------------------------------------------------------------


class _PleEmbeddingMetadata(nn.Module):
    """Small checkpoint-backed buffers for the runtime n-gram lookup."""

    def __init__(self, ple: PleConfig, hc_dim: int) -> None:
        super().__init__()
        self.register_buffer(
            "layer_multipliers",
            torch.zeros(ple.ngram_size, dtype=torch.int64))
        self.register_buffer(
            "ngram_heads_offsets",
            torch.zeros(ple.num_ngram_heads, dtype=torch.int64))
        self.register_buffer(
            "ngram_heads_vocab_sizes",
            torch.zeros(ple.num_ngram_heads, dtype=torch.int64))
        self.ngram_embedding = nn.Module()
        self.ngram_embedding.register_buffer(
            "weight_scale_2", torch.ones(1, dtype=torch.float32))


class Qwen4ExpPleLayer(nn.Module):
    """N-gram embedding gate plus dilated depthwise short convolution."""

    def __init__(self, config: ModelConfig, module_prefix: str) -> None:
        super().__init__()
        ple = config.ple_cfg
        if ple is None:
            raise ValueError("Qwen4ExpPleLayer requires config.ple_cfg")
        self.ple = ple
        self.hidden_size = config.hidden_size
        self.hc_dim = config.hc_count * config.hidden_size
        self.key_proj = make_linear(config,
                                    ple.embed_dim,
                                    self.hc_dim,
                                    module_name=f"{module_prefix}.key_proj")
        self.value_proj = make_linear(config,
                                      ple.embed_dim,
                                      self.hidden_size,
                                      module_name=f"{module_prefix}.value_proj")
        self.norm_key = GroupedRMSNorm(self.hc_dim,
                                       group_size=self.hidden_size,
                                       eps=config.rms_norm_eps)
        self.norm_query = GroupedRMSNorm(self.hc_dim,
                                         group_size=self.hidden_size,
                                         eps=config.rms_norm_eps)
        self.norm_conv = GroupedRMSNorm(self.hc_dim,
                                        group_size=self.hidden_size,
                                        eps=config.rms_norm_eps)
        self.conv1d = nn.Conv1d(self.hc_dim,
                                self.hc_dim,
                                ple.conv_kernel_size,
                                groups=self.hc_dim,
                                bias=False,
                                dtype=torch.float16)
        nn.init.zeros_(self.conv1d.weight)
        self.ple_embedding = _PleEmbeddingMetadata(ple, self.hc_dim)

    def _gate(self, key: torch.Tensor, value: torch.Tensor,
              hidden_states: torch.Tensor) -> Tuple[torch.Tensor,
                                                    torch.Tensor]:
        """Port of vLLM's ``ple_gate`` reference math.

        ``key`` and ``hidden_states`` are ``[T, hc_count * H]``; the dot
        product is formed per stream over ``H`` values, producing one gate
        scalar per ``(token, stream)``.
        """
        hc = self.hc_dim // self.hidden_size
        hs = self.hidden_size
        key_streams = key.unflatten(-1, (hc, hs))
        hidden_streams = hidden_states.unflatten(-1, (hc, hs))
        k = key_streams.float()
        q = hidden_streams.float()
        nk = (1.0 + self.norm_key.weight.float()).view(hc, hs)
        nq = (1.0 + self.norm_query.weight.float()).view(hc, hs)
        k_n = (k * torch.rsqrt(k.pow(2).mean(-1, keepdim=True)
                               + self.norm_key.variance_epsilon) * nk)
        q_n = (q * torch.rsqrt(q.pow(2).mean(-1, keepdim=True)
                               + self.norm_query.variance_epsilon) * nq)
        dot = (k_n * q_n).sum(-1, keepdim=False) / (hs**0.5)
        sign = torch.sign(dot)
        magnitude = torch.sqrt(torch.clamp(dot.abs(), min=1e-6))
        gate = torch.sigmoid(sign * magnitude).to(value.dtype)
        gated = gate.unsqueeze(-1) * value.unsqueeze(-2)
        gated = gated.flatten(-2)
        conv = self.norm_conv(gated)
        return gated, conv

    def forward(self,
                hidden_states: torch.Tensor,
                ple_embeds: torch.Tensor,
                ple_conv_state: torch.Tensor,
                state_indices: torch.Tensor,
                query_start_offsets: torch.Tensor,
                query_lengths: torch.Tensor) -> Tuple[torch.Tensor,
                                                      torch.Tensor]:
        key = self.key_proj(ple_embeds)
        value = self.value_proj(ple_embeds)
        gated, conv_input = self._gate(key, value, hidden_states)
        token_count = hidden_states.shape[0]
        batch = query_lengths.shape[0]
        if token_count % batch != 0:
            raise ValueError(
                "Qwen4-Exp PLE requires equal query lengths per context step; "
                f"got {token_count} tokens for batch {batch}.")
        seq_len = token_count // batch
        state_len = self.ple.conv_state_len
        slots = state_indices.to(torch.int64)
        active = torch.index_select(ple_conv_state, 0, slots)
        conv_x = conv_input.reshape(batch, seq_len,
                                    self.hc_dim).transpose(1, 2)
        history = torch.cat((active, conv_x), dim=2)
        conv = torch.nn.functional.conv1d(history,
                                          self.conv1d.weight,
                                          bias=None,
                                          stride=1,
                                          padding=0,
                                          dilation=self.ple.ngram_size,
                                          groups=self.hc_dim)
        conv = conv.transpose(1, 2).reshape(token_count, self.hc_dim)
        out = (gated + torch.nn.functional.silu(conv)) + hidden_states
        state_out = ple_conv_state.clone()
        state_out.index_copy_(0, slots, history[:, :, -state_len:])
        return out, state_out


# ---------------------------------------------------------------------------
# QSA full attention
# ---------------------------------------------------------------------------


class _RawNormWeight(nn.Module):
    """Checkpoint holder for a Gemma RMSNorm weight stored as a delta."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(dim, dtype=torch.float16))


class Qwen4ExpQsaAttention(nn.Module):
    """Qwen Sparse Attention layer with its block-compressed indexer."""

    def __init__(self, config: ModelConfig, layer_idx: int) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.attention_scale = config.attention_scaling
        self.rms_norm_eps = config.rms_norm_eps
        qsa = config.qsa_cfg
        if qsa is None:
            raise ValueError("Qwen4ExpQsaAttention requires config.qsa_cfg")
        self.indexer_n_heads = qsa.indexer_n_heads
        self.indexer_head_dim = qsa.indexer_head_dim
        self.indexer_budget = qsa.indexer_budget
        self.indexer_compress_ratio = qsa.indexer_compress_ratio
        module_prefix = f"layers.{layer_idx}.self_attn"
        self.q_proj = make_linear(config,
                                  config.hidden_size,
                                  self.num_heads * self.head_dim * 2,
                                  bias=config.attention_bias,
                                  module_name=f"{module_prefix}.q_proj",
                                  tp_mode=TPMode.COL)
        self.k_proj = make_linear(config,
                                  config.hidden_size,
                                  self.num_kv_heads * self.head_dim,
                                  bias=config.attention_bias,
                                  module_name=f"{module_prefix}.k_proj",
                                  tp_mode=TPMode.COL)
        self.v_proj = make_linear(config,
                                  config.hidden_size,
                                  self.num_kv_heads * self.head_dim,
                                  bias=config.attention_bias,
                                  module_name=f"{module_prefix}.v_proj",
                                  tp_mode=TPMode.COL)
        self.o_proj = make_linear(config,
                                  self.num_heads * self.head_dim,
                                  config.hidden_size,
                                  module_name=f"{module_prefix}.o_proj",
                                  tp_mode=TPMode.ROW)
        self.q_norm = _RawNormWeight(self.head_dim)
        self.k_norm = _RawNormWeight(self.head_dim)
        self.indexer = nn.Module()
        self.indexer.index_qk_proj = make_linear(
            config,
            config.hidden_size,
            (self.indexer_n_heads + 1) * self.indexer_head_dim,
            module_name=f"{module_prefix}.indexer.index_qk_proj",
            tp_mode=TPMode.REPLICATED)
        self.indexer.q_layernorm = _RawNormWeight(self.indexer_head_dim)
        self.indexer.k_layernorm = _RawNormWeight(self.indexer_head_dim)

    def cache_export_gammas(self) -> None:
        """Freeze the four RMSNorm gamma vectors for ONNX export.

        ``torch.export`` turns parameter tensors into fake tensors, so
        ``Tensor.tolist()`` inside the traced forward would produce symbolic
        floats that the plugin schema cannot accept.  The weights are final
        before export, so snapshot them as Python floats here.
        """

        def _values(weight: torch.Tensor, fold: bool) -> List[float]:
            gamma = weight.detach().float().cpu()
            if fold:
                gamma = 1.0 + gamma
            return gamma.tolist()

        self._export_q_gamma = _values(self.q_norm.weight, True)
        self._export_k_gamma = _values(self.k_norm.weight, True)
        self._export_indexer_q_gamma = _values(
            self.indexer.q_layernorm.weight, False)
        self._export_indexer_k_gamma = _values(
            self.indexer.k_layernorm.weight, False)

    def _gamma(self, weight: torch.Tensor, fold: bool) -> List[float]:
        gamma = weight.detach().float()
        if fold:
            gamma = 1.0 + gamma
        return gamma.tolist()

    def _gamma_arg(self, cached_attr: str, weight: torch.Tensor,
                   fold: bool) -> List[float]:
        """Return the export-time gamma snapshot when ``prepare_for_export`` ran.

        The fallback path runs on real checkpoint weights (unit tests, eager
        use); it must not be evaluated while tracing, where ``tolist`` on the
        fake parameter produces symbolic values the plugin schema rejects.
        """
        cached = getattr(self, cached_attr, None)
        return cached if cached is not None else self._gamma(weight, fold)

    def forward_ragged(
        self,
        hidden_states: torch.Tensor,
        past_key_value: torch.Tensor,
        qsa_rope_table: torch.Tensor,
        kvcache_start_index: torch.Tensor,
        query_start_offsets: torch.Tensor,
        query_lengths: torch.Tensor,
        past_lengths: torch.Tensor,
        kv_page_table: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        del query_start_offsets
        num_tokens = hidden_states.shape[0]
        batch = query_lengths.shape[0]
        if num_tokens % batch != 0:
            raise ValueError(
                "Qwen4-Exp QSA prefill currently requires equal query lengths "
                "per context step; heterogeneous or chunked prefill is not "
                "implemented.")
        seq_len = num_tokens // batch
        q_output = self.q_proj(hidden_states).reshape(num_tokens,
                                                      self.num_heads,
                                                      self.head_dim * 2)
        query_states, gate_states = q_output.chunk(2, dim=-1)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)
        qkv = torch.cat([query_states.reshape(num_tokens, -1),
                         key_states, value_states], dim=-1)
        qkv = qkv.reshape(batch, seq_len, -1)
        index_qk = self.indexer.index_qk_proj(hidden_states).reshape(
            batch, seq_len, -1)
        context_lengths = query_lengths + past_lengths
        attn_output, present = qsa_attention_plugin(
            qkv,
            index_qk,
            past_key_value,
            context_lengths,
            qsa_rope_table,
            kvcache_start_index,
            kv_page_table,
            num_q_heads=self.num_heads,
            num_kv_heads=self.num_kv_heads,
            head_size=self.head_dim,
            indexer_n_heads=self.indexer_n_heads,
            indexer_head_dim=self.indexer_head_dim,
            indexer_budget=self.indexer_budget,
            indexer_compress_ratio=self.indexer_compress_ratio,
            attention_scale=self.attention_scale,
            rms_norm_eps=self.rms_norm_eps,
            q_norm_gamma=self._gamma_arg("_export_q_gamma", self.q_norm.weight,
                                         True),
            k_norm_gamma=self._gamma_arg("_export_k_gamma", self.k_norm.weight,
                                         True),
            indexer_q_norm_gamma=self._gamma_arg(
                "_export_indexer_q_gamma",
                self.indexer.q_layernorm.weight, False),
            indexer_k_norm_gamma=self._gamma_arg(
                "_export_indexer_k_gamma",
                self.indexer.k_layernorm.weight, False),
        )
        attn_output = attn_output.reshape(num_tokens,
                                          self.num_heads * self.head_dim)
        gate_states = gate_states.reshape(num_tokens, -1)
        attn_output = attn_output * torch.sigmoid(gate_states)
        return self.o_proj(attn_output), present


# ---------------------------------------------------------------------------
# Decoder layer
# ---------------------------------------------------------------------------


class Qwen4ExpDecoderLayer(nn.Module):
    """Hyper-Connection decoder block with a GDN or QSA mixer and MoE FFN."""

    def __init__(self, config: ModelConfig, layer_idx: int,
                 layer_type: str) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.layer_type = layer_type
        self.hc_count = config.hc_count
        self.hidden_size = config.hidden_size
        self.hc_dim = self.hc_count * self.hidden_size
        self.attn_hyper_connection = Qwen4ExpHyperConnection(
            self.hc_count,
            self.hidden_size,
            config.hc_lowrank,
            eps=config.rms_norm_eps)
        self.mlp_hyper_connection = Qwen4ExpHyperConnection(
            self.hc_count,
            self.hidden_size,
            config.hc_lowrank,
            eps=config.rms_norm_eps)
        if layer_type == LAYER_GDN:
            self.linear_attn = GdnMixer(config,
                                        config.gdn_cfg,
                                        f"layers.{layer_idx}.linear_attn")
        else:
            self.self_attn = Qwen4ExpQsaAttention(config, layer_idx)
        self.mlp = Qwen3SparseMoeBlock(config,
                                       module_prefix=f"layers.{layer_idx}.mlp")
        self.ple: Optional[Qwen4ExpPleLayer] = None
        if config.ple_cfg is not None and (
                layer_idx + 1) in config.ple_cfg.layer_ids:
            self.ple = Qwen4ExpPleLayer(config,
                                        f"layers.{layer_idx}.ple")

    def forward_ragged(
        self,
        hidden_states: torch.Tensor,
        prev_block_output: Optional[torch.Tensor],
        prev_residual: Optional[Tuple[torch.Tensor, torch.Tensor]],
        *,
        ple_embeds: torch.Tensor,
        query_start_offsets: torch.Tensor,
        query_lengths: torch.Tensor,
        past_lengths: torch.Tensor,
        state_indices: torch.Tensor,
        execution_phase_marker: torch.Tensor,
        context_sequence_count_carrier: torch.Tensor,
        kv_page_table: torch.Tensor,
        qsa_rope_table: torch.Tensor,
        kvcache_start_index: torch.Tensor,
        past_key_value: Optional[torch.Tensor] = None,
        conv_state: Optional[torch.Tensor] = None,
        recurrent_state: Optional[torch.Tensor] = None,
        ple_conv_state: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor,
               Tuple[torch.Tensor, torch.Tensor], Optional[torch.Tensor],
               Optional[torch.Tensor], Optional[torch.Tensor],
               Optional[torch.Tensor]]:
        if self.ple is not None:
            if prev_block_output is not None:
                hidden_states = self.attn_hyper_connection.combine(
                    prev_block_output, prev_residual)
                prev_block_output = None
                prev_residual = None
            hidden_states, present_ple = self.ple(
                hidden_states, ple_embeds, ple_conv_state, state_indices,
                query_start_offsets, query_lengths)
            block_input, residual = self.attn_hyper_connection.mix(
                hidden_states)
        else:
            present_ple = None
            if prev_block_output is not None:
                (hidden_states, block_input,
                 residual) = self.attn_hyper_connection.combine_and_mix(
                     prev_block_output, prev_residual)
            else:
                block_input, residual = self.attn_hyper_connection.mix(
                    hidden_states)

        present_kv = present_conv = present_recurrent = None
        if self.layer_type == LAYER_GDN:
            (mixer_out, present_conv, present_recurrent, _, _) = (
                self.linear_attn.forward_ragged(
                    block_input,
                    conv_state,
                    recurrent_state,
                    query_start_offsets,
                    query_lengths,
                    state_indices,
                    execution_phase_marker,
                    context_sequence_count_carrier,
                    None,
                    None,
                    False))
            mixer_out = mixer_out.reshape(-1, self.hidden_size)
        else:
            mixer_out, present_kv = self.self_attn.forward_ragged(
                block_input,
                past_key_value,
                qsa_rope_table,
                kvcache_start_index,
                query_start_offsets,
                query_lengths,
                past_lengths,
                kv_page_table)

        hidden_states, block_input, residual = (
            self.mlp_hyper_connection.combine_and_mix(mixer_out, residual))
        mlp_out = self.mlp(block_input)
        return (hidden_states, mlp_out, residual, present_kv, present_conv,
                present_recurrent, present_ple)


# ---------------------------------------------------------------------------
# Backbone
# ---------------------------------------------------------------------------


class Qwen4ExpBackbone(nn.Module):
    """Hybrid Qwen4-Exp text decoder."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.hc_count = config.hc_count
        self.hidden_size = config.hidden_size
        self.embed_tokens = nn.Embedding(config.vocab_size,
                                         config.hidden_size)
        self.layers = nn.ModuleList([
            Qwen4ExpDecoderLayer(config, i, layer_type)
            for i, layer_type in enumerate(config.layer_types)
        ])
        self.hyper_connection_mixer = Qwen4ExpHyperConnection(
            self.hc_count,
            self.hidden_size,
            config.hc_lowrank,
            eps=config.rms_norm_eps,
            use_mix=True,
            use_combine=False)
        self.layer_types = config.layer_types

    def forward_ragged(
        self,
        inputs_embeds: torch.Tensor,
        ple_embeds: torch.Tensor,
        past_key_values: Tuple[torch.Tensor, ...],
        qsa_rope_table: torch.Tensor,
        kvcache_start_index: torch.Tensor,
        query_start_offsets: torch.Tensor,
        query_lengths: torch.Tensor,
        past_lengths: torch.Tensor,
        state_indices: torch.Tensor,
        execution_phase_marker: torch.Tensor,
        context_sequence_count_carrier: torch.Tensor,
        kv_page_table: torch.Tensor,
        conv_states: Tuple[torch.Tensor, ...],
        recurrent_states: Tuple[torch.Tensor, ...],
        ple_conv_state: torch.Tensor,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, ...],
               Tuple[torch.Tensor, ...], Tuple[torch.Tensor, ...],
               torch.Tensor]:
        hidden_states = inputs_embeds.repeat(1, self.hc_count)
        block_output: Optional[torch.Tensor] = None
        residual: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
        present_key_values: List[torch.Tensor] = []
        present_conv_states: List[torch.Tensor] = []
        present_recurrent_states: List[torch.Tensor] = []
        present_ple: Optional[torch.Tensor] = None
        attn_idx = gdn_idx = 0
        ple_state = ple_conv_state
        for layer in self.layers:
            if layer.layer_type == LAYER_GDN:
                layer_kwargs = {
                    "conv_state": conv_states[gdn_idx],
                    "recurrent_state": recurrent_states[gdn_idx],
                }
                gdn_idx += 1
            else:
                layer_kwargs = {"past_key_value": past_key_values[attn_idx]}
                attn_idx += 1
            (hidden_states, block_output, residual, present_kv, present_conv,
             present_recurrent, present_ple) = layer.forward_ragged(
                 hidden_states,
                 block_output,
                 residual,
                 ple_embeds=ple_embeds,
                 query_start_offsets=query_start_offsets,
                 query_lengths=query_lengths,
                 past_lengths=past_lengths,
                 state_indices=state_indices,
                 execution_phase_marker=execution_phase_marker,
                 context_sequence_count_carrier=context_sequence_count_carrier,
                 kv_page_table=kv_page_table,
                 qsa_rope_table=qsa_rope_table,
                 kvcache_start_index=kvcache_start_index,
                 ple_conv_state=ple_state,
                 **layer_kwargs)
            if present_kv is not None:
                present_key_values.append(present_kv)
            if present_conv is not None:
                present_conv_states.append(present_conv)
            if present_recurrent is not None:
                present_recurrent_states.append(present_recurrent)
            if present_ple is not None:
                ple_state = present_ple
        if block_output is None or residual is None:
            raise RuntimeError("Qwen4ExpBackbone requires at least one layer")
        hidden_states = self.layers[-1].mlp_hyper_connection.combine(
            block_output, residual)
        sample_hidden, _ = self.hyper_connection_mixer.mix(hidden_states)
        return (sample_hidden, tuple(present_key_values),
                tuple(present_conv_states), tuple(present_recurrent_states),
                ple_state)


# ---------------------------------------------------------------------------
# Causal LM wrapper and ONNX export contract
# ---------------------------------------------------------------------------


def _make_flat_wrapper_qwen4_exp(model: nn.Module, Na: int, Ng: int):
    param_names: List[str] = (
        ["inputs_embeds", "ple_embeds"] +
        [f"past_key_values_{i}" for i in range(Na)] + [
            "qsa_rope_table", "kvcache_start_index", "query_start_offsets",
            "query_lengths", "past_lengths", "state_indices",
            "execution_phase_marker", "context_sequence_count_carrier",
            "kv_page_table", "logits_indices"
        ] + [f"conv_state_{i}" for i in range(Ng)] +
        [f"recurrent_state_{i}" for i in range(Ng)] + ["ple_conv_state"])
    past_kv_tuple = "({},)".format(", ".join(
        f"past_key_values_{i}" for i in range(Na)))
    conv_tuple = "({},)".format(", ".join(
        f"conv_state_{i}" for i in range(Ng)))
    rec_tuple = "({},)".format(", ".join(
        f"recurrent_state_{i}" for i in range(Ng)))
    body = (
        f"    logits, present_key_values, present_conv_states, "
        f"present_recurrent_states, present_ple_conv = self._model.forward_ragged(\n"
        f"        inputs_embeds, ple_embeds, {past_kv_tuple}, "
        f"qsa_rope_table, kvcache_start_index, query_start_offsets, "
        f"query_lengths, past_lengths, state_indices, execution_phase_marker, "
        f"context_sequence_count_carrier, kv_page_table, logits_indices, "
        f"{conv_tuple}, {rec_tuple}, ple_conv_state)\n"
        f"    return ((logits,) + tuple(present_key_values)\n"
        f"            + tuple(present_conv_states)\n"
        f"            + tuple(present_recurrent_states)\n"
        f"            + (present_ple_conv,))\n")
    src = "def _forward(self, {}):\n{}".format(", ".join(param_names), body)
    globs: dict = {}
    exec(src, globs)  # noqa: S102

    class _Wrapper(nn.Module):

        def __init__(self, m: nn.Module) -> None:
            super().__init__()
            self._model = m

    _Wrapper.forward = globs["_forward"]
    return _Wrapper(model)


class Qwen4ExpCausalLM(nn.Module):
    """Text-only Qwen3.8-Flash-Next causal LM."""

    RECURRENT_STATE_DTYPE = torch.float32
    CONV_STATE_DTYPE = torch.float16

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.model = Qwen4ExpBackbone(config)
        self.lm_head = make_linear(config,
                                   config.hidden_size,
                                   config.vocab_size,
                                   bias=False,
                                   module_name="lm_head")

    def prepare_for_export(self) -> None:
        """Snapshot non-tensor plugin attributes before ONNX tracing."""
        for layer in self.model.layers:
            attn = getattr(layer, "self_attn", None)
            if attn is not None:
                attn.cache_export_gammas()

    def materialize_checkpoint_defaults(self, device: str) -> None:
        """Fill buffers a checkpoint omits (GDN conv bias) under meta init."""
        for layer in self.model.layers:
            mixer = getattr(layer, "linear_attn", None)
            if mixer is None or mixer.conv1d.bias.device.type != "meta":
                continue
            bias = mixer.conv1d.bias
            mixer.conv1d.bias = torch.zeros(bias.shape,
                                            dtype=bias.dtype,
                                            device=device)

    def forward_ragged(
        self,
        inputs_embeds: torch.Tensor,
        ple_embeds: torch.Tensor,
        past_key_values: Tuple[torch.Tensor, ...],
        qsa_rope_table: torch.Tensor,
        kvcache_start_index: torch.Tensor,
        query_start_offsets: torch.Tensor,
        query_lengths: torch.Tensor,
        past_lengths: torch.Tensor,
        state_indices: torch.Tensor,
        execution_phase_marker: torch.Tensor,
        context_sequence_count_carrier: torch.Tensor,
        kv_page_table: torch.Tensor,
        logits_indices: torch.Tensor,
        conv_states: Tuple[torch.Tensor, ...],
        recurrent_states: Tuple[torch.Tensor, ...],
        ple_conv_state: torch.Tensor,
    ) -> Tuple:
        (hidden_states, present_kv, present_conv, present_recurrent,
         present_ple) = self.model.forward_ragged(
             inputs_embeds,
             ple_embeds,
             past_key_values,
             qsa_rope_table,
             kvcache_start_index,
             query_start_offsets,
             query_lengths,
             past_lengths,
             state_indices,
             execution_phase_marker,
             context_sequence_count_carrier,
             kv_page_table,
             conv_states,
             recurrent_states,
             ple_conv_state)
        selected = torch.index_select(hidden_states, 0, logits_indices)
        logits = self.lm_head(selected).to(torch.float32)
        return logits, present_kv, present_conv, present_recurrent, present_ple

    def onnx_export_spec(self) -> OnnxSpec:
        config = self.config
        qsa = config.qsa_cfg
        ple = config.ple_cfg
        if qsa is None or ple is None:
            raise ValueError("Qwen4ExpCausalLM requires qsa_cfg and ple_cfg")
        Na = config.num_attn_layers
        Ng = config.num_gdn_layers
        gc = config.gdn_cfg
        if gc is None:
            raise ValueError("Qwen4ExpCausalLM requires gdn_cfg")
        device = next(itertools.chain(self.parameters(),
                                      self.buffers())).device
        dtype16 = torch.float16
        batch = _BATCH_SIZE
        seq = _SEQ_LEN
        tokens = batch * seq
        pool_rows = 4
        rotary_dim = int(config.head_dim * config.partial_rotary_factor)
        kv_dim = config.head_dim + qsa.indexer_head_dim
        past_key_values = [
            torch.zeros(2,
                        2,
                        KV_PAGE_SIZE,
                        config.num_key_value_heads,
                        kv_dim,
                        dtype=dtype16,
                        device=device) for _ in range(Na)
        ]
        inputs_embeds = torch.zeros(tokens,
                                    config.hidden_size,
                                    dtype=dtype16,
                                    device=device)
        ple_embeds = torch.zeros(tokens, ple.embed_dim,
                                 dtype=dtype16,
                                 device=device)
        qsa_rope_table = torch.zeros(1,
                                     _MAX_POS,
                                     rotary_dim,
                                     dtype=torch.float32,
                                     device=device)
        kvcache_start_index = torch.zeros(0,
                                          dtype=torch.int32,
                                          device=device)
        query_start_offsets = torch.arange(0,
                                           tokens + 1,
                                           seq,
                                           dtype=torch.int32,
                                           device=device)
        query_lengths = torch.full((batch, ),
                                   seq,
                                   dtype=torch.int32,
                                   device=device)
        past_lengths = torch.zeros(batch, dtype=torch.int32, device=device)
        state_indices = torch.tensor([2, 0],
                                     dtype=torch.int32,
                                     device=device)
        execution_phase_marker = torch.zeros(2,
                                             dtype=torch.int32,
                                             device=device)
        context_sequence_count_carrier = torch.zeros(batch,
                                                     dtype=torch.int32,
                                                     device=device)
        kv_page_table = torch.zeros(batch,
                                    2,
                                    2,
                                    dtype=torch.int32,
                                    device=device)
        logits_indices = (query_start_offsets[1:].to(torch.int64) - 1)
        conv_states = [
            torch.zeros(pool_rows,
                        gc.conv_dim,
                        gc.conv_kernel,
                        dtype=self.CONV_STATE_DTYPE,
                        device=device) for _ in range(Ng)
        ]
        recurrent_states = [
            torch.zeros(pool_rows,
                        gc.num_value_heads,
                        gc.key_head_dim,
                        gc.value_head_dim,
                        dtype=self.RECURRENT_STATE_DTYPE,
                        device=device) for _ in range(Ng)
        ]
        ple_conv_state = torch.zeros(pool_rows,
                                     config.hc_count * config.hidden_size,
                                     ple.conv_state_len,
                                     dtype=self.CONV_STATE_DTYPE,
                                     device=device)
        args = (inputs_embeds, ple_embeds, *past_key_values, qsa_rope_table,
                kvcache_start_index, query_start_offsets, query_lengths,
                past_lengths, state_indices, execution_phase_marker,
                context_sequence_count_carrier, kv_page_table, logits_indices,
                *conv_states, *recurrent_states, ple_conv_state)
        input_names = (
            ["inputs_embeds", "ple_embeds"] +
            [f"past_key_values_{i}" for i in range(Na)] + [
                "qsa_rope_table", "kvcache_start_index",
                "query_start_offsets", "query_lengths", "past_lengths",
                "state_indices", "execution_phase_marker",
                "context_sequence_count_carrier", "kv_page_table",
                "logits_indices"
            ] + [f"conv_state_{i}" for i in range(Ng)] +
            [f"recurrent_state_{i}" for i in range(Ng)] +
            ["ple_conv_state"])
        output_names = (["logits"] +
                        [f"present_key_values_{i}" for i in range(Na)] +
                        [f"present_conv_state_{i}" for i in range(Ng)] +
                        [f"present_recurrent_state_{i}" for i in range(Ng)] +
                        ["present_ple_conv_state"])

        tokens_dim = torch.export.Dim("physical_tokens", min=1, max=8_388_608)
        sequences = torch.export.Dim("num_sequences", min=1, max=256)
        state_pool = torch.export.Dim("state_pool_rows", min=1, max=256)
        max_pages = torch.export.Dim("max_pages_per_seq", min=1, max=32768)
        num_pages = torch.export.Dim("num_pages", min=1, max=1048576)
        rope_positions = torch.export.Dim("qsa_rope_positions",
                                          min=1,
                                          max=1048576)
        start_len = torch.export.Dim("qsa_start_len", min=0, max=256)
        phase_extent = torch.export.Dim("execution_phase_extent",
                                        min=1,
                                        max=8)
        context_sequences = torch.export.Dim("num_context_sequences",
                                             min=0,
                                             max=256)
        all_shapes: list = [{0: tokens_dim}, {0: tokens_dim}]
        all_shapes.extend({1: num_pages} for _ in range(Na))
        all_shapes.extend([
            {
                1: rope_positions
            },
            {
                0: start_len
            },
            {
                0: sequences + 1
            },
            {
                0: sequences
            },
            {
                0: sequences
            },
            {
                0: sequences
            },
            {
                0: phase_extent
            },
            {
                0: context_sequences
            },
            {
                0: sequences,
                2: max_pages
            },
            {
                0: sequences
            },
        ])
        all_shapes.extend({0: state_pool} for _ in range(Ng))
        all_shapes.extend({0: state_pool} for _ in range(Ng))
        all_shapes.append({0: state_pool})

        wrapped = _make_flat_wrapper_qwen4_exp(self, Na, Ng)
        wrapped.eval()
        return OnnxSpec(wrapped=wrapped,
                        args=args,
                        input_names=input_names,
                        output_names=output_names,
                        dynamic_shapes=all_shapes)
