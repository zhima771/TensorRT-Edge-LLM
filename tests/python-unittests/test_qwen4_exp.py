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
"""CPU unit tests for the Qwen3.8-Flash-Next text model pieces.

The full checkpoint is ~99 GiB, so these tests build a tiny synthetic config
and exercise the Hyper-Connection, PLE gate/short-conv and the token-major
forward contract without touching the real weights.
"""

from __future__ import annotations

import math
import os
import sys

import onnx
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from tensorrt_edgellm.checkpoint.loader import \
    apply_all_repacking  # noqa: E402
from tensorrt_edgellm.config import GdnConfig  # noqa: E402
from tensorrt_edgellm.config import (LAYER_GDN, ModelConfig, PleConfig,
                                     QsaConfig, QuantConfig)
from tensorrt_edgellm.model import load_model_config  # noqa: E402
from tensorrt_edgellm.models.qwen4_exp.modeling_qwen4_exp import (  # noqa: E402
    GroupedRMSNorm, Qwen4ExpCausalLM, Qwen4ExpHyperConnection,
    Qwen4ExpPleLayer)
from tensorrt_edgellm.onnx.export import _export_model  # noqa: E402


def _tiny_config() -> ModelConfig:
    return ModelConfig(
        model_type="qwen4_exp_text",
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=1,
        intermediate_size=64,
        head_dim=16,
        rms_norm_eps=1e-6,
        vocab_size=128,
        rope_theta=10000.0,
        max_position_embeddings=128,
        default_attention_scale=1.0 / math.sqrt(16),
        partial_rotary_factor=0.5,
        layer_types=[LAYER_GDN, "full_attention"],
        gdn_cfg=GdnConfig(num_key_heads=2,
                          num_value_heads=4,
                          key_head_dim=8,
                          value_head_dim=8,
                          conv_kernel=4),
        qsa_cfg=QsaConfig(indexer_n_heads=2,
                          indexer_head_dim=8,
                          indexer_kv_heads=1,
                          indexer_budget=8,
                          indexer_compress_ratio=4),
        ple_cfg=PleConfig(layer_ids=[2],
                          ngram_size=3,
                          heads_per_ngram=2,
                          embed_dim=32,
                          conv_kernel_size=2,
                          ngram_vocab_size_base=100,
                          vocab_size_divisible_by=4,
                          split_parts=4,
                          embedding_dtype="nvfp4"),
        hc_count=2,
        hc_lowrank=8,
        gdn_output_gate="sigmoid",
        attn_output_gate=True,
        num_experts=4,
        num_experts_per_tok=2,
        moe_intermediate_size=64,
        moe_shared_expert_intermediate_size=64,
        quant=QuantConfig(),
    )


def _ref_grouped_rms_norm(x: torch.Tensor, weight: torch.Tensor,
                          group_size: int, eps: float) -> torch.Tensor:
    dtype = x.dtype
    xf = x.float()
    groups = xf.unflatten(-1, (-1, group_size))
    groups = groups * torch.rsqrt(groups.pow(2).mean(-1, keepdim=True) + eps)
    return (groups.flatten(-2) * (1.0 + weight.float())).to(dtype)


def test_grouped_rms_norm_matches_reference():
    torch.manual_seed(0)
    norm = GroupedRMSNorm(8, group_size=4, eps=1e-6)
    norm.weight.data.normal_()
    x = torch.randn(3, 8, dtype=torch.float16)
    got = norm(x)
    want = _ref_grouped_rms_norm(x, norm.weight, 4, 1e-6)
    torch.testing.assert_close(got, want, rtol=1e-2, atol=1e-2)


def test_hyper_connection_mix_combine():
    torch.manual_seed(0)
    hc = Qwen4ExpHyperConnection(2, 8, 4, eps=1e-6)
    for linear in (hc.input_mix_weight_down, hc.input_mix_weight_up,
                   hc.block_inject_weight):
        linear.weight.data.normal_(0.0, 0.02)
    hc.hc_norm.weight.data.normal_()
    x = torch.randn(3, 16, dtype=torch.float16)
    mixed, residual = hc.mix(x)
    normed = _ref_grouped_rms_norm(x, hc.hc_norm.weight, 8, 1e-6)
    down = normed @ hc.input_mix_weight_down.weight.T
    gate = torch.nn.functional.silu(down / 2.0)
    gate = gate @ hc.input_mix_weight_up.weight.T
    want_mixed = (torch.sigmoid(gate).unflatten(-1, (2, 8)) *
                  normed.unflatten(-1, (2, 8))).mean(-2)
    torch.testing.assert_close(mixed, want_mixed, rtol=2e-2, atol=2e-2)
    block = torch.randn(3, 8, dtype=torch.float16)
    combined = hc.combine(block, residual)
    inject = normed @ hc.block_inject_weight.weight.T
    want_combined = (x.unflatten(-1, (2, 8)) + block.unsqueeze(-2) *
                     (2.0 * torch.sigmoid(inject / 2.0)).unsqueeze(-1))
    torch.testing.assert_close(combined,
                               want_combined.flatten(-2),
                               rtol=2e-2,
                               atol=2e-2)


def test_ple_gate_matches_reference():
    torch.manual_seed(0)
    cfg = _tiny_config()
    ple = Qwen4ExpPleLayer(cfg, "layers.1.ple")
    for linear in (ple.key_proj, ple.value_proj):
        linear.weight.data.normal_(0.0, 0.02)
    for norm in (ple.norm_key, ple.norm_query, ple.norm_conv):
        norm.weight.data.normal_()
    hidden = torch.randn(3,
                         cfg.hc_count * cfg.hidden_size,
                         dtype=torch.float16)
    key = torch.randn_like(hidden)
    value = torch.randn(3, cfg.hidden_size, dtype=torch.float16)
    gated, conv = ple._gate(key, value, hidden)

    hc, hs = cfg.hc_count, cfg.hidden_size
    k = key.float().unflatten(-1, (hc, hs))
    q = hidden.float().unflatten(-1, (hc, hs))
    nk = (1.0 + ple.norm_key.weight.float()).view(hc, hs)
    nq = (1.0 + ple.norm_query.weight.float()).view(hc, hs)
    k_n = k * torch.rsqrt(k.pow(2).mean(-1, keepdim=True) + 1e-6) * nk
    q_n = q * torch.rsqrt(q.pow(2).mean(-1, keepdim=True) + 1e-6) * nq
    dot = (k_n * q_n).sum(-1) / math.sqrt(hs)
    gate = torch.sigmoid(
        torch.sign(dot) * torch.sqrt(torch.clamp(dot.abs(), min=1e-6)))
    want_gated = (gate.unsqueeze(-1) * value.float().unsqueeze(-2)).flatten(-2)
    nc = (1.0 + ple.norm_conv.weight.float()).view(hc, hs)
    grouped = want_gated.unflatten(-1, (hc, hs))
    want_conv = (grouped *
                 torch.rsqrt(grouped.pow(2).mean(-1, keepdim=True) + 1e-6) *
                 nc).flatten(-2)
    torch.testing.assert_close(gated.float(), want_gated, rtol=3e-2, atol=3e-2)
    torch.testing.assert_close(conv.float(), want_conv, rtol=3e-2, atol=3e-2)


def test_tiny_model_forward_shapes():
    torch.manual_seed(0)
    cfg = _tiny_config()
    model = Qwen4ExpCausalLM(cfg)
    apply_all_repacking(model)
    tokens, hs, hc = 2, cfg.hidden_size, cfg.hc_count
    dtype = torch.float16
    out = model.forward_ragged(
        torch.randn(tokens, hs, dtype=dtype),
        torch.randn(tokens, cfg.ple_cfg.embed_dim, dtype=dtype),
        (torch.zeros(2,
                     2,
                     128,
                     cfg.num_key_value_heads,
                     cfg.head_dim + cfg.qsa_cfg.indexer_head_dim,
                     dtype=dtype), ),
        torch.randn(1, 8, int(cfg.head_dim * cfg.partial_rotary_factor)),
        torch.zeros(0, dtype=torch.int32),
        torch.tensor([0, 1, 2], dtype=torch.int32),
        torch.tensor([1, 1], dtype=torch.int32),
        torch.zeros(2, dtype=torch.int32),
        torch.tensor([0, 1], dtype=torch.int32),
        torch.zeros(2, dtype=torch.int32),
        torch.zeros(2, dtype=torch.int32),
        torch.zeros(2, 2, 2, dtype=torch.int32),
        torch.tensor([0, 1], dtype=torch.int64),
        (torch.zeros(
            4, cfg.gdn_cfg.conv_dim, cfg.gdn_cfg.conv_kernel, dtype=dtype), ),
        (torch.zeros(4,
                     cfg.gdn_cfg.num_value_heads,
                     cfg.gdn_cfg.key_head_dim,
                     cfg.gdn_cfg.value_head_dim,
                     dtype=torch.float32), ),
        torch.zeros(4, hc * hs, cfg.ple_cfg.conv_state_len, dtype=dtype),
    )
    logits, pkv, pconv, prec, pple = out
    assert logits.shape == (2, cfg.vocab_size)
    assert len(pkv) == cfg.num_attn_layers
    assert len(pconv) == cfg.num_gdn_layers
    assert len(prec) == cfg.num_gdn_layers
    assert pple.shape == (4, hc * hs, cfg.ple_cfg.conv_state_len)


def test_qwen4_exp_tiny_export_abi(tmp_path):
    torch.manual_seed(0)
    model = Qwen4ExpCausalLM(_tiny_config()).eval()
    apply_all_repacking(model)
    model.prepare_for_export()
    output = tmp_path / "model.onnx"
    _export_model(model, str(output))

    graph = onnx.load(str(output), load_external_data=False).graph
    inputs = {tensor.name for tensor in graph.input}
    outputs = {tensor.name for tensor in graph.output}

    # QSA resolves absolute positions from the resident RoPE table, so the
    # token-major positions / attention_sequence_lengths ABI tensors are absent.
    assert "qsa_rope_table" in inputs
    assert "positions" not in inputs
    assert "attention_sequence_lengths" not in inputs
    assert "ple_embeds" in inputs
    assert "ple_conv_state" in inputs

    assert "present_ple_conv_state" in outputs
    assert "present_key_values_0" in outputs


def test_qwen4_exp_checkpoint_config():
    model_dir = os.environ.get(
        "QWEN4_EXP_MODEL_DIR", "/home/sobeyxx/hdd/models/local-inference-lab/"
        "Qwen3.8-Flash-Next-NVFP4")
    if not os.path.isfile(os.path.join(model_dir, "config.json")):
        pytest.skip("Qwen3.8-Flash-Next checkpoint not available")
    cfg = load_model_config(model_dir)
    assert cfg.model_type == "qwen4_exp_text"
    assert cfg.num_hidden_layers == 48
    assert cfg.num_attn_layers == 12
    assert cfg.num_gdn_layers == 36
    assert cfg.gdn_cfg.num_key_heads == 16
    assert cfg.gdn_cfg.num_value_heads == 48
    assert cfg.qsa_cfg.indexer_budget == 2048
    assert cfg.ple_cfg.layer_ids == [2]
    assert cfg.ple_cfg.num_ngram_heads == 16
    assert cfg.ple_cfg.head_dim == 160
    assert cfg.hc_count == 4
    assert cfg.hc_lowrank == 320
    assert cfg.quant.is_mixed_precision
    assert cfg.quant.layer_overrides["layers.0.mlp.experts"] == "nvfp4"
    assert cfg.quant.layer_overrides["layers.11.self_attn.q_proj"] == "mxfp8"


def test_tiny_checkpoint_round_trip(tmp_path):
    """Exercise module naming, HYBRID-free loading and MoE repacking."""
    import json

    from safetensors.torch import save_file

    import tensorrt_edgellm._export_api  # noqa: F401  (registers qwen4_exp)
    from tensorrt_edgellm.model import AutoModel

    cfg = _tiny_config()
    model = Qwen4ExpCausalLM(cfg)
    with torch.no_grad():
        for tensor in (*model.parameters(), *model.buffers()):
            if tensor.dtype.is_floating_point:
                tensor.nan_to_num_(0.0)
    before = {
        "gdn":
        model.model.layers[0].linear_attn.in_proj_qkv.weight.detach().clone(),
        "ple_conv":
        model.model.layers[1].ple.conv1d.weight.detach().clone(),
        "hc_norm":
        model.model.layers[0].attn_hyper_connection.hc_norm.weight.detach().
        clone(),
    }
    save_file({
        k: v.contiguous()
        for k, v in model.state_dict().items()
    }, str(tmp_path / "model.safetensors"))
    text_config = {
        "model_type":
        "qwen4_exp_text",
        "architectures": ["Qwen4ExpForConditionalGeneration"],
        "hidden_size":
        cfg.hidden_size,
        "num_hidden_layers":
        cfg.num_hidden_layers,
        "num_attention_heads":
        cfg.num_attention_heads,
        "num_key_value_heads":
        cfg.num_key_value_heads,
        "intermediate_size":
        cfg.intermediate_size,
        "head_dim":
        cfg.head_dim,
        "rms_norm_eps":
        cfg.rms_norm_eps,
        "vocab_size":
        cfg.vocab_size,
        "rope_theta":
        cfg.rope_theta,
        "max_position_embeddings":
        cfg.max_position_embeddings,
        "partial_rotary_factor":
        cfg.partial_rotary_factor,
        "dtype":
        "float16",
        "layer_types":
        ["linear_attention" if t == LAYER_GDN else t for t in cfg.layer_types],
        "linear_num_key_heads":
        cfg.gdn_cfg.num_key_heads,
        "linear_num_value_heads":
        cfg.gdn_cfg.num_value_heads,
        "linear_key_head_dim":
        cfg.gdn_cfg.key_head_dim,
        "linear_value_head_dim":
        cfg.gdn_cfg.value_head_dim,
        "linear_conv_kernel_dim":
        cfg.gdn_cfg.conv_kernel,
        "indexer_n_heads":
        cfg.qsa_cfg.indexer_n_heads,
        "indexer_head_dim":
        cfg.qsa_cfg.indexer_head_dim,
        "indexer_kv_heads":
        cfg.qsa_cfg.indexer_kv_heads,
        "indexer_budget":
        cfg.qsa_cfg.indexer_budget,
        "indexer_compress_ratio":
        cfg.qsa_cfg.indexer_compress_ratio,
        "ple_layer_ids":
        cfg.ple_cfg.layer_ids,
        "ngram_size":
        cfg.ple_cfg.ngram_size,
        "heads_per_ngram":
        cfg.ple_cfg.heads_per_ngram,
        "ple_embed_dim":
        cfg.ple_cfg.embed_dim,
        "ple_conv_kernel_size":
        cfg.ple_cfg.conv_kernel_size,
        "ngram_vocab_size_base":
        cfg.ple_cfg.ngram_vocab_size_base,
        "make_ngram_vocab_size_divisible_by":
        cfg.ple_cfg.vocab_size_divisible_by,
        "split_ngram_parts":
        cfg.ple_cfg.split_parts,
        "ple_embedding_dtype":
        cfg.ple_cfg.embedding_dtype,
        "hc_count":
        cfg.hc_count,
        "hc_lowrank":
        cfg.hc_lowrank,
        "output_gate_type":
        "sigmoid",
        "num_experts":
        cfg.num_experts,
        "num_experts_per_tok":
        cfg.num_experts_per_tok,
        "moe_intermediate_size":
        cfg.moe_intermediate_size,
        "shared_expert_intermediate_size":
        cfg.moe_shared_expert_intermediate_size,
        "eos_token_id":
        2,
    }
    (tmp_path / "config.json").write_text(
        json.dumps({
            "model_type": "qwen4_exp",
            "architectures": ["Qwen4ExpForConditionalGeneration"],
            "text_config": text_config,
        }))
    loaded = AutoModel.from_pretrained(str(tmp_path), device="cpu")
    torch.testing.assert_close(
        loaded.model.layers[0].linear_attn.in_proj_qkv.weight,
        before["gdn"].to(torch.float16))
    torch.testing.assert_close(loaded.model.layers[1].ple.conv1d.weight,
                               before["ple_conv"].to(torch.float16))
    torch.testing.assert_close(
        loaded.model.layers[0].attn_hyper_connection.hc_norm.weight,
        before["hc_norm"].to(torch.float32))
    assert hasattr(loaded.model.layers[0].mlp, "fc1_weights")


def test_ple_layer_forward_and_state():
    torch.manual_seed(0)
    cfg = _tiny_config()
    ple = Qwen4ExpPleLayer(cfg, "layers.1.ple")
    for linear in (ple.key_proj, ple.value_proj):
        linear.weight.data.normal_(0.0, 0.02)
    for norm in (ple.norm_key, ple.norm_query, ple.norm_conv):
        norm.weight.data.normal_()
    ple.conv1d.weight.data.normal_(0.0, 0.02)
    hidden = torch.randn(2,
                         cfg.hc_count * cfg.hidden_size,
                         dtype=torch.float16)
    embeds = torch.randn(2, cfg.ple_cfg.embed_dim, dtype=torch.float16)
    state = torch.zeros(4,
                        cfg.hc_count * cfg.hidden_size,
                        cfg.ple_cfg.conv_state_len,
                        dtype=torch.float16)
    out, new_state = ple(
        hidden,
        embeds,
        state,
        torch.tensor([0, 1], dtype=torch.int32),
        torch.tensor([0, 1, 2], dtype=torch.int32),
        torch.tensor([1, 1], dtype=torch.int32),
    )
    assert out.shape == hidden.shape
    assert new_state.shape == state.shape
    # The state must contain the first token's conv input in its newest column.
    gated, conv_input = ple._gate(ple.key_proj(embeds), ple.value_proj(embeds),
                                  hidden)
    torch.testing.assert_close(new_state[0, :, -1], conv_input[0])
    torch.testing.assert_close(new_state[1, :, -1], conv_input[1])
