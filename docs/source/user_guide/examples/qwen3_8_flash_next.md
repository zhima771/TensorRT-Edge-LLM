# Qwen3.8-Flash-Next on Jetson Thor

Text-only deployment of `local-inference-lab/Qwen3.8-Flash-Next-NVFP4`
(`qwen4_exp_text`): a hybrid GDN + Qwen Sparse Attention decoder with
four-stream hyper-connections, a 512-expert NVFP4 MoE FFN and the n-gram PLE
side path.

| Capability | Status |
|---|---|
| Text backbone (GDN + QSA + MoE + hyper-connections) | Supported |
| Vision tower / MTP head | Not exported |
| n-gram PLE table | **Not wired yet** — `ple_embeds` is zero-filled |
| QSA chunked prefill | Rejected by the plugin; one prompt must fit a single prefill |
| Prefix / KV reuse | Not implemented |

Verified on Jetson Thor (aarch64, 122 GB unified memory) with TensorRT
10.13.3, CUDA 13.0, g++ 13.3 and CMake 3.28.

## Step 1: Python environment

`nvidia-modelopt` is only needed to quantize a checkpoint; it publishes no
aarch64 wheel, so install the export dependencies directly.

```bash
cd /path/to/TensorRT-Edge-LLM
uv venv --python /usr/bin/python3.12 .venv
uv pip install --python .venv/bin/python \
  "torch==2.13.0" "transformers==5.14.1" "onnx==1.19.0" "onnxscript==0.7.1" \
  "safetensors==0.8.0" "numpy==2.2.6" "onnx-graphsurgeon==0.6.1" \
  "huggingface-hub>=1.5.0,<2.0"
```

## Step 2: Build the C++ runtime

```bash
git submodule update --init --recursive

export TRT_PACKAGE_DIR=/usr
cmake -S . -B build \
  -DCMAKE_TOOLCHAIN_FILE=cmake/aarch64_linux_toolchain.cmake \
  -DEMBEDDED_TARGET=jetson-thor \
  -DTRT_PACKAGE_DIR=$TRT_PACKAGE_DIR \
  -DENABLE_CUTE_DSL=ALL \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)

export LD_LIBRARY_PATH=$TRT_PACKAGE_DIR/lib/aarch64-linux-gnu:$LD_LIBRARY_PATH
export EDGELLM_PLUGIN_PATH=$PWD/build/libNvInfer_edgellm_plugin.so
```

The apt TensorRT package keeps headers and libraries under the multiarch
directories, which `FindTensorRT.cmake` resolves from `TRT_PACKAGE_DIR=/usr`.
`EDGELLM_PLUGIN_PATH` must point at the freshly built plugin DSO; the default
lookup only covers a directory named `build/` relative to the working tree.

## Step 3: Export ONNX

```bash
export MODEL_DIR=/path/to/Qwen3.8-Flash-Next-NVFP4
export ONNX_DIR=/path/to/workspace/onnx

.venv/bin/python -m tensorrt_edgellm.scripts.export "$MODEL_DIR" "$ONNX_DIR" \
  --skip-visual \
  --externalize-weights nvfp4_moe \
  --no-quantize-activations
```

`--no-quantize-activations` is required on Thor: the checkpoint's dense
projections are MXFP8, and TensorRT 10.13 cannot pass the E8M0 block-scale
tensors across a plugin boundary, so the W8A8 graph fails in
`isPluginApiSupported`. The flag selects FP16 activations (W4A16 MoE,
W8A16 dense). `--externalize-weights nvfp4_moe` keeps the ~63 GiB expert
weights out of the engine plan.

On a host with 128 GB of RAM the exporter is OOM-killed after `prog.save()`
(`EXIT=137`). The ONNX file is complete at that point; continue with Step 4.

## Step 4: Finish the export

The remaining post-processing passes each load a full copy of the graph, which
does not fit next to the still-resident PyTorch model. Run them in a fresh
process with the script in [Appendix A](#appendix-a-finalize-script):

```bash
PYTHONPATH=$PWD .venv/bin/python finalize_export.py "$MODEL_DIR" "$ONNX_DIR/llm"
```

The script also dequantizes the MXFP8 dense weights to FP16 (removing the last
E8M0 tensors) and writes `config.json`, `embedding.safetensors` and the
external-weight manifest.

## Step 5: Build the engine

```bash
./build/examples/llm/llm_build \
  --onnx-dir "$ONNX_DIR/llm" \
  --engine-dir /path/to/workspace/engine \
  --max-input-len 32768 \
  --max-kv-cache-capacity 262144 \
  --max-batch-size 2
```

The builder hard-links `external_*_weights.safetensors` into the engine
directory, so the ONNX and engine directories must share a filesystem,
otherwise the 63 GiB file is copied.

## Step 6: Run inference

```bash
cat > request.json <<'JSON'
{
  "batch_size": 1,
  "temperature": 0.0,
  "top_p": 1.0,
  "top_k": 1,
  "requests": [
    {
      "messages": [{"role": "user", "content": "Introduce yourself in one sentence."}],
      "max_generate_length": 32
    }
  ]
}
JSON

./build/examples/llm/llm_inference \
  --engine-dir /path/to/workspace/engine \
  --input-file request.json \
  --output-file response.json \
  --dump-output
```

`--output-file` is mandatory even with `--dump-output`.

## Disk budget

| Artifact | Size |
|---|---|
| ONNX graph + dense weights | ~9 GiB |
| External MoE weights | ~63 GiB |
| `llm.engine` | ~8 GiB |
| `embedding.safetensors` | ~1.2 GiB |
| Peak while exporting | ~90 GiB |

## Known limitations

- `ple_embeds` is zero-filled: the PLE gate/conv path runs but contributes
  nothing, so accuracy is below the reference implementation.
- QSA has no chunked-prefill kernel. A prompt must fit within
  `--max-input-len`; long-context work needs a new kernel, not a bigger profile.
- Batch entries must share one query length per context step (`PleLayer` and
  `Qwen4ExpQsaAttention` reject ragged prefill).
- MTP and the vision tower are not part of this export.

## Appendix A: Finalize script

Save as `finalize_export.py` next to the repository checkout.

```python
#!/usr/bin/env python3
"""Finish a Qwen3.8-Flash-Next ONNX export after the exporter is OOM-killed."""

import json
import os
import sys

import numpy as np
import onnx
import torch
from onnx import numpy_helper
from safetensors import safe_open

from tensorrt_edgellm._safetensors_io import save_file
from tensorrt_edgellm.checkpoint.checkpoint_utils import write_runtime_artifacts
from tensorrt_edgellm.external_weights import (externalize_model_weights,
                                               patch_external_weight_manifest)
from tensorrt_edgellm.model import load_model_config
from tensorrt_edgellm.models.qwen4_exp.modeling_qwen4_exp import Qwen4ExpCausalLM
from tensorrt_edgellm.onnx import export as E

MODEL_DIR, ONNX_DIR = sys.argv[1], sys.argv[2]
ONNX_PATH = os.path.join(ONNX_DIR, "model.onnx")
MXFP8_BLOCK = 32
E8M0_BIAS = 127


def _save_graph(model):
    """Rewrite the protobuf and the external sidecar without stale tensors."""
    os.remove(ONNX_PATH + ".data")
    onnx.save_model(model,
                    ONNX_PATH,
                    save_as_external_data=True,
                    all_tensors_to_one_file=True,
                    location="model.onnx.data",
                    size_threshold=0)


def run_metadata_passes():
    E._fix_nvfp4_weight_dtype(ONNX_PATH)
    E._strip_onnxscript_internal_attrs(ONNX_PATH)
    E._fix_initializer_dtypes(
        ONNX_PATH,
        dedup_dql_scales=True,
        preserve_fp32_patterns=Qwen4ExpCausalLM.
        preserve_fp32_initializer_patterns,
    )
    E._strip_attention_plugin_optional_inputs(ONNX_PATH)
    E._fix_zero_volume_initializers(ONNX_PATH)


def bake_mxfp8_weights():
    """Dequantize MXFP8 weights to FP16 so no E8M0 tensor reaches a plugin."""
    model = onnx.load(ONNX_PATH)
    inits = {init.name: init for init in model.graph.initializer}
    baked, dropped, dead = {}, [], set()
    for node in model.graph.node:
        if node.op_type != "TRT_MXFP8DequantizeLinear" or len(node.input) < 2:
            continue
        weight_name, scale_name = node.input[0], node.input[1]
        if weight_name not in inits or scale_name not in inits:
            continue
        weight = numpy_helper.to_array(inits[weight_name]).astype(np.float32)
        scale = numpy_helper.to_array(inits[scale_name]).astype(np.int32)
        out_features, in_features = weight.shape
        blocks = in_features // MXFP8_BLOCK
        shared = np.exp2(scale - E8M0_BIAS).reshape(out_features, blocks,
                                                    1).astype(np.float32)
        dequant = (weight.reshape(out_features, blocks, MXFP8_BLOCK) *
                   shared).reshape(out_features, in_features).astype(np.float16)
        baked[node.output[0]] = numpy_helper.from_array(dequant,
                                                        name=node.output[0])
        dropped.append(node)
        dead.update((weight_name, scale_name))
    if not baked:
        return
    print(f"baked {len(baked)} MXFP8 weights to FP16")
    kept_nodes = [node for node in model.graph.node if node not in dropped]
    del model.graph.node[:]
    model.graph.node.extend(kept_nodes)
    kept_inits = [i for i in model.graph.initializer if i.name not in dead]
    kept_inits.extend(baked.values())
    del model.graph.initializer[:]
    model.graph.initializer.extend(kept_inits)
    _save_graph(model)


def widen_delta_norms():
    """Keep (1 + w) in FP32; FP16 weights produce an Add(Half, Float)."""
    model = onnx.load(ONNX_PATH)
    inits = {init.name: init for init in model.graph.initializer}
    types = {init.name: init.data_type for init in model.graph.initializer}
    widened = set()
    for node in model.graph.node:
        if node.op_type != "Add" or len(node.input) != 2:
            continue
        for left, right in ((node.input[0], node.input[1]),
                            (node.input[1], node.input[0])):
            if (left in inits
                    and types.get(left) == onnx.TensorProto.FLOAT16
                    and types.get(right) == onnx.TensorProto.FLOAT):
                widened.add(left)
    if not widened:
        return
    print(f"widened {len(widened)} delta norms to FP32")
    for name in widened:
        value = numpy_helper.to_array(inits[name]).astype(np.float32)
        inits[name].CopyFrom(numpy_helper.from_array(value, name=name))
    for info in (list(model.graph.input) + list(model.graph.value_info) +
                 list(model.graph.output)):
        if info.name in widened:
            info.type.tensor_type.elem_type = onnx.TensorProto.FLOAT
    _save_graph(model)


def externalize_experts():
    manifest = externalize_model_weights(ONNX_PATH, None, ["nvfp4_moe"])
    model = onnx.load(ONNX_PATH)
    _save_graph(model)
    del model
    if manifest:
        print("externalized:", [entry["file"] for entry in manifest])
    return manifest


def write_artifacts(manifest):
    config = load_model_config(MODEL_DIR)

    class _Shim:
        RECURRENT_STATE_DTYPE = Qwen4ExpCausalLM.RECURRENT_STATE_DTYPE
        CONV_STATE_DTYPE = Qwen4ExpCausalLM.CONV_STATE_DTYPE

        def __init__(self, cfg):
            self.config = cfg

    write_runtime_artifacts(_Shim(config), MODEL_DIR, ONNX_DIR)
    if manifest:
        patch_external_weight_manifest(ONNX_DIR, manifest)

    index_path = os.path.join(MODEL_DIR, "model.safetensors.index.json")
    with open(index_path) as handle:
        weight_map = json.load(handle)["weight_map"]
    key = next(k for k in weight_map if k.endswith("embed_tokens.weight"))
    with safe_open(os.path.join(MODEL_DIR, weight_map[key]),
                   framework="pt") as handle:
        embedding = handle.get_tensor(key).to(torch.float16).contiguous()
    save_file({"embedding": embedding},
              os.path.join(ONNX_DIR, "embedding.safetensors"))
    print("wrote embedding.safetensors", list(embedding.shape))


run_metadata_passes()
bake_mxfp8_weights()
widen_delta_norms()
write_artifacts(externalize_experts())
```
