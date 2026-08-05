# Deploying CAT-Q Models with Packed Ternary Weights

The [Hugging Face export](../README.md#hugging-face-export) writes a *fake-quantized*
model: the weights are ternary, but each one still occupies 16 bits and runs through
ordinary floating-point matrix multiplication.

This document covers the other export path, which stores the quantized projections as
**group-128 packed ternary weights** and runs them through the ternary kernels of
[llama.cpp](https://github.com/PrismML-Eng/llama.cpp) as used by the
[Bonsai demo](https://github.com/PrismML-Eng/Bonsai-demo).

CAT-Q quantizes a weight group as

```
W_g = s_g * T_g,    T_g in {-1, 0, +1},    |g| = 128
```

which is exactly the `Q2_0` block type of that runtime (`block_q2_0`: one fp16 scale
plus 128 two-bit codes, 34 bytes, 2.125 bits per weight). The exporter takes the codes
and scales straight out of the CAT-Q quantizer and packs them, so the deployed weights
are bit-for-bit the ones the fake-quantized model uses. Embeddings, norms, the LM head
and MoE routers stay in floating point, as they are outside the quantized set.

The implementation lives with the rest of the quantization code:
`quantize/ternary_export.py` recovers the codes and scales, `quantize/q2_0.py` packs
them into `Q2_0` blocks, and `quantize/gguf_export.py` writes the GGUF.

## 1. Export the model

Exporting is a pure CAT-Q step: it reads the checkpoint and writes the GGUF directly.
No inference runtime is involved, and no intermediate model is produced. Beyond the
project requirements it only needs the `gguf` package (`pip install gguf`, already in
`pyproject.toml`).

```bash
cd BitTern/projects/cat-q

# select the checkpoint, exactly as for evaluation
#   result_dir=configs/qwen3-4b     in task_list.conf
./export_gguf.sh
```

This writes `configs/<model>/export-gguf/<net>-catq-q2_0.gguf`.

The launcher is a thin wrapper; the same thing can be run directly:

```bash
python main.py \
  --config configs/qwen3-4b/config.yaml \
  --checkpoint configs/qwen3-4b/parameters.pth \
  --output_dir configs/qwen3-4b/export-gguf \
  --export_gguf_path configs/qwen3-4b/export-gguf/Qwen3-4B-catq-q2_0.gguf
```

`--export_gguf_path` takes either a `.gguf` file or a directory, in which case the file
is named `<net>-catq-q2_0.gguf`. `--gguf_float_type {f16,bf16,f32}` selects the dtype of
the tensors CAT-Q leaves in floating point and defaults to `f16`; norms and MoE routers
are always `F32`, as in a stock llama.cpp conversion.

Dense (Qwen3, LLaMA) and MoE (Qwen3-MoE) checkpoints are both supported. For MoE models
the per-expert `gate_proj`/`up_proj`/`down_proj` weights are packed and stacked into the
`ffn_*_exps` tensors the runtime expects, while the router stays in `F32`.

## 2. Get a runtime with ternary kernels

The GGUF needs a llama.cpp build with group-128 ternary kernels:

```bash
git clone -b prism https://github.com/PrismML-Eng/llama.cpp.git
export LLAMA_CPP_DIR=$PWD/llama.cpp
```

The easiest way to build it is with the Bonsai demo's own scripts, which also fetch the
runtime for you if it is missing:

```bash
git clone https://github.com/PrismML-Eng/Bonsai-demo.git
cd Bonsai-demo
./scripts/build_cuda_linux.sh "$LLAMA_CPP_DIR"   # CUDA; build_cpu_linux.sh / build_mac.sh also exist
```

Binaries land in `Bonsai-demo/bin/<backend>/`. Prebuilt binaries and other backends
(Metal, Vulkan, ROCm) are described in the Bonsai demo README.

## 3. Run it

The result is a standard GGUF file, so any tool from the runtime works with it:

```bash
BIN=/path/to/Bonsai-demo/bin/cuda
export LD_LIBRARY_PATH="$BIN:$LD_LIBRARY_PATH"

# one-off generation
"$BIN/llama-cli" -m Qwen3-4B-catq-q2_0.gguf -ngl 99 -p "Explain ternary quantization."

# OpenAI-compatible server + web UI on http://localhost:8080
"$BIN/llama-server" -m Qwen3-4B-catq-q2_0.gguf -ngl 99 -c 8192 -fa on

# throughput and memory
"$BIN/llama-bench" -m Qwen3-4B-catq-q2_0.gguf -ngl 99
```

Notes for Qwen3 checkpoints, which are thinking models:

- `--reasoning-format deepseek` keeps `<think>` blocks out of `message.content` and puts
  them in `message.reasoning_content`.
- `--chat-template-kwargs '{"enable_thinking": false}'` turns thinking off.
- `-fit off` stops the server from growing the KV cache to fill the device memory, which
  is worth setting when measuring the memory footprint.

For model management, the web UI, tool calling, speculative decoding and non-Linux
platforms, follow the [Bonsai demo](https://github.com/PrismML-Eng/Bonsai-demo)
documentation; a CAT-Q GGUF can be used wherever it expects a Bonsai ternary model.

## Reference numbers

Qwen3-4B, single NVIDIA L40, context 2048, measured with `llama-bench`:

| | packed ternary | fake-quantized `F16` | ratio |
| --- | ---: | ---: | ---: |
| file size | 1.63 GiB | 7.50 GiB | 4.60x |
| device memory | 2093 MiB | 8745 MiB | 4.18x |
| decode (tg128) | 285.1 t/s | 93.5 t/s | 3.05x |

252 of the 398 tensors are packed ternary and hold 3.63 B of the weights at 2.125 bits
each; the remaining floating-point tensors (mostly the token embedding) account for most
of what is left, which is why the whole-file ratio is below the 7.53x of the quantized
part alone. Task accuracy matches the fake-quantized model to within run-to-run noise.

## Acknowledgement

The packed ternary format and the kernels used here come from
[Bonsai](https://github.com/PrismML-Eng/Bonsai-demo) by PrismML and its
[llama.cpp fork](https://github.com/PrismML-Eng/llama.cpp) (branch `prism`), built on
[llama.cpp](https://github.com/ggml-org/llama.cpp). We thank their authors for making
efficient ternary inference available to the community.
