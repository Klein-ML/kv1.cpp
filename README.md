# KV1.cpp

**llama.cpp with KV1's 2-bit KV cache.** KV1 models are trained to run with a 2-bit KV cache. KV1.cpp adds that
cache format, `BQ2_0`, so a KV1 model runs with exactly the cache it was trained on.

```bash
llama-cli -m KV1-1.2B-Prerelease-BF16.gguf -fa on -ctk bq2_0 -ctv bq2_0 -c 32768
```

Everything else is unchanged llama.cpp. See [README.llama.cpp.md](README.llama.cpp.md) for the full upstream
documentation.

## Models

| Model | Hugging Face | Base |
| :-- | :-- | :-- |
| KV1-1.2B (prerelease) | [klein-ml/KV1-1.2B-Prerelease](https://huggingface.co/klein-ml/KV1-1.2B-Prerelease) (safetensors), [klein-ml/KV1-1.2B-GGUF-Prerelease](https://huggingface.co/klein-ml/KV1-1.2B-GGUF-Prerelease) (GGUF) | LFM2.5-1.2B-Instruct |

Only use `bq2_0` with KV1 models. Other models were not trained for a 2-bit cache and lose most of their quality
with it.

## Build

```bash
git clone https://github.com/Klein-ML/kv1.cpp && cd kv1.cpp

# NVIDIA GPU
cmake -B build -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release
# or CPU only
cmake -B build -DCMAKE_BUILD_TYPE=Release

cmake --build build -j
```

Binaries end up in `build/bin/`. On a single GPU, `-DCMAKE_CUDA_ARCHITECTURES=<your sm>` (e.g. `120` for RTX 50xx /
RTX Pro 6000 Blackwell) shortens the build a lot.

## Run

```bash
# chat
./build/bin/llama-cli -m KV1-1.2B-Prerelease-BF16.gguf -fa on -ctk bq2_0 -ctv bq2_0 -ngl 99 -c 32768

# OpenAI-compatible server
./build/bin/llama-server -m KV1-1.2B-Prerelease-BF16.gguf -fa on -ctk bq2_0 -ctv bq2_0 -ngl 99 -c 32768
```

To convert the safetensors model yourself:

```bash
python3 convert_hf_to_gguf.py /path/to/KV1-1.2B-Prerelease --outtype bf16 --outfile KV1-1.2B-Prerelease-BF16.gguf
```

Requirements for `bq2_0`:

- `-ctk bq2_0 -ctv bq2_0` together. Mixing it with another cache type is refused.
- Flash attention: pass `-fa on`.
- Head dimension 64 (true for every KV1 model).

## The BQ2_0 format

Each block is one head vector of one token: 64 post-RoPE key or value numbers.

```c
#define QK_BQ2_0 64
typedef struct {
    ggml_half d;               // scale = (max - min) / 3
    ggml_half m;               // min
    uint8_t qs[QK_BQ2_0 / 4];  // 2-bit codes, 4 per byte: q0 | q1<<2 | q2<<4 | q3<<6
} block_bq2_0;                 // 20 bytes per 64 values = 2.5 bits per value
```

- **Quantize** (fp32): `d = fp16(max((max - min) / 3, 1e-6))`, `m = fp16(min)`,
  `q = clamp(round_half_to_even((x - m) / d), 0, 3)` using the fp16-rounded `m` and `d`.
- **Dequantize:** `x = q * d + m`.

This matches the KV1 training quantizer (`kv_ternary.py` in every KV1 model repo) byte for byte. Two more parts of
the training setup are applied automatically:

- **Hadamard rotation.** Q, K and V are multiplied by the normalized 64×64 Hadamard matrix after RoPE, and the
  attention output is rotated back before the output projection. This spreads outlier channels over all 64 values
  of a block. It reuses llama.cpp's built-in KV rotation.
- **Attention sinks.** The first 4 tokens of every sequence keep full-precision K and V. They are stored in a small
  F16 side table per layer (4 slots per sequence) and patched into the cache view for attention.

## Memory

For KV1-1.2B (6 attention layers, 8 KV heads, head dim 64), per sequence:

| Context | fp16 KV cache | BQ2_0 KV cache |
| ------: | ------------: | -------------: |
| 8K  | 96 MiB  | 15 MiB |
| 32K | 384 MiB | 60 MiB |

## Verification

| Check | Result |
| :-- | :-- |
| `tools/kv1/test_bq2_0_exact.py`: C quantizer vs. `kv_ternary.py` (random data, outlier channels, Hadamard-rotated, constant vectors, exact .5 ties, tiny values, bf16 inputs) | byte-identical |
| `test-backend-ops -o SET_ROWS` / `-o CPY`: CUDA vs. CPU for the new cases | pass |

```bash
python3 tools/kv1/test_bq2_0_exact.py --ref path/to/kv_ternary.py --lib build/bin/libggml-base.so
./build/bin/test-backend-ops test -b CUDA0 -o SET_ROWS
./build/bin/test-backend-ops test -b CUDA0 -o CPY
```

## Known limitations

- **Attention reads an F16 copy.** For each attention layer, the 2-bit cache view is dequantized to a temporary F16
  buffer (reused across layers) before flash attention. The KV cache itself is 2.5 bits per value, but peak memory
  includes that buffer: about 64 MiB extra at 32K context for KV1-1.2B. Flash-attention kernels that read `BQ2_0`
  directly are planned.
- **No context shift.** `bq2_0` disables context shifting, because the sink tokens are tied to positions 0–3.
- **Saved sessions** do not include the sink side table. Restored sequences use the 2-bit values for positions 0–3
  until they are decoded again (a warning is printed).
- **Backends:** CPU and CUDA. Other backends fall back to the CPU for the `bq2_0` operations.

## License

MIT, like llama.cpp. KV1 model weights have their own license (LFM Open License v1.0, following the base model).
