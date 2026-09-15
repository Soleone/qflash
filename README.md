# Qwen3.8 Flash Next CUDA Lab

A reproducible Linux/CUDA runner and benchmark notebook for **Qwen3.8-Flash-Next** on a single 24 GB RTX 4090.

This project is **not Ollama**. It runs `llama.cpp` directly, with the model's MoE expert weights placed in system RAM (`--cpu-moe`) and attention/KV work placed on the GPU. The server exposes an OpenAI-compatible API for Pi, curl, and other clients.

## Quick start

```bash
./qflash --help
./qflash
```

The default server is:

- 250k context
- F16 KV cache
- CPU MoE (`--cpu-moe`)
- batch/ubatch 2048
- one sequence
- `http://127.0.0.1:8081/v1`

The model shards are intentionally not committed. Place the four GGUF files at:

```text
models/qwen38/UD-Q4_K_XL/
```

`docs/model/UD-Q4_K_XL` is a convenience symlink to that directory.

## Presets

```bash
./qflash                                  # recommended default
./qflash --preset 250k-safe               # 250k, batch 1024
./qflash --preset 250k-conservative       # 250k, batch 512
./qflash --preset 180k-max                # 180k, batch 4096
./qflash --preset 80k-fast                 # 80k, batch 4096, fastest prompt path
./qflash --eager                          # eagerly read and warm model at startup
```

Measured on the RTX 4090:

| Preset | Context | Batch | VRAM | Prompt speed | Decode speed |
|---|---:|---:|---:|---:|---:|
| 250k-safe | 250k | 1024 | ~15.6 GB | ~465–475 tok/s | ~18–20 tok/s |
| 250k-balanced | 250k | 2048 | ~19.0 GB | ~675–700 tok/s | ~18–20 tok/s |
| 250k-conservative | 250k | 512 | ~14.0 GB | not fully benchmarked | ~18–20 tok/s |
| 180k-max | 180k | 4096 | ~20.2 GB | highest candidate | ~18–20 tok/s |
| 80k-fast | 80k | 4096 | ~12.5 GB | ~521 tok/s on diverse 16k input | ~18 tok/s |

Batch size primarily affects prompt ingestion and temporary CUDA buffers. It has little effect on decode speed. Batch 4096 does not fit at 250k. `--eager` additionally sends one small throwaway request after loading to prime CUDA graphs before the server is handed to the user.

## What this project established

- A 125B-class Qwen MoE model can run at 250k configured context on a 24 GB RTX 4090.
- All-CPU MoE is stable at 250k with F16 KV cache and no speculative decoding.
- The practical balanced run uses about 19 GB VRAM and decodes around 18–20 tokens/s.
- `--n-cpu-moe 40` is not practical on this machine/build: it needs too much GPU memory and only started at 65k context with batch 128.
- The public result's lower VRAM usage is explained primarily by batch/ubatch sizing, not by KV quantization.
- The current `llama.cpp-latest` checkout also passes the 250k/batch-2048 preset: 18,928 MiB and 19.33 tok/s in a smoke test. It remains an experimental alternative until the full matrix is repeated.

Detailed measurements are in `logs/batch-vram-matrix-memory.csv` and `logs/performance-matrix.csv`.

## Requirements

- Linux with an NVIDIA GPU and CUDA-capable driver
- 24 GB VRAM recommended for the published presets
- Approximately 110 GB disk space for the four GGUF shards
- `llama.cpp` built with CUDA support; the validated checkout is recorded in `docs/runtime.md`

The local `llama.cpp` checkout is an experiment dependency, not a fork of Ollama. Setup details and the validated commit are in [`docs/runtime.md`](docs/runtime.md).

## Repository layout

```text
qflash                         one public entry point
scripts/llama-server-control.sh  legacy environment-based launcher
docs/model-manifest.json       model shard sizes and hashes
docs/reproduction-plan.md      original experiment plan
results/                       host and model verification records
logs/                          benchmark measurements and ignored runtime logs
.pi/models.json                optional Pi provider configuration
```

The project is a **thin, reproducible llama.cpp launcher plus experiment record**, not a new inference runtime and not an Ollama wrapper.
