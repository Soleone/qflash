# Qwen3.8 CUDA Lab

A reproducible Linux/CUDA runner and benchmark notebook for Qwen3.8 models on a single 24 GB RTX 4090.

This project is **not Ollama**. The default profile uses ExLlamav3/TabbyAPI for Qwen3.8-27B EXL3. The original Flash-Next llama.cpp profile remains available explicitly. Both expose an OpenAI-compatible API for Pi, curl, and other clients.

## Quick start

```bash
./qflash --help
./qflash
```

The default server is:

- 256k context
- EXL3 Q4 KV cache
- MTP depth 4
- one session
- `http://127.0.0.1:8082/v1`

## Current recommendation

Use the default EXL3 Qwen3.8-27B profile for normal Pi coding-agent work. It is
much faster than Flash-Next while retaining a practical 260k-token session.
Keep Flash-Next as the explicit quality fallback for unusually difficult tasks.

| Model | Runtime and cache | Practical context | Representative speed |
|---|---|---:|---|
| **Qwen3.8-27B EXL3** | 3.5 bpw, Q4 KV, MTP4 | **262k configured; 260k tested** | 2,334 / 156 tok/s at 8k; 1,319 / 102 at 131k; 890 / 78 at 260k |
| Qwen3.8-Flash-Next | GGUF, F16 KV, CPU MoE | 250k configured | roughly 675–700 / 18–20 tok/s on balanced runs; roughly 300 / 20 at very long context |

Speed pairs are prompt/decode tok/s. The 27B model is the recommended default
because its decode rate is several times higher and its incremental cached-agent
latency is much lower. Flash-Next remains the larger, higher-quality model, but
its CPU MoE execution makes it substantially slower.

The Flash-Next model shards are only required for the explicit
`--profile flash-next` profile. They are intentionally not committed. Place the
four GGUF files at:

```text
models/qwen38/UD-Q4_K_XL/
```

`docs/model/UD-Q4_K_XL` is a convenience symlink to that directory.

## Presets

```bash
./qflash                                  # EXL3 256k profile (default)
./qflash --profile flash-next --preset 250k-safe
./qflash --profile flash-next --preset 250k-conservative
./qflash --profile flash-next --preset 180k-max
./qflash --profile flash-next --preset 80k-fast
./qflash --profile flash-next --parallel 2
./qflash --profile flash-next --eager
./qflash --profile 27b                     # Q8 KV, safer 27B alternative
./qflash --profile 27b-fast                # Q4 KV, throughput experiment
./qflash --profile 27b-q3-fast             # Q3 weights, maximum speed experiment
./qflash --profile 27b-long                # Q4 KV, longer-context experiment
```

## Qwen3.8-27B GGUF alternatives

The `27b` profile is intended for the RTX 4090 and uses a single
`Qwen3.8-27B-UD-Q4_K_XL.gguf` file. Put it at:

```text
models/qwen38-27b/Qwen3.8-27B-UD-Q4_K_XL.gguf
```

Start it with:

```bash
./qflash --profile 27b
```

The `27b` profile uses 64k context and Q8 KV as the safer baseline. The
`27b-fast` profile uses 80k context, matched Q4 (`q4_0`) KV, and MTP depth 4.
`27b-q3-fast` uses the smaller Q3 weight quantization at 128k context and MTP
depth 5. It is the most promising speed profile, with a modest additional
quality tradeoff. `27b-long` uses 128k matched Q4 KV without MTP, prioritizing
context over output speed. Unlike Flash-Next, these profiles do not use
`--cpu-moe`.

System RAM is useful as pinned backing for `mmap+mlock` and startup/page-cache
behavior, but moving dense layers into RAM generally reduces decode speed over
PCIe. It is not a second VRAM pool that improves tokens/second.

The upstream 24 GB recipe reports roughly 38-40 tok/s on an RTX A5000 at short
context and about 32 tok/s at 128k with MTP. Our first local measurements reached 100-125 tok/s on Q4 at 64-96k context
and roughly 150 tok/s on Q3 at 128k, using synthetic deterministic prompts.
Random
sampling and less predictable coding output will be lower. Treat those as
speed ceilings, not daily-chat guarantees. The benchmark helper below records
prompt and decode throughput separately.

`--parallel N` sets llama.cpp's number of independent server slots. The configured
`--context` is shared across slots, so each slot gets approximately `context / N`
tokens. Parallel slots share the model and can reduce prompt and generation
throughput when active simultaneously.

## EXL3 default runtime

The default `qflash` profile starts the ExLlamav3/TabbyAPI Qwen3.8-27B EXL3
3.5 bpw model. The optimized presets are:

```bash
./qflash --profile 27b-exl3    # recommended 256k EXL3 profile
./scripts/exl3-server.sh q4-128k
./scripts/exl3-server.sh q8-128k
```

EXL3 is launched by `qflash` through the separate TabbyAPI runtime because it
uses a different weight format and inference backend. The matched measurements
and setup notes are in [`docs/alternatives.md`](docs/alternatives.md).

### Reasoning controls

Both the EXL3 and Flash-Next Pi profiles advertise reasoning and pass the
thinking level through the Qwen chat template:

```bash
pi --model qflash-exl3/qwen38-27b-exl3-3.5bpw --thinking medium
pi --model qflash-exl3/qwen38-27b-exl3-3.5bpw --thinking off
```

Qwen3.8 accepts `low`, `medium`, and `xhigh`; Pi's `minimal`/`low` map to
`low`, `medium` maps to `medium`, and `high`/`xhigh`/`max` map to `xhigh`.

## Flash-Next reference presets

These measurements apply to the explicit `--profile flash-next` profile, not the
EXL3 default:

| Preset | Context | Batch | VRAM | Prompt speed | Decode speed |
|---|---:|---:|---:|---:|---:|
| 250k-safe | 250k | 1024 | ~15.6 GB | ~465–475 tok/s | ~18–20 tok/s |
| 250k-balanced | 250k | 2048 | ~19.0 GB | ~675–700 tok/s | ~18–20 tok/s |
| 250k-conservative | 250k | 512 | ~14.0 GB | not fully benchmarked | ~18–20 tok/s |
| 180k-max | 180k | 4096 | ~20.2 GB | highest candidate | ~18–20 tok/s |
| 80k-fast | 80k | 4096 | ~12.5 GB | ~521 tok/s on diverse 16k input | ~18 tok/s |

Batch size primarily affects prompt ingestion and temporary CUDA buffers. It has little effect on decode speed. Batch 4096 does not fit at 250k. `--eager` only changes model loading; llama.cpp's own built-in warm-up remains enabled.

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
```

The project is a **thin, reproducible llama.cpp launcher plus experiment record**, not a new inference runtime and not an Ollama wrapper. NInfer-4090 is tracked as an optional alternative in the benchmark notes because it uses a separate `.ninfer` artifact and specialized engine rather than the GGUF path.
