# 27B speed alternatives

This project keeps `llama.cpp` as the reproducible baseline, but the RTX 4090
has better options than the original 18-20 tok/s Flash-Next result.

## What to measure first

For the Qwen3.8-27B GGUF path, use one server slot and compare these arms:

```bash
# safer baseline: Q8 KV, MTP depth 2
./qflash --profile 27b

# speed arm: matched Q4 KV, MTP depth 4
./qflash --profile 27b-fast

# smaller Q3 weights: deeper MTP and 128k capacity
./qflash --profile 27b-q3-fast

# same Q4 speed arm, larger logical prompt batch but conservative ubatch
./qflash --profile 27b-fast --batch 4096 --ubatch 1024
```

Do not run two servers at once on a 24 GB card. With each server running,
measure the same workloads from another terminal:

```bash
python3 scripts/benchmark-api.py \
  --url http://127.0.0.1:8081 \
  --prompt-tokens 2048 \
  --prompt-tokens 8192 \
  --prompt-tokens 32768 \
  --prompt-tokens 65536 \
  --output-tokens 512 \
  --runs 3 \
  --csv logs/qwen38-27b-current.csv
```

The result has separate `prompt_tps` and `decode_tps` columns. Batch and ubatch
mostly change `prompt_tps`; they should not be expected to improve single-stream
`decode_tps`. MTP, KV-cache kernels, GPU residency, and output predictability
are the important output-speed variables.

Initial local results, using two-run medians:

| Profile | Workload | Capacity | Batch/ubatch | Prompt range | Decode range |
|---|---|---:|---:|---:|---:|
| Q4, MTP 4 | deterministic raw prompt | 80-96k | 512/512 | 2.15-2.53k | 108-123 tok/s |
| Q4, MTP 4 | deterministic raw prompt | 64k | 4096/1024 | 2.13-2.45k | 101-119 tok/s |
| Q3, MTP 5 | deterministic raw prompt | 128k | 512/512 | 2.24-2.55k | 93-150 tok/s |
| Q4, MTP 4 | coding-style task | 80k | 512/512 | 2.12-2.53k | 66-86 tok/s |
| Q3, MTP 5 | coding-style task | 128k | 512/512 | 2.17-2.61k | 68-96 tok/s |
| Q4, no MTP | coding-style task | 128k | 512/512 | 2.46-2.79k | 41-46 tok/s |

The Q3 deterministic 8k result reached about 171 tok/s, while its 32k result
was about 150 tok/s. The coding-style task is a more honest estimate for an
agent workload and shows why 100+ tok/s is not yet a daily-use result. MTP
acceptance depends heavily on output predictability.

For a matched 64k Q4 comparison, 512/512 produced 2,151, 2,509, and 2,389
prompt tok/s at 2k, 8k, and 32k input. 4096/1024 produced 2,130, 2,446, and
2,320 tok/s. In this run, the larger logical batch did not improve input speed.
The full 4096/4096 variant failed its CUDA allocation, so ubatch remains the
next knob to investigate rather than interpreting 4096 as an output setting.

For the existing Flash-Next model, batch 4096 is already represented by
`180k-max`. The safer comparison is to keep the same 180k context and compare
these two configurations, rather than risking 250k + batch 4096:

```bash
./qflash --preset 180k-max --batch 2048 --ubatch 2048
./qflash --preset 180k-max --batch 4096 --ubatch 1024
```

The current measurements show 250k + batch 4096 does not start, while 180k +
batch 4096 uses about 20.2 GiB. A 4096 logical batch with 1024 ubatch is a
useful intermediate test because it separates the maximum prompt batch from
the size of each CUDA graph/work buffer.

## MTP and KV settings

The current CUDA build has Flash Attention kernels for matched `q4_0/q4_0` and
`q8_0/q8_0` caches. The 27B profiles therefore deliberately use matching K and
V types. Avoid mixed types on this build: unsupported combinations can fall
back to CPU attention or fail to initialize.

`27b` is the quality/headroom arm. `27b-fast` uses `q4_0` K/V and MTP depth 4.
The smaller `27b-q3-fast` weight file leaves enough VRAM for 128k context and
MTP depth 5, at a larger weight-quantization quality tradeoff. All use one
server slot. MTP depth can be swept with `--spec-depth`.

`27b-long` is a context-capacity experiment. It uses 128k capacity and Q4 KV
but disables MTP to leave room for context. It is not expected to be the
fastest profile.

## NInfer-4090

NInfer is no longer only a 5090 idea. The `sergiuszm/ninfer-4090` project is a
specialized Linux CUDA engine targeting `sm_89` RTX 4090 cards. It uses a
separate groupwise `.ninfer` artifact rather than the GGUF and supports paged
KV, CUDA graphs, ReplaySSM state handling, and MTP. Its published 4090 results
include 50.5 tok/s without speculation and 106.5 tok/s on a mixed benchmark
with MTP, with higher code-generation results.

This is the most promising route toward a reliable 60 tok/s target, but it is
a separate engine, model artifact, and Docker build. It is not yet wired into
`qflash` because silently making `qflash` download or convert a second model
would weaken reproducibility. The next sensible integration is an optional
`qflash-ninfer` wrapper after the GGUF baseline has been measured locally.

Source: <https://github.com/sergiuszm/ninfer-4090>

## EXL3

EXL3 is the default `qflash` profile, backed by a separate runtime because
ExLlamav3 uses a different weight format and CUDA backend. TabbyAPI exposes the
model through the same OpenAI-compatible API:

```bash
./qflash --profile 27b-exl3          # recommended 256k long-context preset
./scripts/exl3-server.sh q4-128k      # lower-VRAM throughput preset
./scripts/exl3-server.sh q8-128k      # conservative KV-cache preset
```

The launcher expects the EXL3 model at
`models/qwen38-27b-exl3-3.5bpw/` and the optional runtime under `.exl3/`.
The `.exl3/` directory is intentionally ignored by Git. The 3.5 bpw model
comes from `Mia-AiLab/Qwen3.8-27B-EXL3-3.5bpw`; install ExLlamav3 and TabbyAPI
according to their upstream instructions before using the wrapper.

### Matched RTX 4090 results

These are single-session medians with the same temperature/top-p/top-k settings
and 256 generated tokens. EXL3 used Q4 K/V, MTP depth 4, and the `q4-256k`
preset. GGUF used `27b-fast`, Q4 K/V, and MTP depth 4:

| Prompt tokens | EXL3 prompt / decode | GGUF prompt / decode |
|---:|---:|---:|
| 2,048 | 2,226 / 113.1 | 2,214 / 124.8 |
| 8,192 | 2,334 / 155.8 | 2,567 / 140.8 |
| 32,768 | 2,114 / 143.8 | 2,432 / 122.8 |
| 65,536 | 1,755 / 126.5 | 2,134 / 105.8 |

Rates are server-reported prompt and completion rates, not wall-clock rates
that combine the two phases. EXL3 used about 20.9 GiB (21.4 GB) total GPU
memory with
the desktop still active and loaded successfully at 260k prompt tokens plus a
short completion. At that extreme length, prompt processing fell to about 890
tok/s and decode to about 78 tok/s.

A deterministic coding-style request at 8k input and 512 output measured
117.6 tok/s on EXL3 versus 89.4 tok/s on the matched GGUF run. Treat this as
an upper bound: the request was greedy, and sampled coding responses can have
lower MTP acceptance or stop early. TabbyAPI's ExLlamav3 backend also does not
honor llama.cpp's `ignore_eos` option, so sampled output-length comparisons need
careful interpretation.

Benchmark EXL3 with:

```bash
PATH=.exl3/venv/bin:$PATH \
  .exl3/venv/bin/python scripts/benchmark-exl3.py \
  --url http://127.0.0.1:8082 --output-tokens 256 --runs 3
```

### Preliminary 3.0 bpw trial

The optional 3.0 bpw model is downloaded under
`models/qwen38-27b-exl3-3.0bpw/` from
[turboderp/Qwen3.8-27B-exl3](https://huggingface.co/turboderp/Qwen3.8-27B-exl3/tree/SC_3.00bpw_H4_V4),
pinned to revision `004a887127d8304ca2d5475d3a3c41f1761fdd27`. Select it with
`./qflash --profile 27b-exl3-3.0bpw`; the 3.5 bpw model remains the default
under `./qflash --profile 27b-exl3`.

At the same `q4-256k` settings on the RTX 4090, the EXL3 process used 18,266
MiB with 3.5 bpw and 16,366 MiB with 3.0 bpw, a measured saving of 1,900 MiB
(about 1.86 GiB). Total GPU use fell from 21,386 to 19,434 MiB with the desktop
active. The model directories occupy 15.36 GB and 13.01 GB respectively.

A one-session, 8k synthetic-prompt smoke benchmark (three measured runs, one
warmup, 256 output tokens) gave medians of 2,288 / 143 tok/s prompt/decode at
3.5 bpw and 2,308 / 160 tok/s at 3.0 bpw. The prefill result is effectively
unchanged; decode was about 12% faster in this small sample, but MTP acceptance
varies and this is not enough evidence to promise a speedup. Raw rows are in
[`logs/qwen38-27b-exl3-35bpw-8k-synth-t1.csv`](../logs/qwen38-27b-exl3-35bpw-8k-synth-t1.csv)
and [`logs/qwen38-27b-exl3-30bpw-8k-synth-t1.csv`](../logs/qwen38-27b-exl3-30bpw-8k-synth-t1.csv).

This is an artifact comparison, not an isolated bitrate test: the 3.0 config
uses a 4-bit head and 3-bit MTP weights, versus the current 3.5 model's 6-bit
head and 4-bit MTP weights. The existing 3.5 quant is workload-calibrated for
coding and math reasoning; the 3.0 build uses a separate quantization recipe.
The 3.0 server loaded with the 262k Q4-cache setting and served an 8k request;
a 260k-prompt run and task quality have not yet been evaluated.

The `q4-256k` preset is the best default for this 24 GB card. Use `q4-128k`
when prompt latency matters more than capacity, and `q8-128k` when preserving
KV precision matters more than VRAM headroom. Keep one runtime loaded at a
time. DFlash2 EXL3 drafts are a separate fork-specific option, not the same as
llama.cpp's embedded MTP.

## TurboQuant, DSpark, and system RAM

TurboQuant KV work is promising for long context, but the public evidence is
stronger on Apple Silicon, RTX 5090, and Qwen3.6 than on this RTX 4090 plus
Qwen3.8 combination. It should be treated as an experiment after the matched
Q4 KV baseline, not assumed to be a free speedup.

DSpark is primarily a Blackwell/SGLang path using NVFP4 and a specialized draft
model. It is not a sensible RTX 4090 target. NInfer and mainline llama.cpp MTP
are the relevant Ada paths today.

System RAM can hold mmap'ed weights, CPU state, or an offloaded layer, but PCIe
transfers are not a route to 60 tok/s. For this dense 27B model, keep weights,
KV, and recurrent state on the GPU whenever the selected context fits.
