# Reproduce the Qwen3.8-Flash-Next speed comparison on Omarchy

This is a handoff plan for another LLM session. It assumes “Omachi” means the Omarchy/Linux environment. Keep all model data on a data drive with at least 130 GB free.

## Objective

Measure the same model under native Linux and compare:

1. TabbyAPI plus ExL3.
2. llama.cpp plus GGUF.
3. CPU/GPU MoE placement and thread settings.

The key question is whether native Linux CUDA removes enough Windows WDDM overhead to approach the roughly 20 tok/s result reported by Alok for a 4090 using llama.cpp, Ubuntu, and -ncmoe40.

## Known hardware and Windows control

Record the exact Omarchy hardware before testing:

- GPU: RTX 4090, 24 GB VRAM.
- CPU: AMD Ryzen 7 5800X3D, 8 physical cores / 16 threads.
- RAM: 128 GB DDR4.
- Model: Qwen3.8-Flash-Next.
- EXL3 model: Qwen3.8-Flash-Next-exl3-4.05bpw.
- GGUF: Unsloth UD-Q4_K_XL, four shards, about 111.33 GB.

The Windows llama.cpp all-CPU-MoE control used eight generation threads, eight batch threads, F16 KV cache, no speculative decoding, one sequence, batch 4096, ubatch 4096, and context capacity 125184:

- 12.77 tok/s at an 1,838-token prompt.
- 13.93 tok/s at a 28,000-token prompt.
- 12.95 tok/s at a 100,000-token prompt.

The original TabbyAPI plus ExL3 result was about 3.4–4 tok/s.

## Model files

Use a Linux path such as:

~~~text
/data/models/qwen38/
  model/UD-Q4_K_XL/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf
  model/UD-Q4_K_XL/Qwen3.8-Flash-Next-UD-Q4_K_XL-00002-of-00004.gguf
  model/UD-Q4_K_XL/Qwen3.8-Flash-Next-UD-Q4_K_XL-00003-of-00004.gguf
  model/UD-Q4_K_XL/Qwen3.8-Flash-Next-UD-Q4_K_XL-00004-of-00004.gguf
~~~

Copy the four shards from the verified D: bundle or download the pinned Hugging Face revision again. Do not convert or recompress them. Preserve the manifest and verify every shard size and SHA-256 value before running.

Do not copy Tabby API keys into logs or reports. Use local-only endpoints and redact credentials from command output.

## Build llama.cpp

For the exact first comparison, build the same release used on Windows: b10948. A newer build can be tested afterward.

Verify the NVIDIA driver and CUDA device:

~~~bash
nvidia-smi
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
~~~

Build CUDA llama.cpp:

~~~bash
git clone https://github.com/ggml-org/llama.cpp.git
cd llama.cpp
git checkout b10948
cmake -B build -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release -j
build/bin/llama-server --version
build/bin/llama-server --list-devices
~~~

Confirm that llama.cpp sees the RTX 4090.

## Exact llama.cpp control

Start the first GGUF shard. llama.cpp discovers the other shards in the same directory.

~~~bash
./build/bin/llama-server \
  -m /data/models/qwen38/model/UD-Q4_K_XL/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf \
  -c 125184 \
  --host 127.0.0.1 --port 8081 \
  --fit off \
  -ngl 999 \
  --cpu-moe \
  -np 1 \
  -t 8 -tb 8 \
  -b 4096 -ub 4096 \
  --spec-type none \
  --cache-type-k f16 --cache-type-v f16 \
  --lazy-mode off \
  --jinja
~~~

Capture the complete command, llama.cpp version, startup log, GPU memory, and CPU information. Stop if the server reports a context smaller than 125184.

The local OpenAI-compatible API is:

~~~text
http://127.0.0.1:8081/v1
~~~

## Prompt tests

Use deterministic synthetic records so no private conversation text enters the benchmark. Construct actual prompt lengths of:

- 1,838 tokens.
- 28,000 tokens, matching the Alok comparison.
- 100,000 tokens.

Generate 256 output tokens for each test, with a 64-token warmup first. Record:

- Actual prompt tokens.
- Cached and newly evaluated prompt tokens.
- Prompt tokens/second.
- Decode tokens/second.
- First-token latency.
- Total elapsed time.
- Stop reason.
- Output sanity and coherence.
- GPU memory/utilization and CPU telemetry.

For a cold-prefill measurement, restart the server or disable prompt reuse between tests. For an interactive-use measurement, keep the sequence continuous and record cache counts separately. Do not call a 125k capacity test a 125k actual prompt test unless 125k tokens were evaluated.

## MoE placement sweep

With thread count fixed, test one placement at a time:

~~~text
--n-cpu-moe 46   # last 2 layers' experts on GPU
--n-cpu-moe 44   # last 4 layers' experts on GPU
--n-cpu-moe 42   # last 6 layers' experts on GPU
--n-cpu-moe 40   # last 8 layers' experts on GPU; matches Alok's placement
~~~

Start with the short and 28k tests. Monitor VRAM:

~~~bash
watch -n 1 nvidia-smi
~~~

Accept a placement only if it completes without paging, runaway prefill time, CUDA errors, or a stalled server. A 24 GB card may need a less aggressive split.

Windows observations:

- ncmoe44: about 15.3 tok/s short and 14.1 tok/s at 28k, but slower long-prompt prefill.
- ncmoe46: about 13.7 tok/s and near-baseline prefill.
- ncmoe42 and ncmoe40: first-request stalls under WDDM at nearly full VRAM.

Native Linux may behave differently; that difference is the main experiment.

## CPU thread sweep

Keep MoE placement fixed and test:

~~~text
-t 4 -tb 4
-t 8 -tb 8
-t 12 -tb 12
-t 16 -tb 16
~~~

Use the same 28k prompt and 256 generated tokens. Do not assume more threads are faster. Eight threads is the initial control because 16 was slower on Windows.

Test generation and batch threads independently only after the basic sweep. Record CPU package power, clock, temperature, and whether the run is CPU- or GPU-limited.

## TabbyAPI and ExL3

Install TabbyAPI and ExLlamaV3 in a separate Linux environment. Load Qwen3.8-Flash-Next-exl3-4.05bpw with:

- max sequence length 125184.
- Q8 KV cache.
- The same CPU MoE split as the Windows configuration.
- Eight CPU MoE threads.
- No speculative decoding for the first comparison.

Repeat the 1,838, 28k, and 100k tests with the same template and output limit. Record first-token latency separately from decode speed. The Windows logs showed very large first-token costs even when prompt throughput looked high.

## Compare with the public report

Alok's run used Ubuntu, an RTX 4090, DDR4, a UD-Q4_K_XL GGUF, 28k actual prompt depth, and -ncmoe40, reporting about 20.8 tok/s. Treat that as a reference rather than a guarantee: CPU model, llama.cpp commit, driver, CUDA build, cache state, and thread settings all matter.

Report both decode speed and end-to-end time. A higher decode rate is not an improvement if prompt prefill becomes several times slower at 100k context.

## Results record

Store one JSON or Markdown result per run with:

- Date and host.
- Kernel, NVIDIA driver, CUDA version.
- CPU model, RAM speed, GPU PCIe link.
- llama.cpp commit and build flags.
- Model revision and shard hashes.
- Full server command.
- Context size, batch, ubatch, and KV types.
- MoE placement and thread counts.
- Actual prompt, cached prompt, prompt tok/s, and decode tok/s.
- First-token and total time.
- GPU and CPU telemetry.
- Output sanity result.

The final report should identify:

1. Best long-context configuration.
2. Best short-context decode configuration.
3. Whether -ncmoe40 is stable and faster on native Linux.
4. Whether Linux closes the gap to about 20 tok/s.
5. Whether TabbyAPI plus ExL3 remains materially slower than llama.cpp.

Keep the all-CPU llama.cpp run as the control until every optimized result has been validated at 100k actual prompt depth.


## Specifically resolve -ncmoe40 versus -cmoe

The two flags are different operating points, and near-full VRAM is desirable only when the run remains stable:

- `-cmoe` / `--cpu-moe`: keep all MoE expert weights in system RAM; GPU memory is available for attention and KV cache.
- `-ncmoe40`: keep the first 40 layers' experts in system RAM and the final 8 layers' experts on the GPU. This can improve decode speed but consumes almost all 24 GB of VRAM.

The public reference reports `-ncmoe40` at about 22.5 tok/s with 23.85 GB VRAM at an 80k configured capacity. It reports `-cmoe` at about 20.8 tok/s at 80k, 21.0 tok/s at 180k, and 21.0 tok/s at 250k, with 18.3 GB VRAM at 250k. These are configured-capacity results; do not assume they evaluated a filled 250k prompt.

Under Omarchy, run this staged comparison:

1. Start `-ncmoe40` at 80k context capacity with an actual 28k prompt. Record VRAM, RAM, prefill, decode, and stability.
2. If stable, repeat `-ncmoe40` at 180k and 250k capacity.
3. Run full `-cmoe` at the same three capacities.
4. Only after capacity tests succeed, evaluate an actual 100k and then actual 250k prompt. Monitor RAM and swap during the full-prompt tests.

A near-full VRAM reading is not itself a failure. Treat it as successful only if CUDA allocations complete, the server answers normally, and throughput does not collapse into paging or stalls. The Windows `-ncmoe40` test reached about 23.7 GB and stalled on the first request, so Linux stability is the key comparison.
