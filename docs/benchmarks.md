# Benchmark summary

Hardware: RTX 4090 24 GB, Linux, Qwen3.8-Flash-Next UD-Q4_K_XL, F16 KV, one sequence, eight threads, no speculative decoding.

The recommended `250k-balanced` configuration uses `--cpu-moe`, context 250000, and batch/ubatch 2048:

- VRAM: approximately 19.0 GB
- Prompt processing: approximately 675–700 tok/s on 8k–64k synthetic prompts
- Decode: approximately 17.6–19.5 tok/s

Batch 1024 uses approximately 15.6 GB at 250k and has similar decode speed, but prompt processing is approximately 465–475 tok/s. Batch 4096 fails at 250k because the temporary CUDA compute buffers do not fit.

`--n-cpu-moe 40` did not produce a useful operating point on this machine. It only started at 65k context with batch 128, using approximately 21.2 GB VRAM and decoding at 19.7 tok/s.

Raw measurements are in `logs/performance-matrix.csv` and `logs/batch-vram-matrix-memory.csv`. Runtime logs are ignored by Git.
