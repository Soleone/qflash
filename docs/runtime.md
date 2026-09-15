# Runtime setup

The launcher expects a CUDA-built `llama-server` at:

```text
llama.cpp-latest/build/bin/llama-server
```

The validated checkout currently is:

```text
llama.cpp commit 4c9233c034fc450dcf34c7c0988aebe6da5cdf1
```

Build it with:

```bash
git clone https://github.com/ggml-org/llama.cpp.git llama.cpp-latest
cmake -S llama.cpp-latest -B llama.cpp-latest/build \
  -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release
cmake --build llama.cpp-latest/build --config Release -j
```

The launcher supports `LLAMA_SERVER_BIN=/path/to/llama-server` if a different checkout is desired. The older b10948 checkout was used as a historical apples-to-apples control, but is not required by the default setup.

`qflash --eager` uses `--load-mode none`, waits for model loading to finish, then sends one small throwaway request to initialize CUDA graphs and kernels. Use `--eager --no-warmup` if only eager model loading is wanted. Eager loading requires substantial system RAM and takes longer at startup.

## Model

Place the four verified shards under `models/qwen38/UD-Q4_K_XL/`. The model is deliberately ignored by Git because it occupies approximately 111 GB. See `docs/model-manifest.json` and `results/model-sha256.txt` for expected files and hashes.

## API

The server listens only on loopback:

```text
http://127.0.0.1:8081/v1
```

No remote API credentials are required. The optional `.pi/models.json` file contains only this local endpoint and a placeholder local API key used by the OpenAI-compatible client protocol.
