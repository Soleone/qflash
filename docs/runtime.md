# Runtime setup

The launcher expects a CUDA-built `llama-server` at:

```text
llama.cpp-latest/build/bin/llama-server
```

The validated checkout currently is:

```text
llama.cpp commit `4c9233c03` (short commit ID)
```

Build it with:

```bash
git clone https://github.com/ggml-org/llama.cpp.git llama.cpp-latest
cmake -S llama.cpp-latest -B llama.cpp-latest/build \
  -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_BUILD_RPATH_USE_ORIGIN=ON -DCMAKE_INSTALL_RPATH='$ORIGIN'
cmake --build llama.cpp-latest/build --config Release -j
```

`CMAKE_BUILD_RPATH_USE_ORIGIN` keeps the build-tree runtime library path
relative to each executable instead of embedding the checkout's absolute
path. The launcher also sets `LD_LIBRARY_PATH` to the server's sibling library
directory, so an existing build remains usable after the project is moved.

The launcher supports `LLAMA_SERVER_BIN=<path-to-llama-server>` if a different checkout is desired; no machine-specific path is required by the project. The older b10948 checkout was used as a historical apples-to-apples control, but is not required by the default setup.

`qflash --eager` uses `--load-mode none`, so model loading completes before the server starts listening. It does not send a synthetic request; llama.cpp's own built-in warm-up remains enabled. Eager loading requires substantial system RAM and takes longer at startup.

## Model

Place the four verified shards under `models/qwen38/UD-Q4_K_XL/`. The model is deliberately ignored by Git because it occupies approximately 111 GB. See `docs/model-manifest.json` and `results/model-sha256.txt` for expected files and hashes.

## API

The server listens only on loopback:

```text
http://127.0.0.1:8081/v1
```

No remote API credentials are required. The optional `.pi/models.json` file contains only this local endpoint and a placeholder local API key used by the OpenAI-compatible client protocol.
