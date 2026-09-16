#!/usr/bin/env bash
# Launch the optional ExLlamav3/TabbyAPI Qwen3.8-27B profiles.
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PROFILE=${1:-q4-256k}
HOST=${EXL3_HOST:-127.0.0.1}
PORT=${EXL3_PORT:-8082}

case "$PROFILE" in
  q4-256k)
    CONTEXT=262144
    CACHE=Q4
    CHUNK=2048
    DRAFT=4
    ;;
  q4-128k)
    CONTEXT=131072
    CACHE=Q4
    CHUNK=2048
    DRAFT=4
    ;;
  q8-128k)
    CONTEXT=131072
    CACHE=Q8
    CHUNK=2048
    DRAFT=4
    ;;
  *)
    echo "usage: $0 {q4-256k|q4-128k|q8-128k}" >&2
    exit 2
    ;;
esac

PYTHON="$ROOT/.exl3/venv/bin/python"
TABBY="$ROOT/.exl3/TabbyAPI"
MODEL_DIR="$ROOT/models"
if [[ ! -x "$PYTHON" || ! -f "$TABBY/main.py" ]]; then
  echo "EXL3 runtime is not installed under .exl3; see docs/alternatives.md" >&2
  exit 1
fi
if [[ ! -d "$MODEL_DIR/qwen38-27b-exl3-3.5bpw" ]]; then
  echo "EXL3 model not found: $MODEL_DIR/qwen38-27b-exl3-3.5bpw" >&2
  exit 1
fi

cat > "$TABBY/config.yml" <<YAML
network:
  host: $HOST
  port: $PORT
  disable_auth: true
  api_servers: ["OAI"]
  access_log: false

logging:
  log_prompt: false
  log_generation_params: false
  log_requests: false
  log_live_status: false
  log_timestamps: true

model:
  model_dir: "$MODEL_DIR"
  model_name: qwen38-27b-exl3-3.5bpw
  backend: exllamav3
  max_seq_len: $CONTEXT
  cache_size: $CONTEXT
  cache_mode: $CACHE
  chunk_size: $CHUNK
  max_batch_size: 1
  output_chunking: true
  reasoning: true
  reasoning_start_token: "<think>"
  reasoning_end_token: "</think>"
  start_in_reasoning: auto
  tool_format: qwen3_coder

draft_model:
  draft_mode: mtp
  draft_cache_mode: Q8
  draft_num_tokens: $DRAFT

sampling:
  override_preset: safe_defaults

memory:
  sysmem_recurrent_cache: 4096
  sysmem_kv_cache: 0
  cuda_malloc_async: false
YAML

cd "$TABBY"
exec env PATH="$ROOT/.exl3/venv/bin:$PATH" "$PYTHON" main.py
