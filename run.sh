#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

usage() {
  cat <<'EOF'
Qwen3.8 Flash Next CUDA runner

Usage:
  ./run.sh                 Start the default 250k-balanced server
  ./run.sh serve [preset]  Start a named preset
  ./run.sh check           Check CUDA, model, and llama.cpp prerequisites
  ./run.sh help            Show this help

Presets:
  250k-balanced       250k context, batch 2048 (default)
  250k-safe            250k context, batch 1024
  250k-conservative    250k context, batch 512
  180k-max             180k context, batch 4096

The server exposes an OpenAI-compatible API at http://127.0.0.1:8081/v1.
Environment variables may override preset values; see scripts/llama-server-control.sh.
EOF
}

case "${1:-serve}" in
  serve)
    preset="${2:-250k-balanced}"
    exec env PRESET="$preset" ./scripts/llama-server-control.sh
    ;;
  check)
    exec ./scripts/check-environment.sh
    ;;
  help|-h|--help)
    usage
    ;;
  *)
    echo "Unknown command: $1" >&2
    usage >&2
    exit 2
    ;;
esac
