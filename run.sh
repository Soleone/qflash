#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

# Compatibility alias. The public entry point is ./qflash.
if [[ "${1:-}" == "serve" ]]; then
  shift
fi
if [[ "${1:-}" == "help" || "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  exec ./qflash --help
fi
if [[ -n "${1:-}" ]]; then
  exec ./qflash --preset "$1"
fi
exec ./qflash
