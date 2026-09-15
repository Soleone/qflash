#!/usr/bin/env bash
set -euo pipefail
printf '%s\n' '=== Host ==='
uname -a
printf '\n=== CPU ===\n'
lscpu | grep -E 'Model name|CPU\(s\)|Core\(s\) per socket|Thread\(s\) per core' || true
printf '\n=== GPU ===\n'
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
printf '\n=== Model link ===\n'
readlink -f "$(dirname "$0")/../docs/model/UD-Q4_K_XL"
ls -lh "$(dirname "$0")/../docs/model/UD-Q4_K_XL"/*.gguf
