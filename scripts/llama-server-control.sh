#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

# Presets are intentionally explicit about the memory/performance trade-off.
# Individual environment variables override the selected preset.
case "${PRESET:-250k-balanced}" in
  baseline|180k-max)
    preset_context=180000; preset_batch=4096; preset_ubatch=4096 ;;
  80k-fast)
    preset_context=80000; preset_batch=4096; preset_ubatch=4096 ;;
  250k-balanced)
    preset_context=250000; preset_batch=2048; preset_ubatch=2048 ;;
  250k-safe)
    preset_context=250000; preset_batch=1024; preset_ubatch=1024 ;;
  250k-conservative)
    preset_context=250000; preset_batch=512; preset_ubatch=512 ;;
  *)
    echo "unknown PRESET: ${PRESET}" >&2
    exit 2 ;;
esac

moe_args=(--cpu-moe)
if [[ "${MOE_PLACEMENT:-all-cpu}" == "ncmoe40" ]]; then
  moe_args=(--n-cpu-moe 40)
fi
lazy_args=(--lazy-mode off)
if [[ "${OMIT_LAZY_MODE:-0}" == "1" ]]; then
  lazy_args=()
fi
exec "${LLAMA_SERVER_BIN:-./llama.cpp-latest/build/bin/llama-server}" \
  -m "$PWD/docs/model/UD-Q4_K_XL/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf" \
  -c "${CONTEXT_CAPACITY:-$preset_context}" \
  --host 127.0.0.1 --port "${PORT:-8081}" \
  --fit "${FIT_MODE:-off}" \
  --load-mode "${LOAD_MODE:-auto}" \
  ${GPU_LAYERS:+-ngl "$GPU_LAYERS"} \
  "${moe_args[@]}" \
  -np 1 \
  -t "${THREADS:-8}" -tb "${BATCH_THREADS:-8}" \
  -b "${BATCH_SIZE:-$preset_batch}" -ub "${UBATCH_SIZE:-$preset_ubatch}" \
  -fa "${FLASH_ATTN:-auto}" \
  --spec-type none \
  --cache-type-k "${CACHE_TYPE_K:-f16}" --cache-type-v "${CACHE_TYPE_V:-f16}" \
  "${lazy_args[@]}" \
  --jinja
