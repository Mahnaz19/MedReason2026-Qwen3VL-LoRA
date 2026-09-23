#!/usr/bin/env bash
# Merge LoRA adapter into base Qwen3-VL checkpoint (requires GPU for load/merge).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
APP_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
REPO_ROOT="$(cd "${APP_DIR}/.." && pwd)"
ADAPTER_DIR="${1:-${REPO_ROOT}/data/outputs/lora/qwen3vl_medreason/final}"
MERGED_DIR="${2:-${REPO_ROOT}/data/outputs/lora/qwen3vl_medreason_merged}"

cd "${APP_DIR}"
export PYTHONPATH="${APP_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

python tools/merge_lora.py \
  --base-model-path "${APP_DIR}/models/Qwen3-VL-32B-Instruct" \
  --adapter-path "${ADAPTER_DIR}" \
  --output-dir "${MERGED_DIR}"

echo "[merge] set MEDREASON_MODEL_PATH=${MERGED_DIR} for LoRA-tuned inference"
