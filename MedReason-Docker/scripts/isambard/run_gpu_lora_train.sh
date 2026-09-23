#!/usr/bin/env bash
# LoRA fine-tune on MedReason train_lora.jsonl (requires GPU).
set -euo pipefail

LIMIT="${1:-}"
EPOCHS="${2:-1}"
MAX_STEPS="${3:-}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
APP_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
REPO_ROOT="$(cd "${APP_DIR}/.." && pwd)"
OUTPUT_DIR="${REPO_ROOT}/data/outputs/lora/qwen3vl_medreason"

cd "${APP_DIR}"
export PYTHONPATH="${APP_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

ARGS=(
  --model-path "${APP_DIR}/models/Qwen3-VL-32B-Instruct"
  --dataset "${REPO_ROOT}/data/derived/train_lora.jsonl"
  --image-root "${REPO_ROOT}/data/raw/train/imgs"
  --repo-root "${REPO_ROOT}"
  --output-dir "${OUTPUT_DIR}"
  --num-train-epochs "${EPOCHS}"
  --per-device-train-batch-size 1
  --gradient-accumulation-steps 16
  --learning-rate 1e-4
  --lora-r 16
  --lora-alpha 32
  --logging-steps 10
  --save-steps 500
  --save-total-limit 3
)

if [[ -n "${LIMIT}" ]]; then
  ARGS+=(--limit "${LIMIT}")
fi
if [[ -n "${MAX_STEPS}" ]]; then
  ARGS+=(--max-steps "${MAX_STEPS}")
fi

python tools/train_lora.py "${ARGS[@]}"
