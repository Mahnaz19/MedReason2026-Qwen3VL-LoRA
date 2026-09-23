#!/usr/bin/env bash
# Continued LoRA training with open-ended upsampling (requires GPU).
# Usage:
#   bash scripts/isambard/run_gpu_lora_open_continue.sh
#   bash scripts/isambard/run_gpu_lora_open_continue.sh 1 5e-5
# Args:
#   $1 = epochs (default 1)
#   $2 = learning rate (default 5e-5)
#   $3 = open-repeat (default 3)
set -euo pipefail

EPOCHS="${1:-1}"
LR="${2:-5e-5}"
OPEN_REPEAT="${3:-3}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
APP_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
REPO_ROOT="$(cd "${APP_DIR}/.." && pwd)"

DATASET_OUT="${REPO_ROOT}/data/derived/train_lora_open_up.jsonl"
RESUME_ADAPTER="${REPO_ROOT}/data/outputs/lora/qwen3vl_medreason/final"
OUTPUT_DIR="${REPO_ROOT}/data/outputs/lora/qwen3vl_medreason_open"

cd "${APP_DIR}"
export PYTHONPATH="${APP_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

if [[ ! -d "${RESUME_ADAPTER}" ]]; then
  echo "[error] missing resume adapter: ${RESUME_ADAPTER}" >&2
  exit 1
fi

echo "[prepare] building open-upsampled dataset (open_repeat=${OPEN_REPEAT})"
python tools/prepare_lora_dataset.py \
  --train-json "${REPO_ROOT}/data/raw/train/medreason_train_selection.json" \
  --image-root "${REPO_ROOT}/data/raw/train/imgs" \
  --output "${DATASET_OUT}" \
  --open-repeat "${OPEN_REPEAT}" \
  --mcq-repeat 1 \
  --shuffle-seed 42

echo "[train] continuing LoRA from ${RESUME_ADAPTER}"
python tools/train_lora.py \
  --model-path "${APP_DIR}/models/Qwen3-VL-32B-Instruct" \
  --dataset "${DATASET_OUT}" \
  --image-root "${REPO_ROOT}/data/raw/train/imgs" \
  --repo-root "${REPO_ROOT}" \
  --output-dir "${OUTPUT_DIR}" \
  --resume-adapter "${RESUME_ADAPTER}" \
  --num-train-epochs "${EPOCHS}" \
  --per-device-train-batch-size 1 \
  --gradient-accumulation-steps 16 \
  --learning-rate "${LR}" \
  --lora-r 16 \
  --lora-alpha 32 \
  --logging-steps 10 \
  --save-steps 500 \
  --save-total-limit 3

echo "[done] adapter -> ${OUTPUT_DIR}/final"
