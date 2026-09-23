#!/usr/bin/env bash
# Run inference on train MCQ subset and score locally (requires GPU).
# Usage:
#   run_gpu_train_mcq_eval.sh           # first 500 cases
#   run_gpu_train_mcq_eval.sh 500       # first 500 cases (backward compatible)
#   run_gpu_train_mcq_eval.sh 500 500   # cases 501-1000 (offset 500, count 500)
set -euo pipefail

if [[ $# -eq 1 ]]; then
  CASE_OFFSET=0
  MAX_CASES="$1"
elif [[ $# -ge 2 ]]; then
  CASE_OFFSET="$1"
  MAX_CASES="$2"
else
  CASE_OFFSET=0
  MAX_CASES=500
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
APP_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
REPO_ROOT="$(cd "${APP_DIR}/.." && pwd)"
START=$((CASE_OFFSET + 1))
END=$((CASE_OFFSET + MAX_CASES))
OUTPUT_DIR="${REPO_ROOT}/data/outputs/isambard/train_mcq_${START}_${END}"

mkdir -p "${OUTPUT_DIR}"

cd "${APP_DIR}"
python tools/build_train_ground_truth.py

export MEDREASON_SYSTEM=custom
export MEDREASON_MODEL_PATH="${APP_DIR}/models/Qwen3-VL-32B-Instruct"
export MEDREASON_INPUT_DIR="${REPO_ROOT}/data/raw/train"
export MEDREASON_OUTPUT_FILE="${OUTPUT_DIR}/results.json"
export MEDREASON_CASE_OFFSET="${CASE_OFFSET}"
export MEDREASON_MAX_CASES="${MAX_CASES}"

python process.py

cd "${REPO_ROOT}/MedReason-Evaluation"
SCORE_ARGS=(--predictions "${OUTPUT_DIR}/results.json" --ground-truth "${REPO_ROOT}/data/derived/train_ground_truth_mcq.json" --pretty --allow-subset)
python scoring.py "${SCORE_ARGS[@]}"
