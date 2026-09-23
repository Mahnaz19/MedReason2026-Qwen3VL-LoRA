#!/usr/bin/env bash
# Final pre-submit safety checks with the open-continued LoRA adapter.
# 1) Train MCQ cases 1-500
# 2) Train MCQ cases 501-1000
# 3) Open-ended format smoke (100 train open cases)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
APP_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
REPO_ROOT="$(cd "${APP_DIR}/.." && pwd)"

LORA_PATH="${MEDREASON_LORA_PATH:-${REPO_ROOT}/data/outputs/lora/qwen3vl_medreason_open/final}"
MODEL_PATH="${MEDREASON_MODEL_PATH:-${APP_DIR}/models/Qwen3-VL-32B-Instruct}"
OUT_ROOT="${REPO_ROOT}/data/outputs/isambard/final_precheck"

cd "${APP_DIR}"
export PYTHONPATH="${APP_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
export MEDREASON_SYSTEM=custom
export MEDREASON_MODEL_PATH="${MODEL_PATH}"
export MEDREASON_LORA_PATH="${LORA_PATH}"
export MEDREASON_TEMPERATURE=0.0
export MEDREASON_MAX_NEW_TOKENS=512

if [[ ! -d "${LORA_PATH}" ]]; then
  echo "[error] missing LoRA adapter: ${LORA_PATH}" >&2
  exit 1
fi

echo "[precheck] using LoRA: ${LORA_PATH}"
mkdir -p "${OUT_ROOT}"

# Ensure ground truth exists
python tools/build_train_ground_truth.py

run_mcq() {
  local offset="$1"
  local count="$2"
  local tag="$3"
  local out_dir="${OUT_ROOT}/${tag}"
  mkdir -p "${out_dir}"
  echo "[precheck] MCQ ${tag} (offset=${offset}, count=${count})"
  export MEDREASON_INPUT_DIR="${REPO_ROOT}/data/raw/train"
  export MEDREASON_OUTPUT_FILE="${out_dir}/results.json"
  export MEDREASON_CASE_OFFSET="${offset}"
  export MEDREASON_MAX_CASES="${count}"
  python process.py
  python "${REPO_ROOT}/MedReason-Evaluation/scoring.py" \
    --predictions "${out_dir}/results.json" \
    --ground-truth "${REPO_ROOT}/data/derived/train_ground_truth_mcq.json" \
    --pretty \
    --allow-subset \
    | tee "${out_dir}/score.txt"
}

run_mcq 0 500 "mcq_1_500"
run_mcq 500 500 "mcq_501_1000"

echo "[precheck] building open subset (100 cases)"
OPEN_DIR="${OUT_ROOT}/open_100"
OPEN_INPUT="${OUT_ROOT}/open_100_input"
mkdir -p "${OPEN_DIR}" "${OPEN_INPUT}"
python tools/make_subset_cases.py \
  --input-json "${REPO_ROOT}/data/raw/train/cases.json" \
  --output-json "${OPEN_INPUT}/cases.json" \
  --task-type open \
  --limit 100

# Point image_path entries at the shared train imgs directory via symlink.
if [[ ! -e "${OPEN_INPUT}/imgs" ]]; then
  ln -sfn "${REPO_ROOT}/data/raw/train/imgs" "${OPEN_INPUT}/imgs"
fi

unset MEDREASON_CASE_OFFSET || true
unset MEDREASON_MAX_CASES || true
export MEDREASON_INPUT_DIR="${OPEN_INPUT}"
export MEDREASON_OUTPUT_FILE="${OPEN_DIR}/results.json"

echo "[precheck] open inference (100 cases)"
python process.py
python tools/validate_output.py "${OPEN_DIR}/results.json" --input-json "${OPEN_INPUT}/cases.json"
python tools/analyze_open_outputs.py \
  --predictions "${OPEN_DIR}/results.json" \
  --train-json "${REPO_ROOT}/data/raw/train/medreason_train_selection.json" \
  | tee "${OPEN_DIR}/analyze.txt"

echo "[precheck] DONE"
echo "  MCQ 1-500:     ${OUT_ROOT}/mcq_1_500/score.txt"
echo "  MCQ 501-1000:  ${OUT_ROOT}/mcq_501_1000/score.txt"
echo "  Open analyze:  ${OPEN_DIR}/analyze.txt"
