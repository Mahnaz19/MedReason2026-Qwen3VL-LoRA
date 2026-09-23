#!/usr/bin/env bash
# Run real Qwen VL inference on validation (requires GPU).
set -euo pipefail

MAX_CASES="${1:-10}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
APP_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
REPO_ROOT="$(cd "${APP_DIR}/.." && pwd)"
OUTPUT_DIR="${REPO_ROOT}/data/outputs/isambard/validation_${MAX_CASES}"

mkdir -p "${OUTPUT_DIR}"

export MEDREASON_SYSTEM=custom
export MEDREASON_MODEL_PATH="${APP_DIR}/models/Qwen3-VL-32B-Instruct"
export MEDREASON_INPUT_DIR="${REPO_ROOT}/data/raw/validation"
export MEDREASON_OUTPUT_FILE="${OUTPUT_DIR}/results.json"
export MEDREASON_MAX_CASES="${MAX_CASES}"

cd "${APP_DIR}"
python process.py
VALIDATE_ARGS=(--input-json "${MEDREASON_INPUT_DIR}/cases.json")
if [[ -n "${MEDREASON_MAX_CASES:-}" ]]; then
  VALIDATE_ARGS+=(--allow-subset)
fi
python tools/validate_output.py "${MEDREASON_OUTPUT_FILE}" "${VALIDATE_ARGS[@]}"
echo "Done. Output: ${MEDREASON_OUTPUT_FILE}"
