#!/usr/bin/env bash
# Download Qwen3-VL-32B-Instruct into models/ (run on Isambard login or GPU node).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
APP_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
MODEL_DIR="${APP_DIR}/models/Qwen3-VL-32B-Instruct"

mkdir -p "${MODEL_DIR}"
hf download Qwen/Qwen3-VL-32B-Instruct --local-dir "${MODEL_DIR}"
echo "Downloaded to ${MODEL_DIR}"
