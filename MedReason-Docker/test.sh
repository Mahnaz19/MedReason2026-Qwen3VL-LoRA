#!/usr/bin/env bash
set -euo pipefail

IMAGE_NAME="${1:-medreason-docker:latest}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Git Bash on Windows converts paths in a way Docker cannot mount; use Windows paths.
if command -v pwd >/dev/null 2>&1 && pwd -W >/dev/null 2>&1; then
  WIN_SCRIPT_DIR="$(cd "${SCRIPT_DIR}" && pwd -W | sed 's|\\|/|g')"
  TEST_DIR="${WIN_SCRIPT_DIR}/test"
  HOST_OUTPUT_DIR="${2:-${WIN_SCRIPT_DIR}/output}"
else
  TEST_DIR="${SCRIPT_DIR}/test"
  HOST_OUTPUT_DIR="${2:-${SCRIPT_DIR}/output}"
fi
mkdir -p "${HOST_OUTPUT_DIR}"

# Use the lightweight smoke system for the template test.
docker run --rm \
  -e MEDREASON_SYSTEM=smoke \
  -v "${TEST_DIR}:/input:ro" \
  -v "${HOST_OUTPUT_DIR}:/output" \
  "${IMAGE_NAME}"

python tools/validate_output.py "${HOST_OUTPUT_DIR}/results.json" --input-json test/cases.json

echo "Smoke test completed successfully. Output: ${HOST_OUTPUT_DIR}/results.json"
