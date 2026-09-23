#!/usr/bin/env bash
set -euo pipefail
IMAGE_TAG="${1:-medreason-qwen25vl}"
ARCHIVE_NAME="${2:-MedReason-Qwen25VL.tar.gz}"
./build.sh "${IMAGE_TAG}"
docker save "${IMAGE_TAG}" | gzip -c > "${ARCHIVE_NAME}"
echo "Exported ${ARCHIVE_NAME}"
