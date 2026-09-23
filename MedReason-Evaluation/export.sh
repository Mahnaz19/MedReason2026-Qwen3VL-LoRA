#!/usr/bin/env bash
set -euo pipefail
IMAGE_NAME="${1:-medreason-evaluation:latest}"
OUTPUT_TAR="${2:-medreason-evaluation.tar.gz}"
docker save "${IMAGE_NAME}" | gzip > "${OUTPUT_TAR}"
echo "Exported Docker image to ${OUTPUT_TAR}"
