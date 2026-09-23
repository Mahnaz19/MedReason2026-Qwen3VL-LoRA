#!/usr/bin/env bash
set -euo pipefail
IMAGE_NAME="${1:-medreason-evaluation:latest}"
docker build -t "${IMAGE_NAME}" .
