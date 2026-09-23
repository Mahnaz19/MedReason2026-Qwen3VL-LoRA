#!/usr/bin/env bash
set -euo pipefail
SCRIPTPATH="$( cd "$(dirname "$0")" ; pwd -P )"
IMAGE_TAG="${1:-medreason-qwen25vl}"
docker build -t "${IMAGE_TAG}" "${SCRIPTPATH}"
