#!/usr/bin/env bash
set -euo pipefail

IMAGE_NAME="${1:-medreason-docker:latest}"
docker build -t "${IMAGE_NAME}" .
