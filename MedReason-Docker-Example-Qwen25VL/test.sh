#!/usr/bin/env bash
set -euo pipefail

SCRIPTPATH="$( cd "$(dirname "$0")" ; pwd -P )"
IMAGE_TAG="${1:-medreason-qwen25vl}"

./build.sh "${IMAGE_TAG}"

VOLUME_SUFFIX=$(dd if=/dev/urandom bs=32 count=1 2>/dev/null | md5sum | cut --delimiter=' ' --fields=1)
MEM_LIMIT="30g"
OUTPUT_VOLUME="medreason-output-${VOLUME_SUFFIX}"

docker volume create "${OUTPUT_VOLUME}" >/dev/null

cleanup() {
    docker volume rm "${OUTPUT_VOLUME}" >/dev/null || true
}
trap cleanup EXIT

# Do not change the core docker run parameters for official-style local testing.
docker run --rm         --gpus all         --memory="${MEM_LIMIT}"         --memory-swap="${MEM_LIMIT}"         --network="bridge"         --cap-drop="ALL"         --security-opt="no-new-privileges"         --shm-size="2g"         --pids-limit="512"         -e MEDREASON_SMOKE_TEST=1         -v "${SCRIPTPATH}/test/":/input/         -v "${OUTPUT_VOLUME}":/output/         "${IMAGE_TAG}"


docker run --rm         -v "${OUTPUT_VOLUME}":/output/         python:3.11-slim cat /output/results.json | python -m json.tool


docker run --rm         -v "${OUTPUT_VOLUME}":/output/         -v "${SCRIPTPATH}/test/":/input/         python:3.11-slim python -c "import json, sys; f1=json.load(open('/output/results.json')); f2=json.load(open('/input/expected_output.json')); sys.exit(f1 != f2)"

if [ $? -eq 0 ]; then
    echo "Tests successfully passed..."
else
    echo "Expected output was not found..."
    exit 1
fi
