# MedReason-Docker

This repository is the participant-facing Docker template for the **MedReason Challenge**.
It defines the required container interface for submitting a complete medical MLLM reasoning system.

Participants are **not** restricted to a single HuggingFace VLM. A valid submission may use a single MLLM, multiple models, retrieval modules, agentic reasoning, self-verification, uncertainty estimation, rule-based post-processing, or any other self-contained workflow, as long as it reads the official input files and writes the required `results.json` output.

## Container contract

The official evaluator will run the submitted Docker image with mounted `/input` and `/output` directories.

Your container must:

1. Read `/input/cases.json`.
2. Read all referenced image files under `/input`.
3. Write `/output/results.json`.
4. Return exactly one prediction for every input `case_id`.

No internet access, external API calls, or manual interaction should be required during official evaluation.

## Main files

```text
process.py                              # official container entry point
medreason_docker/systems/base.py         # generic MedReasonSystem interface
medreason_docker/systems/custom_system.py # participant implementation entry point
medreason_docker/systems/hf_vlm_system.py # optional single-model HF VLM example
medreason_docker/systems/smoke_system.py  # lightweight I/O smoke-test system
tools/validate_output.py                 # local output validator
```

## Implement your system

Edit:

```text
medreason_docker/systems/custom_system.py
```

The default `custom` system wraps **Qwen3-VL** (or other Qwen VL checkpoints) with MedReason-specific prompts and parsing.
Place a checkpoint under `models/` (see `models/README.md`) and set `MEDREASON_MODEL_PATH`.

For a quick I/O test without a checkpoint:

```bash
MEDREASON_SYSTEM=smoke python process.py
```

For real inference:

```bash
MEDREASON_SYSTEM=custom \
MEDREASON_MODEL_PATH=./models/Qwen3-VL-32B-Instruct \
MEDREASON_INPUT_DIR=../data/raw/validation \
MEDREASON_OUTPUT_FILE=../data/outputs/validation/results.json \
MEDREASON_MAX_CASES=10 \
python process.py
```

See `docs/pre_evaluation_plan.md` for the full pre-evaluation workflow.

## Local Python smoke test

```bash
MEDREASON_SYSTEM=smoke \
MEDREASON_INPUT_DIR=./test \
MEDREASON_OUTPUT_FILE=./output/results.json \
python process.py

python tools/validate_output.py output/results.json --input-json test/cases.json
```

## Docker test

```bash
./build.sh medreason-docker:latest
./test.sh medreason-docker:latest
```

## Export for submission

```bash
./export.sh medreason-docker:latest medreason-docker.tar.gz
```

## Optional HuggingFace VLM example

`medreason_docker/systems/hf_vlm_system.py` is a reference implementation for a single HuggingFace VLM. It is not a challenge restriction. To use it, install the required dependencies and set:

```bash
MEDREASON_SYSTEM=hf_vlm
MEDREASON_MODEL_PATH=/path/or/hf/model/id
```

The default Dockerfile installs only lightweight dependencies so that the template remains easy to test. Add your model-specific dependencies to `requirements.txt` when preparing a real submission.

## Reference baseline example

This repository is the minimal, model-agnostic submission template. A separate reference baseline is provided in:

```text
../MedReason-Docker-Example-Qwen25VL
```

That example implements a single-model Qwen2.5-VL MedReason system with task-specific prompting, robust JSON parsing, MCQ answer normalization, and open-ended answer formatting. It is provided as a starting point only. Participants may submit any complete MLLM system as long as the official input/output contract is satisfied.
