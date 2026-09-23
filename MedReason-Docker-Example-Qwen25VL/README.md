# MedReason Qwen2.5-VL Example Docker

This repository is the **MedReason example Docker baseline**. It is intentionally aligned with the VLM3D `reportgen_example_docker` style: a complete baseline system is packaged inside a Docker container, and the challenge evaluator only depends on a fixed input/output contract.

The baseline system is:

```text
Qwen2.5-VL checkpoint
        ↓
MedReason MCQ/open-ended prompt
        ↓
model generation
        ↓
JSON/text parser
        ↓
MCQ answer normalization
        ↓
/output/results.json
```

This is a reference baseline, not a restriction. MedReason submissions may be single-model MLLMs, agentic systems, retrieval-augmented systems, ensembles, visual-verification pipelines, or rule-based post-processing systems, provided that the submitted Docker image is self-contained and writes the required `results.json`.

## What this follows from VLM3D

The VLM3D example Docker implements a real CT-CHAT report-generation pipeline rather than a toy placeholder. It includes hard-wired container paths, baked model checkpoints, preprocessing/model-loading/generation code in `process.py`, a fixed JSON output schema, and `build.sh`/`test.sh`/`export.sh` scripts.

This MedReason example follows the same pattern:

```text
/input/cases.json          # official cases mounted by evaluator
/input/images/...          # official images mounted by evaluator
/opt/app/models/Qwen2.5-VL # model checkpoint baked into the image
/output/results.json       # predictions written by process.py
```

## Input Specification

Mount a directory to `/input` containing:

```text
/input/
  cases.json
  images/
    case_0001.png
    case_0002.png
    ...
```

`cases.json` must contain either a list of cases or an object with a `cases` list. Each case should include:

```json
{
  "case_id": "case_0001",
  "task_type": "mcq",
  "image_path": "images/case_0001.png",
  "question": "...",
  "options": [
    {"label": "A", "text": "..."},
    {"label": "B", "text": "..."}
  ]
}
```

For open-ended cases, `options` is omitted and `task_type` is `open`.

The baseline supports both `image_path` and `image_paths`.

## Model Checkpoint

For a real baseline image, place a self-contained Qwen2.5-VL checkpoint under:

```text
models/Qwen2.5-VL/
```

The Dockerfile copies this directory into:

```text
/opt/app/models/Qwen2.5-VL
```

During official evaluation, the container should not rely on internet access or external APIs.

## Output Specification

The container writes `/output/results.json`:

```json
{
  "name": "MedReason Qwen2.5-VL baseline predictions",
  "type": "Medical visual reasoning",
  "answers": [
    {
      "case_id": "case_0001",
      "task_type": "mcq",
      "answer": "A",
      "reasoning_trace": "brief image-grounded rationale",
      "confidence": 0.0
    },
    {
      "case_id": "case_0002",
      "task_type": "open",
      "answer": "concise final answer",
      "reasoning_trace": "image-grounded evidence and reasoning",
      "confidence": 0.0
    }
  ],
  "version": {"major": 1, "minor": 0}
}
```

## Testing

A smoke test is included so that the Docker I/O can be verified without a large model checkpoint:

```bash
./test.sh
```

The smoke test sets:

```bash
MEDREASON_SMOKE_TEST=1
```

This mode should not be used for real submissions.

## Running with a real checkpoint

After placing model files in `models/Qwen2.5-VL`, build and run normally:

```bash
./build.sh

docker run --rm   --gpus all   -v $PWD/test:/input   -v $PWD/output:/output   medreason-qwen25vl
```

## Exporting

```bash
./export.sh
```

This creates:

```text
MedReason-Qwen25VL.tar.gz
```

## Where to modify

The main baseline implementation is intentionally visible in:

```text
process.py
```

To build a stronger MedReason system, replace the `run_baseline`, `generate_with_qwen`, or parsing/normalization blocks while preserving the input/output schema.
