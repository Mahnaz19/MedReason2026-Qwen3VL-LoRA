# MedReason-Evaluation

This repository contains the public submission validator and metric aggregation code for the **MedReason Challenge**.

It is intentionally separate from `MedReason-Docker`:

- `MedReason-Docker` shows how participants package a complete MLLM system.
- `MedReason-Evaluation` defines submission format validation and public metric aggregation.

## Official metrics

The challenge reports three official leaderboard metrics:

1. **MCQ Accuracy**: exact-match accuracy for closed-ended multiple-choice questions.
2. **Open-ended GT**: mean organizer-side ground-truth correctness score for open-ended responses.
3. **Open-ended VA**: mean visual-accuracy score after threshold-gated aggregation.

For open-ended cases, participants submit both `reasoning_trace` and `answer`. During official evaluation, organizer-side judges assign case-level `GT_final`, `VA_answer`, and `RVF_trace` scores. The public `scoring.py` defines how these scores are aggregated.

## Validate submission format

```bash
python validate_submission.py sample_submission.zip --cases-json test/cases.json
```

The expected archive structure is:

```text
submission.zip
└── results.json
```

Do not place `results.json` inside an extra folder.

## Run public scoring example

```bash
python scoring.py \
  --predictions sample_submission/results.json \
  --ground-truth test/ground_truth.json \
  --judge-scores test/judge_scores.json \
  --pretty
```

## VA aggregation rule

```python
if RVF_trace <= 1:
    VA_final = min(VA_answer, 1)
elif RVF_trace == 2:
    VA_final = min(VA_answer, 3)
else:
    VA_final = VA_answer
```

The public script does not call an LLM judge and does not expose hidden-test labels.
