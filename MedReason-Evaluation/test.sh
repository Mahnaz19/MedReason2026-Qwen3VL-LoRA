#!/usr/bin/env bash
set -euo pipefail
python validate_submission.py sample_submission.zip --cases-json test/cases.json
python scoring.py \
  --predictions sample_submission/results.json \
  --ground-truth test/ground_truth.json \
  --judge-scores test/judge_scores.json \
  --output test/metrics.json \
  --pretty
cat test/metrics.json
