#!/usr/bin/env python3
"""Public MedReason metric aggregation script.

This script defines the official aggregation procedure for the public metrics:
MCQ Accuracy, Open-ended GT, and Open-ended VA.

For official hidden-test evaluation, case-level open-ended judge scores are
produced organizer-side and then aggregated by this script. The public script
therefore does not call any external LLM judge and does not expose hidden labels.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any


@dataclass(frozen=True)
class Prediction:
    case_id: str
    task_type: str
    answer: str
    reasoning_trace: str


@dataclass(frozen=True)
class GroundTruthCase:
    case_id: str
    task_type: str
    answer: str | None = None


@dataclass(frozen=True)
class OpenJudgeScore:
    case_id: str
    gt_final: float
    va_answer: float
    rvf_trace: float


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def normalize_task_type(task_type: str) -> str:
    key = str(task_type).strip().lower()
    mapping = {
        "mcq": "mcq",
        "closed": "mcq",
        "closed_ended": "mcq",
        "closed-ended": "mcq",
        "multiple_choice": "mcq",
        "open": "open",
        "open_ended": "open",
        "open-ended": "open",
        "free_text": "open",
    }
    if key not in mapping:
        raise ValueError(f"Unsupported task_type: {task_type!r}")
    return mapping[key]


def load_predictions(path: Path) -> dict[str, Prediction]:
    payload = load_json(path)
    answers = payload.get("answers") if isinstance(payload, dict) else payload
    if not isinstance(answers, list):
        raise ValueError("Predictions must contain an 'answers' list")

    predictions: dict[str, Prediction] = {}
    for idx, item in enumerate(answers):
        if not isinstance(item, dict):
            raise ValueError(f"answers[{idx}] must be an object")
        case_id = str(item.get("case_id", "")).strip()
        if not case_id:
            raise ValueError(f"answers[{idx}].case_id is required")
        if case_id in predictions:
            raise ValueError(f"Duplicate prediction for case_id={case_id}")
        task_type = normalize_task_type(str(item.get("task_type", "")))
        answer = str(item.get("answer", "")).strip()
        reasoning_trace = str(item.get("reasoning_trace", "")).strip()
        if not answer:
            raise ValueError(f"case_id={case_id} answer is required")
        if task_type == "open" and not reasoning_trace:
            raise ValueError(f"case_id={case_id} reasoning_trace is required for open-ended cases")
        predictions[case_id] = Prediction(case_id, task_type, answer, reasoning_trace)
    return predictions


def load_ground_truth(path: Path) -> dict[str, GroundTruthCase]:
    payload = load_json(path)
    raw_cases = payload.get("cases") if isinstance(payload, dict) else payload
    if not isinstance(raw_cases, list):
        raise ValueError("Ground truth must contain a 'cases' list")

    cases: dict[str, GroundTruthCase] = {}
    for idx, item in enumerate(raw_cases):
        case_id = str(item.get("case_id", "")).strip()
        if not case_id:
            raise ValueError(f"ground_truth cases[{idx}].case_id is required")
        if case_id in cases:
            raise ValueError(f"Duplicate ground-truth case_id={case_id}")
        task_type = normalize_task_type(str(item.get("task_type", "")))
        answer = item.get("answer")
        cases[case_id] = GroundTruthCase(
            case_id=case_id,
            task_type=task_type,
            answer=str(answer).strip() if answer is not None else None,
        )
    return cases


def load_judge_scores(path: Path | None) -> dict[str, OpenJudgeScore]:
    if path is None:
        return {}
    payload = load_json(path)
    raw_scores = payload.get("scores") if isinstance(payload, dict) else payload
    if not isinstance(raw_scores, list):
        raise ValueError("Judge scores must contain a 'scores' list")

    scores: dict[str, OpenJudgeScore] = {}
    for idx, item in enumerate(raw_scores):
        case_id = str(item.get("case_id", "")).strip()
        if not case_id:
            raise ValueError(f"judge_scores[{idx}].case_id is required")
        if case_id in scores:
            raise ValueError(f"Duplicate judge score for case_id={case_id}")
        scores[case_id] = OpenJudgeScore(
            case_id=case_id,
            gt_final=float(item["gt_final"]),
            va_answer=float(item["va_answer"]),
            rvf_trace=float(item["rvf_trace"]),
        )
    return scores


def compute_va_final(va_answer: float, rvf_trace: float) -> float:
    """Threshold-gated visual-accuracy aggregation."""
    if rvf_trace <= 1:
        return min(va_answer, 1)
    if rvf_trace == 2:
        return min(va_answer, 3)
    return va_answer


def score(
    predictions: dict[str, Prediction],
    ground_truth: dict[str, GroundTruthCase],
    judge_scores: dict[str, OpenJudgeScore],
    *,
    allow_subset: bool = False,
) -> dict[str, Any]:
    missing = sorted(set(ground_truth) - set(predictions))
    extra = sorted(set(predictions) - set(ground_truth))
    if not allow_subset and missing:
        raise ValueError(f"Missing predictions for case_id(s): {missing[:10]}{'...' if len(missing) > 10 else ''}")
    if extra:
        raise ValueError(f"Predictions contain unknown case_id(s): {extra[:10]}{'...' if len(extra) > 10 else ''}")

    scored_ground_truth = (
        {case_id: ground_truth[case_id] for case_id in predictions if case_id in ground_truth}
        if allow_subset
        else ground_truth
    )

    mcq_correct: list[float] = []
    open_gt: list[float] = []
    open_va: list[float] = []
    per_case: list[dict[str, Any]] = []

    for case_id, gt in scored_ground_truth.items():
        pred = predictions[case_id]
        if pred.task_type != gt.task_type:
            raise ValueError(f"case_id={case_id} task_type mismatch: expected {gt.task_type}, got {pred.task_type}")

        if gt.task_type == "mcq":
            if gt.answer is None:
                raise ValueError(f"case_id={case_id} MCQ ground truth answer is missing")
            correct = float(pred.answer.strip() == gt.answer.strip())
            mcq_correct.append(correct)
            per_case.append({"case_id": case_id, "task_type": "mcq", "mcq_correct": correct})
        else:
            if case_id not in judge_scores:
                raise ValueError(
                    f"case_id={case_id} is open-ended but no organizer-side judge score was provided. "
                    "Pass --judge-scores for open-ended metric aggregation."
                )
            js = judge_scores[case_id]
            va_final = compute_va_final(js.va_answer, js.rvf_trace)
            open_gt.append(js.gt_final)
            open_va.append(va_final)
            per_case.append(
                {
                    "case_id": case_id,
                    "task_type": "open",
                    "GT_final": js.gt_final,
                    "VA_answer": js.va_answer,
                    "RVF_trace": js.rvf_trace,
                    "VA_final": va_final,
                }
            )

    metrics = {
        "MCQ Accuracy": mean(mcq_correct) if mcq_correct else None,
        "Open-ended GT": mean(open_gt) if open_gt else None,
        "Open-ended VA": mean(open_va) if open_va else None,
    }
    counts = {
        "num_cases": len(scored_ground_truth),
        "num_mcq": len(mcq_correct),
        "num_open": len(open_gt),
    }
    return {"metrics": metrics, "counts": counts, "per_case": per_case}


def main() -> int:
    parser = argparse.ArgumentParser(description="Aggregate MedReason public metrics.")
    parser.add_argument("--predictions", type=Path, required=True, help="Path to submitted results.json")
    parser.add_argument("--ground-truth", type=Path, required=True, help="Path to ground truth JSON")
    parser.add_argument("--judge-scores", type=Path, default=None, help="Organizer-side open-ended judge scores JSON")
    parser.add_argument("--output", type=Path, default=None, help="Optional output metrics JSON")
    parser.add_argument("--pretty", action="store_true", help="Pretty-print JSON output")
    parser.add_argument(
        "--allow-subset",
        action="store_true",
        help="Score only cases present in predictions (for subset inference runs)",
    )
    args = parser.parse_args()

    result = score(
        predictions=load_predictions(args.predictions),
        ground_truth=load_ground_truth(args.ground_truth),
        judge_scores=load_judge_scores(args.judge_scores),
        allow_subset=args.allow_subset,
    )
    text = json.dumps(result, ensure_ascii=False, indent=2 if args.pretty else None)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
