#!/usr/bin/env python3
"""Build a ground-truth JSON file from the Synapse train package for local MCQ scoring."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _load_cases(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    cases = payload.get("cases") if isinstance(payload, dict) else payload
    if not isinstance(cases, list):
        raise ValueError("Expected a JSON object with a 'cases' list.")
    return cases


def _normalize_task_type(question_type: str) -> str:
    key = str(question_type).strip().lower()
    if key in {"mcq", "multiple-choice", "multiple_choice", "closed", "closed-ended", "closed_ended"}:
        return "mcq"
    if "open" in key:
        return "open"
    raise ValueError(f"Unsupported question type: {question_type!r}")


def build_ground_truth(train_json: Path, *, mcq_only: bool) -> dict[str, Any]:
    cases = _load_cases(train_json)
    out_cases: list[dict[str, Any]] = []
    for raw in cases:
        task_type = _normalize_task_type(str(raw.get("question type", "")))
        if mcq_only and task_type != "mcq":
            continue
        answer = raw.get("answer")
        if answer is None or not str(answer).strip():
            continue
        out_cases.append(
            {
                "case_id": str(raw["case_id"]).strip(),
                "task_type": task_type,
                "answer": str(answer).strip(),
            }
        )
    return {"cases": out_cases}


def main() -> int:
    parser = argparse.ArgumentParser(description="Build train ground-truth JSON for local scoring.")
    parser.add_argument(
        "--train-json",
        type=Path,
        default=Path("../data/raw/train/medreason_train_selection.json"),
        help="Path to medreason_train_selection.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("../data/derived/train_ground_truth_mcq.json"),
        help="Output ground-truth JSON path",
    )
    parser.add_argument("--mcq-only", action="store_true", default=True, help="Include MCQ cases only (default).")
    parser.add_argument("--include-open", action="store_true", help="Also include open-ended reference answers.")
    args = parser.parse_args()

    mcq_only = not args.include_open
    payload = build_ground_truth(args.train_json, mcq_only=mcq_only)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"[build_train_ground_truth] wrote {len(payload['cases'])} cases -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
