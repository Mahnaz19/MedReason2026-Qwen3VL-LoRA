#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import tempfile
import zipfile
from pathlib import Path
from typing import Any


def fail(message: str) -> None:
    raise SystemExit(f"[validate_submission][ERROR] {message}")


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
    }
    if key not in mapping:
        fail(f"Unsupported task_type: {task_type!r}")
    return mapping[key]


def extract_submission(path: Path) -> Path:
    if path.suffix.lower() == ".json":
        return path
    if path.suffix.lower() != ".zip":
        fail("Submission must be either results.json or a .zip archive containing results.json at the top level")

    with zipfile.ZipFile(path, "r") as zf:
        names = [n for n in zf.namelist() if not n.endswith("/")]
        if "results.json" not in names:
            fail("Zip archive must contain results.json directly at the top level")
        nested = [n for n in names if "/" in n]
        if nested:
            fail("Zip archive must not contain nested folders or extra files")
        tmpdir = Path(tempfile.mkdtemp(prefix="medreason_submission_"))
        zf.extract("results.json", tmpdir)
        return tmpdir / "results.json"


def load_answers(results_json: Path) -> list[dict[str, Any]]:
    payload = load_json(results_json)
    answers = payload.get("answers") if isinstance(payload, dict) else payload
    if not isinstance(answers, list):
        fail("results.json must contain an 'answers' list")
    if not all(isinstance(item, dict) for item in answers):
        fail("Every item in answers must be an object")
    return answers


def validate_basic(answers: list[dict[str, Any]]) -> None:
    seen: set[str] = set()
    for i, item in enumerate(answers):
        case_id = str(item.get("case_id", "")).strip()
        if not case_id:
            fail(f"answers[{i}].case_id is required")
        if case_id in seen:
            fail(f"Duplicate case_id: {case_id}")
        seen.add(case_id)
        task_type = normalize_task_type(str(item.get("task_type", "")))
        answer = item.get("answer")
        if not isinstance(answer, str) or not answer.strip():
            fail(f"answers[{i}].answer must be a non-empty string")
        reasoning = item.get("reasoning_trace", "")
        if task_type == "open" and (not isinstance(reasoning, str) or not reasoning.strip()):
            fail(f"answers[{i}].reasoning_trace is required and non-empty for open-ended cases")


def validate_against_cases(answers: list[dict[str, Any]], cases_json: Path) -> None:
    payload = load_json(cases_json)
    cases = payload.get("cases") if isinstance(payload, dict) else payload
    if not isinstance(cases, list):
        fail("cases_json must contain a 'cases' list")
    answer_by_id = {str(item["case_id"]): item for item in answers}
    case_by_id = {str(item["case_id"]): item for item in cases}

    missing = sorted(set(case_by_id) - set(answer_by_id))
    extra = sorted(set(answer_by_id) - set(case_by_id))
    if missing:
        fail(f"Missing predictions for case_id(s): {missing[:10]}{'...' if len(missing) > 10 else ''}")
    if extra:
        fail(f"Submission contains unknown case_id(s): {extra[:10]}{'...' if len(extra) > 10 else ''}")

    for case_id, case in case_by_id.items():
        item = answer_by_id[case_id]
        gt_task = normalize_task_type(str(case.get("task_type", "")))
        pred_task = normalize_task_type(str(item.get("task_type", "")))
        if gt_task != pred_task:
            fail(f"case_id={case_id} task_type mismatch: expected {gt_task}, got {pred_task}")
        if gt_task == "mcq":
            labels = {str(opt.get("label", "")).strip() for opt in case.get("options", []) if isinstance(opt, dict)}
            answer = str(item.get("answer", "")).strip()
            if labels and answer not in labels:
                fail(f"case_id={case_id} answer must be one of {sorted(labels)}, got {answer!r}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate a MedReason validation submission archive.")
    parser.add_argument("submission", type=Path, help="submission.zip or results.json")
    parser.add_argument("--cases-json", type=Path, default=None, help="Optional validation cases.json for strict validation")
    args = parser.parse_args()

    results_json = extract_submission(args.submission)
    answers = load_answers(results_json)
    validate_basic(answers)
    if args.cases_json is not None:
        validate_against_cases(answers, args.cases_json)
    print(f"[validate_submission] OK: {len(answers)} predictions validated")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
