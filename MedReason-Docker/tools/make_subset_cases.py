#!/usr/bin/env python3
"""Create a cases.json subset, optionally filtered by task type."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _normalize_task(task: str) -> str:
    key = str(task).strip().lower()
    if key in {"mcq", "multiple-choice", "multiple_choice", "closed", "closed-ended", "closed_ended"}:
        return "mcq"
    if "open" in key:
        return "open"
    return key


def main() -> int:
    parser = argparse.ArgumentParser(description="Write a cases.json subset for quick tests.")
    parser.add_argument("--input-json", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument(
        "--task-type",
        type=str,
        default="",
        help='Optional filter: "mcq" or "open".',
    )
    parser.add_argument("--offset", type=int, default=0, help="Skip this many matching cases first.")
    args = parser.parse_args()

    payload: Any = json.loads(args.input_json.read_text(encoding="utf-8"))
    cases = payload.get("cases") if isinstance(payload, dict) else payload
    if not isinstance(cases, list):
        raise ValueError("Input must contain a 'cases' list.")

    wanted = _normalize_task(args.task_type) if args.task_type.strip() else ""
    filtered: list[Any] = []
    for case in cases:
        if not isinstance(case, dict):
            continue
        task = _normalize_task(str(case.get("task_type", case.get("question type", ""))))
        if wanted and task != wanted:
            continue
        filtered.append(case)

    selected = filtered[args.offset : args.offset + args.limit]
    subset = {"cases": selected}
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(subset, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        f"[make_subset_cases] wrote {len(selected)} cases "
        f"(filter={wanted or 'all'}, offset={args.offset}) -> {args.output_json}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
