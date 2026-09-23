#!/usr/bin/env python3
"""Prepare train split JSONL for LoRA / SFT fine-tuning (no GPU required)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterator


def _iter_train_cases(train_json: Path) -> Iterator[dict[str, Any]]:
    payload = json.loads(train_json.read_text(encoding="utf-8"))
    cases = payload.get("cases") if isinstance(payload, dict) else payload
    if not isinstance(cases, list):
        raise ValueError("Expected train JSON with a 'cases' list.")
    for raw in cases:
        if isinstance(raw, dict):
            yield raw


def _normalize_task_type(question_type: str) -> str:
    key = str(question_type).strip().lower()
    if key in {"mcq", "multiple-choice", "multiple_choice", "closed", "closed-ended", "closed_ended"}:
        return "mcq"
    if "open" in key:
        return "open"
    raise ValueError(f"Unsupported question type: {question_type!r}")


def _image_rel_path(image_path_value: str) -> str:
    return Path(str(image_path_value)).name


def _build_target(raw: dict[str, Any], task_type: str) -> dict[str, str]:
    if task_type == "mcq":
        answer = str(raw.get("answer", "")).strip()
        return {
            "reasoning_trace": f"Based on the image evidence, option {answer} is most consistent with the findings.",
            "answer": answer,
        }
    answer = str(raw.get("answer", "")).strip()
    visual = str(raw.get("visual_description", "")).strip()
    reasoning = visual or "Image-grounded reasoning based on the visible findings."
    return {"reasoning_trace": reasoning, "answer": answer}


def convert_case(raw: dict[str, Any], *, image_root: Path) -> dict[str, Any] | None:
    case_id = str(raw.get("case_id", "")).strip()
    if not case_id:
        return None
    task_type = _normalize_task_type(str(raw.get("question type", "")))
    question = str(raw.get("question", "")).strip()
    filename = _image_rel_path(str(raw.get("image_path", "")))
    image_abs = image_root / filename
    if not image_abs.exists():
        return None

    record: dict[str, Any] = {
        "case_id": case_id,
        "task_type": task_type,
        "image_path": str(image_abs),
        "question": question,
    }
    if task_type == "mcq":
        options = []
        for label in ["A", "B", "C", "D", "E"]:
            text = str(raw.get(label, "")).strip()
            if text:
                options.append({"label": label, "text": text})
        record["options"] = options
    record["target"] = _build_target(raw, task_type)
    return record


def main() -> int:
    parser = argparse.ArgumentParser(description="Build JSONL fine-tuning dataset from Synapse train JSON.")
    parser.add_argument(
        "--train-json",
        type=Path,
        default=Path("../data/raw/train/medreason_train_selection.json"),
    )
    parser.add_argument(
        "--image-root",
        type=Path,
        default=Path("../data/raw/train/imgs"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("../data/derived/train_lora.jsonl"),
    )
    parser.add_argument("--limit", type=int, default=None, help="Optional cap for quick experiments.")
    parser.add_argument(
        "--open-repeat",
        type=int,
        default=1,
        help="Repeat each open-ended record this many times to upsample open cases.",
    )
    parser.add_argument(
        "--mcq-repeat",
        type=int,
        default=1,
        help="Repeat each MCQ record this many times (usually keep at 1).",
    )
    parser.add_argument(
        "--shuffle-seed",
        type=int,
        default=42,
        help="Shuffle seed after upsampling (set <0 to disable shuffle).",
    )
    args = parser.parse_args()

    if args.open_repeat < 1 or args.mcq_repeat < 1:
        raise ValueError("--open-repeat and --mcq-repeat must be >= 1")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    skipped = 0
    n_mcq = 0
    n_open = 0
    for raw in _iter_train_cases(args.train_json):
        record = convert_case(raw, image_root=args.image_root)
        if record is None:
            skipped += 1
            continue
        repeats = args.open_repeat if record["task_type"] == "open" else args.mcq_repeat
        for _ in range(repeats):
            records.append(record)
        if record["task_type"] == "open":
            n_open += 1
        else:
            n_mcq += 1
        if args.limit is not None and len(records) >= args.limit:
            records = records[: args.limit]
            break

    if args.shuffle_seed >= 0:
        import random

        rng = random.Random(args.shuffle_seed)
        rng.shuffle(records)

    with args.output.open("w", encoding="utf-8") as out:
        for record in records:
            out.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(
        f"[prepare_lora_dataset] wrote {len(records)} records -> {args.output} "
        f"(unique mcq={n_mcq}, unique open={n_open}, "
        f"open_repeat={args.open_repeat}, mcq_repeat={args.mcq_repeat})"
    )
    if skipped:
        print(f"[prepare_lora_dataset] skipped {skipped} records (missing image or fields)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
