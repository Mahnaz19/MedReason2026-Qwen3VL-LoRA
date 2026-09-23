#!/usr/bin/env python3
"""Summarize MedReason train/validation packages (no GPU required)."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


def _load_cases(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    cases = payload.get("cases") if isinstance(payload, dict) else payload
    if not isinstance(cases, list):
        raise ValueError(f"{path} must contain a 'cases' list.")
    return cases


def summarize(name: str, cases: list[dict[str, Any]]) -> dict[str, Any]:
    task_counts = Counter(str(c.get("question type", "unknown")).lower() for c in cases)
    modality_counts = Counter(str(c.get("primary_modality", c.get("modality", "unknown"))) for c in cases)
    organ_counts = Counter(str(c.get("organ system", "unknown")) for c in cases)
    return {
        "split": name,
        "total_cases": len(cases),
        "task_type_counts": dict(task_counts),
        "top_modalities": dict(modality_counts.most_common(10)),
        "top_organ_systems": dict(organ_counts.most_common(10)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Print dataset statistics.")
    parser.add_argument("--train-json", type=Path, default=Path("../data/raw/train/medreason_train_selection.json"))
    parser.add_argument("--validation-json", type=Path, default=Path("../data/raw/validation/medreason_validation_participant_facing.json"))
    parser.add_argument("--output", type=Path, default=Path("../data/derived/dataset_stats.json"))
    args = parser.parse_args()

    stats = [
        summarize("train", _load_cases(args.train_json)),
        summarize("validation", _load_cases(args.validation_json)),
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for item in stats:
        print(json.dumps(item, ensure_ascii=False, indent=2))
    print(f"[dataset_stats] saved -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
