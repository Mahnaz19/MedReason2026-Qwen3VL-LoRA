#!/usr/bin/env python3
"""Analyze open-ended predictions for format / length sanity checks."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


def _load_answers(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    answers = payload.get("answers") if isinstance(payload, dict) else payload
    if not isinstance(answers, list):
        raise ValueError("Expected results.json with an answers list.")
    return [a for a in answers if isinstance(a, dict)]


def _load_gt_open(path: Path) -> dict[str, str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    cases = payload.get("cases") if isinstance(payload, dict) else payload
    out: dict[str, str] = {}
    if not isinstance(cases, list):
        return out
    for raw in cases:
        if not isinstance(raw, dict):
            continue
        qtype = str(raw.get("question type", "")).lower()
        if "open" not in qtype:
            continue
        case_id = str(raw.get("case_id", "")).strip()
        answer = str(raw.get("answer", "")).strip()
        if case_id and answer:
            out[case_id] = answer
    return out


def _norm(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"\s+", " ", text)
    text = text.strip(" .:;,!?'\"")
    return text


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize open-ended prediction quality signals.")
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument(
        "--train-json",
        type=Path,
        default=None,
        help="Optional Synapse train JSON for rough exact-match rate on open answers.",
    )
    args = parser.parse_args()

    answers = _load_answers(args.predictions)
    open_items = [a for a in answers if str(a.get("task_type", "")).lower() in {"open", "open-ended", "open_ended"}]
    if not open_items:
        open_items = answers

    lengths = [len(str(a.get("answer", "")).strip()) for a in open_items]
    trace_lens = [len(str(a.get("reasoning_trace", "")).strip()) for a in open_items]
    empty_answer = sum(1 for a in open_items if not str(a.get("answer", "")).strip())
    empty_trace = sum(1 for a in open_items if not str(a.get("reasoning_trace", "")).strip())
    hedged = sum(
        1
        for a in open_items
        if re.match(
            r"^(the\s+)?(final\s+)?answer\s*(is|=|:)|based on (the )?(image|findings)",
            str(a.get("answer", "")).strip(),
            flags=re.IGNORECASE,
        )
    )

    def _pct(xs: list[int], p: float) -> float:
        if not xs:
            return 0.0
        ordered = sorted(xs)
        idx = min(len(ordered) - 1, max(0, int(round((len(ordered) - 1) * p))))
        return float(ordered[idx])

    print(f"[analyze_open] n={len(open_items)}")
    print(f"[analyze_open] empty_answer={empty_answer} empty_trace={empty_trace} hedged_prefix={hedged}")
    if lengths:
        print(
            "[analyze_open] answer_len "
            f"mean={sum(lengths)/len(lengths):.1f} median={_pct(lengths, 0.5):.0f} "
            f"p90={_pct(lengths, 0.9):.0f}"
        )
    if trace_lens:
        print(
            "[analyze_open] reasoning_len "
            f"mean={sum(trace_lens)/len(trace_lens):.1f} median={_pct(trace_lens, 0.5):.0f}"
        )

    if args.train_json is not None and args.train_json.exists():
        gt = _load_gt_open(args.train_json)
        matched = 0
        scored = 0
        for item in open_items:
            case_id = str(item.get("case_id", "")).strip()
            if case_id not in gt:
                continue
            scored += 1
            pred = _norm(str(item.get("answer", "")))
            gold = _norm(gt[case_id])
            if pred == gold:
                matched += 1
        rate = (matched / scored) if scored else 0.0
        print(f"[analyze_open] rough_exact_match={matched}/{scored} ({100*rate:.1f}%)")
        print("[analyze_open] note: exact-match is a local proxy only, not organizer GT.")

    print("[analyze_open] sample answers:")
    for item in open_items[:5]:
        print(f"  - {item.get('case_id')}: {str(item.get('answer', ''))[:160]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
