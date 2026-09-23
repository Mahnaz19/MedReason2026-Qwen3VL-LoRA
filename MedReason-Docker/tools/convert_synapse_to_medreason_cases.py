#!/usr/bin/env python3
"""
Convert Synapse MedReason participant-facing train/validation JSON packages into
the MedReason Docker input format:

  /input/cases.json

This script does NOT download or move image files. By default, it writes
`cases.json` next to the raw Synapse package and references images under the
existing `imgs/` directory.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any, Iterable


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _cases_list(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, dict) and isinstance(payload.get("cases"), list):
        return payload["cases"]
    if isinstance(payload, list):
        return payload
    raise ValueError("Expected a JSON object with a 'cases' list, or a raw list of cases.")


def _normalize_task_type(question_type: str) -> str:
    key = str(question_type).strip().lower()
    if key in {"mcq", "multiple-choice", "multiple_choice", "closed", "closed-ended", "closed_ended"}:
        return "mcq"
    if "open" in key:
        return "open"
    # Synapse uses "open-ended" / "mcq" today, but keep a small amount of tolerance.
    raise ValueError(f"Unsupported Synapse question type: {question_type!r}")


def _image_filename(image_path_value: str) -> str:
    # Synapse image_path looks like: "dataset/imgs/<filename>.png"
    # In this repo's local extraction we keep files under: imgs/<filename>.png
    return Path(str(image_path_value)).name


def convert_split(
    raw_root: Path,
    raw_cases_json: Path,
    out_cases_json: Path,
    *,
    image_subdir_name: str,
    copy_images: bool,
) -> None:
    payload = _load_json(raw_cases_json)
    cases = _cases_list(payload)

    raw_imgs_dir = raw_root / image_subdir_name
    if not raw_imgs_dir.exists():
        raise FileNotFoundError(f"Expected image directory: {raw_imgs_dir}")

    if out_cases_json.exists():
        out_cases_json.unlink()

    out_cases: list[dict[str, Any]] = []
    needed_images: set[str] = set()

    for raw in cases:
        if not isinstance(raw, dict):
            continue
        case_id = str(raw.get("case_id", "")).strip()
        question_type_raw = str(raw.get("question type", "")).strip()
        question = str(raw.get("question", "")).strip()
        raw_image_path = raw.get("image_path")

        if not case_id:
            raise ValueError("Found a case without case_id.")
        if not question_type_raw:
            raise ValueError(f"case_id={case_id} missing 'question type'.")
        if not question:
            raise ValueError(f"case_id={case_id} missing 'question'.")
        if not raw_image_path:
            raise ValueError(f"case_id={case_id} missing 'image_path'.")

        task_type = _normalize_task_type(question_type_raw)
        filename = _image_filename(str(raw_image_path))
        needed_images.add(filename)

        image_rel_for_cases = f"{image_subdir_name}/{filename}"

        case_out: dict[str, Any] = {
            "case_id": case_id,
            "task_type": task_type,
            "image_path": image_rel_for_cases,
            "question": question,
        }

        if task_type == "mcq":
            options: list[dict[str, str]] = []
            for label in ["A", "B", "C", "D", "E"]:
                text = str(raw.get(label, "")).strip()
                if text:
                    # The Synapse option strings already include the label prefix ("A) ...").
                    # We keep the full text as-is for maximum prompt fidelity.
                    options.append({"label": label, "text": text})
            if len(options) < 2:
                raise ValueError(f"case_id={case_id} does not have enough MCQ options.")
            case_out["options"] = options

        out_cases.append(case_out)

    out_cases_json.parent.mkdir(parents=True, exist_ok=True)
    out_cases_json.write_text(json.dumps({"cases": out_cases}, ensure_ascii=False, indent=2), encoding="utf-8")

    if copy_images:
        # Copy images into the output directory's image subdir (useful if you
        # choose an out dir different from raw_root).
        out_imgs_dir = out_cases_json.parent / image_subdir_name
        out_imgs_dir.mkdir(parents=True, exist_ok=True)
        for filename in needed_images:
            src = raw_imgs_dir / filename
            dst = out_imgs_dir / filename
            if not dst.exists():
                shutil.copy2(src, dst)


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert Synapse MedReason packages to cases.json format.")
    parser.add_argument("--train-raw-root", type=Path, default=None)
    parser.add_argument("--validation-raw-root", type=Path, default=None)
    parser.add_argument("--image-subdir", type=str, default="imgs", help="Subdir containing image files (default: imgs).")
    parser.add_argument("--copy-images", action="store_true", help="If set, copy images into the out folder.")
    parser.add_argument(
        "--in-place",
        action="store_true",
        help="If set, write cases.json directly into each raw root (raw_root/cases.json).",
    )
    parser.add_argument("--out-dir", type=Path, default=None, help="Output base dir when not using --in-place.")
    args = parser.parse_args()

    if args.in_place:
        if not args.train_raw_root and not args.validation_raw_root:
            raise ValueError("Use --in-place with at least one of --train-raw-root or --validation-raw-root.")
    else:
        if args.out_dir is None:
            raise ValueError("When not using --in-place, you must provide --out-dir.")

    if args.train_raw_root:
        raw_root = args.train_raw_root
        raw_cases_json = raw_root / "medreason_train_selection.json"
        if not raw_cases_json.exists():
            raise FileNotFoundError(raw_cases_json)
        if args.in_place:
            out_cases_json = raw_root / "cases.json"
        else:
            out_cases_json = args.out_dir / "train" / "cases.json"
        convert_split(
            raw_root=raw_root,
            raw_cases_json=raw_cases_json,
            out_cases_json=out_cases_json,
            image_subdir_name=args.image_subdir,
            copy_images=args.copy_images,
        )
        print(f"[convert] Train written: {out_cases_json}")

    if args.validation_raw_root:
        raw_root = args.validation_raw_root
        raw_cases_json = raw_root / "medreason_validation_participant_facing.json"
        if not raw_cases_json.exists():
            raise FileNotFoundError(raw_cases_json)
        if args.in_place:
            out_cases_json = raw_root / "cases.json"
        else:
            out_cases_json = args.out_dir / "validation" / "cases.json"
        convert_split(
            raw_root=raw_root,
            raw_cases_json=raw_cases_json,
            out_cases_json=out_cases_json,
            image_subdir_name=args.image_subdir,
            copy_images=args.copy_images,
        )
        print(f"[convert] Validation written: {out_cases_json}")


if __name__ == "__main__":
    main()

