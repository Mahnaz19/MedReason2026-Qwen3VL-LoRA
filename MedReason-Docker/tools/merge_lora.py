#!/usr/bin/env python3
"""Merge a LoRA adapter into the base Qwen VL checkpoint for Docker submission."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoProcessor


def _load_base(model_path: str) -> torch.nn.Module:
    load_errors: list[str] = []
    for class_name in (
        "Qwen3VLForConditionalGeneration",
        "Qwen2_5_VLForConditionalGeneration",
        "AutoModelForImageTextToText",
        "AutoModelForVision2Seq",
    ):
        try:
            transformers_mod = __import__("transformers", fromlist=[class_name])
            model_cls = getattr(transformers_mod, class_name)
            return model_cls.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True,
            )
        except Exception as exc:  # noqa: BLE001
            load_errors.append(f"{class_name}: {exc}")
    raise RuntimeError(
        f"Could not load base checkpoint from {model_path}. Tried:\n" + "\n".join(load_errors)
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Merge MedReason LoRA adapter into base model.")
    parser.add_argument(
        "--base-model-path",
        type=Path,
        default=Path("models/Qwen3-VL-32B-Instruct"),
    )
    parser.add_argument(
        "--adapter-path",
        type=Path,
        required=True,
        help="LoRA adapter directory (e.g. data/outputs/lora/.../final).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("../data/outputs/lora/qwen3vl_medreason_merged"),
    )
    parser.add_argument(
        "--max-shard-size",
        type=str,
        default="4GB",
        help="Shard size when saving the merged checkpoint.",
    )
    args = parser.parse_args()

    base_path = str(args.base_model_path.resolve())
    adapter_path = str(args.adapter_path.resolve())
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise SystemExit(
            f"Output dir already exists and is not empty: {output_dir}\n"
            "Remove it first if a previous merge failed: rm -rf <output_dir>"
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    processor = AutoProcessor.from_pretrained(base_path, trust_remote_code=True)
    base_model = _load_base(base_path)
    model = PeftModel.from_pretrained(base_model, adapter_path)
    del base_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    merged = model.merge_and_unload()
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    merged.save_pretrained(
        str(output_dir),
        safe_serialization=True,
        max_shard_size=args.max_shard_size,
    )
    del merged
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    processor.save_pretrained(str(output_dir))
    print(f"[merge_lora] wrote merged checkpoint -> {output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
