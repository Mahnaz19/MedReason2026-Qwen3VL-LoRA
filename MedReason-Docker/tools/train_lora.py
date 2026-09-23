#!/usr/bin/env python3
"""LoRA fine-tune Qwen3-VL (or Qwen2.5-VL) on MedReason train_lora.jsonl."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

APP_ROOT = Path(__file__).resolve().parents[1]
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

import torch
from peft import LoraConfig, get_peft_model
from transformers import Trainer, TrainingArguments

from medreason_docker.training.lora_dataset import (
    MedReasonLoraCollator,
    MedReasonLoraDataset,
    load_lora_records,
)


def _load_model(model_path: str, *, attn_impl: str | None) -> torch.nn.Module:
    load_errors: list[str] = []
    extra_kwargs: dict = {}
    if attn_impl:
        extra_kwargs["attn_implementation"] = attn_impl

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
                **extra_kwargs,
            )
        except Exception as exc:  # noqa: BLE001
            load_errors.append(f"{class_name}: {exc}")

    raise RuntimeError(
        f"Could not load VLM checkpoint from {model_path}. Tried:\n" + "\n".join(load_errors)
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="LoRA fine-tune MedReason Qwen VL model.")
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("models/Qwen3-VL-32B-Instruct"),
        help="Local HuggingFace model directory.",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("../data/derived/train_lora.jsonl"),
        help="JSONL produced by tools/prepare_lora_dataset.py.",
    )
    parser.add_argument(
        "--image-root",
        type=Path,
        default=Path("../data/raw/train/imgs"),
        help="Directory containing train images.",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(".."),
        help="Repository root for resolving relative image paths in JSONL.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("../data/outputs/lora/qwen3vl_medreason"),
        help="Directory for LoRA checkpoints.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Optional record cap for smoke tests.")
    parser.add_argument("--num-train-epochs", type=float, default=1.0)
    parser.add_argument("--max-steps", type=int, default=-1, help="Override epochs when > 0.")
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--save-steps", type=int, default=500)
    parser.add_argument("--save-total-limit", type=int, default=3)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument(
        "--resume-adapter",
        type=Path,
        default=None,
        help="Continue training from an existing PEFT adapter directory.",
    )
    parser.add_argument(
        "--attn-implementation",
        type=str,
        default="",
        help='Optional attention backend, e.g. "flash_attention_2" if installed.',
    )
    args = parser.parse_args()

    if args.per_device_train_batch_size != 1:
        raise ValueError("Use --per-device-train-batch-size 1 for Qwen VL LoRA training.")

    model_path = str(args.model_path.resolve())
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    attn_impl = args.attn_implementation.strip() or None
    model = _load_model(model_path, attn_impl=attn_impl)
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()

    if args.resume_adapter is not None:
        from peft import PeftModel

        adapter_path = str(args.resume_adapter.resolve())
        print(f"[train_lora] continuing from adapter {adapter_path}", flush=True)
        model = PeftModel.from_pretrained(model, adapter_path, is_trainable=True)
    else:
        lora_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
        )
        model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    records = load_lora_records(
        args.dataset.resolve(),
        image_root=args.image_root.resolve(),
        repo_root=args.repo_root.resolve(),
        limit=args.limit,
    )
    if not records:
        raise ValueError(f"No training records loaded from {args.dataset}")

    dataset = MedReasonLoraDataset(records)
    collator = MedReasonLoraCollator(processor)

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        bf16=True,
        remove_unused_columns=False,
        dataloader_pin_memory=False,
        gradient_checkpointing=True,
        report_to="none",
        optim="adamw_torch",
        lr_scheduler_type="cosine",
        label_names=["labels"],
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=collator,
    )

    print(
        f"[train_lora] records={len(dataset)} output_dir={output_dir} "
        f"effective_batch={args.per_device_train_batch_size * args.gradient_accumulation_steps}",
        flush=True,
    )
    trainer.train()
    trainer.save_model(str(output_dir / "final"))
    processor.save_pretrained(str(output_dir / "final"))
    print(f"[train_lora] saved adapter -> {output_dir / 'final'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
