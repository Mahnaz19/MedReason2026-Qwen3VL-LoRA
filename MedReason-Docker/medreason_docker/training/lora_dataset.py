from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from torch.utils.data import Dataset

from medreason_docker.prompting import SYSTEM_PROMPT, build_text_prompt
from medreason_docker.schema import MedReasonCase, MedReasonOption


def resolve_image_path(raw_path: str, *, image_root: Path, repo_root: Path | None = None) -> Path | None:
    """Resolve image paths from JSONL (handles Windows-style relative paths)."""
    normalized = str(raw_path).replace("\\", "/").strip()
    if not normalized:
        return None

    candidates: list[Path] = []
    path = Path(normalized)
    if path.is_absolute():
        candidates.append(path)
    else:
        candidates.append(path.resolve())
        if repo_root is not None:
            candidates.append((repo_root / path).resolve())
            candidates.append((repo_root / path.name).resolve())
        candidates.append((image_root / path.name).resolve())
        candidates.append((image_root / path).resolve())

    seen: set[Path] = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        if candidate.exists():
            return candidate
    return None


def record_to_messages(record: dict[str, Any], image_path: Path) -> list[dict[str, Any]]:
    """Build Qwen chat messages aligned with inference-time prompting."""
    options = tuple(
        MedReasonOption(str(opt["label"]), str(opt["text"]))
        for opt in record.get("options", [])
        if isinstance(opt, dict) and opt.get("label") and opt.get("text")
    )
    case = MedReasonCase(
        case_id=str(record["case_id"]),
        task_type=record["task_type"],
        question=str(record["question"]),
        image_paths=(image_path,),
        options=options,
    )
    target = record.get("target") or {}
    assistant_text = json.dumps(
        {
            "reasoning_trace": str(target.get("reasoning_trace", "")).strip(),
            "answer": str(target.get("answer", "")).strip(),
        },
        ensure_ascii=False,
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": str(image_path)},
                {"type": "text", "text": build_text_prompt(case)},
            ],
        },
        {"role": "assistant", "content": assistant_text},
    ]


def load_lora_records(
    dataset_path: Path,
    *,
    image_root: Path,
    repo_root: Path | None = None,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    skipped = 0
    with dataset_path.open(encoding="utf-8") as handle:
        for line in handle:
            if limit is not None and len(records) >= limit:
                break
            line = line.strip()
            if not line:
                continue
            raw = json.loads(line)
            image_path = resolve_image_path(
                str(raw.get("image_path", "")),
                image_root=image_root,
                repo_root=repo_root,
            )
            if image_path is None:
                skipped += 1
                continue
            records.append(
                {
                    "case_id": str(raw.get("case_id", "")),
                    "messages": record_to_messages(raw, image_path),
                }
            )
    if skipped:
        print(f"[lora_dataset] skipped {skipped} record(s) with missing images", flush=True)
    return records


class MedReasonLoraDataset(Dataset):
    def __init__(self, records: list[dict[str, Any]]) -> None:
        self.records = records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        item = self.records[index]
        return {"case_id": item["case_id"], "messages": item["messages"]}


class MedReasonLoraCollator:
    """Collate Qwen VL SFT batches with label masking on the prompt tokens."""

    def __init__(self, processor: Any) -> None:
        self.processor = processor

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        if len(features) != 1:
            raise ValueError(
                "MedReason LoRA training currently supports batch_size=1 only "
                "(set --per-device-train-batch-size 1)."
            )

        from qwen_vl_utils import process_vision_info

        messages = features[0]["messages"]
        full_text = self.processor.apply_chat_template(messages, tokenize=False)
        prompt_text = self.processor.apply_chat_template(
            messages[:-1],
            tokenize=False,
            add_generation_prompt=True,
        )
        image_inputs, video_inputs = process_vision_info(messages)

        model_inputs = self.processor(
            text=[full_text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        prompt_inputs = self.processor(
            text=[prompt_text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )

        labels = model_inputs["input_ids"].clone()
        prompt_len = prompt_inputs["input_ids"].shape[1]
        labels[:, :prompt_len] = -100
        pad_id = self.processor.tokenizer.pad_token_id
        if pad_id is not None:
            labels[labels == pad_id] = -100

        model_inputs["labels"] = labels
        return model_inputs
