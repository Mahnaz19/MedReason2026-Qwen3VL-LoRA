from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RuntimeConfig:
    """Runtime configuration for the participant container.

    The official evaluator will mount `/input` and `/output`. Environment variables
    are provided mainly to make local testing easier.
    """

    input_dir: Path
    output_dir: Path
    output_file: Path
    system_name: str
    submission_name: str
    model_path: str | None
    lora_path: str | None
    max_new_tokens: int
    temperature: float
    top_p: float
    dtype: str
    max_cases: int | None
    case_offset: int

    @property
    def cases_file(self) -> Path:
        return self.input_dir / "cases.json"

    @classmethod
    def from_env(cls) -> "RuntimeConfig":
        input_dir = Path(os.environ.get("MEDREASON_INPUT_DIR", "/input"))
        output_dir = Path(os.environ.get("MEDREASON_OUTPUT_DIR", "/output"))
        output_file = Path(os.environ.get("MEDREASON_OUTPUT_FILE", str(output_dir / "results.json")))
        max_cases_raw = os.environ.get("MEDREASON_MAX_CASES", "").strip()
        max_cases = int(max_cases_raw) if max_cases_raw else None
        case_offset_raw = os.environ.get("MEDREASON_CASE_OFFSET", "").strip()
        case_offset = int(case_offset_raw) if case_offset_raw else 0
        return cls(
            input_dir=input_dir,
            output_dir=output_dir,
            output_file=output_file,
            system_name=os.environ.get("MEDREASON_SYSTEM", "custom").strip().lower(),
            submission_name=os.environ.get("MEDREASON_SUBMISSION_NAME", "MedReason system submission"),
            model_path=os.environ.get("MEDREASON_MODEL_PATH", "/opt/app/models/Qwen3-VL-32B-Instruct"),
            lora_path=os.environ.get("MEDREASON_LORA_PATH", "").strip() or None,
            max_new_tokens=int(os.environ.get("MEDREASON_MAX_NEW_TOKENS", "512")),
            temperature=float(os.environ.get("MEDREASON_TEMPERATURE", "0.0")),
            top_p=float(os.environ.get("MEDREASON_TOP_P", "1.0")),
            dtype=os.environ.get("MEDREASON_DTYPE", "auto"),
            max_cases=max_cases,
            case_offset=case_offset,
        )
