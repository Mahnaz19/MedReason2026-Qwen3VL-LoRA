from __future__ import annotations

from typing import Any

from medreason_docker.parsing import normalize_mcq_answer, normalize_open_answer, parse_model_output
from medreason_docker.prompting import build_qwen_messages
from medreason_docker.schema import MedReasonCase, MedReasonPrediction
from medreason_docker.systems.base import MedReasonSystem


class QwenVLMedReasonSystem(MedReasonSystem):
    """Qwen VL MedReason inference system (Qwen3-VL and Qwen2.5-VL checkpoints)."""

    def setup(self) -> None:
        model_path = self.config.model_path
        if not model_path:
            raise ValueError(
                "MEDREASON_MODEL_PATH must point to a local Qwen VL checkpoint. "
                "See MedReason-Docker/models/README.md."
            )

        try:
            import torch
            from transformers import AutoProcessor
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                "Qwen VL mode requires torch, transformers>=4.57, and qwen-vl-utils."
            ) from exc

        self.torch = torch
        self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        self.model = self._load_model(model_path)
        if self.config.lora_path:
            self.model = self._load_lora_adapter(self.config.lora_path)
        self.model.eval()

    def _load_lora_adapter(self, lora_path: str) -> Any:
        try:
            from peft import PeftModel
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError("LoRA inference requires the peft package.") from exc
        print(f"[MedReason] loading LoRA adapter from {lora_path}", flush=True)
        return PeftModel.from_pretrained(self.model, lora_path)

    def _load_model(self, model_path: str) -> Any:
        torch = self.torch
        dtype = self.config.dtype
        torch_dtype = "auto" if dtype == "auto" else getattr(torch, dtype, "auto")

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
                    torch_dtype=torch_dtype,
                    device_map="auto",
                    trust_remote_code=True,
                )
            except Exception as exc:  # noqa: BLE001
                load_errors.append(f"{class_name}: {exc}")

        raise RuntimeError(
            f"Could not load Qwen VL checkpoint from {model_path}. Tried:\n" + "\n".join(load_errors)
        )

    def predict_case(self, case: MedReasonCase) -> MedReasonPrediction:
        raw_text = self._generate(case)
        reasoning_trace, answer = parse_model_output(raw_text, case)
        if case.task_type == "mcq":
            answer = normalize_mcq_answer(answer, case)
        else:
            answer = normalize_open_answer(answer)

        return MedReasonPrediction(
            case_id=case.case_id,
            task_type=case.task_type,
            reasoning_trace=reasoning_trace,
            answer=answer,
            confidence=None,
            metadata={"system": "qwen_vl", "raw_response": raw_text[:2000]},
        )

    def _generate(self, case: MedReasonCase) -> str:
        try:
            from qwen_vl_utils import process_vision_info
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError("Qwen VL mode requires qwen-vl-utils.") from exc

        torch = self.torch
        processor = self.processor
        model = self.model

        messages = build_qwen_messages(case)
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(model.device)

        do_sample = self.config.temperature > 0
        generation_kwargs: dict[str, Any] = {
            "max_new_tokens": self.config.max_new_tokens,
            "do_sample": do_sample,
        }
        if do_sample:
            generation_kwargs["temperature"] = self.config.temperature
            generation_kwargs["top_p"] = self.config.top_p

        with torch.inference_mode():
            generated_ids = model.generate(**inputs, **generation_kwargs)

        trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
        decoded = processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        return decoded[0].strip() if decoded else ""


# Backward-compatible alias for older scripts and registry keys.
Qwen25VLMedReasonSystem = QwenVLMedReasonSystem
