from __future__ import annotations

from typing import Any

from .schema import MedReasonCase

SYSTEM_PROMPT = """You are a medical visual reasoning assistant for the MedReason Challenge.
Use the provided image evidence and the question. Do not invent findings that are not supported by the image.
If the evidence is insufficient, state the limitation clearly in reasoning_trace.
Return only valid JSON with two keys: reasoning_trace and answer."""

MCQ_INSTRUCTION = """Task: closed-ended multiple-choice medical visual question answering.

Question:
{question}

Options:
{options}

Instructions:
1. Inspect the image evidence relevant to the question.
2. Choose exactly one option label from the provided options.
3. The answer field must contain only the selected label, for example "A".
4. The reasoning_trace should be concise and image-grounded.

Return only JSON:
{{
  "reasoning_trace": "brief visual rationale",
  "answer": "A"
}}"""

OPEN_INSTRUCTION = """Task: open-ended medical visual reasoning.

Question:
{question}

Instructions:
1. Put image-grounded evidence and step-by-step clinical reasoning ONLY in reasoning_trace.
2. Put uncertainty or evidence limitations ONLY in reasoning_trace.
3. The answer field must be a short, definitive clinical statement (about one sentence, typically under ~25 words).
4. Do NOT put explanations, hedging, or phrases like "The answer is" / "Based on the image" in answer.
5. Prefer a direct finding/diagnosis/relationship statement that a reference answer would use.
6. reasoning_trace is required and must remain non-empty and image-grounded.

Return only JSON:
{{
  "reasoning_trace": "image-grounded evidence and reasoning",
  "answer": "short definitive clinical answer"
}}"""


def format_options(case: MedReasonCase) -> str:
    return "\n".join(f"{opt.label}. {opt.text}" for opt in case.options)


def build_text_prompt(case: MedReasonCase) -> str:
    if case.task_type == "mcq":
        return MCQ_INSTRUCTION.format(question=case.question, options=format_options(case))
    return OPEN_INSTRUCTION.format(question=case.question)


def build_prompt(case: MedReasonCase) -> str:
    """Generic text prompt used by HuggingFace VLM fallback systems."""
    return build_text_prompt(case)


def build_qwen_messages(case: MedReasonCase) -> list[dict[str, Any]]:
    content: list[dict[str, Any]] = []
    for image_path in case.image_paths:
        content.append({"type": "image", "image": str(image_path)})
    content.append({"type": "text", "text": build_text_prompt(case)})
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": content},
    ]
