from __future__ import annotations

import json
import re
from typing import Any

from .schema import MedReasonCase

_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def extract_json_object(text: str) -> dict[str, Any] | None:
    stripped = (text or "").strip()
    stripped = re.sub(r"^```(?:json)?\s*", "", stripped, flags=re.IGNORECASE)
    stripped = re.sub(r"\s*```$", "", stripped)
    try:
        obj = json.loads(stripped)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass

    start = stripped.find("{")
    end = stripped.rfind("}")
    if 0 <= start < end:
        try:
            obj = json.loads(stripped[start : end + 1])
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            return None
    return None


def parse_reasoning_and_answer(text: str, case: MedReasonCase) -> tuple[str, str]:
    """Parse a model response into `(reasoning_trace, answer)`."""

    raw = (text or "").strip()
    obj = extract_json_object(raw)
    if obj is not None:
        reasoning = str(obj.get("reasoning_trace", obj.get("rationale", ""))).strip()
        answer = str(obj.get("answer", obj.get("final_answer", ""))).strip()
        return reasoning, answer

    reasoning = raw
    answer = raw
    for marker in ["Final answer:", "Answer:", "ANSWER:", "Final:"]:
        if marker in raw:
            prefix, suffix = raw.rsplit(marker, 1)
            reasoning = prefix.strip()
            answer = suffix.strip()
            break
    return reasoning, answer


def parse_model_output(text: str, case: MedReasonCase) -> tuple[str, str]:
    reasoning, answer = parse_reasoning_and_answer(text, case)

    if not answer:
        match = re.search(r"(?:final answer|answer)\s*[:：]\s*(.+)", text, flags=re.IGNORECASE | re.DOTALL)
        answer = match.group(1).strip().splitlines()[0] if match else text.strip()
    if not reasoning:
        reasoning = text.strip() or "No reasoning trace was generated."

    if case.task_type == "open":
        answer = normalize_open_answer(answer)
        if not answer:
            answer = "Unable to determine from the available visual evidence."

    return reasoning, answer


_OPEN_PREFIX_RE = re.compile(
    r"^(?:"
    r"(?:the\s+)?(?:final\s+)?answer\s*(?:is|=|:)\s*"
    r"|based on (?:the )?(?:image|findings|evidence)[,:]?\s*"
    r"|according to (?:the )?(?:image|findings)[,:]?\s*"
    r"|from the (?:image|findings)[,:]?\s*"
    r"|in (?:conclusion|summary)[,:]?\s*"
    r"|therefore[,:]?\s*"
    r"|overall[,:]?\s*"
    r")+",
    flags=re.IGNORECASE,
)
_OPEN_WRAPPER_RE = re.compile(r'^[`"\'(\[]+|[`"\')\]]+$')


def normalize_open_answer(answer: str, *, max_chars: int = 220) -> str:
    """Tighten open-ended answers toward short reference-style statements."""
    clean = (answer or "").strip()
    if not clean:
        return ""

    clean = clean.splitlines()[0].strip()
    clean = _OPEN_PREFIX_RE.sub("", clean).strip()
    clean = _OPEN_WRAPPER_RE.sub("", clean).strip()
    clean = re.sub(r"\s+", " ", clean).strip(" -:;,")

    # If the model returned a long explanation, keep the first sentence.
    if len(clean) > max_chars:
        sentence_end = re.search(r"[.!?](?:\s|$)", clean)
        if sentence_end and sentence_end.end() <= max_chars + 40:
            clean = clean[: sentence_end.end()].strip()
        else:
            cut = clean[:max_chars].rsplit(" ", 1)[0].strip()
            clean = cut or clean[:max_chars].strip()

    return clean.strip(" -:;,")


def normalize_mcq_answer(answer: str, case: MedReasonCase) -> str:
    labels = [option.label for option in case.options]
    label_map = {label.upper(): label for label in labels}
    clean = (answer or "").strip()

    if clean in case.option_labels:
        return clean
    if clean.upper() in label_map:
        return label_map[clean.upper()]

    match = re.search(
        r"(?:answer|option|choice)?\s*[:：\(\[]?\s*([A-Z])\s*[\)\].:]?",
        clean,
        flags=re.IGNORECASE,
    )
    if match and match.group(1).upper() in label_map:
        return label_map[match.group(1).upper()]

    lowered = clean.lower()
    for option in case.options:
        if option.text and option.text.lower() in lowered:
            return option.label

    # Deterministic fallback keeps submissions valid under time pressure.
    return labels[0]
