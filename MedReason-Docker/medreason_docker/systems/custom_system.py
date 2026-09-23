from __future__ import annotations

from medreason_docker.schema import MedReasonCase, MedReasonPrediction
from medreason_docker.systems.base import MedReasonSystem
from medreason_docker.systems.qwen_vl_system import QwenVLMedReasonSystem


class CustomMedReasonSystem(MedReasonSystem):
    """Participant submission system.

    The default implementation delegates to Qwen3-VL (or another Qwen VL checkpoint)
    with MedReason-specific prompting and parsing. Extend or replace this class with
    ensembles, self-verification, or fine-tuned / LoRA-merged checkpoints.
    """

    def setup(self) -> None:
        self._backend = QwenVLMedReasonSystem(config=self.config)
        self._backend.setup()

    def teardown(self) -> None:
        if hasattr(self, "_backend"):
            self._backend.teardown()

    def predict_case(self, case: MedReasonCase) -> MedReasonPrediction:
        pred = self._backend.predict_case(case)
        return MedReasonPrediction(
            case_id=pred.case_id,
            task_type=pred.task_type,
            answer=pred.answer,
            reasoning_trace=pred.reasoning_trace,
            confidence=pred.confidence,
            metadata={**(pred.metadata or {}), "system": "custom_qwen_vl"},
        )
