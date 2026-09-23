# MedReason 2026 Model Description (Draft Template)

Fill this in for the official submission form (PDF required by **15 July 2026**).

## Team information
- Team name:
- Synapse username:
- Method name:
- Contact email:

## Task overview
Briefly describe the medical VQA system for MedReason (MCQ + open-ended, image-grounded reasoning).

## Model architecture
- Base model: Qwen2.5-VL-7B-Instruct (initial baseline)
- Fine-tuning: [none / LoRA / full fine-tune — update after experiments]
- Additional components: [RAG / self-verification / ensemble — if any]

## Training data
- Train split: 17,722 cases (14,398 MCQ + 3,324 open-ended)
- Validation split: used only for local pipeline checks (no labels)
- Hidden leaderboard splits: not used (organizer-side only)

## Training procedure
- Hardware: 4× RTX 3090 (or specify actual setup)
- Framework:
- Hyperparameters (LR, epochs, batch size, LoRA rank):
- Training time:

## Inference procedure
- Prompting strategy for MCQ vs open-ended
- Decoding: temperature=0, max_new_tokens=512
- Post-processing: JSON parsing + MCQ label normalization
- Runtime per case (approx.):

## Docker submission
- Base image: python:3.11
- Entry point: `process.py`
- Model baked into image at: `/opt/app/models/Qwen2.5-VL-7B-Instruct`
- No internet access at inference time

## Expected strengths / limitations
- Strengths:
- Limitations:
- Failure modes (low-quality images, OOD modalities):

## References
- MedReason challenge: https://medreason26.github.io/challenge.html
- Qwen2.5-VL: https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct
