#!/usr/bin/env python3
"""Fill the official MedReason model description Word template."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from docx import Document


# ---------------------------------------------------------------------------
# Edit these fields before regenerating the Word document.
# If you already edited the .docx manually, see docs/submission_update_reminders.md
# instead of regenerating (to avoid overwriting your team/contact fields).
# ---------------------------------------------------------------------------
TEAM_INFO = {
    "team_name": "Mahnaz",
    "primary_contact": "Mahnaz Mohammed Salih  mahnaz.mohammed@hotmail.com",
    "synapse_username": "Mahnaz Mohammed",
    "method_name": "MedReason-Qwen3VL",
}

TEAM_MEMBERS = [
    ("Mahnaz Mohammed Salih", "m.mohammedsalih19@aston.ac.uk", "Aston University"),
    ("Ziyang Wang", "z.wang47@aston.ac.uk", "Aston University"),
]

DOCKER_LINK = (
    "https://1drv.ms/f/c/6c0b3dc173416b21/IgCU335bDZivQZzkbB1BGLyoAc_ejL8fXoJPtl1NssZdG78?e=C7qkaU"
)
CODE_REPO = "Not publicly released (private development repository)"
PREPARED_BY = "Mahnaz Mohammed Salih"
PREPARED_DATE = "2026-07-31"

# Set True when submitting for pre-evaluation; False when submitting final evaluation.
PRE_EVALUATION_PHASE = False


METHOD_SUMMARY = (
    "We submit a single-model medical visual reasoning system: Qwen3-VL-32B-Instruct with a LoRA "
    "adapter trained on the official MedReason train split and then continued with open-ended "
    "upsampling, packaged in the official Docker template. At inference, the base checkpoint and "
    "LoRA adapter are loaded separately (MEDREASON_MODEL_PATH + MEDREASON_LORA_PATH). The system "
    "reads cases.json and medical images, applies task-specific prompts for MCQ and open-ended "
    "questions, and writes results.json with normalized multiple-choice labels plus image-grounded "
    "reasoning_trace and answer fields. Open-ended answers are post-processed toward short "
    "reference-style clinical statements while retaining a non-empty reasoning_trace."
)

MODEL_INFO = {
    "Foundation model / backbone": (
        "Qwen3-VL-32B-Instruct (HuggingFace: Qwen/Qwen3-VL-32B-Instruct) with a LoRA adapter "
        "(PEFT) trained on the official MedReason train split. The adapter is not merged into "
        "the base weights; both are loaded at runtime from the Docker image."
    ),
    "Vision encoder or image-processing component": (
        "Native Qwen3-VL vision tower and processor (AutoProcessor + qwen-vl-utils). "
        "Images are loaded as RGB PNG inputs referenced by cases.json."
    ),
    "Reasoning or prompting strategy": (
        "Task-specific JSON-output prompts: MCQ prompts include all official options and require a single "
        "option label; open-ended prompts require image-grounded reasoning_trace and a short definitive "
        "clinical answer (uncertainty kept in reasoning_trace only). "
        "Decoding uses temperature=0 for deterministic inference."
    ),
    "Frameworks / libraries / APIs used": (
        "Python 3.11, PyTorch (CUDA 12.8), HuggingFace Transformers (≥4.57), PEFT, accelerate, "
        "qwen-vl-utils, Pillow, Docker. No external APIs or internet access at official inference time."
    ),
    "Hardware used for training or adaptation": (
        "Isambard-AI Phase 2: 1× NVIDIA GH200 (~96 GB VRAM) per job via Slurm (workq partition). "
        "Initial LoRA on the full train split (17,722 cases, 1 epoch), then continued LoRA with "
        "open-ended upsampling (open×3) for 1 additional epoch."
    ),
    "Hardware used for inference testing": (
        "Isambard-AI Phase 2: 1× NVIDIA GH200 (~96 GB VRAM), single-GPU inference with "
        "Qwen3-VL-32B-Instruct + continued LoRA adapter (device_map=auto). "
        "Train MCQ safety checks and open-format smoke completed before packaging. "
        "Final Docker image smoke-tested on Windows (Docker Desktop)."
    ),
    "Number of models used in the final submission": (
        "1 system: Qwen3-VL-32B-Instruct base checkpoint plus one LoRA adapter "
        "(/opt/app/models/lora/medreason_open_final). Adapter weights are shipped alongside the base model "
        "in the Docker image, not merged into a single checkpoint."
    ),
    "Post-processing / ensembling steps (if any)": (
        "JSON extraction from model output; MCQ answer normalization to official option labels; "
        "open-ended answer cleanup (strip hedging prefixes / tighten short answers); "
        "contract validation against cases.json. No ensembling in the submission."
    ),
}

TASK_OUTPUT_DESIGN = """Closed-ended (MCQ):
• Input: medical image + question + official options from cases.json.
• The model receives a MedReason MCQ prompt requiring JSON with reasoning_trace and answer.
• Post-processing maps the answer field to exactly one official option label (A–E).

Open-ended:
• Input: medical image + question.
• The model receives an open-ended prompt requiring JSON with reasoning_trace and a short definitive answer.
• Both fields are mandatory in results.json; reasoning_trace supports visual grounding (VA); answer is cleaned toward concise clinical phrasing for GT agreement.

Input formatting:
• Official Docker input: /input/cases.json and /input/imgs/*.png (or paths referenced in cases.json).
• Each case is processed independently in a single-pass generation loop.

Inference pipeline:
• Single-pass VLM generation per case (no agent loop, no retrieval, no external API).
• Entry point: process.py → CustomMedReasonSystem → QwenVLMedReasonSystem.
• Output: /output/results.json following the official MedReason schema."""

TRAINING_SECTION = """Training data sources:
• Official MedReason released train split only (17,722 cases: 14,398 MCQ + 3,324 open-ended), downloaded from Synapse.
• No private, proprietary, or non-public data are used.

LoRA fine-tuning (completed):
• Base model: Qwen3-VL-32B-Instruct.
• Stage 1: all 17,722 official train cases (MCQ + open-ended), 1 epoch, learning rate 1e-4 (~6 hours).
• Stage 2 (final submission): continue from Stage-1 adapter with open-ended upsampling (open×3; 24,370 records), 1 epoch, learning rate 5e-5 (~7–8 hours).
• LoRA rank r=16, alpha=32, dropout=0.05; target modules: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj.
• Optimizer: AdamW; warmup ratio 0.03; weight decay 0.01.
• Batch size: per-device train batch size 1, gradient accumulation steps 16 (effective batch 16).
• Hardware: Isambard-AI Phase 2, 1× NVIDIA GH200 (~96 GB VRAM), Slurm workq partition.
• Final adapter checkpoint: models/lora/medreason_open_final (shipped in Docker at /opt/app/models/lora/medreason_open_final).
• Stage-2 final training loss: ~0.37 (reported by HuggingFace Trainer).

Validation strategy:
• Participant-facing validation split (2,532 cases, no public labels) used for local pipeline and output-format checking only.
• Full validation inference (2,532 cases) completed earlier with the Stage-1 LoRA system.
• Before final packaging, Stage-2 adapter was safety-checked on train MCQ subsets and an open-format smoke set.
• Hidden public/private leaderboard splits are not accessed locally (organizer-side evaluation only).

Local train-set monitoring (development checks — not held-out validation):
• Zero-shot baseline (no LoRA): 42.6% MCQ accuracy on train MCQ cases 1–500.
• Stage-2 LoRA precheck: 99.8% MCQ on train cases 1–500 and 99.8% on train cases 501–1000.
• Open smoke (100 train open cases): schema OK; empty answer/trace = 0; answer length median ~79 (aligned with train reference style).
• These scores are train-set development monitors; they do not estimate hidden-test performance.
• Pre-evaluation organizer result (reference): MCQ 97.14%, Open GT 1.638/4, Open VA 2.487/4, participant rank 1.

External public datasets:
• None beyond the official MedReason released train package.

Open-source pre-trained models:
• Qwen3-VL-32B-Instruct (base checkpoint).

Preprocessing:
• Synapse JSON converted to official cases.json format; image paths resolved to imgs/ directory.
• LoRA training examples built from train cases with task-specific prompts and ground-truth answers.
• Images used as-is (RGB); no additional normalization beyond model processor defaults.

Data augmentation:
• None.

Hyperparameter tuning:
• Inference: max_new_tokens=512, temperature=0.0, top_p=1.0.
• LoRA: fixed configuration above (r=16, lr=1e-4, 1 epoch); no extensive hyperparameter search.

Training / adaptation time:
• Zero-shot MCQ evaluation (train cases 1–500): completed on Isambard-AI, no training.
• Stage-1 LoRA (17,722 cases, 1 epoch): ~6 hours on Isambard-AI GH200.
• Stage-2 continued LoRA (open-upsampled, 1 epoch): ~7–8 hours on Isambard-AI GH200.
• Final precheck MCQ + open smoke: completed on Isambard-AI before Docker packaging.

Docker submission package:
• Image name: medreason-docker:latest; export file: medreason_final.tar.gz.
• Base model baked at /opt/app/models/Qwen3-VL-32B-Instruct; LoRA adapter at /opt/app/models/lora/medreason_open_final.
• Environment: MEDREASON_SYSTEM=custom, MEDREASON_MODEL_PATH, MEDREASON_LORA_PATH set in Dockerfile.
• Compressed export size: ~55 GB; Docker image disk usage: ~136 GB.
• Cloud download link: see Section 2 (OneDrive folder containing medreason_final.tar.gz)."""

EXTERNAL_RESOURCES = [
    (
        "Challenge dataset",
        "MedReason train + validation (Synapse)",
        "https://www.synapse.org/Synapse:syn74522250.1/datasets/",
        "Train split for development/adaptation; validation split for local I/O checks.",
    ),
    (
        "Pre-trained model",
        "Qwen3-VL-32B-Instruct",
        "https://huggingface.co/Qwen/Qwen3-VL-32B-Instruct",
        "Foundation VLM for medical VQA inference; checkpoint baked into Docker image.",
    ),
    (
        "Submission template",
        "MedReason-Docker official repository",
        "https://medreason26.github.io/challenge.html",
        "Docker I/O contract, validation tools, and participant system interface.",
    ),
    (
        "Software",
        "PyTorch + HuggingFace Transformers",
        "https://pytorch.org / https://huggingface.co/docs/transformers",
        "Model loading, generation, and containerized inference.",
    ),
    (
        "Software",
        "PEFT (LoRA)",
        "https://huggingface.co/docs/peft",
        "Parameter-efficient fine-tuning of Qwen3-VL-32B-Instruct on MedReason train data.",
    ),
]

OPTIONAL_NOTES = (
    "Final submission prepared 2026-07-31. Method: Qwen3-VL-32B-Instruct + continued LoRA "
    "(Stage-1 full train; Stage-2 open-upsampled continue on Isambard-AI GH200), with updated "
    "open prompts and open-answer post-processing. "
    "Local Stage-2 MCQ precheck: 99.8% on train cases 1–500 and 501–1000 (train-set monitors only). "
    "Pre-evaluation organizer scores (reference): MCQ 97.14%, Open GT 1.638/4, Open VA 2.487/4, rank 1. "
    "Docker package medreason_final.tar.gz (~55 GB) hosted on OneDrive; link in Section 2."
)

CHECKLIST_ITEMS = [
    "The submitted Docker package runs end-to-end under the official challenge I/O contract.",
    "The Docker package link in this document matches the final package sent to the organizers.",
    "All public external resources used by the method are listed above.",
    "No private, proprietary, or non-public data were used.",
    "The description reflects the actual final submission, not an earlier experiment.",
]

# Checklist marks for Word doc (index → checked).
CHECKLIST_DONE = {
    0: True,   # Docker smoke test passed with full image
    1: True,   # OneDrive link added
    2: True,   # external resources listed
    3: True,   # no private data
    4: True,   # reflects actual submission
}


def _set_cell(table, row: int, col: int, text: str) -> None:
    table.rows[row].cells[col].text = text


def fill_document(template: Path, output: Path) -> None:
    shutil.copy2(template, output)
    doc = Document(output)

    # SECTION 1 — team info (TABLE 1)
    t1 = doc.tables[1]
    _set_cell(t1, 0, 1, TEAM_INFO["team_name"])
    _set_cell(t1, 1, 1, TEAM_INFO["primary_contact"])
    _set_cell(t1, 2, 1, TEAM_INFO["synapse_username"])
    _set_cell(t1, 3, 1, TEAM_INFO["method_name"])

    # Team members (TABLE 2)
    t2 = doc.tables[2]
    for i, (name, email, affil) in enumerate(TEAM_MEMBERS, start=1):
        if i < len(t2.rows):
            _set_cell(t2, i, 0, name)
            _set_cell(t2, i, 1, email)
            _set_cell(t2, i, 2, affil)

    # Submission phase paragraph
    for p in doc.paragraphs:
        if "Submission phase" in p.text:
            if PRE_EVALUATION_PHASE:
                p.text = "Submission phase (check one): ☒ Pre-evaluation   ☐ Final evaluation"
            else:
                p.text = "Submission phase (check one): ☐ Pre-evaluation   ☒ Final evaluation"
            break

    # SECTION 2 (TABLE 3, TABLE 4)
    t3 = doc.tables[3]
    _set_cell(t3, 0, 1, DOCKER_LINK)
    _set_cell(t3, 1, 1, CODE_REPO)
    doc.tables[4].rows[0].cells[0].text = METHOD_SUMMARY

    # SECTION 3 (TABLE 5)
    t5 = doc.tables[5]
    for i, (label, value) in enumerate(MODEL_INFO.items()):
        _set_cell(t5, i, 1, value)

    # SECTION 4 (TABLE 6)
    doc.tables[6].rows[0].cells[0].text = TASK_OUTPUT_DESIGN

    # SECTION 5 (TABLE 8, TABLE 9)
    doc.tables[8].rows[0].cells[0].text = TRAINING_SECTION
    t9 = doc.tables[9]
    for i, row in enumerate(EXTERNAL_RESOURCES, start=1):
        if i < len(t9.rows):
            for j, val in enumerate(row):
                _set_cell(t9, i, j, val)

    # SECTION 6 — checklist
    for i, item in enumerate(CHECKLIST_ITEMS):
        for p in doc.paragraphs:
            if item in p.text:
                mark = "☒" if CHECKLIST_DONE.get(i, False) else "☐"
                p.text = f"{mark} {item}"
                break

    for p in doc.paragraphs:
        if p.text.startswith("Prepared by"):
            p.text = f"Prepared by {PREPARED_BY}"
        if p.text.startswith("Date"):
            p.text = f"Date {PREPARED_DATE}"

    doc.tables[10].rows[0].cells[0].text = OPTIONAL_NOTES

    cp = doc.core_properties
    cp.author = PREPARED_BY
    cp.last_modified_by = PREPARED_BY
    cp.comments = f"MedReason 2026 methods description — {PREPARED_BY}"
    cp.title = "MedReason 2026 Methods Description"

    doc.save(output)


def write_markdown_copy(output_md: Path) -> None:
  lines = [
    "# MedReason 2026 — Methods Description (Filled Draft)",
    "",
    "> MedReason 2026 pre-evaluation methods description.",
    "",
    "## SECTION 1 — Team Information",
    f"- **Team name:** {TEAM_INFO['team_name']}",
    f"- **Primary contact:** {TEAM_INFO['primary_contact']}",
    f"- **Synapse username:** {TEAM_INFO['synapse_username']}",
    f"- **Method name:** {TEAM_INFO['method_name']}",
    "",
    "### Team members",
  ]
  for name, email, affil in TEAM_MEMBERS:
    lines.append(f"- {name} | {email} | {affil}")
  lines += [
    "",
    "## SECTION 2 — Submission Summary",
    f"- **Phase:** {'Pre-evaluation' if PRE_EVALUATION_PHASE else 'Final evaluation'}",
    f"- **Docker link:** {DOCKER_LINK}",
    f"- **Code repo:** {CODE_REPO}",
    "",
    "### Short method summary",
    METHOD_SUMMARY,
    "",
    "## SECTION 3 — Model and Inference Information",
  ]
  for k, v in MODEL_INFO.items():
    lines.append(f"### {k}\n{v}\n")
  lines += ["## SECTION 4 — Task-Specific Output Design", TASK_OUTPUT_DESIGN, "", "## SECTION 5 — Training, Adaptation, and External Resources", TRAINING_SECTION, "", "### External resources log", "| Type | Name | URL | Usage |", "|------|------|-----|-------|"]
  for row in EXTERNAL_RESOURCES:
    lines.append(f"| {row[0]} | {row[1]} | {row[2]} | {row[3]} |")
  lines += ["", "## SECTION 6 — Checklist", ""]
  for i, item in enumerate(CHECKLIST_ITEMS):
    mark = "x" if CHECKLIST_DONE.get(i, False) else " "
    lines.append(f"- [{mark}] {item}")
  lines += ["", f"**Prepared by:** {PREPARED_BY}", f"**Date:** {PREPARED_DATE}", "", "### Optional notes", OPTIONAL_NOTES]
  output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Fill MedReason model description Word template.")
    parser.add_argument(
        "--template",
        type=Path,
        default=Path("../data/MedReason26_Model_Description_Documents.docx"),
    )
    parser.add_argument(
        "--output-docx",
        type=Path,
        default=Path("../data/MedReason26_Model_Description_FILLED.docx"),
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=Path("../data/derived/MedReason26_Model_Description_FILLED.md"),
    )
    args = parser.parse_args()
    fill_document(args.template, args.output_docx)
    write_markdown_copy(args.output_md)
    print(f"[fill_model_description] wrote {args.output_docx}")
    print(f"[fill_model_description] wrote {args.output_md}")
    print("Next: export Word to PDF (File -> Save As -> PDF) if not auto-generated.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
