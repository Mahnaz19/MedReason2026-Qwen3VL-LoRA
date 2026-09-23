# MedReason 2026 — Two-Stage LoRA on Qwen3-VL

**Team Mahnaz (Team_17)** · MICCAI 2026 MedReason Challenge

This repository contains the **code** for our MedReason submission: a frozen [Qwen3-VL-32B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-32B-Instruct) backbone plus a **two-stage LoRA** adapter for image-grounded medical visual question answering (MCQ and open-ended).

> **Not included in this repo (by design)**  
> - Challenge train/validation images and labels (download from Synapse)  
> - Base model weights (~65 GB) and LoRA adapters  
> - Docker image exports (`.tar.gz`)  

A methods paper describing this work will be published (Springer LNCS, MedReason / MICCAI 2026 workshop track).

---

## Repository layout

```text
MedReason-Docker/                 # Our submission system (train + infer + Docker)
  medreason_docker/               # Runtime: prompts, parsing, Qwen3-VL + LoRA
  tools/                          # Data prep, LoRA train/merge, validators
  scripts/isambard/               # GPU / Slurm helper scripts
  models/README.md                # Where to place downloaded checkpoints
  Dockerfile, build.sh, test.sh, export.sh, process.py
MedReason-Evaluation/             # Public scoring / submission validation helpers
MedReason-Docker-Example-Qwen25VL/# Organiser Qwen2.5-VL baseline example
```

Expected **local** folders (gitignored — you create them):

```text
data/raw/train/                   # Official train package from Synapse
data/raw/validation/              # Official validation package from Synapse
data/derived/                     # JSONL we build for LoRA
data/outputs/lora/                # Trained adapters
MedReason-Docker/models/Qwen3-VL-32B-Instruct/
MedReason-Docker/models/lora/     # Optional copy of adapters for Docker bake-in
```

---

## Requirements

- Python **3.11+**
- A large GPU for training/inference of Qwen3-VL-32B (we used **1× NVIDIA GH200 ~96 GB** on Isambard-AI). Smaller GPUs may need quantisation or a smaller backbone (not covered here).
- Docker (optional, for packaging a submission image)
- Hugging Face CLI (`huggingface_hub`) to download the base model
- Synapse account with MedReason data access

Install Python deps from the submission package:

```bash
cd MedReason-Docker
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
pip install huggingface_hub
```

---

## 1. Get the challenge data (Synapse)

Data is **not** redistributed here. Register for MedReason and download from Synapse:

- Participation / overview: https://www.synapse.org/Synapse:syn74403682/wiki/640168  
- Data access: https://www.synapse.org/Synapse:syn74403682/wiki/640171  

Place the official packages so images live under `imgs/` and the selection JSON is present, for example:

```text
data/raw/train/medreason_train_selection.json
data/raw/train/imgs/...
data/raw/validation/...   # validation package as released
```

Convert Synapse JSON into the Docker `cases.json` contract (does not move images by default):

```bash
cd MedReason-Docker

python tools/convert_synapse_to_medreason_cases.py \
  --raw-root ../data/raw/train \
  --raw-cases ../data/raw/train/medreason_train_selection.json \
  --out-cases ../data/raw/train/cases.json

# Repeat for validation with the released validation JSON path.
```

Optional stats:

```bash
python tools/dataset_stats.py --train-json ../data/raw/train/medreason_train_selection.json
```

---

## 2. Download the base model (Hugging Face)

```bash
cd MedReason-Docker
bash scripts/isambard/download_model.sh
```

Or manually:

```bash
hf download Qwen/Qwen3-VL-32B-Instruct \
  --local-dir MedReason-Docker/models/Qwen3-VL-32B-Instruct
```

Size is roughly **~65 GB**. You need a Hugging Face account and acceptance of the model terms if required.

---

## 3. Prepare the LoRA training dataset

Build Stage-1 JSONL (MCQ + open targets from the official train labels):

```bash
cd MedReason-Docker

python tools/prepare_lora_dataset.py \
  --train-json ../data/raw/train/medreason_train_selection.json \
  --image-root ../data/raw/train/imgs \
  --output ../data/derived/train_lora.jsonl \
  --open-repeat 1 \
  --mcq-repeat 1 \
  --shuffle-seed 42
```

Stage 2 rebuilds an **open-upsampled** JSONL inside `scripts/isambard/run_gpu_lora_open_continue.sh` (default: each open-ended case repeated **3×**).

---

## 4. Two-stage LoRA training

Hyper-parameters used in our submission:

| | Stage 1 | Stage 2 (submitted) |
|---|---|---|
| Resume from | — | Stage-1 adapter |
| Data | full train mix | open×3 upsampling |
| Epochs | 1 | 1 (continued) |
| Learning rate | `1e-4` | `5e-5` |
| LoRA rank / alpha | 16 / 32 | 16 / 32 |
| Effective batch | 16 (1 × 16 accum) | same |
| Optimiser | AdamW + cosine schedule | same |

### Stage 1

```bash
cd MedReason-Docker
bash scripts/isambard/run_gpu_lora_train.sh "" 1
```

Adapter written to:

```text
data/outputs/lora/qwen3vl_medreason/final/
```

Smoke test (optional, 500 records):

```bash
bash scripts/isambard/run_gpu_lora_train.sh 500 1
```

### Stage 2 (continue + open upsampling)

```bash
cd MedReason-Docker
bash scripts/isambard/run_gpu_lora_open_continue.sh 1 5e-5 3
```

Adapter written to:

```text
data/outputs/lora/qwen3vl_medreason_open/final/
```

Copy (or symlink) the Stage-2 adapter into the Docker models tree if you will bake it into an image:

```bash
mkdir -p models/lora
cp -r ../data/outputs/lora/qwen3vl_medreason_open/final models/lora/medreason_open_final
```

Optional merge into a full checkpoint (not required for PEFT runtime loading):

```bash
bash scripts/isambard/run_gpu_lora_merge.sh
```

---

## 5. Run inference (local)

Smoke test without a large model:

```bash
cd MedReason-Docker
MEDREASON_SYSTEM=smoke \
MEDREASON_INPUT_DIR=./test \
MEDREASON_OUTPUT_FILE=./output/results.json \
python process.py

python tools/validate_output.py output/results.json --input-json test/cases.json
```

Real inference with base model + Stage-2 LoRA:

```bash
cd MedReason-Docker

export MEDREASON_SYSTEM=custom
export MEDREASON_MODEL_PATH=./models/Qwen3-VL-32B-Instruct
export MEDREASON_LORA_PATH=./models/lora/medreason_open_final
export MEDREASON_INPUT_DIR=../data/raw/validation
export MEDREASON_OUTPUT_FILE=../data/outputs/validation/results.json
export MEDREASON_MAX_NEW_TOKENS=512
export MEDREASON_TEMPERATURE=0.0

python process.py

python tools/validate_output.py \
  ../data/outputs/validation/results.json \
  --input-json ../data/raw/validation/cases.json
```

On Windows PowerShell, set env vars with `$env:MEDREASON_MODEL_PATH="..."` instead of `export`.

Helper scripts also exist under `scripts/isambard/` for train-slice MCQ checks and validation runs on a GPU cluster.

---

## 6. Docker build, smoke test, and export

The Dockerfile expects weights under `/opt/app/models/` and defaults to:

- `MEDREASON_MODEL_PATH=/opt/app/models/Qwen3-VL-32B-Instruct`
- `MEDREASON_LORA_PATH=/opt/app/models/lora/medreason_open_final`

```bash
cd MedReason-Docker

# Ensure models/ and models/lora/medreason_open_final/ are populated first
./build.sh medreason-docker:latest
./test.sh medreason-docker:latest

# Export (large file — store outside git)
./export.sh medreason-docker:latest /path/to/medreason_final.tar.gz
```

The container reads `/input/cases.json` (+ images) and writes `/output/results.json`. No network at evaluation time.

---

## 7. Evaluation helpers

Public aggregation / submission checks live in `MedReason-Evaluation/`:

```bash
cd MedReason-Evaluation
python scoring.py --help
python validate_submission.py --help
```

Hidden MedReason leaderboard metrics (LLM judges for open-ended scores) are run by the organisers, not fully reproduced by these public scripts alone.

---

## Method summary

1. **Single model** — Qwen3-VL-32B-Instruct (frozen) + one unmerged LoRA adapter.  
2. **LoRA targets** — attention and MLP projections (`q/k/v/o`, `gate/up/down`).  
3. **Prompts** — separate contracts for MCQ vs open-ended: label-only answers for MCQ; short definitive answers + image-grounded reasoning traces for open.  
4. **Two-stage training** — Stage 1 on the full official mix; Stage 2 continues from that adapter with open-ended upsampling and a lower learning rate.  
5. **Post-processing** — JSON parse with fallbacks, MCQ label normalisation, hedge stripping on open answers (reasoning left intact).

---

## Citation / challenge links

- Synapse project: https://www.synapse.org/Synapse:syn74403682  
- OpenReview (challenge): https://openreview.net/group?id=MICCAI.org/2026/Challenge/MedReason  
- Base model: https://huggingface.co/Qwen/Qwen3-VL-32B-Instruct  

If you use this code, please cite the MedReason challenge and the Qwen3-VL model card, and respect Synapse / challenge data terms.

---

## Licence / notice

Challenge template and evaluation utilities originate from the MedReason organisers’ public Docker starter. Our training scripts, prompting, parsing, and two-stage LoRA pipeline are provided as-is for research reproducibility. **Do not upload restricted challenge data or unpublished paper materials to public forks.**
