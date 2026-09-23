# Pre-evaluation plan (MedReason @ MICCAI 2026)

Target: submit a **competition-ready** Docker package for organizer-side **pre-evaluation**.
Pre-evaluation scores on the hidden public split **count equally** with the hidden private
split toward the **final official ranking** — not a throwaway test.

Final submission deadline: **15 July 2026** (update Docker + PDF if the method improves after pre-eval feedback).

## Hardware profile

- **Isambard-AI**: 1× GH200 (96 GB GPU memory) per Slurm GPU allocation
- **Competition model**: **Qwen3-VL-32B-Instruct** + LoRA on train split

## Timeline

| Phase | Goal | Status |
|-------|------|--------|
| 1. Data + I/O | Synapse → `cases.json`, smoke test on validation | Done |
| 2. Code prep for Qwen3-VL | Loader, requirements, scripts | Done |
| 3. Model download | Download Qwen3-VL-32B on Isambard | **Your next step** |
| 4. Zero-shot baseline | Run 10-case subset, then 500-case train MCQ | Done (42.6% on 500 MCQ) |
| 5. Train MCQ score | Measure MCQ accuracy on train split | Done (500-case subset) |
| 6. LoRA fine-tune | Fine-tune on 17k train cases | **Next — scripts ready** |
| 7. Docker package | Build, test, export `.tar.gz` | **Before pre-eval** |
| 8. Pre-evaluation submit | Email Docker link + PDF to organizers | Once |
| 9. Final submission | Update if method improved after feedback | By 15 July 2026 |

## Commands (Isambard-AI)

### 1) Setup environment

```bash
cd ~/medreason/MedReason-Challenge-Docker/MedReason-Docker
conda activate medreason   # or your env name
pip install -r requirements.txt
hf auth login
bash scripts/isambard/download_model.sh
```

### 2) Request GPU and run 10-case validation test

```bash
srun --nodes=1 --gpus=1 --partition=workq --time=02:00:00 --pty bash -i
conda activate medreason
cd ~/medreason/MedReason-Challenge-Docker/MedReason-Docker
bash scripts/isambard/run_gpu_validation.sh 10
```

### 3) Train MCQ baseline (500 cases)

```bash
sbatch --wrap="bash scripts/isambard/run_gpu_train_mcq_eval.sh 500"
```

Or interactively:

```bash
bash scripts/isambard/run_gpu_train_mcq_eval.sh 500
```

### 4) LoRA smoke test (500 records)

```bash
sbatch --wrap="bash scripts/isambard/run_gpu_lora_train.sh 500 1"
```

### 5) LoRA full train + merge

```bash
# Full train (~24h); use sbatch with --time=24:00:00 or longer
bash scripts/isambard/run_gpu_lora_train.sh "" 1
bash scripts/isambard/run_gpu_lora_merge.sh
export MEDREASON_MODEL_PATH=../../data/outputs/lora/qwen3vl_medreason_merged
```

### 6) Full validation run (2532 cases)

```bash
unset MEDREASON_MAX_CASES
export MEDREASON_MODEL_PATH=$PWD/models/Qwen3-VL-32B-Instruct
export MEDREASON_INPUT_DIR=../../data/raw/validation
export MEDREASON_OUTPUT_FILE=../../data/outputs/isambard/validation_full/results.json
python process.py
python tools/validate_output.py $MEDREASON_OUTPUT_FILE --input-json $MEDREASON_INPUT_DIR/cases.json
```

### 5) Docker build + export (after LoRA merge, on Windows or cluster)

```bash
docker build -t medreason-docker .
./test.sh medreason-docker
./export.sh medreason-docker medreason_pre_eval.tar.gz
```

Upload the archive and email the link to **medreason26@googlegroups.com**.

## What to improve before pre-evaluation

1. **LoRA fine-tune** on full train split (biggest expected gain)
2. **Prompt tuning** on a 100–200 case train subset
3. **Open-ended faithfulness** — reasoning_trace quality affects official VA via RVF_trace gating
4. **Self-verification pass** — optional second VLM check against the image for open-ended cases

## System entry point

```text
medreason_docker/systems/custom_system.py
medreason_docker/systems/qwen_vl_system.py
```

After LoRA, point `MEDREASON_MODEL_PATH` at the merged checkpoint directory.
