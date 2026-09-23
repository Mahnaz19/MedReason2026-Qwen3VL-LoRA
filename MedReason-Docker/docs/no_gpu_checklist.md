# Work you can do without GPU (while waiting for Isambard access)

## Already done
- [x] Train + validation data organized under `data/raw/`
- [x] Synapse → `cases.json` converter
- [x] Submission system (`custom_system.py` + Qwen VL backend)
- [x] Smoke test on full validation split (2532 cases, format OK)
- [x] Train MCQ ground truth built (14,398 cases)
- [x] LoRA dataset prepared (`data/derived/train_lora.jsonl`)
- [x] Phase A code update for **Qwen3-VL-32B-Instruct**
- [x] Docker smoke test passed locally

## Do on Isambard (GPU)

### 1. Download model (~64 GB)
```bash
cd MedReason-Docker
bash scripts/isambard/download_model.sh
```

### 2. First GPU runs
```bash
srun --nodes=1 --gpus=1 --partition=workq --time=02:00:00 --pty bash -i
conda activate medreason
cd MedReason-Docker
bash scripts/isambard/run_gpu_validation.sh 10
bash scripts/isambard/run_gpu_train_mcq_eval.sh 500
```

### 3. LoRA fine-tune
- Data: `data/derived/train_lora.jsonl`
- Framework: LLaMA-Factory or HuggingFace TRL + PEFT
- Merge adapter → set `MEDREASON_MODEL_PATH` to merged checkpoint

## Copy to Isambard

Copy the repository and `data/` folder. **Do not** copy a local model checkpoint — download Qwen3-VL-32B on Isambard instead.

## First hour on GPU (priority order)

1. `nvidia-smi` — confirm GH200 visible
2. `pip install -r requirements.txt`
3. `bash scripts/isambard/download_model.sh` (if not done)
4. `bash scripts/isambard/run_gpu_validation.sh 10`
5. `bash scripts/isambard/run_gpu_train_mcq_eval.sh 500`
6. Start LoRA fine-tune

## Do not block on

- Full validation inference (2532 cases) — run overnight once GPU works
- Docker export with full model — do after first successful GPU inference + LoRA merge
