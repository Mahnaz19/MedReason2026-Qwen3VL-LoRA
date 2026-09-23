# Model checkpoints for MedReason submission

Place a self-contained Qwen VL HuggingFace checkpoint in this directory before building the Docker image for organizer-side evaluation.

## Competition model

```text
models/Qwen3-VL-32B-Instruct/
```

HuggingFace: https://huggingface.co/Qwen/Qwen3-VL-32B-Instruct

Download on your GPU machine (recommended: Isambard-AI):

```bash
pip install huggingface_hub
hf download Qwen/Qwen3-VL-32B-Instruct --local-dir models/Qwen3-VL-32B-Instruct
```

Approximate size: **~64 GB**.

## Runtime configuration

```bash
export MEDREASON_MODEL_PATH=/path/to/models/Qwen3-VL-32B-Instruct
```

After LoRA fine-tuning, point `MEDREASON_MODEL_PATH` at the merged checkpoint directory.

To test an adapter before merging:

```bash
export MEDREASON_MODEL_PATH=/path/to/models/Qwen3-VL-32B-Instruct
export MEDREASON_LORA_PATH=/path/to/data/outputs/lora/qwen3vl_medreason/final
```

## LoRA training (Isambard)

Install training deps (once per conda env):

```bash
pip install peft>=0.13.0
```

Smoke test (500 records, 1 epoch):

```bash
bash scripts/isambard/run_gpu_lora_train.sh 500 1
```

Full train (all 17,722 records):

```bash
bash scripts/isambard/run_gpu_lora_train.sh "" 1
```

Merge adapter for Docker / inference:

```bash
bash scripts/isambard/run_gpu_lora_merge.sh
export MEDREASON_MODEL_PATH=../data/outputs/lora/qwen3vl_medreason_merged
```

Submit long jobs with `sbatch` (recommended). Example:

```bash
cat > ~/run_lora_smoke.sh << 'EOF'
#!/bin/bash
#SBATCH --job-name=lora500
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --partition=workq
#SBATCH --time=12:00:00
#SBATCH --output=/home/u6mn/mahnaz.u6mn/lora_smoke.log
source ~/miniforge3/bin/activate
conda activate medreason
cd ~/MedReason-Challenge-Docker/MedReason-Docker
bash scripts/isambard/run_gpu_lora_train.sh 500 1
EOF
chmod +x ~/run_lora_smoke.sh
sbatch ~/run_lora_smoke.sh
```

## Hardware notes

| Environment | Notes |
|-------------|-------|
| Isambard-AI (1× GH200, 96 GB) | Fits Qwen3-VL-32B-Instruct with `device_map=auto` |
| Docker submission | Bake the full (merged) checkpoint into the image; no internet at eval time |

## Supported checkpoints

The inference system loads **Qwen3-VL** and **Qwen2.5-VL** HuggingFace checkpoints via `AutoProcessor` and `Qwen3VLForConditionalGeneration` / `Qwen2_5_VLForConditionalGeneration`.

Requires `transformers>=4.57.0`.
