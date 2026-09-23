# Run inference on train MCQ subset and score locally (requires GPU for inference).
param(
    [int]$MaxCases = 500
)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot\..

python tools\build_train_ground_truth.py

$env:MEDREASON_SYSTEM = "custom"
$env:MEDREASON_MODEL_PATH = "$PWD\models\Qwen3-VL-32B-Instruct"
$env:MEDREASON_INPUT_DIR = "..\data\raw\train"
$env:MEDREASON_OUTPUT_FILE = "..\data\outputs\train_mcq\results.json"
$env:MEDREASON_MAX_CASES = "$MaxCases"

New-Item -ItemType Directory -Force -Path ..\data\outputs\train_mcq | Out-Null
python process.py

Set-Location ..\MedReason-Evaluation
python scoring.py `
  --predictions ..\data\outputs\train_mcq\results.json `
  --ground-truth ..\data\derived\train_ground_truth_mcq.json `
  --pretty
