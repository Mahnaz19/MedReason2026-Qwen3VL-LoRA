# Run real Qwen inference on validation (requires GPU).
param(
    [int]$MaxCases = 10,
    [string]$OutputDir = "..\data\outputs\validation_gpu"
)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot\..

$env:MEDREASON_SYSTEM = "custom"
$env:MEDREASON_MODEL_PATH = "$PWD\models\Qwen3-VL-32B-Instruct"
$env:MEDREASON_INPUT_DIR = "..\data\raw\validation"
$env:MEDREASON_OUTPUT_FILE = "$OutputDir\results.json"

if ($MaxCases -gt 0) {
    $env:MEDREASON_MAX_CASES = "$MaxCases"
} else {
    Remove-Item Env:MEDREASON_MAX_CASES -ErrorAction SilentlyContinue
}

New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null
python process.py
python tools\validate_output.py $env:MEDREASON_OUTPUT_FILE --input-json ..\data\raw\validation\cases.json
Write-Host "Done. Output: $env:MEDREASON_OUTPUT_FILE"
