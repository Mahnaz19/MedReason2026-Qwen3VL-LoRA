# Alignment with VLM3D `reportgen_example_docker`

The uploaded VLM3D example is not just a Docker skeleton. It implements a full reference baseline:

| VLM3D `reportgen_example_docker` | MedReason aligned example |
|---|---|
| `/input` contains CT volumes | `/input` contains `cases.json` and images |
| `/output/results.json` is fixed | `/output/results.json` is fixed |
| `process.py` is the no-CLI entrypoint | `process.py` is the no-CLI entrypoint |
| model files copied into `/opt/app/models` | Qwen2.5-VL checkpoint copied into `/opt/app/models/Qwen2.5-VL` |
| loads CTViT visual encoder + CT-CHAT generator | loads Qwen2.5-VL baseline system |
| performs preprocessing, model inference, JSON assembly | performs prompt construction, model inference, parsing, normalization, JSON assembly |
| `build.sh`, `test.sh`, `export.sh` scripts | same script pattern |
| Docker uses non-root user and fixed mounted dirs | same pattern |

The main task difference is that VLM3D produces one report per CT volume, while MedReason produces one answer per case and supports both MCQ and open-ended tasks.
