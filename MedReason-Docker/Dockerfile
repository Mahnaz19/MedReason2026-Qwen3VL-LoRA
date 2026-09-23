FROM python:3.11

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsm6 \
    libxext6 \
    libglib2.0-0 \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    MEDREASON_INPUT_DIR=/input \
    MEDREASON_OUTPUT_DIR=/output \
    MEDREASON_OUTPUT_FILE=/output/results.json \
    MEDREASON_SYSTEM=custom \
    MEDREASON_MODEL_PATH=/opt/app/models/Qwen3-VL-32B-Instruct \
    MEDREASON_LORA_PATH=/opt/app/models/lora/medreason_open_final \
    MEDREASON_MAX_NEW_TOKENS=512 \
    MEDREASON_TEMPERATURE=0.0

WORKDIR /opt/app

COPY requirements.txt /opt/app/requirements.txt
RUN pip install --no-cache-dir -r /opt/app/requirements.txt

COPY . /opt/app
RUN chmod +x /opt/app/process.py

ENTRYPOINT ["python", "/opt/app/process.py"]
