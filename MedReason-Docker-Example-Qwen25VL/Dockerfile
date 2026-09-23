FROM python:3.11

RUN apt-get update && apt-get install -y --no-install-recommends     ffmpeg     libsm6     libxext6     git     && rm -rf /var/lib/apt/lists/*

RUN groupadd -r user && useradd -m --no-log-init -r -g user user

RUN mkdir -p /opt/app/gc_job/ /input /output /opt/app/models     && chown -R user:user /opt/app/ /input /output

USER user
WORKDIR /opt/app/

ENV PATH="/home/user/.local/bin:${PATH}"
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV MEDREASON_INPUT_DIR=/input
ENV MEDREASON_OUTPUT_FILE=/output/results.json
ENV MEDREASON_MODEL_PATH=/opt/app/models/Qwen2.5-VL
ENV MEDREASON_MAX_NEW_TOKENS=512
ENV MEDREASON_TEMPERATURE=0.0

COPY --chown=user:user requirements.txt /opt/app/requirements.txt
RUN python -m pip install --user -U pip &&     python -m pip install --user pip-tools &&     python -m pip install --user -r /opt/app/requirements.txt

COPY --chown=user:user models /opt/app/models
COPY --chown=user:user process.py /opt/app/process.py

ENTRYPOINT ["python", "process.py"]
