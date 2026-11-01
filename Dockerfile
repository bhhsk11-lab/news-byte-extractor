# CPU voice server (Whisper STT + Kokoro TTS) for Cloud Run.
# Models are downloaded at BUILD time and baked into the image, so a cold start only loads files from disk.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 HF_HUB_DISABLE_TELEMETRY=1

RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

# ---- speech-to-text model --------------------------------------------------------------
# small  = good English + usable Hindi, ~1 GB RAM       <- default
# base   = faster, weaker Hindi, ~0.5 GB RAM
# medium = much better Hindi, but too slow on 2 vCPU for live talking
ARG WHISPER_MODEL=small
RUN python -c "from faster_whisper import download_model; download_model('${WHISPER_MODEL}', output_dir='/models/whisper')"

# ---- text-to-speech model (Kokoro-82M, English + Hindi) --------------------------------
# Use the full fp32 file. The int8 file is smaller but returns broken audio (NaN) on ordinary sentences.
RUN mkdir -p /models \
    && curl -fL --retry 3 -o /models/kokoro.onnx https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx \
    && curl -fL --retry 3 -o /models/voices.bin  https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin

COPY main.py .

# One worker: the models live in memory once. Cloud Run scales by adding instances, not workers.
CMD exec uvicorn main:app --host 0.0.0.0 --port ${PORT:-8080} --workers 1
