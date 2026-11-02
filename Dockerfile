# CPU voice server (fast speech-to-text + Kokoro TTS) for Cloud Run.
# Models are downloaded at BUILD time and baked into the image, so a cold start only loads files from disk.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 HF_HUB_DISABLE_TELEMETRY=1

RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

# ---- speech-to-text engine (English) ----------------------------------------------------
# parakeet       = NVIDIA Parakeet TDT 0.6B v2. Fast and very accurate, ~1 GB RAM       <- default
# moonshine-base = ~2x faster than parakeet, a bit less accurate in noise, ~0.5 GB RAM
# moonshine-tiny = fastest and smallest (~0.3 GB RAM), least accurate
# whisper        = the old engine (faster-whisper small.en). Slowest of the four.
# Change it with:  --build-arg STT_ENGINE=moonshine-base
ARG STT_ENGINE=parakeet
ARG WHISPER_MODEL=small.en
ENV STT_ENGINE=${STT_ENGINE}
COPY fetch_stt.py .
RUN python fetch_stt.py

# ---- text-to-speech model (Kokoro-82M, only English voices are used) --------------------
# Use the full fp32 file. The int8 file is smaller but returns broken audio (NaN) on ordinary sentences.
RUN mkdir -p /models \
    && curl -fL --retry 5 --retry-all-errors -o /models/kokoro.onnx https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx \
    && curl -fL --retry 5 --retry-all-errors -o /models/voices.bin  https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin

COPY main.py .
COPY audio_clean.py .

# One worker: the models live in memory once. Cloud Run scales by adding instances, not workers.
CMD exec uvicorn main:app --host 0.0.0.0 --port ${PORT:-8080} --workers 1
