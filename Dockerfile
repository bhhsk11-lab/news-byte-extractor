# CPU voice server (Moondream Parakeet Redux speech-to-text + Kokoro TTS) for Cloud Run.
# Models are downloaded at BUILD time and baked into the image, so a cold start only loads files from disk.
FROM python:3.12-slim

# STT_BACKEND: redux = Moondream Parakeet Redux (default, small RAM) | sherpa = Parakeet TDT 0.6B v2 int8 (English only, ~1.2-1.65 GB RAM).
# Change the default here to switch; the same value is used at build time (what to download) and at run time (what to load).
ARG STT_BACKEND=redux
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 HF_HUB_DISABLE_TELEMETRY=1 STT_BACKEND=${STT_BACKEND} HF_HOME=/models/hf

RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

# ---- speech-to-text: Moondream Parakeet Redux (25 languages, language picked automatically, CPU) - the only STT model in this image ----
# Weights are cached at BUILD time. License: CC-BY-4.0, same as NVIDIA's Parakeet (attribution required, see README).
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
