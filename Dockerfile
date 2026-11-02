# CPU voice server (faster-whisper speech-to-text + Kokoro TTS) for Cloud Run.
# Models are downloaded at BUILD time and baked into the image, so a cold start only loads files from disk.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 HF_HUB_DISABLE_TELEMETRY=1

RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

# ---- speech-to-text: faster-whisper (CPU, int8) - the only STT engine in this image ----
# small.en = default: good accuracy, about 1 GB RAM.   base.en = about twice as fast, a little less accurate.
# medium.en = more accurate but too slow for live talking on 2 vCPU.   Change it with:  --build-arg WHISPER_MODEL=base.en
ARG WHISPER_MODEL=small.en
ENV WHISPER_MODEL=${WHISPER_MODEL}
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
