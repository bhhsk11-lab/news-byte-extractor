"""Build-time download of the speech-to-text model (run by the Dockerfile).
faster-whisper only (CTranslate2 format, runs on CPU). Retries on network errors, fails loudly if a file is missing."""
import os
import sys
import time

NAME = os.environ.get("WHISPER_MODEL", "small.en")
OUT = os.path.join(os.environ.get("STT_OUT", "/models"), "whisper")


def main():
    from faster_whisper import download_model
    for i in range(1, 7):
        try:
            print(f"download whisper {NAME} (try {i})", flush=True)
            download_model(NAME, output_dir=OUT)
            break
        except Exception as e:  # network hiccup: wait and retry
            print("  failed:", e, flush=True)
            if i == 6:
                raise
            time.sleep(4 * i)
    need = ["model.bin", "config.json", "tokenizer.json"]
    missing = [f for f in need if not os.path.exists(os.path.join(OUT, f))]
    if missing:
        sys.exit(f"whisper files missing after download: {missing}")
    size = os.path.getsize(os.path.join(OUT, "model.bin")) / 1e6
    print(f"Whisper {NAME} ready in {OUT} (model.bin {size:.0f} MB)", flush=True)


main()
