"""Build-time download of the speech-to-text model (run by the Dockerfile).
NVIDIA Parakeet TDT 0.6B v2 (English, int8) is the only speech-to-text model. Silero VAD (a ~2 MB neural voice detector, not a
speech-to-text model) is downloaded with it. Retries on network errors, fails loudly if a file is missing."""
import os
import sys
import tarfile
import time
import urllib.request

OUT = os.environ.get("STT_OUT", "/models")
URL = "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8.tar.bz2"
KEEP = ["encoder.int8.onnx", "decoder.int8.onnx", "joiner.int8.onnx", "tokens.txt"]
VAD_URL = "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/silero_vad.onnx"
TEN_URL = "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/ten-vad.onnx"   # optional (VAD_NET=ten); a failed download is not fatal


def download(url: str, dest: str, tries: int = 6):
    for i in range(1, tries + 1):
        try:
            print(f"download {url} (try {i})", flush=True)
            urllib.request.urlretrieve(url, dest)
            return
        except Exception as e:  # network hiccup: wait and retry
            print("  failed:", e, flush=True)
            if i == tries:
                raise
            time.sleep(3 * i)


def main():
    dest_dir = os.path.join(OUT, "stt")
    os.makedirs(dest_dir, exist_ok=True)
    tmp = os.path.join(OUT, "stt.tar.bz2")
    download(URL, tmp)
    with tarfile.open(tmp, "r:bz2") as tf:
        for m in tf.getmembers():
            base = os.path.basename(m.name)
            if m.isfile() and base in KEEP and "test_wavs" not in m.name:
                m.name = base
                tf.extract(m, dest_dir)
    os.remove(tmp)
    missing = [f for f in KEEP if not os.path.exists(os.path.join(dest_dir, f))]
    if missing:
        sys.exit(f"model files missing after download: {missing}")
    download(VAD_URL, os.path.join(dest_dir, "silero_vad.onnx"))
    if os.path.getsize(os.path.join(dest_dir, "silero_vad.onnx")) < 100000:
        sys.exit("silero_vad.onnx looks wrong (too small)")
    try:
        download(TEN_URL, os.path.join(dest_dir, "ten-vad.onnx"), tries=2)
    except Exception as e:
        print("note: optional TEN VAD not downloaded:", e, flush=True)
    print("Parakeet TDT 0.6B v2 + VAD ready in", dest_dir, sorted(os.listdir(dest_dir)), flush=True)


main()
