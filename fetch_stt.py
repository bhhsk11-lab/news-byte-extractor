"""Build-time download of the speech-to-text model (run by the Dockerfile). Backend chosen by STT_BACKEND (default: redux).

redux  : Moondream Parakeet Redux. Loaded once through Photon so the weights land in the Hugging Face cache (HF_HOME) inside the image.
         If this step cannot download (network, access), the build CONTINUES and the server downloads the weights at its first start
         instead (slower cold start). The warning is printed in the build log.
sherpa : NVIDIA Parakeet TDT 0.6B v2 int8 (English only) + Silero VAD, through sherpa-onnx. Retries on network errors, fails loudly if a file is missing.
"""
import os
import sys
import tarfile
import time
import urllib.request

BACKEND = os.environ.get("STT_BACKEND", "redux").strip().lower()
OUT = os.environ.get("STT_OUT", "/models")
REDUX_MODEL = os.environ.get("REDUX_MODEL", "moondream/parakeet-redux")
URL = "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8.tar.bz2"
KEEP = ["encoder.int8.onnx", "decoder.int8.onnx", "joiner.int8.onnx", "tokens.txt"]
VAD_URL = "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/silero_vad.onnx"   # tiny neural voice detector (not a speech-to-text model)


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


def fetch_vad():
    """Silero VAD (about 2 MB) finds the pauses where speech is cut into pieces. Needed by both backends."""
    dest_dir = os.path.join(OUT, "stt")
    os.makedirs(dest_dir, exist_ok=True)
    path = os.path.join(dest_dir, "silero_vad.onnx")
    download(VAD_URL, path)
    if os.path.getsize(path) < 100000:
        sys.exit("silero_vad.onnx looks wrong (too small)")


def fetch_redux():
    fetch_vad()
    try:
        import moondream as md
        for i in range(1, 4):
            try:
                print(f"loading {REDUX_MODEL} once to cache its weights (try {i}), HF_HOME={os.environ.get('HF_HOME')}", flush=True)
                with md.photon(REDUX_MODEL, device="cpu"):
                    pass
                print("Moondream Parakeet Redux cached in the image", flush=True)
                return
            except Exception as e:
                print("  failed:", e, flush=True)
                time.sleep(3 * i)
        raise RuntimeError("3 tries failed")
    except Exception as e:
        print(f"WARNING: could not pre-download {REDUX_MODEL} ({e}). The server will download it at first start (slow cold start).", flush=True)


def fetch_sherpa():
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
    fetch_vad()
    print("Parakeet TDT 0.6B v2 + Silero VAD ready in", dest_dir, sorted(os.listdir(dest_dir)), flush=True)


def main():
    print("STT_BACKEND =", BACKEND, flush=True)
    if BACKEND == "redux":
        fetch_redux()
    elif BACKEND == "sherpa":
        fetch_sherpa()
    else:
        sys.exit(f"STT_BACKEND must be redux or sherpa, not {BACKEND!r}")


main()
