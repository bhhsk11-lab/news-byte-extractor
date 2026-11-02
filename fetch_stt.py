"""Build-time download of the speech-to-text model (run by the Dockerfile). Retries on network errors, fails loudly if a file is missing."""
import os
import sys
import tarfile
import time
import urllib.request

ENGINE = os.environ.get("STT_ENGINE", "parakeet").lower()
OUT = os.environ.get("STT_OUT", "/models")
REL = "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/"

SHERPA = {   # engine -> (archive name, files we keep)
    "parakeet": ("sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8",
                 ["encoder.int8.onnx", "decoder.int8.onnx", "joiner.int8.onnx", "tokens.txt"]),
    "moonshine-base": ("sherpa-onnx-moonshine-base-en-int8",
                       ["preprocess.onnx", "encode.int8.onnx", "uncached_decode.int8.onnx", "cached_decode.int8.onnx", "tokens.txt"]),
    "moonshine-tiny": ("sherpa-onnx-moonshine-tiny-en-int8",
                       ["preprocess.onnx", "encode.int8.onnx", "uncached_decode.int8.onnx", "cached_decode.int8.onnx", "tokens.txt"]),
}


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
    if ENGINE == "whisper":
        from faster_whisper import download_model
        name = os.environ.get("WHISPER_MODEL", "small.en")
        for i in range(1, 6):
            try:
                download_model(name, output_dir=os.path.join(OUT, "whisper"))
                break
            except Exception as e:
                print("whisper download failed:", e, flush=True)
                if i == 5:
                    raise
                time.sleep(4 * i)
        return
    if ENGINE not in SHERPA:
        sys.exit(f"unknown STT_ENGINE '{ENGINE}'. Use: parakeet, moonshine-base, moonshine-tiny, whisper")
    archive, keep = SHERPA[ENGINE]
    dest_dir = os.path.join(OUT, "stt")
    os.makedirs(dest_dir, exist_ok=True)
    tmp = os.path.join(OUT, "stt.tar.bz2")
    download(REL + archive + ".tar.bz2", tmp)
    with tarfile.open(tmp, "r:bz2") as tf:
        for m in tf.getmembers():
            base = os.path.basename(m.name)
            if m.isfile() and base in keep and "test_wavs" not in m.name:
                m.name = base
                tf.extract(m, dest_dir)
    os.remove(tmp)
    missing = [f for f in keep if not os.path.exists(os.path.join(dest_dir, f))]
    if missing:
        sys.exit(f"model files missing after download: {missing}")
    print("STT model ready in", dest_dir, sorted(os.listdir(dest_dir)), flush=True)


main()
