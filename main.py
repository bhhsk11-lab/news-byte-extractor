"""
CPU voice server for Cloud Run  -  OpenAI-compatible speech API.

    POST /v1/audio/transcriptions   speech -> text   (NVIDIA Parakeet TDT 0.6B v2, int8, CPU, English only - the only STT model)
    POST /v1/audio/speech           text -> speech   (Kokoro-82M ONNX, CPU)
    GET  /health                    warm-up / status

The AI Chat Firefox extension already speaks this API: in Settings > Voice set the
speech-to-text and text-to-speech "base URL" to  https://<your-service>.run.app/v1
and the "key" to the API_KEY you chose.
"""
import hmac
import io
import logging
import os
import queue
import re
import threading
import time
import wave
from collections import OrderedDict
from contextlib import asynccontextmanager, contextmanager

import lameenc
import numpy as np

import audio_clean as ac
from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse, Response
from pydantic import BaseModel

log = logging.getLogger("voice")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# ------------------------------------------------------------------ settings (all overridable with env vars)
API_KEY = os.getenv("API_KEY", "").strip()                    # clients send  Authorization: Bearer <API_KEY>
STT_ENGINE = "parakeet-tdt-0.6b-v2"                            # the ONLY speech-to-text model: NVIDIA Parakeet TDT 0.6B v2, 600 M parameters, English only, int8, CPU
STT_DIR = os.getenv("STT_DIR", "/models/stt")                  # baked into the image (see Dockerfile / fetch_stt.py)
STT_THREADS = int(os.getenv("STT_THREADS", "2"))               # = number of vCPUs, shared by all running recordings
STT_WORKERS = max(1, int(os.getenv("STT_WORKERS", str(max(1, min(2, STT_THREADS))))))  # recordings transcribed AT THE SAME TIME
VAD_MODE = os.getenv("VAD_MODE", "neural").strip().lower()     # neural = Silero VAD network (falls back to energy if it cannot load) | energy = simple loudness rule
VAD_NET = os.getenv("VAD_NET", "silero").strip().lower()        # silero (default, MIT) | ten (TEN VAD, optional; falls back to silero if it cannot load)
VAD_THRESHOLD = float(os.getenv("VAD_THRESHOLD", "0.4"))       # Silero speech probability needed to count as speech (0.3 = more sensitive, 0.7 = stricter)
STT_SEPARATE_COPIES = os.getenv("STT_SEPARATE_COPIES", "0") == "1"  # 0 = all workers share ONE loaded model (about 1.2 GB). 1 = one model per worker (safest, +1.2 GB each)
LEVELER = os.getenv("LEVELER", "1") == "1"                       # even out the loudness BEFORE the voice detector, so quiet sentences are not skipped
TARGET_DBFS = float(os.getenv("TARGET_DBFS", "-20"))            # every speech piece is set to this level (dB below full scale) before decoding
MAX_GAIN_DB = float(os.getenv("MAX_GAIN_DB", "40"))             # most a quiet passage may be lifted
PIECE_MAX_S = float(os.getenv("PIECE_MAX_S", "12"))             # speech is decoded in pieces of at most this many seconds, cut at pauses (long windows lose sentences)
RETRY_DROPS = os.getenv("RETRY_DROPS", "1") == "1"              # decode a piece a second time (with quiet edges) when it looks like words were lost
STT_QUEUE_MAX = int(os.getenv("STT_QUEUE_MAX", "8"))           # more than this many recordings waiting -> HTTP 429 (client retries) instead of a long hang
TTS_WORKERS = max(1, int(os.getenv("TTS_WORKERS", "1")))       # sentences spoken at the same time (each copy of Kokoro needs ~0.55 GB RAM)
MAX_AUDIO_SECONDS = float(os.getenv("MAX_AUDIO_SECONDS", "120"))  # protects the memory limit
TRIM_TTS = os.getenv("TRIM_TTS", "1") == "1"                  # cut dead air at the start/end of speech -> denser, snappier audio
TTS_CACHE_MAX = int(os.getenv("TTS_CACHE_MAX", "64"))         # repeated sentences are served instantly
KOKORO_MODEL = os.getenv("KOKORO_MODEL", "/models/kokoro.onnx")     # fp32 build. Do NOT use the int8 file: it outputs garbage
KOKORO_VOICES = os.getenv("KOKORO_VOICES", "/models/voices.bin")
KOKORO_THREADS = int(os.getenv("KOKORO_THREADS", "2"))         # threads per Kokoro copy
MP3_KBPS = int(os.getenv("MP3_KBPS", "48"))                   # small = cheap egress. 64 for a bit more quality
MAX_AUDIO_MB = float(os.getenv("MAX_AUDIO_MB", "25"))

# Voices. Any Kokoro voice id (af_heart, am_michael, hf_alpha, hm_omega ...) is accepted as-is.
# OpenAI names that the extension's picker uses (marin, cedar ...) are mapped by gender.
EN_FEMALE, EN_MALE = os.getenv("VOICE_EN_FEMALE", "af_heart"), os.getenv("VOICE_EN_MALE", "am_michael")
HI_FEMALE, HI_MALE = os.getenv("VOICE_HI_FEMALE", "hf_alpha"), os.getenv("VOICE_HI_MALE", "hm_omega")
MALE_NAMES = {"cedar", "ash", "echo", "ballad", "verse", "onyx", "fable"}

class Busy(Exception):
    pass


class Pool:
    """A few ready-to-use model copies. A request takes one, uses it alone, gives it back. When all copies are busy the
    request waits in line (up to `max_wait` others); this is what lets several people be served at the same moment."""

    def __init__(self, items, max_wait: int = 1000):
        self._q: "queue.Queue" = queue.Queue()
        for it in items:
            self._q.put(it)
        self.size = len(items)
        self.max_wait, self.waiting, self.active = max_wait, 0, 0
        self._n = threading.Lock()

    @contextmanager
    def take(self):
        with self._n:
            if self._q.empty() and self.waiting >= self.max_wait:
                raise Busy()
            self.waiting += 1
        try:
            item = self._q.get()
        finally:
            with self._n:
                self.waiting -= 1
        with self._n:
            self.active += 1
        try:
            yield item
        finally:
            with self._n:
                self.active -= 1
            self._q.put(item)


_stt = {"pool": None}
_stats = {"recovered": 0}
_tts = {"pool": None, "voices": set()}
_tts_cache: "OrderedDict" = OrderedDict()


# ------------------------------------------------------------------ model loading
def _load_parakeet():
    """NVIDIA Parakeet TDT 0.6B v2 (English) through sherpa-onnx (int8 ONNX, CPU)."""
    import sherpa_onnx as so
    d = STT_DIR
    return so.OfflineRecognizer.from_transducer(
        encoder=f"{d}/encoder.int8.onnx", decoder=f"{d}/decoder.int8.onnx", joiner=f"{d}/joiner.int8.onnx",
        tokens=f"{d}/tokens.txt", num_threads=STT_THREADS, sample_rate=16000, feature_dim=80,
        decoding_method="greedy_search", model_type="nemo_transducer", provider="cpu")


def _log_memory(where: str):
    """Resident memory of this process, so you can read the real GB per model copy in the Cloud Run logs."""
    try:
        with open("/proc/self/status") as f:
            kb = next(int(l.split()[1]) for l in f if l.startswith("VmRSS:"))
        log.info("MEMORY %s: %.2f GB resident", where, kb / 1048576)
    except Exception:
        pass


def load_models():
    t = time.time()
    first = _load_parakeet()
    copies = [first] + [_load_parakeet() if STT_SEPARATE_COPIES else first for _ in range(STT_WORKERS - 1)]
    _stt["pool"] = Pool(copies, max_wait=STT_QUEUE_MAX)      # at most STT_WORKERS recordings run at once, the rest wait in line
    log.info("%s x%d worker(s), %s model copy/copies, loaded in %.1fs (%s)", STT_ENGINE, STT_WORKERS,
             STT_WORKERS if STT_SEPARATE_COPIES else 1, time.time() - t, STT_DIR)
    _log_memory("after STT load")

    import onnxruntime as ort
    from kokoro_onnx import Kokoro
    t = time.time()
    copies = []
    for _ in range(TTS_WORKERS):
        so = ort.SessionOptions()
        so.intra_op_num_threads = KOKORO_THREADS
        so.inter_op_num_threads = 1
        sess = ort.InferenceSession(KOKORO_MODEL, sess_options=so, providers=["CPUExecutionProvider"])
        copies.append(Kokoro.from_session(sess, KOKORO_VOICES))
    _tts["pool"] = Pool(copies)
    _tts["voices"] = set(copies[0].get_voices())
    log.info("kokoro x%d loaded in %.1fs (%d voices)", TTS_WORKERS, time.time() - t, len(_tts["voices"]))
    _log_memory("after TTS load")


def warm_up():
    """One tiny job through each engine so the first real request is not slow."""
    try:
        rng = np.random.default_rng(0)
        probe = (0.05 * rng.standard_normal(16000 * 2)).astype(np.float32)   # not silence, so the encoder really runs
        ths = []
        def _one():
            with _stt["pool"].take() as rec:
                _stt_decode(rec, probe)
        ths = [threading.Thread(target=_one) for _ in range(STT_WORKERS)]    # every worker gets warmed
        [t.start() for t in ths]
        [t.join() for t in ths]
        _synthesize_blocking("Hello.", EN_FEMALE, 1.0)
        _log_memory("after warm-up")
        log.info("warm-up done")
    except Exception as e:  # never block startup because of the warm-up
        log.warning("warm-up skipped: %s", e)


_ready = threading.Event()
_boot = {"error": None, "t0": time.time()}


def _boot_models():
    try:
        load_models()
        warm_up()
        _ready.set()
        log.info("READY after %.1fs", time.time() - _boot["t0"])
    except Exception as e:
        _boot["error"] = f"{type(e).__name__}: {e}"
        log.exception("MODEL LOADING FAILED")


@asynccontextmanager
async def lifespan(app):
    # PRELOAD=1 (default): load the models BEFORE the port opens. Cloud Run gives a starting container full CPU, but once the startup
    # check has passed it slows an idle container to almost nothing, so a background load stalled for 6-14 minutes (seen in the logs:
    # "parakeet loaded in 374.1s", "READY after 830.4s") and the first recordings waited 16 s for it. Loading first takes ~20-40 s at full CPU.
    # PRELOAD=0: the old behaviour (open at once, load in the background; requests wait up to 90 s).
    if os.getenv("PRELOAD", "1") == "1":
        await run_in_threadpool(_boot_models)
    else:
        threading.Thread(target=_boot_models, daemon=True).start()
    yield


async def _require_ready():
    if _ready.is_set():
        return
    if _boot["error"]:
        raise HTTPException(500, "Voice models failed to load: " + _boot["error"])
    await run_in_threadpool(_ready.wait, 90)
    if not _ready.is_set():
        raise HTTPException(503, "Voice models are still loading, try again in a few seconds.")


app = FastAPI(title="CPU voice server", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"], max_age=3600)


# ------------------------------------------------------------------ auth + OpenAI-style errors
def auth(authorization: str = Header(default="")):
    if not API_KEY:
        return
    tok = authorization[7:] if authorization.lower().startswith("bearer ") else authorization
    if not hmac.compare_digest(tok.strip().encode(), API_KEY.encode()):
        raise HTTPException(status_code=401, detail="Invalid API key.")


@app.exception_handler(HTTPException)
async def _http_err(_: Request, e: HTTPException):
    return JSONResponse({"error": {"message": str(e.detail), "type": "invalid_request_error"}}, status_code=e.status_code)


@app.exception_handler(RequestValidationError)
async def _val_err(_: Request, e: RequestValidationError):
    return JSONResponse({"error": {"message": "Bad request: " + str(e.errors())[:300], "type": "invalid_request_error"}}, status_code=400)


# ------------------------------------------------------------------ speech -> text
def _silence_wav(sec=0.5, sr=16000) -> bytes:
    b = io.BytesIO()
    with wave.open(b, "wb") as w:
        w.setnchannels(1), w.setsampwidth(2), w.setframerate(sr)
        w.writeframes(b"\x00\x00" * int(sec * sr))
    return b.getvalue()


def _decode(data: bytes) -> np.ndarray:
    """Any audio the browser can record (webm/opus, ogg, mp4/m4a, mp3, wav) -> 16 kHz mono float32."""
    import av
    chunks, total, cap = [], 0, int(MAX_AUDIO_SECONDS * 16000)
    with av.open(io.BytesIO(data), mode="r") as box:
        stream = next((x for x in box.streams if x.type == "audio"), None)
        if stream is None:
            raise ValueError("no audio track in this file")
        rs = av.AudioResampler(format="s16", layout="mono", rate=16000)
        for frame in box.decode(stream):
            for f in rs.resample(frame):
                a = f.to_ndarray().reshape(-1)
                chunks.append(a)
                total += a.size
            if total > cap:
                raise ValueError(f"recording is longer than {MAX_AUDIO_SECONDS:g} seconds")
        for f in rs.resample(None) or []:
            chunks.append(f.to_ndarray().reshape(-1))
    if not chunks:
        raise ValueError("the recording is empty")
    return np.concatenate(chunks).astype(np.float32) / 32768.0


def _prepare(audio: np.ndarray) -> np.ndarray:
    """Clean the recording before the voice detector and the speech model: remove DC offset and rumble (high-pass, no denoiser: published
    tests show denoising networks make Parakeet/Whisper worse), then even out the loudness so quiet sentences are as strong as loud ones.
    Returns an empty array when there is no real sound."""
    if audio.size == 0:
        return audio
    audio = audio - float(np.mean(audio))
    if float(np.abs(audio).max()) < 0.0005:           # digital silence (below -66 dBFS)
        return audio[:0]
    audio = ac.clean(audio, gate=False)                # high-pass only (about 40-80 Hz)
    if LEVELER:
        audio = ac.level(audio, target_db=TARGET_DBFS, max_gain_db=MAX_GAIN_DB)
    else:
        peak = float(np.abs(audio).max())
        audio = np.clip(audio * min(0.85 / max(peak, 1e-6), 12.0), -1.0, 1.0)
    return audio.astype(np.float32)


def _sherpa_decode(rec, audio: np.ndarray) -> str:
    stream = rec.create_stream()                        # a stream belongs to ONE thread; the recognizer itself is shared
    stream.accept_waveform(16000, audio)
    rec.decode_stream(stream)
    return stream.result.text.strip()


def _stt_decode(rec, audio: np.ndarray) -> str:
    """One piece of 16 kHz mono float32 audio -> text (Parakeet TDT 0.6B v2)."""
    return _sherpa_decode(rec, audio)


def _decode_piece(rec, piece: np.ndarray) -> str:
    """Decode one speech piece. Parakeet-type models sometimes return nothing, or only part of a piece, without any error.
    When the result looks too short for the length of the piece, decode it once more with faint noise on both edges and keep
    whichever result has more words."""
    text = _stt_decode(rec, piece)
    if not RETRY_DROPS:
        return text
    dur, words = len(piece) / 16000, len(text.split())
    if (words == 0 and dur >= 0.6) or (dur >= 4.0 and words / dur < 0.6):
        again = _stt_decode(rec, ac.pad_edges(piece))
        if len(again.split()) > words:
            log.info("STT retry recovered words: %d -> %d in a %.1fs piece", words, len(again.split()), dur)
            _stats["recovered"] += 1
            return again
    return text


def _energy_segments(audio: np.ndarray):
    """Fallback voice detection (numpy only): one span from the first to the last loud moment. [] when there is no speech."""
    n = 320                                              # 20 ms frames
    m = len(audio) // n
    if m < 3:
        return []
    e = np.sqrt((audio[: m * n].reshape(m, n) ** 2).mean(axis=1))      # RMS per frame
    noise = float(np.percentile(e, 10))
    peak = float(np.percentile(e, 95))
    thr = max(noise * 3.0, peak * 0.08, 0.004)           # well above the background, and a share of the loud part
    voiced = np.flatnonzero(e > thr)
    if voiced.size < 4 or peak < 0.01:                   # < ~80 ms of sound: not speech
        return []
    pad = 12                                             # 240 ms of padding: never clip the first / last word
    a, b = max(0, voiced[0] - pad), min(m, voiced[-1] + 1 + pad)
    return [audio[a * n: b * n]]


def _neural_segments(audio: np.ndarray):
    """Silero VAD (a small neural network, ~2 MB, run through sherpa-onnx) finds where people really talk.
    Returns a list of speech pieces ([] = no speech at all), or None when the network is not available."""
    try:
        import sherpa_onnx as so
        def _make(net: str):
            cfg = so.VadModelConfig()
            sub = cfg.ten_vad if net == "ten" else cfg.silero_vad
            sub.model = os.path.join(STT_DIR, "ten-vad.onnx" if net == "ten" else "silero_vad.onnx")
            sub.threshold = VAD_THRESHOLD
            sub.min_silence_duration = 0.5               # pauses shorter than this stay inside one piece (sentences are not chopped)
            sub.min_speech_duration = 0.15
            sub.max_speech_duration = PIECE_MAX_S        # long speech is cut by the network at a pause
            cfg.sample_rate = 16000
            cfg.num_threads = 1
            cfg.provider = "cpu"
            return so.VoiceActivityDetector(cfg, buffer_size_in_seconds=MAX_AUDIO_SECONDS + 10)
        try:
            vad = _make(VAD_NET)
        except Exception as e:
            if VAD_NET == "silero":
                raise
            log.warning("VAD_NET=%s unavailable (%s), using silero", VAD_NET, e)
            vad = _make("silero")
        w = 512                                          # Silero window at 16 kHz
        for k in range(0, len(audio), w):
            vad.accept_waveform(audio[k:k + w])
        if hasattr(vad, "flush"):
            vad.flush()
        pad, pieces = 3200, []                           # 0.2 s of real audio on each side: never clip the first / last word
        while not vad.empty():
            seg = vad.front
            start, n = getattr(seg, "start", None), len(seg.samples)
            if start is None:
                pieces.append(np.asarray(seg.samples, dtype=np.float32))
            else:
                pieces.append(audio[max(0, start - pad): min(len(audio), start + n + pad)])
            vad.pop()
        return pieces
    except Exception as e:
        log.warning("neural VAD unavailable, using the energy rule: %s", e)
        return None


def _speech_pieces(audio: np.ndarray):
    """Pieces of the recording that contain speech. [] = only noise / silence."""
    if VAD_MODE == "neural":
        pieces = _neural_segments(audio)
        if pieces is not None:
            return pieces
    return _energy_segments(audio)


def _split_long(audio: np.ndarray, max_s: float = 0.0):
    """Cut long recordings into pieces of at most PIECE_MAX_S, always at the quietest moment (never mid-word)."""
    max_s = max_s or PIECE_MAX_S
    n, k = int(max_s * 16000), 1600
    out, start = [], 0
    while len(audio) - start > n:
        lo, hi = start + int(n * 0.7), start + n
        level = np.convolve(np.abs(audio[lo:hi]), np.ones(k) / k, mode="valid")
        cut = lo + int(np.argmin(level)) + k // 2
        out.append(audio[start:cut])
        start = cut
    out.append(audio[start:])
    return out


def _transcribe_blocking(data: bytes, prompt: str = ""):
    t0 = time.time()
    raw = _decode(data)
    qual = ac.describe(ac.stats(raw))                  # what the browser really sent (before our own leveling)
    audio = _prepare(raw)
    if audio.size < 1600:
        log.info("STT skipped: no speech in recording")
        return ""
    # No noise remover on purpose: published tests found that denoising networks make modern ASR (Parakeet, Whisper) WORSE.
    pieces = _speech_pieces(audio)
    if not pieces:
        log.info("STT skipped: only noise, no speech")
        return ""
    pieces = [ac.normalize_piece(sub, target_db=TARGET_DBFS) for piece in pieces for sub in _split_long(piece) if len(sub) > 1600]
    audio = np.concatenate(pieces) if pieces else audio[:0]    # only used for the length in the log line below
    if audio.size == 0:
        return ""
    with _stt["pool"].take() as rec:                   # waits here when all workers are busy
        text = " ".join(_decode_piece(rec, sub) for sub in pieces)
    text = re.sub(r"\s+", " ", text).strip()
    dt = time.time() - t0
    log.info("STT[%s] %.1fs audio in %.2fs (x%.1f) %s: %s", STT_ENGINE, len(audio) / 16000, dt, len(audio) / 16000 / max(dt, 1e-3), qual, text[:80])
    if not text:
        log.warning("STT[%s] heard NOTHING in %.1fs of audio (%s). Not English, too distorted, or only background / the assistant's own echo.", STT_ENGINE, len(audio) / 16000, qual)
    return text


@app.post("/v1/audio/transcriptions")
@app.post("/audio/transcriptions")
async def transcriptions(
    file: UploadFile = File(...),
    model: str = Form(""),                  # ignored: Parakeet TDT 0.6B v2 is always used
    language: str = Form(""),             # ignored: English only
    prompt: str = Form(""),               # ignored
    temperature: float = Form(0.0),
    response_format: str = Form("json"),
    _: None = Depends(auth),
):
    await _require_ready()
    data = await file.read()
    if not data:
        raise HTTPException(400, "Empty audio file.")
    if len(data) > MAX_AUDIO_MB * 1024 * 1024:
        raise HTTPException(413, f"Audio larger than {MAX_AUDIO_MB:g} MB.")
    try:
        text = await run_in_threadpool(_transcribe_blocking, data, prompt)
    except Busy:
        raise HTTPException(429, "Too many recordings waiting. Try again in a moment.")
    except HTTPException:
        raise
    except Exception as e:
        log.exception("transcription failed (%d bytes, type %s)", len(data), file.content_type)
        raise HTTPException(422, f"Could not read this audio: {e}")
    return PlainTextResponse(text) if response_format == "text" else {"text": text}


# ------------------------------------------------------------------ text -> speech
_DEV = re.compile(r"[\u0900-\u097F]")
_URL = re.compile(r"https?://\S+")
_MD = re.compile(r"[*_`#>~|]+")
_EMOJI = re.compile("[\U0001F000-\U0001FAFF\u2600-\u27BF\uFE0F\u200d]+")
_BRACKETS = re.compile(r"\[([^\]]*)\]\([^)]*\)")       # [label](url) -> label


def _clean(text: str) -> str:
    text = _BRACKETS.sub(r"\1", text)
    text = _URL.sub(" link ", text)
    text = _EMOJI.sub(" ", text)
    text = _MD.sub(" ", text)
    text = text.replace("&", " and ").replace("%", " percent ")
    return re.sub(r"\s+", " ", text).strip()


def _trim(audio: np.ndarray, sr: int, thr: float = 0.006, keep_ms: int = 60) -> np.ndarray:
    """Cut dead air at both ends of a rendered chunk."""
    loud = np.flatnonzero(np.abs(audio) > thr)
    if loud.size == 0:
        return audio
    pad = sr * keep_ms // 1000
    return audio[max(0, loud[0] - pad): loud[-1] + pad]


def _script_runs(text: str):
    """Split into [(chunk, is_hindi)] so Devanagari and Latin words each get the right phonemiser."""
    runs, cur, cur_hi = [], "", None
    for ch in text:
        if "\u0900" <= ch <= "\u097f":
            hi = True
        elif ch.isalpha():
            hi = False
        else:
            hi = cur_hi                      # punctuation / digits / spaces stay with the current run
        if cur_hi is None:
            cur_hi = hi if hi is not None else False
        elif hi != cur_hi and cur.strip():
            runs.append((cur, cur_hi)); cur = ""; cur_hi = hi
        cur += ch
    if cur.strip():
        runs.append((cur, bool(cur_hi)))
    return runs


def _pick_voice(requested: str, text: str) -> str:
    v = (requested or "").strip()
    letters = [c for c in text if c.isalpha()]
    hindi_text = bool(letters) and sum(1 for c in letters if _DEV.match(c)) / len(letters) > 0.3
    known = _tts["voices"]
    if v in known:
        male = len(v) > 1 and v[1] == "m"
        if hindi_text and v[0] != "h":      # English voice asked for Hindi text -> Hindi voice, same gender
            return HI_MALE if male else HI_FEMALE
        return v
    male = v.lower() in MALE_NAMES          # an OpenAI voice name from the extension's picker
    if hindi_text:
        return HI_MALE if male else HI_FEMALE
    return EN_MALE if male else EN_FEMALE


def _synthesize_blocking(text: str, voice: str, speed: float):
    t0 = time.time()
    parts, sr = [], 24000
    with _tts["pool"].take() as kokoro:
        for chunk, hi in _script_runs(text):
            if not re.search(r"\w", chunk):
                continue
            try:
                audio, sr = kokoro.create(chunk.strip(), voice=voice, speed=speed, lang="hi" if hi else "en-us")
            except Exception as e:           # one unreadable word must not kill the whole sentence
                log.warning("skipped chunk %r: %s", chunk[:30], e)
                continue
            # never let a broken render reach the speaker (the int8 Kokoro build can return NaN / huge values)
            if len(audio) < sr // 20 or not np.isfinite(audio).all() or float(np.abs(audio).max()) > 1.5:
                log.warning("bad audio for chunk %r (len=%d) - skipped", chunk[:30], len(audio))
                continue
            parts.append(_trim(audio, sr) if TRIM_TTS else audio)
    if not parts:
        raise SynthFailed("The voice model produced no usable audio for this text.")
    gap = np.zeros(sr * 70 // 1000, dtype=np.float32)          # short, even pause between chunks
    audio = np.concatenate([x for p in parts for x in (p, gap)][:-1]).astype(np.float32)
    peak = float(np.abs(audio).max())
    if 0 < peak < 0.9:                                         # even loudness from call to call
        audio = audio * min(0.92 / peak, 1.6)
    dt = time.time() - t0
    log.info("TTS %.1fs audio in %.2fs (x%.1f realtime) voice=%s", len(audio) / sr, dt, len(audio) / sr / max(dt, 1e-3), voice)
    return audio, sr


def _to_mp3(audio: np.ndarray, sr: int) -> bytes:
    enc = lameenc.Encoder()
    enc.set_bit_rate(MP3_KBPS); enc.set_in_sample_rate(sr); enc.set_channels(1); enc.set_quality(5)
    pcm = (np.clip(audio, -1, 1) * 32767).astype("<i2").tobytes()
    return bytes(enc.encode(pcm) + enc.flush())


def _to_wav(audio: np.ndarray, sr: int) -> bytes:
    b = io.BytesIO()
    with wave.open(b, "wb") as w:
        w.setnchannels(1), w.setsampwidth(2), w.setframerate(sr)
        w.writeframes((np.clip(audio, -1, 1) * 32767).astype("<i2").tobytes())
    return b.getvalue()


class SynthFailed(Exception):
    pass


class SpeechRequest(BaseModel):
    model: str | None = None            # ignored
    input: str
    voice: str | None = None
    response_format: str | None = "mp3"
    speed: float | None = 1.0
    instructions: str | None = None     # ignored (Kokoro has no style prompt)


@app.post("/v1/audio/speech")
@app.post("/audio/speech")
async def speech(req: SpeechRequest, _: None = Depends(auth)):
    await _require_ready()
    text = _clean(req.input)[:4000]
    if not text:
        raise HTTPException(400, "Empty input.")
    voice = _pick_voice(req.voice or "", text)
    speed = min(2.0, max(0.5, float(req.speed or 1.0)))
    fmt = "wav" if (req.response_format or "mp3").lower() == "wav" else "mp3"
    key = (text, voice, round(speed, 2), fmt)
    hit = _tts_cache.get(key)
    if hit is not None:
        _tts_cache.move_to_end(key)
        return Response(hit, media_type="audio/wav" if fmt == "wav" else "audio/mpeg")
    try:
        audio, sr = await run_in_threadpool(_synthesize_blocking, text, voice, speed)
    except SynthFailed as e:
        raise HTTPException(500, str(e))      # 5xx -> the extension retries / falls back to a browser voice
    body = await run_in_threadpool(_to_wav if fmt == "wav" else _to_mp3, audio, sr)
    if TTS_CACHE_MAX > 0 and len(body) < 2_000_000:
        _tts_cache[key] = body
        while len(_tts_cache) > TTS_CACHE_MAX:
            _tts_cache.popitem(last=False)
    return Response(body, media_type="audio/wav" if fmt == "wav" else "audio/mpeg")


# ------------------------------------------------------------------ misc
@app.get("/v1/voices")
async def voices(_: None = Depends(auth)):
    await _require_ready()
    return {"voices": sorted(_tts["voices"])}


@app.get("/")
@app.get("/health")
async def health():
    return {"ok": _ready.is_set(), "loading": not _ready.is_set() and not _boot["error"], "error": _boot["error"],
            "seconds_since_start": round(time.time() - _boot["t0"], 1), "stt": STT_ENGINE,
            "tts": "kokoro-82m", "auth": bool(API_KEY), "build": "voice-v15-parakeet-v2-leveler", "engine": STT_ENGINE, "lang": "en", "vad": VAD_MODE,
            "stt_recovered": _stats["recovered"], "target_dbfs": TARGET_DBFS if LEVELER else None, "stt_workers": STT_WORKERS, "stt_busy": _stt["pool"].active if _stt["pool"] else 0, "stt_waiting": _stt["pool"].waiting if _stt["pool"] else 0,
            "tts_workers": TTS_WORKERS}
