"""
CPU voice server for Cloud Run  -  OpenAI-compatible speech API.

    POST /v1/audio/transcriptions   speech -> text   (faster-whisper, int8, CPU)
    POST /v1/audio/speech           text -> speech   (Kokoro-82M ONNX, CPU, English + Hindi)
    GET  /health                    warm-up / status

The AI Chat Firefox extension already speaks this API: in Settings > Voice set the
speech-to-text and text-to-speech "base URL" to  https://<your-service>.run.app/v1
and the "key" to the API_KEY you chose.
"""
import hmac
import io
import logging
import os
import re
import threading
import time
import wave
from contextlib import asynccontextmanager

import lameenc
import numpy as np
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
WHISPER_DIR = os.getenv("WHISPER_DIR", "/models/whisper")     # baked into the image (see Dockerfile)
WHISPER_THREADS = int(os.getenv("WHISPER_THREADS", "2"))      # = number of vCPUs
WHISPER_BEAM = int(os.getenv("WHISPER_BEAM", "1"))            # 1 = greedy = fastest. 2-5 = a little better, slower
WHISPER_VAD = os.getenv("WHISPER_VAD", "1") == "1"            # drop silence: fewer "Thank you." hallucinations
KOKORO_MODEL = os.getenv("KOKORO_MODEL", "/models/kokoro.onnx")     # fp32 build. Do NOT use the int8 file: it outputs garbage
KOKORO_VOICES = os.getenv("KOKORO_VOICES", "/models/voices.bin")
KOKORO_THREADS = int(os.getenv("KOKORO_THREADS", "2"))
MP3_KBPS = int(os.getenv("MP3_KBPS", "48"))                   # small = cheap egress. 64 for a bit more quality
MAX_AUDIO_MB = float(os.getenv("MAX_AUDIO_MB", "25"))

# Voices. Any Kokoro voice id (af_heart, am_michael, hf_alpha, hm_omega ...) is accepted as-is.
# OpenAI names that the extension's picker uses (marin, cedar ...) are mapped by gender.
EN_FEMALE, EN_MALE = os.getenv("VOICE_EN_FEMALE", "af_heart"), os.getenv("VOICE_EN_MALE", "am_michael")
HI_FEMALE, HI_MALE = os.getenv("VOICE_HI_FEMALE", "hf_alpha"), os.getenv("VOICE_HI_MALE", "hm_omega")
MALE_NAMES = {"cedar", "ash", "echo", "ballad", "verse", "onyx", "fable"}

_stt = {"model": None}
_tts = {"model": None, "voices": set()}
_stt_lock, _tts_lock = threading.Lock(), threading.Lock()     # one STT and one TTS job at a time -> predictable speed


# ------------------------------------------------------------------ model loading
def load_models():
    from faster_whisper import WhisperModel
    t = time.time()
    _stt["model"] = WhisperModel(WHISPER_DIR, device="cpu", compute_type="int8", cpu_threads=WHISPER_THREADS, num_workers=1)
    log.info("whisper loaded in %.1fs (%s)", time.time() - t, WHISPER_DIR)

    import onnxruntime as ort
    from kokoro_onnx import Kokoro
    t = time.time()
    so = ort.SessionOptions()
    so.intra_op_num_threads = KOKORO_THREADS
    so.inter_op_num_threads = 1
    sess = ort.InferenceSession(KOKORO_MODEL, sess_options=so, providers=["CPUExecutionProvider"])
    _tts["model"] = Kokoro.from_session(sess, KOKORO_VOICES)
    _tts["voices"] = set(_tts["model"].get_voices())
    log.info("kokoro loaded in %.1fs (%d voices)", time.time() - t, len(_tts["voices"]))


def warm_up():
    """One tiny job through each engine so the first real request is not slow."""
    try:
        _transcribe_blocking(_silence_wav(), "en", "")
        _synthesize_blocking("Hello.", EN_FEMALE, 1.0)
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
    # Start listening at once (Cloud Run's startup check passes), load the models in the background.
    # Requests that arrive while loading wait up to 90 s; a loading failure is written to the logs and shown on /health.
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


def _lang_code(language: str):
    """'en-IN' -> 'en', 'hi-IN' -> 'hi', '' / 'hinglish' / unknown -> None (auto-detect)."""
    code = (language or "").strip().lower().replace("_", "-").split("-")[0]
    if code in ("hinglish", "auto", ""):
        return "hi" if code == "hinglish" else None
    return code if code in _stt["model"].supported_languages else None


def _transcribe_blocking(data: bytes, language: str, prompt: str):
    t0 = time.time()
    with _stt_lock:
        segs, info = _stt["model"].transcribe(
            io.BytesIO(data),
            language=_lang_code(language),
            beam_size=WHISPER_BEAM, best_of=1,
            temperature=0.0,
            condition_on_previous_text=False,       # short utterances: no need, and it stops repetition loops
            initial_prompt=(prompt or None),
            vad_filter=WHISPER_VAD,
            vad_parameters={"min_silence_duration_ms": 400},
            no_speech_threshold=0.6,
        )
        text = " ".join(s.text.strip() for s in segs).strip()
    dt = time.time() - t0
    log.info("STT %.1fs audio in %.2fs (x%.1f realtime) lang=%s: %s", info.duration, dt, info.duration / max(dt, 1e-3), info.language, text[:80])
    return text


@app.post("/v1/audio/transcriptions")
@app.post("/audio/transcriptions")
async def transcriptions(
    file: UploadFile = File(...),
    model: str = Form(""),                  # ignored: the model baked into the image is always used
    language: str = Form(""),
    prompt: str = Form(""),
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
        text = await run_in_threadpool(_transcribe_blocking, data, language, prompt)
    except HTTPException:
        raise
    except Exception as e:
        log.exception("transcription failed")
        raise HTTPException(422, f"Could not read this audio: {e}")
    return PlainTextResponse(text) if response_format == "text" else {"text": text}


# ------------------------------------------------------------------ text -> speech
_DEV = re.compile(r"[\u0900-\u097F]")
_URL = re.compile(r"https?://\S+")
_MD = re.compile(r"[*_`#>~|]+")


def _clean(text: str) -> str:
    text = _URL.sub(" link ", text)
    text = _MD.sub(" ", text)
    return re.sub(r"\s+", " ", text).strip()


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
    with _tts_lock:
        for chunk, hi in _script_runs(text):
            if not re.search(r"\w", chunk):
                continue
            try:
                audio, sr = _tts["model"].create(chunk.strip(), voice=voice, speed=speed, lang="hi" if hi else "en-us")
            except Exception as e:           # one unreadable word must not kill the whole sentence
                log.warning("skipped chunk %r: %s", chunk[:30], e)
                continue
            # never let a broken render reach the speaker (the int8 Kokoro build can return NaN / huge values)
            if len(audio) < sr // 20 or not np.isfinite(audio).all() or float(np.abs(audio).max()) > 1.5:
                log.warning("bad audio for chunk %r (len=%d) - skipped", chunk[:30], len(audio))
                continue
            parts.append(audio)
    if not parts:
        raise SynthFailed("The voice model produced no usable audio for this text.")
    audio = np.concatenate(parts).astype(np.float32)
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
    try:
        audio, sr = await run_in_threadpool(_synthesize_blocking, text, voice, speed)
    except SynthFailed as e:
        raise HTTPException(500, str(e))      # 5xx -> the extension retries / falls back to a browser voice
    if (req.response_format or "mp3").lower() == "wav":
        return Response(_to_wav(audio, sr), media_type="audio/wav")
    return Response(await run_in_threadpool(_to_mp3, audio, sr), media_type="audio/mpeg")


# ------------------------------------------------------------------ misc
@app.get("/v1/voices")
async def voices(_: None = Depends(auth)):
    await _require_ready()
    return {"voices": sorted(_tts["voices"])}


@app.get("/")
@app.get("/health")
async def health():
    return {"ok": _ready.is_set(), "loading": not _ready.is_set() and not _boot["error"], "error": _boot["error"],
            "seconds_since_start": round(time.time() - _boot["t0"], 1), "stt": os.path.basename(WHISPER_DIR.rstrip("/")),
            "tts": "kokoro-82m", "auth": bool(API_KEY)}
