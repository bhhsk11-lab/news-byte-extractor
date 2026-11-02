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
from collections import OrderedDict
from contextlib import asynccontextmanager

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
STT_ENGINE = os.getenv("STT_ENGINE", "parakeet").strip().lower()   # parakeet (fast + accurate) | moonshine-base | moonshine-tiny (fastest) | whisper
STT_DIR = os.getenv("STT_DIR", "/models/stt")                  # used by parakeet / moonshine
WHISPER_DIR = os.getenv("WHISPER_DIR", "/models/whisper")     # baked into the image (see Dockerfile)
WHISPER_THREADS = int(os.getenv("WHISPER_THREADS", "2"))      # = number of vCPUs
WHISPER_BEAM = int(os.getenv("WHISPER_BEAM", "2"))            # 1 = greedy = fastest. 2 = good balance. 3-5 = best, slower
WHISPER_VAD = os.getenv("WHISPER_VAD", "1") == "1"            # drop silence: fewer "Thank you." hallucinations
ALLOWED_LANGS = [x.strip() for x in re.split(r"[,;| ]+", os.getenv("ALLOWED_LANGS", "en")) if x.strip()]  # auto-detect picks ONLY from these
WHISPER_HOTWORDS = " ".join(re.split(r"[,;|]+", os.getenv("WHISPER_HOTWORDS", ""))).strip()  # optional: names / terms you say often, separated by spaces
DENOISE = os.getenv("DENOISE", "auto").strip().lower()        # auto = clean only when the room is noisy, with a safety re-check | on | off
SNR_CLEAN_DB = float(os.getenv("SNR_CLEAN_DB", "18"))          # recordings at or above this speech-to-noise ratio are never gated
RETRY_LOGPROB = float(os.getenv("RETRY_LOGPROB", "-0.50"))     # below this confidence, the other audio version is tried too
MAX_AUDIO_SECONDS = float(os.getenv("MAX_AUDIO_SECONDS", "120"))  # protects the 2 GiB memory limit
TRIM_TTS = os.getenv("TRIM_TTS", "1") == "1"                  # cut dead air at the start/end of speech -> denser, snappier audio
TTS_CACHE_MAX = int(os.getenv("TTS_CACHE_MAX", "64"))         # repeated sentences are served instantly
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
_tts_cache: "OrderedDict" = OrderedDict()
_stt_lock, _tts_lock = threading.Lock(), threading.Lock()     # one STT and one TTS job at a time -> predictable speed


# ------------------------------------------------------------------ model loading
def _load_sherpa():
    """Parakeet / Moonshine through sherpa-onnx (int8 ONNX, CPU)."""
    import sherpa_onnx as so
    d = STT_DIR
    if STT_ENGINE == "parakeet":
        return so.OfflineRecognizer.from_transducer(
            encoder=f"{d}/encoder.int8.onnx", decoder=f"{d}/decoder.int8.onnx", joiner=f"{d}/joiner.int8.onnx",
            tokens=f"{d}/tokens.txt", num_threads=WHISPER_THREADS, sample_rate=16000, feature_dim=80,
            decoding_method="greedy_search", model_type="nemo_transducer", provider="cpu")
    if STT_ENGINE.startswith("moonshine"):
        return so.OfflineRecognizer.from_moonshine(
            preprocessor=f"{d}/preprocess.onnx", encoder=f"{d}/encode.int8.onnx",
            uncached_decoder=f"{d}/uncached_decode.int8.onnx", cached_decoder=f"{d}/cached_decode.int8.onnx",
            tokens=f"{d}/tokens.txt", num_threads=WHISPER_THREADS, provider="cpu")
    raise RuntimeError(f"unknown STT_ENGINE '{STT_ENGINE}' (use parakeet, moonshine-base, moonshine-tiny or whisper)")


def load_models():
    t = time.time()
    if STT_ENGINE == "whisper":
        from faster_whisper import WhisperModel
        _stt["model"] = WhisperModel(WHISPER_DIR, device="cpu", compute_type="int8", cpu_threads=WHISPER_THREADS, num_workers=1)
        log.info("whisper loaded in %.1fs (%s)", time.time() - t, WHISPER_DIR)
    else:
        _stt["model"] = _load_sherpa()
        log.info("%s loaded in %.1fs (%s)", STT_ENGINE, time.time() - t, STT_DIR)

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
        rng = np.random.default_rng(0)
        probe = (0.05 * rng.standard_normal(16000 * 2)).astype(np.float32)   # not silence, so the encoder really runs
        with _stt_lock:
            if STT_ENGINE == "whisper":
                list(_stt["model"].transcribe(probe, language="en", beam_size=WHISPER_BEAM, without_timestamps=True)[0])
            else:
                _sherpa_decode(probe)
        ac.clean(probe)                                                       # warms numpy FFT paths too
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


def _lang_code(language: str):
    """'en-IN' -> 'en', 'hi-IN' -> 'hi', 'hinglish' -> 'hi', '' / 'auto' / unknown -> None (we detect it ourselves)."""
    if ALLOWED_LANGS == ["en"]:                         # English-only mode: never detect, never guess
        return "en"
    code = (language or "").strip().lower().replace("_", "-").split("-")[0]
    if code == "hinglish":
        return "hi"
    if code in ("auto", ""):
        return None
    return code if code in _stt["model"].supported_languages else None


def _prepare(audio: np.ndarray) -> np.ndarray:
    """Clean the recording before Whisper sees it: remove DC offset and lift quiet microphones to a healthy level.
    Quiet / low-gain audio is a top cause of wrong words. Returns an empty array when there is no real sound."""
    if audio.size == 0:
        return audio
    audio = audio - float(np.mean(audio))
    peak = float(np.abs(audio).max())
    if peak < 0.004:                                   # digital silence / pure noise floor
        return audio[:0]
    gain = min(0.85 / peak, 12.0)                      # peak-normalise, never boost more than 12x
    if gain > 1.15:
        audio = audio * gain
    return np.clip(audio, -1.0, 1.0).astype(np.float32)


def _detect_language(audio: np.ndarray):
    """Whisper's own language guess is often wrong for Indian speech (Hindi heard as Urdu, English heard as Welsh...).
    We look at its full probability table but only accept a language from ALLOWED_LANGS (default en, hi)."""
    try:
        _, _, probs = _stt["model"].detect_language(audio, vad_filter=WHISPER_VAD, language_detection_segments=2)
        table = dict(probs)
        pool = {k: table.get(k, 0.0) for k in ALLOWED_LANGS}
        if pool:
            best = max(pool, key=pool.get)
            log.info("lang detect: %s", ", ".join(f"{k}={v:.2f}" for k, v in sorted(pool.items(), key=lambda kv: -kv[1])))
            return best
    except Exception as e:
        log.warning("language detection skipped: %s", e)
    return None


# Phrases Whisper invents out of silence / noise. Dropped only when the segment is also low-confidence.
_HALLUCINATIONS = {
    "thank you", "thanks", "thank you.", "thanks for watching", "thank you for watching", "bye", "bye.", "you",
    "subtitles by the amara.org community", "please subscribe", "like and subscribe",
    "धन्यवाद", "शुक्रिया", "थैंक यू", "देखने के लिए धन्यवाद", "सब्सक्राइब करें",
}


def _is_hallucination(seg) -> bool:
    t = re.sub(r"[\s.!,?।]+$", "", seg.text.strip().lower())
    weak = seg.no_speech_prob > 0.35 or seg.avg_logprob < -0.9
    return weak and t in _HALLUCINATIONS


def _sherpa_decode(audio: np.ndarray) -> str:
    rec = _stt["model"]
    stream = rec.create_stream()
    stream.accept_waveform(16000, audio)
    rec.decode_stream(stream)
    return stream.result.text.strip()


def _speech_only(audio: np.ndarray):
    """Silero voice detection (already installed with faster-whisper): trims the silence around the speech and returns
    None when the recording holds no speech at all, so noise alone never turns into invented words."""
    from faster_whisper.vad import VadOptions, get_speech_timestamps
    spans = get_speech_timestamps(audio, VadOptions(threshold=0.45, min_silence_duration_ms=350,
                                                    speech_pad_ms=250, min_speech_duration_ms=120))
    if not spans:
        return None
    return audio[max(0, spans[0]["start"] - 4000): spans[-1]["end"] + 4000]


def _split_long(audio: np.ndarray, max_s: float = 20.0):
    """Cut long recordings into pieces of at most ~20 s, always at the quietest moment (never mid-word)."""
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


def _transcribe_sherpa(data: bytes):
    t0 = time.time()
    raw = _decode(data)
    qual = ac.describe(ac.stats(raw))                  # what the browser really sent (before our own level boost)
    audio = _prepare(raw)
    if audio.size < 1600:
        log.info("STT skipped: no speech in recording")
        return ""
    # No noise gate here on purpose: in our tests it made Parakeet and Moonshine slightly WORSE, not better.
    audio = _speech_only(audio)
    if audio is None:
        log.info("STT skipped: only noise, no speech")
        return ""
    with _stt_lock:
        text = " ".join(_sherpa_decode(piece) for piece in _split_long(audio))
    text = re.sub(r"\s+", " ", text).strip()
    dt = time.time() - t0
    log.info("STT[%s] %.1fs audio in %.2fs (x%.1f) %s: %s", STT_ENGINE, len(audio) / 16000, dt, len(audio) / 16000 / max(dt, 1e-3), qual, text[:80])
    if not text:
        log.warning("STT[%s] heard NOTHING in %.1fs of audio (%s). Not English, too distorted, or only background / the assistant's own echo.", STT_ENGINE, len(audio) / 16000, qual)
    return text


def _run_whisper(audio: np.ndarray, lang, prompt: str):
    """One Whisper pass. Returns (text, confidence, info). Confidence = average token log-probability (closer to 0 = surer)."""
    segs, info = _stt["model"].transcribe(
        audio,
        language=lang,
        task="transcribe",
        beam_size=WHISPER_BEAM, best_of=3,            # best_of only matters in the temperature fallback below
        temperature=(0.0, 0.2, 0.4),                  # retry a failed / garbled window at a higher temperature
        condition_on_previous_text=False,             # short utterances: stops repetition loops
        initial_prompt=(prompt or None),
        hotwords=(WHISPER_HOTWORDS or None),
        vad_filter=WHISPER_VAD,
        vad_parameters={"threshold": 0.45, "min_silence_duration_ms": 350,
                        "speech_pad_ms": 250, "min_speech_duration_ms": 120},   # pad = never clip the first / last word
        without_timestamps=True,                      # faster decoding, we only need text
        no_speech_threshold=0.6,
        log_prob_threshold=-1.0,
        compression_ratio_threshold=2.2,              # catches "the the the the" loops
        repetition_penalty=1.05,
        no_repeat_ngram_size=0,
        suppress_blank=True,
    )
    texts, lp, wt = [], 0.0, 0
    for sg in segs:
        t = sg.text.strip()
        if not t or _is_hallucination(sg):
            continue
        texts.append(t)
        w = max(1, len(sg.tokens or ()))
        lp += sg.avg_logprob * w
        wt += w
    text = re.sub(r"\s+", " ", " ".join(texts)).strip()
    return text, (lp / wt if wt else -9.0), info


def _audio_versions(audio: np.ndarray):
    """The recording in the order we should try it: [(name, audio), ...]. See README 'Noise' for the reasoning."""
    snr = ac.estimate_snr_db(audio)
    plain = ac.clean(audio, gate=False)                       # high-pass only: safe for every recording
    if DENOISE == "off":
        return snr, [("plain", plain)]
    gated = ac.clean(audio, gate=True)                        # + soft gate for steady background noise
    if DENOISE == "on":
        return snr, [("gated", gated), ("plain", plain)]
    return snr, ([("plain", plain), ("gated", gated)] if snr >= SNR_CLEAN_DB else [("gated", gated), ("plain", plain)])


def _transcribe_blocking(data: bytes, language: str, prompt: str):
    if STT_ENGINE != "whisper":
        return _transcribe_sherpa(data)           # Parakeet / Moonshine: English only, no language or prompt needed
    t0 = time.time()
    audio = _prepare(_decode(data))
    if audio.size < 1600:                              # < 0.1 s or silence: nothing to transcribe
        log.info("STT skipped: no speech in recording")
        return ""
    lang = _lang_code(language)
    with _stt_lock:
        if lang is None:
            lang = _detect_language(audio)
        snr, versions = _audio_versions(audio)
        name, ver = versions[0]
        text, conf, info = _run_whisper(ver, lang, prompt)
        used = name
        # Not sure about the result (or nothing heard)? Listen once more to the other version of the same recording and
        # keep whichever Whisper is more confident about. Costs time only on hard recordings; clear speech never pays it.
        if len(versions) > 1 and (conf < RETRY_LOGPROB or not text) and audio.size < 20 * 16000:
            name2, ver2 = versions[1]
            text2, conf2, _ = _run_whisper(ver2, lang, prompt)
            if text2 and (not text or conf2 > conf + 0.05):
                text, conf, used = text2, conf2, name2
    dt = time.time() - t0
    log.info("STT %.1fs audio in %.2fs (x%.1f) lang=%s %s used=%s conf=%.2f: %s",
             info.duration, dt, info.duration / max(dt, 1e-3), lang or info.language, ac.describe(ac.stats(audio)), used, conf, text[:80])
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
            "seconds_since_start": round(time.time() - _boot["t0"], 1), "stt": (os.path.basename(WHISPER_DIR.rstrip("/")) if STT_ENGINE == "whisper" else STT_ENGINE),
            "tts": "kokoro-82m", "auth": bool(API_KEY), "build": "voice-v6-fast-stats", "engine": STT_ENGINE, "beam": WHISPER_BEAM, "langs": ALLOWED_LANGS}
