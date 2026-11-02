"""
Light, dependency-free audio clean-up for speech recognition (numpy only, ~30 MB peak for 2 minutes of audio).

Why so gentle?  Research and our own tests agree that heavy neural denoisers (RNNoise, DeepFilterNet ...) often make
Whisper LESS accurate, because the artefacts they leave behind confuse it. So this module only does what is safe:

  * remove rumble below ~70 Hz (fans, table knocks, wind, mains hum fundamental)
  * soft spectral gate that learns the *steady* background (fan, AC, hiss, hum) from the quietest part of the clip
    and turns it down by at most `max_cut_db` (never to zero -> no "underwater" artefacts)
  * estimate how noisy a clip is, so the caller can decide whether cleaning is worth it
"""
import numpy as np

SR = 16000
N_FFT = 512          # 32 ms window
HOP = 128            # 8 ms hop (75 % overlap -> smooth, artefact-free overlap-add)
_WIN = np.hanning(N_FFT + 1)[:-1].astype(np.float32)   # periodic Hann


def _stft(x: np.ndarray) -> np.ndarray:
    pad = N_FFT // 2
    xp = np.pad(x, (pad, pad + (-(len(x) + 2 * pad - N_FFT) % HOP)), mode="reflect")
    frames = np.lib.stride_tricks.sliding_window_view(xp, N_FFT)[::HOP]
    return np.fft.rfft(frames * _WIN, axis=1).astype(np.complex64)


def _istft(spec: np.ndarray, length: int) -> np.ndarray:
    frames = np.fft.irfft(spec, n=N_FFT, axis=1).astype(np.float32) * _WIN
    n = (len(frames) - 1) * HOP + N_FFT
    out = np.zeros(n, np.float32)
    norm = np.zeros(n, np.float32)
    w2 = _WIN * _WIN
    for i in range(len(frames)):                       # 8 ms hop -> ~125 iterations per second of audio, cheap
        s = i * HOP
        out[s:s + N_FFT] += frames[i]
        norm[s:s + N_FFT] += w2
    out /= np.maximum(norm, 1e-6)
    return out[N_FFT // 2: N_FFT // 2 + length]


def _highpass_curve(sr: int = SR, f_lo: float = 40.0, f_hi: float = 80.0) -> np.ndarray:
    f = np.fft.rfftfreq(N_FFT, 1.0 / sr)
    return np.clip((f - f_lo) / (f_hi - f_lo), 0.0, 1.0) ** 2


def estimate_snr_db(x: np.ndarray, sr: int = SR) -> float:
    """Rough speech-to-background ratio in dB from 20 ms frame energies. ~30+ = studio quiet, <12 = noisy room."""
    n = int(0.02 * sr)
    m = len(x) // n
    if m < 10:
        return 30.0
    p = (x[: m * n].reshape(m, n) ** 2).mean(axis=1) + 1e-12
    p.sort()
    noise = p[: max(2, m // 10)].mean()                 # quietest 10 % of frames
    speech = p[-max(2, m * 3 // 10):].mean()            # loudest 30 % of frames
    return float(10 * np.log10(speech / noise))


def clean(x: np.ndarray, sr: int = SR, gate: bool = True, max_cut_db: float = 10.0, strength: float = 1.0) -> np.ndarray:
    """High-pass (+ optional soft spectral gate). Returns float32 audio of the same length."""
    if x.size < N_FFT * 4:
        return x
    S = _stft(x)
    S *= _highpass_curve(sr)[None, :]
    if gate:
        P = (S.real ** 2 + S.imag ** 2)
        frame_e = P.sum(axis=1)
        k = max(3, len(frame_e) // 6)                       # quietest ~17 % of frames = the background
        quiet = np.argpartition(frame_e, k)[:k]
        N = P[quiet].mean(axis=0) + 1e-12                   # steady background power per frequency bin
        floor = 10 ** (-max_cut_db / 20)
        G = np.sqrt(np.maximum(P - strength * N[None, :], 0.0) / (P + 1e-12))
        G = np.maximum(G, floor)
        # smooth over time (3 frames) and frequency (3 bins): kills "musical noise" artefacts
        G = (G + np.roll(G, 1, axis=0) + np.roll(G, -1, axis=0)) / 3
        G = (G + np.roll(G, 1, axis=1) + np.roll(G, -1, axis=1)) / 3
        S *= G.astype(np.float32)
    y = _istft(S, len(x))
    return np.clip(y, -1.0, 1.0).astype(np.float32)


def stats(x: np.ndarray, sr: int = SR) -> dict:
    """What the recording looks like when it reaches the recognizer (logged for every request, so wrong words can be traced to the audio).
    peak_db / rms_db are dB below full scale; snr_db = loud 30 % of the frames vs the quietest 10 %; clip_pct = share of samples at the limit."""
    if x.size == 0:
        return {"dur": 0.0, "peak_db": -99.0, "rms_db": -99.0, "snr_db": 0.0, "clip_pct": 0.0}
    x = x - float(np.mean(x))
    peak = float(np.abs(x).max()) + 1e-9
    n = int(0.02 * sr)
    m = x.size // n
    if m >= 5:
        p = np.sort((x[: m * n].reshape(m, n) ** 2).mean(axis=1))
        loud = p[-max(1, m * 3 // 10):].mean()
    else:
        loud = float((x ** 2).mean())
    return {"dur": x.size / sr, "peak_db": 20 * float(np.log10(peak)), "rms_db": 10 * float(np.log10(loud + 1e-12)),
            "snr_db": estimate_snr_db(x, sr), "clip_pct": 100.0 * float((np.abs(x) > 0.98).mean())}


def describe(st: dict) -> str:
    """One short line for the log + a plain warning when the recording itself is the likely problem."""
    tips = []
    if st["peak_db"] < -30 or st["rms_db"] < -45:
        tips.append("VERY QUIET (raise the microphone level / move closer)")
    if st["clip_pct"] > 0.5:
        tips.append("CLIPPED (microphone too loud)")
    if st["snr_db"] < 12:
        tips.append("NOISY (voice is not much louder than the background)")
    return "peak=%.0fdB voice=%.0fdB snr=%.0fdB clip=%.1f%%%s" % (st["peak_db"], st["rms_db"], st["snr_db"], st["clip_pct"], (" <-- " + ", ".join(tips)) if tips else "")


# ------------------------------------------------------------------ level control (v14)
# Why: the speech model itself is fairly level-independent (its features are normalised per recording), but everything IN FRONT of it
# (silence gate, voice detector, "loud enough" rules) is not. A sentence spoken 30-40 dB quieter than the one before it is the one that
# gets skipped. So the level is evened out BEFORE the voice detector, and every speech piece is set to the same level before decoding.
# "50 dB" cannot be an absolute value for a file (sound pressure depends on the microphone and speaker), so the target is a fixed
# digital level: the loud frames of each piece are brought to `target_db` dB below full scale (default -20 dBFS, a normal speech level).

def _frame_db(x: np.ndarray, n: int) -> np.ndarray:
    m = len(x) // n
    fr = x[: m * n].reshape(m, n)
    return 20 * np.log10(np.sqrt((fr ** 2).mean(axis=1)) + 1e-9)


def soft_limit(y: np.ndarray, knee: float = 0.6, ceil: float = 0.95) -> np.ndarray:
    """Transparent below `knee`, smoothly squeezes peaks above it (no hard clipping)."""
    a = np.abs(y)
    over = a > knee
    if over.any():
        span = ceil - knee
        y = y.copy()
        y[over] = np.sign(y[over]) * (knee + span * np.tanh((a[over] - knee) / span))
    return y.astype(np.float32)


def level(x: np.ndarray, sr: int = SR, target_db: float = -20.0, max_gain_db: float = 40.0, win_s: float = 0.8) -> np.ndarray:
    """Slow automatic gain control: lifts quiet passages (up to +max_gain_db) and lowers very loud ones (down to -12 dB), following
    the speech level of the last/next ~0.4 s. The gain moves smoothly (about 0.2 s), so there is no pumping and no clicks.
    Frames that are only background (within 8 dB of the quietest 10 % of the clip) never set the gain, so a silent room is not boosted
    on its own. Returns float32 of the same length."""
    n = int(0.02 * sr)                                   # 20 ms frames
    if x.size < n * 10:
        return x
    db = _frame_db(x, n)
    m = len(db)
    noise_db = float(np.percentile(db, 10))
    voiced = db > max(noise_db + 8.0, -75.0)
    if int(voiced.sum()) < 3:
        return x
    k = 2 * max(1, int(win_s / 0.04))                    # even number of frames (~0.8 s)
    d = np.where(voiced, db, -200.0)
    padded = np.pad(d, (k // 2, k // 2), constant_values=-200.0)
    local = np.lib.stride_tricks.sliding_window_view(padded, k + 1).max(axis=1)[:m]
    ok = local > -150.0
    gain = np.clip(target_db + 4.0 - local, -12.0, max_gain_db)      # +4: loudest frame sits ~4 dB above the average speech level
    idx = np.arange(m)
    gain = np.interp(idx, idx[ok], gain[ok])             # frames far from any speech take the gain of the nearest speech
    gain = np.convolve(np.pad(gain, (5, 5), mode="edge"), np.ones(11) / 11, mode="valid")    # ~0.2 s smoothing
    centers = idx * n + n // 2
    g = np.interp(np.arange(len(x)), centers, gain)
    return soft_limit(x * (10 ** (g / 20)).astype(np.float32))


def normalize_piece(x: np.ndarray, sr: int = SR, target_db: float = -20.0, max_gain_db: float = 30.0) -> np.ndarray:
    """Final fixed level for ONE speech piece: its loud frames (the speech, not the pauses) are brought to `target_db` dBFS."""
    n = int(0.02 * sr)
    if x.size < n * 5:
        return x
    db = _frame_db(x, n)
    speech = db[db > db.max() - 20.0]                    # frames within 20 dB of the loudest = the voice
    cur = float(10 * np.log10(np.mean(10 ** (speech / 10)) + 1e-12))     # power-mean level of those frames
    gain = float(np.clip(target_db - cur, -20.0, max_gain_db))
    return soft_limit(x * np.float32(10 ** (gain / 20)))


def pad_edges(x: np.ndarray, sec: float = 0.5, sr: int = SR, seed: int = 1) -> np.ndarray:
    """Put `sec` seconds of faint background-like noise before and after a piece. Used only for a second attempt when the first
    decode of a piece looks like it lost words: Parakeet-type decoders are known to drop speech when a window starts or ends right
    on the voice with no quiet context (reported in several open-source projects, Oct-Nov 2026)."""
    n = int(0.02 * sr)
    noise_db = float(np.clip(np.percentile(_frame_db(x, n), 10), -60.0, -40.0)) if x.size >= n * 5 else -55.0
    rng = np.random.default_rng(seed)
    pad = (rng.standard_normal(int(sec * sr)) * 10 ** (noise_db / 20)).astype(np.float32)
    return np.concatenate([pad, x, pad[::-1]]).astype(np.float32)
