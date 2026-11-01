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
