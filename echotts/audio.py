"""Audio basics: load a WAV, turn it into a log-mel spectrogram, and back.

Analogy: a waveform is a long list of air-pressure readings. The STFT slices
it into short overlapping windows (like frames of a film) and asks "which
pitches are in this slice?". The mel scale then squeezes those pitches the
way human ears do: fine detail in the low notes, coarse in the high notes.

Both the acoustic model and the vocoder import mel_spectrogram from HERE,
so they always see identical mels.
"""
from functools import lru_cache
from math import gcd
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import torch
from scipy.signal import resample_poly


# ---------------------------------------------------------------- loading ---

def load_wav(path: str | Path, audio_cfg: dict[str, Any]) -> torch.Tensor:
    """Load a WAV file as a mono float tensor at the config sample rate.

    Returns shape (samples,) with values in [-1, 1].
    """
    data, sr = sf.read(str(path), dtype="float32", always_2d=True)  # (samples, channels)
    data = data.mean(axis=1)  # mix channels down to mono -> (samples,)

    target_sr = audio_cfg["sample_rate"]
    if sr != target_sr:
        # Polyphase resampling by an exact integer ratio up/down.
        g = gcd(sr, target_sr)
        data = resample_poly(data, target_sr // g, sr // g).astype(np.float32)

    peak = np.abs(data).max()
    if peak > 0:
        data = data / peak * 0.95  # normalise loudness, leave a little headroom
    return torch.from_numpy(np.ascontiguousarray(data))


def save_wav(path: str | Path, wav: torch.Tensor, sample_rate: int) -> None:
    """Save a (samples,) float tensor as a 16-bit PCM WAV."""
    audio = wav.detach().cpu().float().clamp(-1.0, 1.0).numpy()
    sf.write(str(path), audio, sample_rate, subtype="PCM_16")


# ------------------------------------------------------------ mel filters ---

def hz_to_mel(hz: np.ndarray) -> np.ndarray:
    """HTK mel formula: equal steps in mel sound like equal steps in pitch."""
    return 2595.0 * np.log10(1.0 + hz / 700.0)


def mel_to_hz(mel: np.ndarray) -> np.ndarray:
    """Inverse of hz_to_mel."""
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)


@lru_cache(maxsize=8)
def _mel_filterbank_np(sr: int, n_fft: int, n_mels: int, f_min: float, f_max: float) -> np.ndarray:
    """Build triangular mel filters. Returns (n_mels, n_fft // 2 + 1)."""
    n_freqs = n_fft // 2 + 1
    fft_freqs = np.linspace(0, sr / 2, n_freqs)  # centre frequency of each STFT bin

    # n_mels triangles need n_mels + 2 edge points, evenly spaced on the mel scale.
    mel_points = np.linspace(hz_to_mel(np.array(f_min)), hz_to_mel(np.array(f_max)), n_mels + 2)
    hz_points = mel_to_hz(mel_points)

    filters = np.zeros((n_mels, n_freqs), dtype=np.float32)
    for m in range(n_mels):
        left, centre, right = hz_points[m], hz_points[m + 1], hz_points[m + 2]
        rising = (fft_freqs - left) / (centre - left)     # 0 -> 1 from left edge to centre
        falling = (right - fft_freqs) / (right - centre)  # 1 -> 0 from centre to right edge
        filters[m] = np.maximum(0.0, np.minimum(rising, falling))
        # Area normalisation: wide high-frequency triangles do not get extra energy.
        filters[m] *= 2.0 / (right - left)
    return filters


def mel_filterbank(audio_cfg: dict[str, Any]) -> torch.Tensor:
    """Mel filter matrix as a tensor, shape (n_mels, n_fft // 2 + 1)."""
    fb = _mel_filterbank_np(
        int(audio_cfg["sample_rate"]), int(audio_cfg["n_fft"]), int(audio_cfg["n_mels"]),
        float(audio_cfg["f_min"]), float(audio_cfg["f_max"]),
    )
    return torch.from_numpy(fb)


# --------------------------------------------------------------- spectra ----

def _stft_magnitude(wav: torch.Tensor, audio_cfg: dict[str, Any]) -> torch.Tensor:
    """Magnitude STFT with framing chosen so frames == samples // hop exactly.

    wav: (samples,) or (batch, samples)  ->  (..., n_fft // 2 + 1, frames)
    """
    n_fft, hop, win = audio_cfg["n_fft"], audio_cfg["hop_length"], audio_cfg["win_length"]
    # Pad (n_fft - hop) / 2 on each side and use center=False. This gives exactly
    # samples // hop frames, so the vocoder (hop x upsampling) lines up perfectly.
    pad = (n_fft - hop) // 2
    squeeze = wav.dim() == 1
    x = wav.unsqueeze(0) if squeeze else wav  # (batch, samples)
    x = torch.nn.functional.pad(x.unsqueeze(1), (pad, pad), mode="reflect").squeeze(1)
    window = torch.hann_window(win, device=x.device)
    spec = torch.stft(x, n_fft, hop_length=hop, win_length=win, window=window,
                      center=False, return_complex=True)  # (batch, freqs, frames)
    mag = spec.abs()
    return mag.squeeze(0) if squeeze else mag


def mel_spectrogram(wav: torch.Tensor, audio_cfg: dict[str, Any]) -> torch.Tensor:
    """Waveform -> natural-log mel spectrogram.

    wav: (samples,) -> (n_mels, frames)     or (batch, samples) -> (batch, n_mels, frames)
    """
    mag = _stft_magnitude(wav, audio_cfg)                 # (..., freqs, frames)
    fb = mel_filterbank(audio_cfg).to(mag.device)         # (n_mels, freqs)
    mel = torch.matmul(fb, mag)                           # (..., n_mels, frames)
    # Clamp before log: log(0) = -inf would turn into NaN losses later.
    return torch.log(torch.clamp(mel, min=float(audio_cfg["log_clamp_min"])))


def log_mel_floor(audio_cfg: dict[str, Any]) -> float:
    """The value of a totally silent mel bin, used to pad mels."""
    return float(np.log(float(audio_cfg["log_clamp_min"])))


# ----------------------------------------------------------- Griffin-Lim ---

def mel_to_linear(log_mel: torch.Tensor, audio_cfg: dict[str, Any]) -> torch.Tensor:
    """Approximately undo the mel squeeze. (n_mels, frames) -> (freqs, frames)."""
    mel = torch.exp(log_mel)
    fb = mel_filterbank(audio_cfg).to(mel.device)            # (n_mels, freqs)
    inverse = torch.linalg.pinv(fb)                          # (freqs, n_mels) least-squares inverse
    return torch.clamp(inverse @ mel, min=0.0)               # magnitudes cannot be negative


def griffin_lim(log_mel: torch.Tensor, audio_cfg: dict[str, Any], n_iter: int = 60,
                power: float = 1.5) -> torch.Tensor:
    """Turn a log-mel (n_mels, frames) into an approximate waveform (samples,).

    The mel only keeps loudness per pitch, not phase (where each wave is in its
    cycle). Griffin-Lim guesses a phase, converts to audio and back, keeps the
    new phase but forces the known magnitude, and repeats until it settles.
    No training involved: it is a fixed algorithm.
    """
    n_fft, hop, win = audio_cfg["n_fft"], audio_cfg["hop_length"], audio_cfg["win_length"]
    log_mel = log_mel.detach().float().cpu()
    magnitude = mel_to_linear(log_mel, audio_cfg) ** power   # sharpen: reduces GL "phasiness"
    window = torch.hann_window(win)
    frames = magnitude.shape[-1]
    length = frames * hop

    # Start from random phase.
    angles = torch.exp(2j * torch.pi * torch.rand_like(magnitude))
    for _ in range(n_iter):
        wav = torch.istft(magnitude * angles, n_fft, hop_length=hop, win_length=win,
                          window=window, center=True, length=length)
        rebuilt = torch.stft(wav, n_fft, hop_length=hop, win_length=win, window=window,
                             center=True, return_complex=True)[..., :frames]
        angles = torch.exp(1j * torch.angle(rebuilt))
    wav = torch.istft(magnitude * angles, n_fft, hop_length=hop, win_length=win,
                      window=window, center=True, length=length)
    peak = wav.abs().max()
    return wav / peak * 0.95 if peak > 0 else wav
