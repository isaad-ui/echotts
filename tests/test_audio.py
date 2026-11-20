"""Tests for echotts/audio.py on a short synthetic tone (no dataset needed)."""
import numpy as np
import soundfile as sf
import torch

from echotts.audio import griffin_lim, load_wav, mel_spectrogram, save_wav
from echotts.config import load_config

CFG = load_config()["audio"]


def _tone(seconds: float = 0.5, sr: int = 22050) -> torch.Tensor:
    t = torch.arange(int(seconds * sr)) / sr
    return 0.5 * torch.sin(2 * torch.pi * 440 * t)  # (samples,)


def test_mel_shape_and_frames() -> None:
    wav = _tone()
    mel = mel_spectrogram(wav, CFG)
    assert mel.shape[0] == 80
    assert mel.shape[1] == wav.numel() // CFG["hop_length"]


def test_mel_has_no_nan_even_on_silence() -> None:
    mel = mel_spectrogram(torch.zeros(4096), CFG)
    assert not torch.isnan(mel).any()
    floor = float(np.log(float(CFG["log_clamp_min"])))
    assert torch.allclose(mel, torch.full_like(mel, floor))


def test_batched_mel() -> None:
    wav = torch.stack([_tone(), _tone()])          # (2, samples)
    mel = mel_spectrogram(wav, CFG)
    assert mel.shape == (2, 80, wav.shape[1] // 256)


def test_load_wav_resamples_and_mono(tmp_path) -> None:
    sr = 16000
    stereo = np.random.uniform(-0.3, 0.3, size=(sr, 2)).astype(np.float32)  # 1 s stereo
    path = tmp_path / "x.wav"
    sf.write(path, stereo, sr)
    wav = load_wav(path, CFG)
    assert wav.dim() == 1
    assert abs(wav.numel() - 22050) <= 1
    assert wav.abs().max() <= 1.0


def test_griffin_lim_roundtrip(tmp_path) -> None:
    wav = _tone()
    mel = mel_spectrogram(wav, CFG)
    out = griffin_lim(mel, CFG, n_iter=8)
    assert out.shape == (mel.shape[1] * 256,)
    assert not torch.isnan(out).any()
    save_wav(tmp_path / "gl.wav", out, CFG["sample_rate"])
    assert (tmp_path / "gl.wav").exists()
