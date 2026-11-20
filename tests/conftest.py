"""Shared test fixtures: the config and a tiny fake LJSpeech folder."""
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from echotts.config import load_config


@pytest.fixture(scope="session")
def cfg() -> dict:
    return load_config()


FAKE_LINES = [
    ("LJ000-0001", "Dr. Smith paid $12.", "Doctor Smith paid twelve dollars."),
    ("LJ000-0002", "Hello world", "Hello world"),
    ("LJ000-0003", "The 3rd test in 1984, is it?", "The third test in nineteen eighty-four, is it?"),
]


@pytest.fixture()
def fake_ljspeech(tmp_path: Path) -> Path:
    """Three short noisy-tone WAVs plus metadata.csv in LJSpeech layout."""
    root = tmp_path / "LJSpeech-1.1"
    (root / "wavs").mkdir(parents=True)
    sr = 22050
    rng = np.random.default_rng(0)
    for i, (clip_id, raw, norm) in enumerate(FAKE_LINES):
        n = int(sr * (0.4 + 0.2 * i))
        t = np.arange(n) / sr
        wav = 0.4 * np.sin(2 * np.pi * (200 + 50 * i) * t) + 0.02 * rng.standard_normal(n)
        sf.write(root / "wavs" / f"{clip_id}.wav", wav.astype(np.float32), sr)
    with open(root / "metadata.csv", "w", encoding="utf-8") as f:
        for clip_id, raw, norm in FAKE_LINES:
            f.write(f"{clip_id}|{raw}|{norm}\n")
    return root
