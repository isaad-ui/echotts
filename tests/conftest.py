"""Shared test fixtures: the config, a tiny model config and a tiny fake LJSpeech folder."""
import copy
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from echotts.config import load_config


@pytest.fixture(scope="session")
def cfg() -> dict:
    return load_config()


@pytest.fixture(scope="session")
def tiny_cfg(cfg) -> dict:
    """Same audio/text settings, but miniature networks so CPU tests take seconds."""
    c = copy.deepcopy(cfg)
    m = c["model"]
    m["embedding_dim"] = 16
    m["encoder"].update(channels=16, lstm_hidden=8)
    m["attention"].update(dim=8, location_filters=4, location_kernel=5)
    m["prenet"]["sizes"] = [16, 16]
    m["decoder"].update(lstm_units=16, max_decoder_steps=20)
    m["postnet"]["channels"] = 16
    c["vocoder"].update(upsample_initial_channel=16, resblock_kernel_sizes=[3], resblock_dilations=[[1]])
    c["audio"]["griffin_lim_iters"] = 2
    return c


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
