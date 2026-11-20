"""Tests for the dataset pipeline on 3 tiny fake WAV files."""
import numpy as np
import torch

from echotts.audio import log_mel_floor
from echotts.data import (TTSCollate, TTSDataset, build_dataloaders, read_metadata,
                          split_items)
from echotts.precompute_mels import precompute


def test_read_metadata_prefers_normalised(fake_ljspeech) -> None:
    items = read_metadata(fake_ljspeech)
    assert len(items) == 3
    assert items[0] == ("LJ000-0001", "Doctor Smith paid twelve dollars.")


def test_split_is_fixed_and_disjoint() -> None:
    items = [(f"id{i}", "x") for i in range(50)]
    a = split_items(items, 40, 5, 5, seed=1)
    b = split_items(list(reversed(items)), 40, 5, 5, seed=1)
    assert a == b                                            # same seed -> same split
    ids = [set(x for x, _ in a[k]) for k in ("train", "val", "test")]
    assert sum(len(s) for s in ids) == 50
    assert not (ids[0] & ids[1]) and not (ids[0] & ids[2]) and not (ids[1] & ids[2])


def test_batch_shapes_and_padding(fake_ljspeech, cfg) -> None:
    ds = TTSDataset(read_metadata(fake_ljspeech), fake_ljspeech, cfg)
    floor = log_mel_floor(cfg["audio"])
    batch = TTSCollate(floor)([ds[i] for i in range(3)])
    B = 3
    T_text, T_mel = int(batch["text_lengths"].max()), int(batch["mel_lengths"].max())
    assert batch["text"].shape == (B, T_text)
    assert batch["mel"].shape == (B, 80, T_mel)
    assert batch["stop"].shape == (B, T_mel)
    # Sorted longest text first
    assert batch["text_lengths"].tolist() == sorted(batch["text_lengths"].tolist(), reverse=True)
    for i in range(B):
        tl, ml = int(batch["text_lengths"][i]), int(batch["mel_lengths"][i])
        assert (batch["text"][i, tl:] == 0).all()                   # text padded with zeros
        assert torch.allclose(batch["mel"][i, :, ml:], torch.tensor(floor))  # mel padded with floor
        assert (batch["stop"][i, :ml - 1] == 0).all()
        assert (batch["stop"][i, ml - 1:] == 1).all()               # 1 from last real frame on


def test_precompute_and_dataloader(fake_ljspeech, cfg) -> None:
    assert precompute(fake_ljspeech, cfg) == 3
    assert precompute(fake_ljspeech, cfg) == 0                      # second run skips
    mel = np.load(fake_ljspeech / "mels" / "LJ000-0001.npy")
    assert mel.shape[0] == 80 and not np.isnan(mel).any()

    small = {**cfg, "training": {**cfg["training"], "split": {"train": 1, "val": 1, "test": 1, "seed": 0}},
             "data": {**cfg["data"], "num_workers": 0}}
    loaders = build_dataloaders(small, fake_ljspeech, batch_size=2)
    batch = next(iter(loaders["train"]))
    assert batch["mel"].shape[:2] == (1, 80)
