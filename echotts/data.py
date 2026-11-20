"""LJSpeech dataset: (text IDs, mel) pairs, a fixed split, and padded batches.

Analogy: the Dataset is a librarian who fetches one (sentence, recording)
card at a time; the collate function is a packer who squares a stack of
cards of different lengths into one neat box by adding blank padding.
"""
import csv
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from echotts.audio import load_wav, log_mel_floor, mel_spectrogram, trim_silence
from echotts.text import text_to_ids

Item = tuple[str, str]  # (clip id, transcript)


def read_metadata(data_dir: str | Path) -> list[Item]:
    """Read LJSpeech metadata.csv: 'id|raw text|normalised text' per line."""
    items = []
    with open(Path(data_dir) / "metadata.csv", encoding="utf-8") as f:
        for row in csv.reader(f, delimiter="|", quoting=csv.QUOTE_NONE):
            # Prefer the normalised column (numbers already spelled out) when present.
            text = row[2] if len(row) > 2 and row[2].strip() else row[1]
            items.append((row[0], text))
    return items


def split_items(items: list[Item], train: int, val: int, test: int, seed: int) -> dict[str, list[Item]]:
    """Shuffle with a fixed seed and cut into test / val / train.

    The same seed always gives the same split, so test sentences never leak
    into training even across restarts.
    """
    shuffled = sorted(items)                  # start from a known order
    random.Random(seed).shuffle(shuffled)
    test_set = shuffled[:test]
    val_set = shuffled[test:test + val]
    train_set = shuffled[test + val:test + val + train]
    return {"train": train_set, "val": val_set, "test": test_set}


def compute_mel(wav_path: str | Path, cfg: dict[str, Any]) -> torch.Tensor:
    """WAV path -> trimmed log-mel (n_mels, frames). Same steps used everywhere."""
    wav = load_wav(wav_path, cfg["audio"])
    wav = trim_silence(wav, cfg["data"]["trim_top_db"])
    return mel_spectrogram(wav, cfg["audio"])


class TTSDataset(Dataset):
    """Returns (text_ids (T_text,), mel (n_mels, T_mel)) for each clip.

    Uses a precomputed mel from <data_dir>/mels/<id>.npy when it exists,
    otherwise computes it from the WAV on the fly.
    """

    def __init__(self, items: list[Item], data_dir: str | Path, cfg: dict[str, Any]) -> None:
        self.items = items
        self.data_dir = Path(data_dir)
        self.mel_dir = self.data_dir / cfg["data"]["mels_subdir"]
        self.cfg = cfg

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        clip_id, text = self.items[idx]
        ids = torch.tensor(text_to_ids(text, self.cfg["text"]), dtype=torch.long)  # (T_text,)
        mel_path = self.mel_dir / f"{clip_id}.npy"
        if mel_path.exists():
            mel = torch.from_numpy(np.load(mel_path))
        else:
            mel = compute_mel(self.data_dir / "wavs" / f"{clip_id}.wav", self.cfg)
        return ids, mel                                                         # (n_mels, T_mel)


class TTSCollate:
    """Pads a list of (ids, mel) into one batch dict.

    text:         (B, T_text)        padded with 0 (the pad symbol)
    text_lengths: (B,)
    mel:          (B, n_mels, T_mel) padded with the log-mel floor (silence)
    mel_lengths:  (B,)
    stop:         (B, T_mel)         0 while speaking, 1 from the last real frame onward
    """

    def __init__(self, mel_pad_value: float) -> None:
        self.mel_pad_value = mel_pad_value

    def __call__(self, batch: list[tuple[torch.Tensor, torch.Tensor]]) -> dict[str, torch.Tensor]:
        # Longest text first: required by pack_padded_sequence in the encoder.
        batch = sorted(batch, key=lambda pair: pair[0].numel(), reverse=True)
        text_lengths = torch.tensor([ids.numel() for ids, _ in batch], dtype=torch.long)
        mel_lengths = torch.tensor([mel.shape[1] for _, mel in batch], dtype=torch.long)
        n_mels = batch[0][1].shape[0]
        B, T_text, T_mel = len(batch), int(text_lengths.max()), int(mel_lengths.max())

        text = torch.zeros(B, T_text, dtype=torch.long)
        mel = torch.full((B, n_mels, T_mel), self.mel_pad_value)
        stop = torch.zeros(B, T_mel)
        for i, (ids, m) in enumerate(batch):
            text[i, :ids.numel()] = ids
            mel[i, :, :m.shape[1]] = m
            stop[i, m.shape[1] - 1:] = 1.0
        return {"text": text, "text_lengths": text_lengths, "mel": mel,
                "mel_lengths": mel_lengths, "stop": stop}


def build_dataloaders(cfg: dict[str, Any], data_dir: str | Path | None = None,
                      batch_size: int | None = None) -> dict[str, DataLoader]:
    """Build train / val / test DataLoaders from the config."""
    data_dir = Path(data_dir or cfg["data"]["data_dir"])
    split_cfg = cfg["training"]["split"]
    splits = split_items(read_metadata(data_dir), split_cfg["train"], split_cfg["val"],
                         split_cfg["test"], split_cfg["seed"])
    collate = TTSCollate(log_mel_floor(cfg["audio"]))
    bs = batch_size or cfg["training"]["batch_size"]
    loaders = {}
    for name, items in splits.items():
        is_train = name == "train"
        loaders[name] = DataLoader(
            TTSDataset(items, data_dir, cfg), batch_size=bs if is_train else min(bs, 8),
            shuffle=is_train, drop_last=is_train and len(items) > bs, collate_fn=collate,
            num_workers=cfg["data"]["num_workers"], pin_memory=torch.cuda.is_available(),
        )
    return loaders
