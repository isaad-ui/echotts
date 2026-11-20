"""Save / find / load training checkpoints so cut-off GPU sessions can resume.

A checkpoint is one file `step_000123000.pt` holding every model and
optimizer state plus the step number. We keep only the newest few.
"""
import os
import re
from pathlib import Path
from typing import Any

import torch

_PATTERN = re.compile(r"step_(\d+)\.pt$")


def list_checkpoints(ckpt_dir: str | Path) -> list[Path]:
    """All checkpoints in the folder, oldest first."""
    ckpt_dir = Path(ckpt_dir)
    if not ckpt_dir.exists():
        return []
    found = [p for p in ckpt_dir.iterdir() if _PATTERN.search(p.name)]
    return sorted(found, key=lambda p: int(_PATTERN.search(p.name).group(1)))


def latest_checkpoint(ckpt_dir: str | Path) -> Path | None:
    ckpts = list_checkpoints(ckpt_dir)
    return ckpts[-1] if ckpts else None


def save_checkpoint(ckpt_dir: str | Path, step: int, state: dict[str, Any], keep: int) -> Path:
    """Write atomically (temp file then rename) so a cut-off never leaves a half file."""
    ckpt_dir = Path(ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    path = ckpt_dir / f"step_{step:09d}.pt"
    tmp = path.with_suffix(".tmp")
    torch.save({**state, "step": step}, tmp)
    os.replace(tmp, path)
    for old in list_checkpoints(ckpt_dir)[:-keep]:
        old.unlink()
    return path


def load_checkpoint(path: str | Path, device: str | torch.device = "cpu") -> dict[str, Any]:
    # weights_only=True: only tensors and plain Python types are unpickled (safe loading).
    return torch.load(path, map_location=device, weights_only=True)
