"""Postnet: polish the finished mel.

Analogy: proofreading a draft. The decoder wrote the mel one frame at a
time without seeing the future; the postnet looks at the whole mel and
predicts a correction (a residual) that is ADDED to the draft.
"""
from typing import Any

import torch
import torch.nn as nn

from echotts.models.layers import ConvNormBlock


class Postnet(nn.Module):
    """5 conv blocks: tanh on all but the last. (B, n_mels, T) -> (B, n_mels, T) residual."""

    def __init__(self, n_mels: int, post_cfg: dict[str, Any]) -> None:
        super().__init__()
        n, ch, k, p = post_cfg["n_convs"], post_cfg["channels"], post_cfg["kernel_size"], post_cfg["dropout"]
        dims = [n_mels] + [ch] * (n - 1) + [n_mels]
        blocks = []
        for i, (c_in, c_out) in enumerate(zip(dims[:-1], dims[1:])):
            last = i == n - 1
            blocks.append(ConvNormBlock(c_in, c_out, k, None if last else "tanh", p))
        self.blocks = nn.Sequential(*blocks)

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        return self.blocks(mel)
