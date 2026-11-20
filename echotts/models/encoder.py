"""Encoder: character IDs -> one context-aware vector per character.

Analogy: reading the whole sentence before speaking it, so each letter
"knows" its neighbours (the 'c' in "cat" vs "city" sounds different).
"""
from typing import Any

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from echotts.models.layers import ConvNormBlock


class Encoder(nn.Module):
    """Embedding -> 3 conv blocks -> bidirectional LSTM.

    (B, T_text) int IDs -> (B, T_text, 2 * lstm_hidden)
    """

    def __init__(self, n_symbols: int, model_cfg: dict[str, Any]) -> None:
        super().__init__()
        enc = model_cfg["encoder"]
        emb_dim = model_cfg["embedding_dim"]
        # padding_idx=0: the pad symbol always maps to a zero vector.
        self.embedding = nn.Embedding(n_symbols, emb_dim, padding_idx=0)
        channels = [emb_dim] + [enc["channels"]] * enc["n_convs"]
        self.convs = nn.Sequential(*[
            ConvNormBlock(c_in, c_out, enc["kernel_size"], "relu", enc["dropout"])
            for c_in, c_out in zip(channels[:-1], channels[1:])
        ])
        self.lstm = nn.LSTM(enc["channels"], enc["lstm_hidden"], batch_first=True, bidirectional=True)
        self.out_dim = 2 * enc["lstm_hidden"]

    def forward(self, text: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        x = self.embedding(text)                  # (B, T_text, emb)
        x = self.convs(x.transpose(1, 2))         # (B, channels, T_text)   conv wants channels first
        x = x.transpose(1, 2)                     # (B, T_text, channels)
        # Packing tells the LSTM each real length so it never reads padding.
        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        out, _ = self.lstm(packed)
        out, _ = pad_packed_sequence(out, batch_first=True, total_length=text.shape[1])
        return out                                # (B, T_text, 2 * lstm_hidden)
