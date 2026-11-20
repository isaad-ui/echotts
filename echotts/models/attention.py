"""Location-sensitive attention: which character is the decoder saying now?

Analogy: a finger moving along the text. Plain (content-only) attention
looks at every letter fresh at each step, so on long sentences it can jump
to a similar-looking word elsewhere and skip or repeat. Location features
let it see where the finger WAS (previous weights) and where it has
already been (cumulative weights), so it prefers to move a little forward.

Score for character j:  e_j = v^T tanh(W q + V m_j + U f_j + b)
  q   = decoder query (attention LSTM output)
  m_j = encoder output for character j
  f_j = conv features of [previous weights, cumulative weights] around j
weights = softmax(e) over real characters only (padding masked out).
"""
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


class LocationSensitiveAttention(nn.Module):
    """Returns a context vector (B, memory_dim) and new weights (B, T_text)."""

    def __init__(self, query_dim: int, memory_dim: int, attn_cfg: dict[str, Any]) -> None:
        super().__init__()
        dim, n_filters, kernel = attn_cfg["dim"], attn_cfg["location_filters"], attn_cfg["location_kernel"]
        self.query_layer = nn.Linear(query_dim, dim, bias=False)       # W
        self.memory_layer = nn.Linear(memory_dim, dim, bias=False)     # V
        self.location_conv = nn.Conv1d(2, n_filters, kernel, padding=kernel // 2, bias=False)
        self.location_layer = nn.Linear(n_filters, dim, bias=False)    # U
        self.v = nn.Linear(dim, 1, bias=True)                          # v and b

    def process_memory(self, memory: torch.Tensor) -> torch.Tensor:
        """V m_j is the same at every decoder step, so compute it once. (B, T, mem) -> (B, T, dim)"""
        return self.memory_layer(memory)

    def forward(self, query: torch.Tensor, memory: torch.Tensor, processed_memory: torch.Tensor,
                weights_cat: torch.Tensor, mask: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor]:
        """
        query:            (B, query_dim)
        memory:           (B, T_text, memory_dim)
        processed_memory: (B, T_text, dim)
        weights_cat:      (B, 2, T_text)   previous and cumulative weights
        mask:             (B, T_text) True on REAL characters, or None
        """
        q = self.query_layer(query).unsqueeze(1)                       # (B, 1, dim)
        loc = self.location_conv(weights_cat).transpose(1, 2)          # (B, T_text, n_filters)
        loc = self.location_layer(loc)                                 # (B, T_text, dim)
        energies = self.v(torch.tanh(q + processed_memory + loc)).squeeze(-1)  # (B, T_text)
        if mask is not None:
            energies = energies.masked_fill(~mask, float("-inf"))     # padding gets weight 0
        weights = F.softmax(energies, dim=1)                           # (B, T_text), sums to 1
        context = torch.bmm(weights.unsqueeze(1), memory).squeeze(1)   # (B, memory_dim)
        return context, weights
