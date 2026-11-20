"""Shared building blocks for the acoustic model."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvNormBlock(nn.Module):
    """Conv1d -> BatchNorm -> activation -> dropout, keeping the time length.

    Analogy: a sliding magnifying glass that looks at `kernel_size`
    neighbouring positions at once and writes a summary for each position.
    (B, in_channels, T) -> (B, out_channels, T)
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int,
                 activation: str | None, dropout: float) -> None:
        super().__init__()
        # padding = k // 2 on each side keeps T unchanged for odd kernels.
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size // 2)
        self.norm = nn.BatchNorm1d(out_channels)
        self.activation = activation
        self.dropout = nn.Dropout(dropout)
        gain = nn.init.calculate_gain(activation if activation else "linear")
        nn.init.xavier_uniform_(self.conv.weight, gain=gain)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(self.conv(x))
        if self.activation == "relu":
            x = F.relu(x)
        elif self.activation == "tanh":
            x = torch.tanh(x)
        return self.dropout(x)


class Prenet(nn.Module):
    """Two small Linear+ReLU layers with dropout that is ALWAYS on.

    The decoder feeds its previous mel frame through this bottleneck.
    Keeping dropout on even at inference adds a little noise, which stops
    the decoder from copying the last frame and forces it to use attention.
    (..., in_dim) -> (..., sizes[-1])
    """

    def __init__(self, in_dim: int, sizes: list[int], dropout: float) -> None:
        super().__init__()
        dims = [in_dim] + list(sizes)
        self.layers = nn.ModuleList(nn.Linear(a, b, bias=False) for a, b in zip(dims[:-1], dims[1:]))
        self.dropout = dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = F.dropout(F.relu(layer(x)), p=self.dropout, training=True)  # training=True on purpose
        return x


def lengths_to_mask(lengths: torch.Tensor, max_len: int | None = None) -> torch.Tensor:
    """(B,) lengths -> (B, max_len) bool mask that is True on REAL positions."""
    max_len = max_len or int(lengths.max())
    positions = torch.arange(max_len, device=lengths.device)        # (max_len,)
    return positions.unsqueeze(0) < lengths.unsqueeze(1)            # (B, max_len)
