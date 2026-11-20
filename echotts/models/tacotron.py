"""The full acoustic model: text IDs -> mel spectrogram (Tacotron 2 style)."""
from typing import Any

import torch
import torch.nn as nn

from echotts.models.decoder import Decoder
from echotts.models.encoder import Encoder
from echotts.models.layers import lengths_to_mask
from echotts.models.postnet import Postnet
from echotts.text import build_symbols


class Tacotron(nn.Module):
    """Encoder -> attention decoder -> postnet."""

    def __init__(self, cfg: dict[str, Any]) -> None:
        super().__init__()
        n_mels = cfg["audio"]["n_mels"]
        self.encoder = Encoder(len(build_symbols(cfg["text"])), cfg["model"])
        self.decoder = Decoder(cfg["model"], n_mels, self.encoder.out_dim)
        self.postnet = Postnet(n_mels, cfg["model"]["postnet"])

    def forward(self, text: torch.Tensor, text_lengths: torch.Tensor,
                mel: torch.Tensor) -> dict[str, torch.Tensor]:
        """Teacher-forced training pass.

        text (B, T_text), text_lengths (B,), mel (B, n_mels, T_mel)
        """
        memory = self.encoder(text, text_lengths)                     # (B, T_text, 512)
        mask = lengths_to_mask(text_lengths, text.shape[1])           # (B, T_text)
        mel_before, stop, align = self.decoder(memory, mel, mask)     # (B, n_mels, T_mel) ...
        mel_after = mel_before + self.postnet(mel_before)             # residual polish
        return {"mel_before": mel_before, "mel_after": mel_after,
                "stop_logits": stop, "alignments": align}

    @torch.no_grad()
    def infer(self, text: torch.Tensor, max_steps: int | None = None) -> dict[str, Any]:
        """Generate a mel for one sentence. text: (1, T_text)."""
        lengths = torch.tensor([text.shape[1]], device=text.device)
        memory = self.encoder(text, lengths)
        mel_before, stop, align, hit_max = self.decoder.infer(memory, None, max_steps)
        mel_after = mel_before + self.postnet(mel_before)
        return {"mel": mel_after, "stop_logits": stop, "alignments": align, "hit_max_steps": hit_max}


def count_parameters(model: nn.Module) -> int:
    """Number of trainable numbers in the model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
