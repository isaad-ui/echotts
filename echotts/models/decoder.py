"""Decoder: produce the mel one frame at a time, steered by attention.

Analogy: speaking syllable by syllable. At each step it looks at what it
just said (previous frame, through the prenet), asks attention where it is
in the text, and says the next 1/86th of a second.

Per step:
  prenet(prev_frame) + previous context -> attention LSTM -> query
  attention(query) -> new context
  [query, context]  -> decoder LSTM -> hidden
  [hidden, context] -> mel frame (80) and stop logit (1)

Training uses TEACHER FORCING: prev_frame is the REAL previous frame, so
every step gets a correct input even early on. At inference there is no
real frame, so the decoder feeds back its own prediction (infer()).
"""
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from echotts.models.attention import LocationSensitiveAttention
from echotts.models.layers import Prenet

State = dict[str, torch.Tensor]


class Decoder(nn.Module):
    """Autoregressive Tacotron 2 style decoder."""

    def __init__(self, model_cfg: dict[str, Any], n_mels: int, memory_dim: int) -> None:
        super().__init__()
        dec = model_cfg["decoder"]
        self.n_mels = n_mels
        self.units = dec["lstm_units"]
        self.dropout = dec["dropout"]
        self.max_steps = dec["max_decoder_steps"]
        self.stop_threshold = dec["stop_threshold"]
        self.prenet = Prenet(n_mels, model_cfg["prenet"]["sizes"], model_cfg["prenet"]["dropout"])
        prenet_dim = model_cfg["prenet"]["sizes"][-1]
        self.attention_rnn = nn.LSTMCell(prenet_dim + memory_dim, self.units)
        self.attention = LocationSensitiveAttention(self.units, memory_dim, model_cfg["attention"])
        self.decoder_rnn = nn.LSTMCell(self.units + memory_dim, self.units)
        self.mel_proj = nn.Linear(self.units + memory_dim, n_mels)
        self.stop_proj = nn.Linear(self.units + memory_dim, 1)

    # ------------------------------------------------------------ helpers ---

    def init_state(self, memory: torch.Tensor) -> State:
        """All-zero starting state. memory: (B, T_text, memory_dim)."""
        B, T, D = memory.shape
        zeros = lambda *shape: memory.new_zeros(*shape)  # noqa: E731  same device and dtype
        return {
            "att_h": zeros(B, self.units), "att_c": zeros(B, self.units),
            "dec_h": zeros(B, self.units), "dec_c": zeros(B, self.units),
            "weights": zeros(B, T), "weights_cum": zeros(B, T),
            "context": zeros(B, D),
        }

    def step(self, prenet_out: torch.Tensor, memory: torch.Tensor, processed_memory: torch.Tensor,
             mask: torch.Tensor | None, s: State) -> tuple[torch.Tensor, torch.Tensor, State]:
        """One decoder step. prenet_out: (B, prenet_dim) -> mel (B, n_mels), stop logit (B,)."""
        att_in = torch.cat([prenet_out, s["context"]], dim=-1)              # (B, prenet + mem)
        att_h, att_c = self.attention_rnn(att_in, (s["att_h"], s["att_c"]))
        att_h = F.dropout(att_h, self.dropout, self.training)

        weights_cat = torch.stack([s["weights"], s["weights_cum"]], dim=1)  # (B, 2, T_text)
        context, weights = self.attention(att_h, memory, processed_memory, weights_cat, mask)

        dec_in = torch.cat([att_h, context], dim=-1)                        # (B, units + mem)
        dec_h, dec_c = self.decoder_rnn(dec_in, (s["dec_h"], s["dec_c"]))
        dec_h = F.dropout(dec_h, self.dropout, self.training)

        out = torch.cat([dec_h, context], dim=-1)                           # (B, units + mem)
        new_state = {"att_h": att_h, "att_c": att_c, "dec_h": dec_h, "dec_c": dec_c,
                     "weights": weights, "weights_cum": s["weights_cum"] + weights, "context": context}
        return self.mel_proj(out), self.stop_proj(out).squeeze(-1), new_state

    # -------------------------------------------------------------- modes ---

    def forward(self, memory: torch.Tensor, mel_target: torch.Tensor,
                mask: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Teacher-forced pass.

        memory:     (B, T_text, mem)
        mel_target: (B, n_mels, T_mel)
        returns mel (B, n_mels, T_mel), stop logits (B, T_mel), alignments (B, T_mel, T_text)
        """
        B, _, T_mel = mel_target.shape
        go = mel_target.new_zeros(B, self.n_mels, 1)                         # silent "go" frame
        prev = torch.cat([go, mel_target[:, :, :-1]], dim=2)                 # shift right by one
        prenet_out = self.prenet(prev.transpose(1, 2))                       # (B, T_mel, prenet)
        processed = self.attention.process_memory(memory)
        state = self.init_state(memory)
        mels, stops, aligns = [], [], []
        for t in range(T_mel):
            mel, stop, state = self.step(prenet_out[:, t], memory, processed, mask, state)
            mels.append(mel); stops.append(stop); aligns.append(state["weights"])
        return (torch.stack(mels, dim=2), torch.stack(stops, dim=1), torch.stack(aligns, dim=1))

    @torch.no_grad()
    def infer(self, memory: torch.Tensor, mask: torch.Tensor | None = None,
              max_steps: int | None = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, bool]:
        """Free-running pass: feed back own predictions until stop fires for all items.

        returns mel (B, n_mels, T), stop logits (B, T), alignments (B, T, T_text), hit_max_steps
        """
        max_steps = max_steps or self.max_steps
        B = memory.shape[0]
        processed = self.attention.process_memory(memory)
        state = self.init_state(memory)
        prev = memory.new_zeros(B, self.n_mels)                              # go frame
        finished = torch.zeros(B, dtype=torch.bool, device=memory.device)
        mels, stops, aligns = [], [], []
        for _ in range(max_steps):
            mel, stop, state = self.step(self.prenet(prev), memory, processed, mask, state)
            mels.append(mel); stops.append(stop); aligns.append(state["weights"])
            finished |= torch.sigmoid(stop) > self.stop_threshold
            if finished.all():
                break
            prev = mel
        hit_max = not bool(finished.all())
        return torch.stack(mels, 2), torch.stack(stops, 1), torch.stack(aligns, 1), hit_max
