"""Loss functions for the acoustic model (and, from stage 6, the vocoder)."""
import torch
import torch.nn as nn
import torch.nn.functional as F

from echotts.models.layers import lengths_to_mask


def masked_mse(pred: torch.Tensor, target: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    """Mean squared error over REAL frames only.

    pred, target: (B, n_mels, T_mel); lengths: (B,)
    Padded frames are multiplied by 0, so the model is never graded on them.
    """
    mask = lengths_to_mask(lengths, pred.shape[2]).unsqueeze(1).to(pred.dtype)  # (B, 1, T_mel)
    squared = (pred - target) ** 2 * mask
    return squared.sum() / (mask.sum() * pred.shape[1])


class TacotronLoss(nn.Module):
    """MSE before postnet + MSE after postnet + weighted stop-token BCE.

    The stop BCE uses every frame, padding included: padded frames are
    genuinely "already finished" (target 1), which gives the stop token far
    more positive examples than the single real last frame.
    """

    def __init__(self, stop_weight: float) -> None:
        super().__init__()
        self.stop_weight = stop_weight

    def forward(self, out: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        mel, lengths = batch["mel"], batch["mel_lengths"]
        before = masked_mse(out["mel_before"], mel, lengths)
        after = masked_mse(out["mel_after"], mel, lengths)
        stop = F.binary_cross_entropy_with_logits(out["stop_logits"], batch["stop"])
        total = before + after + self.stop_weight * stop
        return {"total": total, "mel_before": before, "mel_after": after, "stop": stop}


# ------------------------------------------------------------ vocoder (GAN) ---
# Least-squares GAN: the critic should output 1 for real and 0 for fake.


def discriminator_loss(real_scores: list[torch.Tensor], fake_scores: list[torch.Tensor]) -> torch.Tensor:
    """Critic loss: push real scores to 1 and fake scores to 0."""
    return sum(torch.mean((1 - r) ** 2) + torch.mean(f ** 2) for r, f in zip(real_scores, fake_scores))


def generator_adversarial_loss(fake_scores: list[torch.Tensor]) -> torch.Tensor:
    """Generator wants the critic to score its audio as real (1)."""
    return sum(torch.mean((1 - f) ** 2) for f in fake_scores)


def feature_matching_loss(real_fmaps: list[list[torch.Tensor]], fake_fmaps: list[list[torch.Tensor]]) -> torch.Tensor:
    """L1 between the critic's inner features for real vs generated audio.

    Like matching not just the judge's verdict but the judge's notes.
    """
    loss = 0.0
    for real_layers, fake_layers in zip(real_fmaps, fake_fmaps):
        for r, f in zip(real_layers, fake_layers):
            loss = loss + torch.mean(torch.abs(r.detach() - f))
    return loss
