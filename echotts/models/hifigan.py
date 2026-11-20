"""HiFi-GAN style vocoder: a generator and two kinds of discriminator.

Generator analogy: the mel is a coarse sketch with one column per 256
samples. Transposed convolutions are "zoom-in" steps (x8, x8, x2, x2);
each one stretches time and fills in finer detail, until one column has
become 256 audio samples. 8*8*2*2 must equal the hop length (256) or the
audio would be too long or too short for its mel.

Transposed convolution: a normal strided conv takes `stride` inputs to 1
output; a transposed conv does the reverse, painting each input onto
`stride` outputs with a learned brush (kernel). kernel = 2 * stride and
padding = (kernel - stride) / 2 give exactly L * stride outputs.

Discriminators are critics that try to tell real audio from generated:
  - Multi-period (MPD): folds audio into a 2D grid of width p (2,3,5,7,11),
    so it can check repeating patterns like pitch periods.
  - Multi-scale (MSD): looks at the raw audio, then 2x and 4x smoothed
    versions, judging both fine texture and overall shape.
"""
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.parametrizations import spectral_norm, weight_norm


def _same_padding(kernel: int, dilation: int = 1) -> int:
    return dilation * (kernel - 1) // 2


class ResBlock(nn.Module):
    """Dilated conv pairs with skip connections. (B, C, T) -> (B, C, T)

    Dilation spaces the kernel taps apart, so a 3-tap kernel with dilation 5
    sees 11 samples: a wide view without more weights.
    """

    def __init__(self, channels: int, kernel: int, dilations: list[int], slope: float) -> None:
        super().__init__()
        self.slope = slope
        self.convs1 = nn.ModuleList(
            weight_norm(nn.Conv1d(channels, channels, kernel, dilation=d, padding=_same_padding(kernel, d)))
            for d in dilations)
        self.convs2 = nn.ModuleList(
            weight_norm(nn.Conv1d(channels, channels, kernel, padding=_same_padding(kernel)))
            for _ in dilations)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for c1, c2 in zip(self.convs1, self.convs2):
            x = x + c2(F.leaky_relu(c1(F.leaky_relu(x, self.slope)), self.slope))
        return x


class Generator(nn.Module):
    """Mel (B, n_mels, T) -> waveform (B, 1, T * prod(upsample_rates))."""

    def __init__(self, cfg: dict[str, Any]) -> None:
        super().__init__()
        v = cfg["vocoder"]
        self.slope = v["leaky_relu_slope"]
        ch = v["upsample_initial_channel"]
        self.conv_pre = weight_norm(nn.Conv1d(cfg["audio"]["n_mels"], ch, 7, padding=3))
        self.ups = nn.ModuleList()
        self.resblocks = nn.ModuleList()
        for i, (rate, kernel) in enumerate(zip(v["upsample_rates"], v["upsample_kernel_sizes"])):
            c_in, c_out = ch // (2 ** i), ch // (2 ** (i + 1))
            self.ups.append(weight_norm(nn.ConvTranspose1d(c_in, c_out, kernel, stride=rate,
                                                           padding=(kernel - rate) // 2)))
            # Several resblocks with different kernel sizes per level; their outputs are averaged.
            self.resblocks.append(nn.ModuleList(
                ResBlock(c_out, k, d, self.slope)
                for k, d in zip(v["resblock_kernel_sizes"], v["resblock_dilations"])))
        self.conv_post = weight_norm(nn.Conv1d(c_out, 1, 7, padding=3))

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        x = self.conv_pre(mel)                                       # (B, ch, T)
        for up, blocks in zip(self.ups, self.resblocks):
            x = up(F.leaky_relu(x, self.slope))                      # time x rate, channels / 2
            x = sum(block(x) for block in blocks) / len(blocks)
        x = self.conv_post(F.leaky_relu(x))                          # (B, 1, T * 256)
        return torch.tanh(x)                                         # audio lives in [-1, 1]


# ---------------------------------------------------------- discriminators ---

Scores = list[torch.Tensor]
FeatureMaps = list[list[torch.Tensor]]


class PeriodDiscriminator(nn.Module):
    """Judges audio folded into rows of length `period`."""

    def __init__(self, period: int, v: dict[str, Any]) -> None:
        super().__init__()
        self.period, self.slope = period, v["leaky_relu_slope"]
        k, s = v["mpd_kernel"], v["mpd_stride"]
        chans = [1] + list(v["mpd_channels"])
        layers = [weight_norm(nn.Conv2d(a, b, (k, 1), (s, 1), padding=(_same_padding(k), 0)))
                  for a, b in zip(chans[:-1], chans[1:])]
        layers.append(weight_norm(nn.Conv2d(chans[-1], chans[-1], (k, 1), 1, padding=(_same_padding(k), 0))))
        self.convs = nn.ModuleList(layers)
        self.post = weight_norm(nn.Conv2d(chans[-1], 1, (3, 1), 1, padding=(1, 0)))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        B, C, T = x.shape
        if T % self.period:                                           # pad to a whole number of rows
            x = F.pad(x, (0, self.period - T % self.period), mode="reflect")
        x = x.view(B, C, -1, self.period)                             # (B, 1, T / p, p)
        fmaps = []
        for conv in self.convs:
            x = F.leaky_relu(conv(x), self.slope)
            fmaps.append(x)
        x = self.post(x)
        fmaps.append(x)
        return x.flatten(1), fmaps


class ScaleDiscriminator(nn.Module):
    """Judges raw audio with grouped strided 1D convs."""

    def __init__(self, v: dict[str, Any], use_spectral_norm: bool) -> None:
        super().__init__()
        norm = spectral_norm if use_spectral_norm else weight_norm
        self.slope = v["leaky_relu_slope"]
        chans = [1] + list(v["msd_channels"])
        self.convs = nn.ModuleList(
            norm(nn.Conv1d(a, b, k, s, groups=g, padding=_same_padding(k)))
            for a, b, k, s, g in zip(chans[:-1], chans[1:], v["msd_kernels"], v["msd_strides"], v["msd_groups"]))
        self.post = norm(nn.Conv1d(chans[-1], 1, 3, 1, padding=1))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        fmaps = []
        for conv in self.convs:
            x = F.leaky_relu(conv(x), self.slope)
            fmaps.append(x)
        x = self.post(x)
        fmaps.append(x)
        return x.flatten(1), fmaps


class MultiPeriodDiscriminator(nn.Module):
    def __init__(self, cfg: dict[str, Any]) -> None:
        super().__init__()
        v = cfg["vocoder"]
        self.discriminators = nn.ModuleList(PeriodDiscriminator(p, v) for p in v["mpd_periods"])

    def forward(self, wav: torch.Tensor) -> tuple[Scores, FeatureMaps]:
        """wav (B, 1, T) -> one score tensor and one feature-map list per sub-discriminator."""
        outs = [d(wav) for d in self.discriminators]
        return [o[0] for o in outs], [o[1] for o in outs]


class MultiScaleDiscriminator(nn.Module):
    def __init__(self, cfg: dict[str, Any]) -> None:
        super().__init__()
        v = cfg["vocoder"]
        # First scale uses spectral norm (more stable on raw audio), the rest weight norm.
        self.discriminators = nn.ModuleList(ScaleDiscriminator(v, i == 0) for i in range(v["msd_scales"]))
        self.pool = nn.AvgPool1d(4, 2, padding=2)                     # halves the sample rate (smoothing)

    def forward(self, wav: torch.Tensor) -> tuple[Scores, FeatureMaps]:
        scores, fmaps = [], []
        for i, d in enumerate(self.discriminators):
            if i > 0:
                wav = self.pool(wav)
            s, f = d(wav)
            scores.append(s); fmaps.append(f)
        return scores, fmaps
