"""Vocoder shape tests and a tiny resume test (CPU)."""
import copy

import torch

from echotts.audio import mel_spectrogram
from echotts.checkpoint import latest_checkpoint
from echotts.losses import discriminator_loss, feature_matching_loss, generator_adversarial_loss
from echotts.models.hifigan import Generator, MultiPeriodDiscriminator, MultiScaleDiscriminator
from echotts.models.tacotron import count_parameters
from echotts.train_vocoder import VocoderDataset, train


def test_upsampling_matches_hop(cfg) -> None:
    rates = cfg["vocoder"]["upsample_rates"]
    assert torch.tensor(rates).prod().item() == cfg["audio"]["hop_length"]


def test_generator_output_length_and_size(cfg) -> None:
    gen = Generator(cfg)
    for T in (1, 7, 32):
        assert gen(torch.randn(2, 80, T)).shape == (2, 1, T * 256)
    assert 0.5e6 < count_parameters(gen) < 1.5e6                     # "about 1 million"


def test_discriminators_return_scores_and_features(cfg) -> None:
    wav = torch.randn(2, 1, 4096)
    for disc, n in ((MultiPeriodDiscriminator(cfg), 5), (MultiScaleDiscriminator(cfg), 3)):
        scores, fmaps = disc(wav)
        assert len(scores) == n and len(fmaps) == n
        assert all(s.shape[0] == 2 for s in scores)
        assert all(len(f) > 1 for f in fmaps)
        loss = discriminator_loss(scores, scores) + generator_adversarial_loss(scores)
        assert loss.dim() == 0
        assert feature_matching_loss(fmaps, fmaps).item() == 0.0


def test_dataset_uses_same_mel_function(fake_ljspeech, cfg) -> None:
    ds = VocoderDataset(sorted((fake_ljspeech / "wavs").glob("*.wav")), cfg, 2048)
    wav, mel = ds[0]
    assert wav.shape == (1, 2048) and mel.shape == (80, 8)
    assert torch.allclose(mel, mel_spectrogram(wav.squeeze(0), cfg["audio"]))


def _tiny_cfg(cfg: dict) -> dict:
    c = copy.deepcopy(cfg)
    v = c["vocoder"]
    v.update(upsample_initial_channel=16, mpd_channels=[4, 8], msd_channels=[4, 8, 8],
             msd_kernels=[5, 5, 5], msd_strides=[1, 2, 1], msd_groups=[1, 4, 1], segment_size=2048,
             checkpoint_every=2, log_every=1)
    c["training"]["split"] = {"train": 2, "val": 1, "test": 0, "seed": 0}
    c["data"]["num_workers"] = 0
    return c


def test_vocoder_training_resumes(fake_ljspeech, cfg, tmp_path) -> None:
    c = _tiny_cfg(cfg)
    ckpt = tmp_path / "ckpt"
    assert train(c, fake_ljspeech, ckpt, tmp_path / "logs", max_steps=2, batch_size=2) == 2
    assert train(c, fake_ljspeech, ckpt, tmp_path / "logs", max_steps=3, batch_size=2) == 3
    state = torch.load(latest_checkpoint(ckpt), weights_only=True)
    assert state["step"] == 3 and {"generator", "mpd", "msd", "opt_g", "opt_d"} <= state.keys()
