"""Train the HiFi-GAN style vocoder on real recordings, with checkpoint/resume.

Each step:
  1. Critic step: discriminators learn to tell real audio from generated.
  2. Generator step: generator learns to fool them (adversarial), to match
     their inner features (feature matching x2) and to reproduce the same
     mel as the real audio (mel L1 x45, the main "be correct" signal).

Run on a cloud GPU (see notebooks/train_on_gpu.ipynb):
    python -m echotts.train_vocoder --data_dir /path/LJSpeech-1.1 \
        --checkpoint_dir /drive/echotts/vocoder --log_dir /drive/echotts/runs/vocoder
"""
import argparse
import itertools
import random
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter

from echotts.audio import load_wav, mel_spectrogram
from echotts.checkpoint import latest_checkpoint, load_checkpoint, save_checkpoint
from echotts.config import load_config
from echotts.data import read_metadata, split_items
from echotts.losses import discriminator_loss, feature_matching_loss, generator_adversarial_loss
from echotts.models.hifigan import Generator, MultiPeriodDiscriminator, MultiScaleDiscriminator
from echotts.models.tacotron import count_parameters
from echotts.plotting import plot_mels


class VocoderDataset(Dataset):
    """Random fixed-length audio segments with their matching mels.

    Returns wav (1, segment) and mel (n_mels, segment // hop). The mel comes
    from echotts.audio.mel_spectrogram, the exact function the acoustic
    model was trained on, so the two models agree.
    """

    def __init__(self, wav_paths: list[Path], cfg: dict[str, Any], segment: int | None) -> None:
        self.paths, self.cfg, self.segment = wav_paths, cfg, segment

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        wav = load_wav(self.paths[idx], self.cfg["audio"])            # (samples,)
        hop = self.cfg["audio"]["hop_length"]
        if self.segment is not None:
            if wav.numel() >= self.segment:
                start = random.randint(0, wav.numel() - self.segment)
                wav = wav[start:start + self.segment]
            else:
                wav = F.pad(wav, (0, self.segment - wav.numel()))
        else:
            wav = wav[: wav.numel() // hop * hop]                     # whole clip, trimmed to whole frames
        return wav.unsqueeze(0), mel_spectrogram(wav, self.cfg["audio"])


def build_vocoder_loaders(cfg: dict[str, Any], data_dir: Path, batch_size: int | None) -> dict[str, DataLoader]:
    """Same fixed split as the acoustic model, so test clips stay unseen."""
    s = cfg["training"]["split"]
    splits = split_items(read_metadata(data_dir), s["train"], s["val"], s["test"], s["seed"])
    paths = {k: [data_dir / "wavs" / f"{cid}.wav" for cid, _ in v] for k, v in splits.items()}
    v = cfg["vocoder"]
    bs = batch_size or v["batch_size"]
    train = DataLoader(VocoderDataset(paths["train"], cfg, v["segment_size"]), batch_size=bs, shuffle=True,
                       drop_last=len(paths["train"]) > bs, num_workers=cfg["data"]["num_workers"],
                       pin_memory=torch.cuda.is_available())
    val = DataLoader(VocoderDataset(paths["val"][:4], cfg, None), batch_size=1)
    return {"train": train, "val": val}


def train(cfg: dict[str, Any], data_dir: Path, ckpt_dir: Path, log_dir: Path, max_steps: int,
          batch_size: int | None = None) -> int:
    """Main GAN loop. Returns the final step. Resumes automatically from ckpt_dir."""
    v = cfg["vocoder"]
    acfg = cfg["audio"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loaders = build_vocoder_loaders(cfg, data_dir, batch_size)

    gen = Generator(cfg).to(device)
    mpd, msd = MultiPeriodDiscriminator(cfg).to(device), MultiScaleDiscriminator(cfg).to(device)
    opt_g = torch.optim.AdamW(gen.parameters(), v["learning_rate"], betas=tuple(v["betas"]))
    opt_d = torch.optim.AdamW(itertools.chain(mpd.parameters(), msd.parameters()),
                              v["learning_rate"], betas=tuple(v["betas"]))
    sched_g = torch.optim.lr_scheduler.ExponentialLR(opt_g, v["lr_decay"])
    sched_d = torch.optim.lr_scheduler.ExponentialLR(opt_d, v["lr_decay"])
    parts = {"generator": gen, "mpd": mpd, "msd": msd, "opt_g": opt_g, "opt_d": opt_d,
             "sched_g": sched_g, "sched_d": sched_d}

    step = 0
    latest = latest_checkpoint(ckpt_dir)
    if latest is not None:
        state = load_checkpoint(latest, device)
        for name, part in parts.items():
            part.load_state_dict(state[name])
        step = state["step"]
        print(f"resumed from {latest.name} at step {step}")
    else:
        print(f"starting fresh: generator {count_parameters(gen):,} parameters on {device}")

    writer = SummaryWriter(str(log_dir))
    w_fm, w_mel = v["loss_weights"]["feature_matching"], v["loss_weights"]["mel_l1"]
    started = time.time()
    while step < max_steps:
        for wav, mel in loaders["train"]:
            wav, mel = wav.to(device), mel.to(device)                 # (B, 1, S), (B, n_mels, S / hop)
            fake = gen(mel)                                           # (B, 1, S)

            # 1. Critic step (fake is detached: do not update the generator here).
            real_p, _ = mpd(wav); fake_p, _ = mpd(fake.detach())
            real_s, _ = msd(wav); fake_s, _ = msd(fake.detach())
            loss_d = discriminator_loss(real_p, fake_p) + discriminator_loss(real_s, fake_s)
            opt_d.zero_grad(set_to_none=True)
            loss_d.backward()
            opt_d.step()

            # 2. Generator step.
            mel_l1 = F.l1_loss(mel_spectrogram(fake.squeeze(1), acfg), mel)
            real_p, real_pf = mpd(wav); fake_p, fake_pf = mpd(fake)
            real_s, real_sf = msd(wav); fake_s, fake_sf = msd(fake)
            adv = generator_adversarial_loss(fake_p) + generator_adversarial_loss(fake_s)
            fm = feature_matching_loss(real_pf, fake_pf) + feature_matching_loss(real_sf, fake_sf)
            loss_g = adv + w_fm * fm + w_mel * mel_l1
            opt_g.zero_grad(set_to_none=True)
            loss_g.backward()
            opt_g.step()
            step += 1

            if step % v["log_every"] == 0:
                for name, value in {"d": loss_d, "g": loss_g, "adv": adv, "fm": fm, "mel_l1": mel_l1}.items():
                    writer.add_scalar(f"train/{name}", value.item(), step)
                print(f"step {step}  D {loss_d.item():.3f}  G {loss_g.item():.3f}  mel_l1 {mel_l1.item():.4f}  "
                      f"{(time.time() - started) / v['log_every']:.2f}s/step", flush=True)
                started = time.time()

            if step % v["checkpoint_every"] == 0 or step == max_steps:
                validate_and_log(gen, loaders["val"], writer, step, cfg, device)
                state = {name: part.state_dict() for name, part in parts.items()}
                print(f"saved {save_checkpoint(ckpt_dir, step, state, v['keep_checkpoints'])}", flush=True)
            if step >= max_steps:
                break
        sched_g.step(); sched_d.step()                                # learning-rate decay per epoch
    writer.close()
    return step


@torch.no_grad()
def validate_and_log(gen: Generator, loader: DataLoader, writer: SummaryWriter, step: int,
                     cfg: dict[str, Any], device: torch.device) -> None:
    """Mel L1 on whole validation clips, plus audio and mel images for the first two."""
    gen.eval()
    errors = []
    for i, (wav, mel) in enumerate(loader):
        fake = gen(mel.to(device)).squeeze(1)                         # (1, samples)
        fake_mel = mel_spectrogram(fake, cfg["audio"])
        errors.append(F.l1_loss(fake_mel, mel.to(device)).item())
        if i < 2:
            sr = cfg["audio"]["sample_rate"]
            writer.add_audio(f"val{i}/generated", fake.cpu(), step, sample_rate=sr)
            if step == cfg["vocoder"]["checkpoint_every"]:
                writer.add_audio(f"val{i}/real", wav.squeeze(1), step, sample_rate=sr)
            writer.add_figure(f"val{i}/mel", plot_mels(fake_mel[0].cpu().numpy(), mel[0].numpy()), step)
    if errors:
        writer.add_scalar("val/mel_l1", sum(errors) / len(errors), step)
    gen.train()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=None)
    parser.add_argument("--data_dir", default=None)
    parser.add_argument("--checkpoint_dir", default=None, help="point at persistent storage (Drive)")
    parser.add_argument("--log_dir", default=None)
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    args = parser.parse_args()
    cfg = load_config(args.config) if args.config else load_config()
    v = cfg["vocoder"]
    train(cfg, Path(args.data_dir or cfg["data"]["data_dir"]), Path(args.checkpoint_dir or v["checkpoint_dir"]),
          Path(args.log_dir or v["log_dir"]), args.max_steps or v["max_steps"], args.batch_size)


if __name__ == "__main__":
    main()
