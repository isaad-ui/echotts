"""Train the acoustic model with checkpoint/resume and TensorBoard logging.

Run on a cloud GPU (see notebooks/train_on_gpu.ipynb), not on a laptop:
    python -m echotts.train_acoustic --data_dir /path/LJSpeech-1.1 \
        --checkpoint_dir /drive/echotts/acoustic --log_dir /drive/echotts/runs/acoustic
"""
import argparse
import time
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from echotts.audio import griffin_lim
from echotts.checkpoint import latest_checkpoint, load_checkpoint, save_checkpoint
from echotts.config import load_config
from echotts.data import build_dataloaders
from echotts.losses import TacotronLoss
from echotts.models.tacotron import Tacotron, count_parameters
from echotts.plotting import plot_alignment, plot_mels


def to_device(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {k: v.to(device, non_blocking=True) for k, v in batch.items()}


@torch.no_grad()
def validate(model: Tacotron, loader: DataLoader, loss_fn: TacotronLoss, device: torch.device) -> float:
    """Average teacher-forced loss on the validation set."""
    model.eval()
    total, n = 0.0, 0
    for batch in loader:
        batch = to_device(batch, device)
        out = model(batch["text"], batch["text_lengths"], batch["mel"])
        total += loss_fn(out, batch)["total"].item() * batch["text"].shape[0]
        n += batch["text"].shape[0]
    model.train()
    return total / max(n, 1)


@torch.no_grad()
def log_samples(model: Tacotron, samples: list[tuple[torch.Tensor, torch.Tensor]], writer: SummaryWriter,
                step: int, cfg: dict[str, Any], device: torch.device) -> None:
    """Alignment + mel images for fixed validation sentences, and one Griffin-Lim clip."""
    model.eval()
    for i, (ids, mel) in enumerate(samples):
        text = ids.unsqueeze(0).to(device)
        out = model(text, torch.tensor([ids.numel()], device=device), mel.unsqueeze(0).to(device))
        writer.add_figure(f"val{i}/alignment_teacher_forced",
                          plot_alignment(out["alignments"][0].cpu().numpy()), step)
        writer.add_figure(f"val{i}/mel_teacher_forced",
                          plot_mels(out["mel_after"][0].cpu().numpy(), mel.numpy()), step)
        free = model.infer(text)                                  # what it sounds like for real
        writer.add_figure(f"val{i}/alignment_inference", plot_alignment(free["alignments"][0].cpu().numpy()), step)
        if i == 0:
            wav = griffin_lim(free["mel"][0].cpu(), cfg["audio"], n_iter=cfg["audio"]["griffin_lim_iters"])
            writer.add_audio("val0/griffin_lim", wav.unsqueeze(0), step, sample_rate=cfg["audio"]["sample_rate"])
    model.train()


def train(cfg: dict[str, Any], data_dir: Path, ckpt_dir: Path, log_dir: Path, max_steps: int,
          batch_size: int | None = None, amp: bool | None = None) -> int:
    """Main loop. Returns the final step. Resumes automatically from ckpt_dir."""
    tcfg = cfg["training"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = (tcfg["mixed_precision"] if amp is None else amp) and device.type == "cuda"
    torch.manual_seed(tcfg["split"]["seed"])

    loaders = build_dataloaders(cfg, data_dir, batch_size)
    model = Tacotron(cfg).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=tcfg["learning_rate"], weight_decay=tcfg["weight_decay"])
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    loss_fn = TacotronLoss(tcfg["stop_loss_weight"])

    step = 0
    latest = latest_checkpoint(ckpt_dir)
    if latest is not None:
        state = load_checkpoint(latest, device)
        try:
            model.load_state_dict(state["model"])
        except RuntimeError as exc:                     # e.g. checkpoint made with another frames_per_step
            raise SystemExit(f"{latest} does not match the current model settings in the config. "
                             f"Use a new --checkpoint_dir for the new settings.\n{exc}") from exc
        optimizer.load_state_dict(state["optimizer"])
        scaler.load_state_dict(state["scaler"])
        step = state["step"]
        print(f"resumed from {latest.name} at step {step}")
    else:
        print(f"starting fresh: {count_parameters(model):,} parameters on {device}")

    writer = SummaryWriter(str(log_dir))
    val_set = loaders["val"].dataset
    samples = [val_set[i] for i in range(min(2, len(val_set)))]
    model.train()
    started = time.time()

    while step < max_steps:
        for batch in loaders["train"]:
            batch = to_device(batch, device)
            with torch.autocast(device.type, enabled=use_amp):
                out = model(batch["text"], batch["text_lengths"], batch["mel"])
            losses = loss_fn({k: v.float() for k, v in out.items()}, batch)   # loss in full precision
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(losses["total"]).backward()
            scaler.unscale_(optimizer)                           # so clipping sees real gradient sizes
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), tcfg["grad_clip"])
            scaler.step(optimizer)
            scaler.update()
            step += 1

            if step % tcfg["log_every"] == 0:
                for name, value in losses.items():
                    writer.add_scalar(f"train/{name}", value.item(), step)
                writer.add_scalar("train/grad_norm", float(grad_norm), step)
                print(f"step {step}  loss {losses['total'].item():.4f}  "
                      f"{(time.time() - started) / tcfg['log_every']:.2f}s/step", flush=True)
                started = time.time()

            if step % tcfg["checkpoint_every"] == 0 or step == max_steps:
                writer.add_scalar("val/loss", validate(model, loaders["val"], loss_fn, device), step)
                if samples:
                    log_samples(model, samples, writer, step, cfg, device)
                path = save_checkpoint(ckpt_dir, step, {"model": model.state_dict(),
                                                        "optimizer": optimizer.state_dict(),
                                                        "scaler": scaler.state_dict()},
                                       tcfg["keep_checkpoints"])
                print(f"saved {path}", flush=True)
            if step >= max_steps:
                break
    writer.close()
    return step


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=None)
    parser.add_argument("--data_dir", default=None)
    parser.add_argument("--checkpoint_dir", default=None, help="point at persistent storage (Drive)")
    parser.add_argument("--log_dir", default=None)
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--amp", action="store_true", help="mixed precision on GPU")
    args = parser.parse_args()
    cfg = load_config(args.config) if args.config else load_config()
    t = cfg["training"]
    train(cfg, Path(args.data_dir or cfg["data"]["data_dir"]), Path(args.checkpoint_dir or t["checkpoint_dir"]),
          Path(args.log_dir or t["log_dir"]), args.max_steps or t["max_steps"], args.batch_size,
          True if args.amp else None)


if __name__ == "__main__":
    main()
