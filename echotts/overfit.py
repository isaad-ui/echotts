"""Sanity check: memorise 8 fixed clips before any long run.

If the model cannot drive the loss near zero on 8 clips and draw a diagonal
attention line, it will not learn 13,100. Run this on CPU or a small GPU.

Usage:  python -m echotts.overfit --data_dir data/LJSpeech-1.1 [--steps 300] [--out_dir outputs/overfit]
"""
import argparse
from pathlib import Path

import torch

from echotts.audio import log_mel_floor
from echotts.config import load_config
from echotts.data import TTSCollate, TTSDataset, read_metadata
from echotts.losses import TacotronLoss
from echotts.models.tacotron import Tacotron, count_parameters
from echotts.plotting import plot_alignment, plot_loss, plot_mels, save


def run_overfit(cfg: dict, data_dir: Path, steps: int, out_dir: Path, device: str) -> list[float]:
    """Train on one fixed batch of clips and save the diagnostic plots."""
    torch.manual_seed(0)
    items = sorted(read_metadata(data_dir))[: cfg["training"]["overfit_clips"]]
    dataset = TTSDataset(items, data_dir, cfg)
    batch = TTSCollate(log_mel_floor(cfg["audio"]))([dataset[i] for i in range(len(dataset))])
    batch = {k: v.to(device) for k, v in batch.items()}

    model = Tacotron(cfg).to(device)
    print(f"parameters: {count_parameters(model):,}")
    loss_fn = TacotronLoss(cfg["training"]["stop_loss_weight"])
    opt = torch.optim.Adam(model.parameters(), lr=cfg["training"]["learning_rate"],
                           weight_decay=cfg["training"]["weight_decay"])

    losses = []
    model.train()
    for step in range(1, steps + 1):
        out = model(batch["text"], batch["text_lengths"], batch["mel"])
        loss = loss_fn(out, batch)
        opt.zero_grad()
        loss["total"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["training"]["grad_clip"])
        opt.step()
        losses.append(loss["total"].item())
        if step % 25 == 0 or step == 1:
            print(f"step {step:4d}  total {losses[-1]:.4f}  mel {loss['mel_after'].item():.4f}  "
                  f"stop {loss['stop'].item():.4f}")

    out_dir.mkdir(parents=True, exist_ok=True)
    T_text, T_mel = int(batch["text_lengths"][0]), int(batch["mel_lengths"][0])
    save(plot_loss(losses), out_dir / "loss_curve.png")
    save(plot_alignment(out["alignments"][0, :T_mel, :T_text].detach().cpu().numpy()),
         out_dir / "attention.png")
    save(plot_mels(out["mel_after"][0, :, :T_mel].detach().cpu().numpy(),
                   batch["mel"][0, :, :T_mel].cpu().numpy()), out_dir / "mel_compare.png")
    print(f"plots saved to {out_dir}")
    return losses


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=None)
    parser.add_argument("--data_dir", default=None)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--out_dir", default="outputs/overfit")
    args = parser.parse_args()
    cfg = load_config(args.config) if args.config else load_config()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    run_overfit(cfg, Path(args.data_dir or cfg["data"]["data_dir"]),
                args.steps or cfg["training"]["overfit_steps"], Path(args.out_dir), device)


if __name__ == "__main__":
    main()
