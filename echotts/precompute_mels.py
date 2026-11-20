"""Compute every LJSpeech mel once and save it as .npy, so training never recomputes them.

Usage:  python -m echotts.precompute_mels [--data_dir data/LJSpeech-1.1]
"""
import argparse
from pathlib import Path

import numpy as np

from echotts.config import load_config
from echotts.data import compute_mel, read_metadata


def precompute(data_dir: Path, cfg: dict) -> int:
    """Write <data_dir>/mels/<id>.npy for each clip that does not have one yet."""
    out_dir = data_dir / cfg["data"]["mels_subdir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    items = read_metadata(data_dir)
    done = 0
    for i, (clip_id, _) in enumerate(items):
        out = out_dir / f"{clip_id}.npy"
        if out.exists():
            continue                                    # resume-friendly
        mel = compute_mel(data_dir / "wavs" / f"{clip_id}.wav", cfg)
        np.save(out, mel.numpy().astype(np.float32))
        done += 1
        if (i + 1) % 500 == 0:
            print(f"{i + 1}/{len(items)}")
    return done


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    parser.add_argument("--data_dir", default=None)
    args = parser.parse_args()
    cfg = load_config(args.config) if args.config else load_config()
    data_dir = Path(args.data_dir or cfg["data"]["data_dir"])
    n = precompute(data_dir, cfg)
    print(f"computed {n} new mels in {data_dir / cfg['data']['mels_subdir']}")


if __name__ == "__main__":
    main()
