"""Strip a training checkpoint down to inference weights only.

Training checkpoints also hold optimizer state (about 3x the model size).
Deployment only needs the weights:
    python -m echotts.export acoustic checkpoints/acoustic/step_000050000.pt acoustic.pt
    python -m echotts.export vocoder  checkpoints/vocoder/step_000200000.pt  vocoder.pt
"""
import argparse

import torch

from echotts.checkpoint import load_checkpoint

KEYS = {"acoustic": "model", "vocoder": "generator"}


def export(kind: str, src: str, dst: str) -> None:
    key = KEYS[kind]
    state = load_checkpoint(src)
    torch.save({key: state[key], "step": state.get("step", 0)}, dst)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("kind", choices=sorted(KEYS))
    parser.add_argument("src")
    parser.add_argument("dst")
    args = parser.parse_args()
    export(args.kind, args.src, args.dst)
    print(f"wrote {args.dst}")


if __name__ == "__main__":
    main()
