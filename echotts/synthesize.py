"""The inference pipeline: text -> cleaned IDs -> acoustic model -> vocoder -> WAV.

Usage:
    python -m echotts.synthesize "Hello world." [--out hello.wav]
        [--acoustic path/to/acoustic.pt] [--vocoder path/to/vocoder.pt]

Checkpoint paths default to $ECHOTTS_ACOUSTIC_CKPT / $ECHOTTS_VOCODER_CKPT,
then to the newest checkpoint in the config folders. With no vocoder
checkpoint it falls back to Griffin-Lim.
"""
import argparse
import os
import re
from pathlib import Path
from typing import Any

import torch

from echotts.audio import griffin_lim, save_wav
from echotts.checkpoint import latest_checkpoint, load_checkpoint
from echotts.config import load_config
from echotts.models.hifigan import Generator
from echotts.models.tacotron import Tacotron
from echotts.text import clean_text, text_to_ids


def split_sentences(text: str) -> list[str]:
    """Split on . ! ? followed by space. Attention is most reliable on short inputs."""
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p for p in parts if p.strip()]


def resolve_checkpoint(explicit: str | None, env_var: str, folder: str) -> Path | None:
    """Pick a checkpoint: explicit path > environment variable > newest in folder."""
    for candidate in (explicit, os.environ.get(env_var)):
        if candidate:
            return Path(candidate)
    return latest_checkpoint(folder)


class Synthesizer:
    """Holds both models in eval mode and turns text into a waveform."""

    def __init__(self, cfg: dict[str, Any], acoustic: Tacotron, vocoder: Generator | None,
                 device: str = "cpu") -> None:
        self.cfg, self.device = cfg, torch.device(device)
        self.acoustic = acoustic.to(self.device).eval()
        self.vocoder = vocoder.to(self.device).eval() if vocoder is not None else None

    @classmethod
    def from_checkpoints(cls, cfg: dict[str, Any], acoustic_path: str | Path,
                         vocoder_path: str | Path | None, device: str = "cpu") -> "Synthesizer":
        acoustic = Tacotron(cfg)
        acoustic.load_state_dict(load_checkpoint(acoustic_path, device)["model"])
        vocoder = None
        if vocoder_path is not None:
            vocoder = Generator(cfg)
            vocoder.load_state_dict(load_checkpoint(vocoder_path, device)["generator"])
        return cls(cfg, acoustic, vocoder, device)

    @torch.no_grad()
    def mel_to_wav(self, mel: torch.Tensor) -> torch.Tensor:
        """(n_mels, T) -> (T * hop,)"""
        if self.vocoder is None:
            return griffin_lim(mel.cpu(), self.cfg["audio"], self.cfg["audio"]["griffin_lim_iters"])
        return self.vocoder(mel.unsqueeze(0).to(self.device)).view(-1).cpu()

    @torch.no_grad()
    def sentence_to_wav(self, sentence: str) -> tuple[torch.Tensor, bool]:
        """One sentence -> (waveform (samples,), hit_max_steps)."""
        ids = torch.tensor([text_to_ids(sentence, self.cfg["text"])], device=self.device)
        out = self.acoustic.infer(ids)
        return self.mel_to_wav(out["mel"][0]), out["hit_max_steps"]

    @torch.no_grad()
    def synthesize(self, text: str) -> tuple[torch.Tensor, dict[str, Any]]:
        """Any text -> (waveform (samples,), info). Long text is done sentence by sentence."""
        sentences = [s for s in split_sentences(text) if clean_text(s, self.cfg["text"])]
        if not sentences:
            raise ValueError("text has no speakable characters")
        pause = torch.zeros(int(self.cfg["inference"]["sentence_pause_s"] * self.cfg["audio"]["sample_rate"]))
        pieces, hit_max = [], False
        for i, sentence in enumerate(sentences):
            wav, failed = self.sentence_to_wav(sentence)
            hit_max |= failed
            pieces.extend([pause, wav] if i else [wav])
        return torch.cat(pieces), {"sentences": len(sentences), "hit_max_steps": hit_max}

    def save(self, wav: torch.Tensor, path: str | Path) -> None:
        save_wav(path, wav, self.cfg["audio"]["sample_rate"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("text")
    parser.add_argument("--out", default="output.wav")
    parser.add_argument("--acoustic", default=None)
    parser.add_argument("--vocoder", default=None)
    parser.add_argument("--config", default=None)
    args = parser.parse_args()
    cfg = load_config(args.config) if args.config else load_config()
    acoustic = resolve_checkpoint(args.acoustic, "ECHOTTS_ACOUSTIC_CKPT", cfg["training"]["checkpoint_dir"])
    if acoustic is None:
        raise SystemExit("No acoustic checkpoint found. Pass --acoustic or set ECHOTTS_ACOUSTIC_CKPT.")
    vocoder = resolve_checkpoint(args.vocoder, "ECHOTTS_VOCODER_CKPT", cfg["vocoder"]["checkpoint_dir"])
    synth = Synthesizer.from_checkpoints(cfg, acoustic, vocoder)
    wav, info = synth.synthesize(args.text)
    synth.save(wav, args.out)
    vocoder_name = vocoder.name if vocoder else "Griffin-Lim"
    print(f"wrote {args.out}: {wav.numel() / cfg['audio']['sample_rate']:.2f}s "
          f"({acoustic.name} + {vocoder_name}){'  WARNING: hit max decoder steps' if info['hit_max_steps'] else ''}")


if __name__ == "__main__":
    main()
