"""Measure the engine on the 500 held-out test sentences.

Reports:
  - Word Error Rate: our audio is transcribed by a PRETRAINED speech
    recognizer (torchaudio wav2vec 2.0). It only grades us; it is never
    part of the engine. This is the one allowed pretrained model.
  - Real-time factor on CPU: seconds of compute per second of audio.
    Below 1.0 = faster than real time.
  - Failed sentences: hit the decoder's max length, or far too short.

Usage:
    python -m echotts.evaluate --acoustic a.pt --vocoder v.pt [--limit 50] [--out_dir outputs/eval]
"""
import argparse
import re
import time
from pathlib import Path
from typing import Any, Callable

import torch

from echotts.config import load_config
from echotts.data import read_metadata, split_items
from echotts.synthesize import Synthesizer, resolve_checkpoint
from echotts.text import clean_text

Transcriber = Callable[[torch.Tensor, int], str]   # (wav, sample_rate) -> text


# ----------------------------------------------------------------- WER ---

def normalise_for_wer(text: str, text_cfg: dict[str, Any]) -> list[str]:
    """Same cleaning as the model's input, then drop punctuation -> list of words."""
    return re.sub(r"[^a-z' ]", " ", clean_text(text, text_cfg)).split()


def edit_distance(ref: list[str], hyp: list[str]) -> int:
    """Fewest word substitutions + deletions + insertions turning ref into hyp."""
    prev = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        cur = [i] + [0] * len(hyp)
        for j, h in enumerate(hyp, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (r != h))
        prev = cur
    return prev[-1]


def word_error_rate(refs: list[list[str]], hyps: list[list[str]]) -> float:
    """Total edits / total reference words, over the whole set."""
    errors = sum(edit_distance(r, h) for r, h in zip(refs, hyps))
    return errors / max(1, sum(len(r) for r in refs))


# ---------------------------------------------------------- recognizer ---

def load_wav2vec_transcriber(device: str = "cpu") -> Transcriber:
    """Pretrained wav2vec 2.0 (LibriSpeech 960h) with greedy CTC decoding."""
    import torchaudio  # only needed here

    bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H
    model = bundle.get_model().to(device).eval()
    labels = bundle.get_labels()                                   # ('-', '|', 'E', 'T', ...)

    @torch.no_grad()
    def transcribe(wav: torch.Tensor, sample_rate: int) -> str:
        wav = torchaudio.functional.resample(wav, sample_rate, bundle.sample_rate)
        emissions, _ = model(wav.unsqueeze(0).to(device))          # (1, frames, labels)
        best = emissions[0].argmax(dim=-1).tolist()
        chars, last = [], None
        for idx in best:                                           # collapse repeats, drop blanks
            if idx != last and labels[idx] != "-":
                chars.append(labels[idx])
            last = idx
        return "".join(chars).replace("|", " ").strip().lower()

    return transcribe


# ------------------------------------------------------------ evaluate ---

def evaluate(synth: Synthesizer, sentences: list[tuple[str, str]], transcribe: Transcriber,
             out_dir: Path, n_samples: int) -> dict[str, Any]:
    """Synthesise every sentence, grade it, write samples and a markdown report."""
    cfg = synth.cfg
    sr = cfg["audio"]["sample_rate"]
    samples_dir = out_dir / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)
    refs, hyps, rows = [], [], []
    compute_s = audio_s = 0.0
    failed = 0
    for i, (clip_id, text) in enumerate(sentences):
        start = time.perf_counter()
        wav, info = synth.synthesize(text)
        compute_s += time.perf_counter() - start
        seconds = wav.numel() / sr
        audio_s += seconds
        n_chars = len(clean_text(text, cfg["text"]))
        too_short = seconds < cfg["inference"]["short_audio_ratio"] * n_chars
        is_failed = info["hit_max_steps"] or too_short
        failed += is_failed
        hyp = transcribe(wav, sr)
        refs.append(normalise_for_wer(text, cfg["text"]))
        hyps.append(hyp.split())
        rows.append((clip_id, text, hyp, edit_distance(refs[-1], hyps[-1]), is_failed))
        if i < n_samples:
            synth.save(wav, samples_dir / f"{i:02d}_{clip_id}.wav")
        if (i + 1) % 25 == 0:
            print(f"{i + 1}/{len(sentences)}", flush=True)

    results = {"sentences": len(sentences), "wer": word_error_rate(refs, hyps),
               "rtf": compute_s / max(audio_s, 1e-9), "failed": failed, "audio_seconds": audio_s}
    write_report(out_dir / "report.md", results, rows, n_samples)
    return results


def write_report(path: Path, results: dict[str, Any], rows: list[tuple], n_samples: int) -> None:
    lines = [
        "# EchoTTS evaluation report", "",
        "| Metric | Value |", "|---|---|",
        f"| Test sentences | {results['sentences']} |",
        f"| Word Error Rate (wav2vec 2.0 ASR) | {results['wer'] * 100:.1f}% |",
        f"| Real-time factor (CPU) | {results['rtf']:.2f} |",
        f"| Failed sentences (max length or far too short) | {results['failed']} |",
        f"| Total audio | {results['audio_seconds']:.0f} s |", "",
        f"## First {n_samples} samples (listen in `samples/`)", "",
        "| # | Clip | Reference | Heard by ASR | Word errors | Failed |", "|---|---|---|---|---|---|",
    ]
    for i, (clip_id, ref, hyp, errs, failed) in enumerate(rows[:n_samples]):
        lines.append(f"| {i} | {clip_id} | {ref} | {hyp} | {errs} | {'yes' if failed else ''} |")
    lines += ["", "## Worst 20 by word errors", "", "| Clip | Reference | Heard by ASR | Word errors |",
              "|---|---|---|---|"]
    for clip_id, ref, hyp, errs, _ in sorted(rows, key=lambda r: -r[3])[:20]:
        lines.append(f"| {clip_id} | {ref} | {hyp} | {errs} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=None)
    parser.add_argument("--data_dir", default=None)
    parser.add_argument("--acoustic", default=None)
    parser.add_argument("--vocoder", default=None)
    parser.add_argument("--limit", type=int, default=None, help="evaluate only the first N test sentences")
    parser.add_argument("--out_dir", default="outputs/eval")
    args = parser.parse_args()
    cfg = load_config(args.config) if args.config else load_config()
    torch.set_num_threads(max(1, torch.get_num_threads()))
    acoustic = resolve_checkpoint(args.acoustic, "ECHOTTS_ACOUSTIC_CKPT", cfg["training"]["checkpoint_dir"])
    if acoustic is None:
        raise SystemExit("No acoustic checkpoint found.")
    vocoder = resolve_checkpoint(args.vocoder, "ECHOTTS_VOCODER_CKPT", cfg["vocoder"]["checkpoint_dir"])
    synth = Synthesizer.from_checkpoints(cfg, acoustic, vocoder, "cpu")   # CPU: that is what we host on

    data_dir = Path(args.data_dir or cfg["data"]["data_dir"])
    s = cfg["training"]["split"]
    test = split_items(read_metadata(data_dir), s["train"], s["val"], s["test"], s["seed"])["test"]
    test = test[: args.limit] if args.limit else test
    results = evaluate(synth, test, load_wav2vec_transcriber(), Path(args.out_dir),
                       cfg["inference"]["eval_samples"])
    print(f"WER {results['wer'] * 100:.1f}%  RTF {results['rtf']:.2f}  failed {results['failed']}/"
          f"{results['sentences']}  -> {args.out_dir}/report.md")


if __name__ == "__main__":
    main()
