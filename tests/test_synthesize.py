"""Inference pipeline and evaluation helpers, with tiny untrained models (no real checkpoints)."""
import soundfile as sf
import torch

from echotts.checkpoint import save_checkpoint
from echotts.evaluate import edit_distance, evaluate, normalise_for_wer, word_error_rate
from echotts.export import export
from echotts.models.hifigan import Generator
from echotts.models.tacotron import Tacotron
from echotts.synthesize import Synthesizer, split_sentences


def _synth(cfg, vocoder: bool = True) -> Synthesizer:
    torch.manual_seed(0)
    return Synthesizer(cfg, Tacotron(cfg), Generator(cfg) if vocoder else None)


def test_split_sentences() -> None:
    assert split_sentences("Hi there. How are you? Fine!") == ["Hi there.", "How are you?", "Fine!"]
    assert split_sentences("  one sentence  ") == ["one sentence"]


def test_synthesize_vocoder_and_griffin_lim(tiny_cfg) -> None:
    for use_vocoder in (True, False):
        wav, info = _synth(tiny_cfg, use_vocoder).synthesize("Hello world. Second one.")
        assert wav.dim() == 1 and wav.numel() > 0 and not torch.isnan(wav).any()
        assert info["sentences"] == 2


def test_checkpoint_roundtrip_and_export(tiny_cfg, tmp_path) -> None:
    synth = _synth(tiny_cfg)
    a = save_checkpoint(tmp_path / "a", 5, {"model": synth.acoustic.state_dict(), "optimizer": {}}, 1)
    v = save_checkpoint(tmp_path / "v", 5, {"generator": synth.vocoder.state_dict(), "opt_g": {}}, 1)
    export("acoustic", a, tmp_path / "acoustic.pt")
    export("vocoder", v, tmp_path / "vocoder.pt")
    loaded = Synthesizer.from_checkpoints(tiny_cfg, tmp_path / "acoustic.pt", tmp_path / "vocoder.pt")
    wav, _ = loaded.synthesize("Hello world.")
    out = tmp_path / "hello.wav"
    loaded.save(wav, out)
    data, sr = sf.read(out)
    assert sr == 22050 and len(data) == wav.numel()


def test_wer() -> None:
    assert edit_distance("a b c".split(), "a x c d".split()) == 2
    assert word_error_rate([["a", "b"]], [["a", "b"]]) == 0.0
    assert word_error_rate([["a", "b", "c", "d"]], [["a", "b"]]) == 0.5


def test_normalise_for_wer(cfg) -> None:
    assert normalise_for_wer("Dr. Smith, 12 times!", cfg["text"]) == ["doctor", "smith", "twelve", "times"]


def test_evaluate_writes_report(tiny_cfg, tmp_path) -> None:
    sentences = [("T1", "Hello world."), ("T2", "Good morning.")]
    results = evaluate(_synth(tiny_cfg), sentences, lambda wav, sr: "hello world", tmp_path, n_samples=1)
    assert results["sentences"] == 2 and 0 <= results["wer"] <= 2 and results["rtf"] > 0
    assert (tmp_path / "report.md").read_text().startswith("# EchoTTS evaluation report")
    assert len(list((tmp_path / "samples").glob("*.wav"))) == 1
