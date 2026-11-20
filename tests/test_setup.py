"""Stage 0 placeholder tests: the package imports and the config loads."""
from pathlib import Path

import yaml

CONFIG_PATH = Path(__file__).resolve().parent.parent / "configs" / "base.yaml"


def test_package_imports() -> None:
    import echotts  # noqa: F401


def test_config_has_all_sections() -> None:
    config = yaml.safe_load(CONFIG_PATH.read_text())
    for section in ("audio", "text", "model", "training"):
        assert section in config


def test_vocoder_upsampling_matches_hop_length() -> None:
    # The vocoder must stretch each mel frame into exactly hop_length samples.
    config = yaml.safe_load(CONFIG_PATH.read_text())
    product = 1
    for rate in config["vocoder"]["upsample_rates"]:
        product *= rate
    assert product == config["audio"]["hop_length"]
