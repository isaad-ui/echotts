"""Load the project settings from configs/base.yaml.

Every number the code uses comes from this one file, so the acoustic model
and the vocoder can never disagree about things like the hop length.
"""
from pathlib import Path
from typing import Any

import yaml

# Repo root = the folder that contains configs/ and echotts/
ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = ROOT / "configs" / "base.yaml"


def load_config(path: str | Path = DEFAULT_CONFIG) -> dict[str, Any]:
    """Read the YAML config file and return it as a nested dict."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)
