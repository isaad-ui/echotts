"""Download the two model files from URLs (e.g. GitHub Release assets) if not present.

Environment variables:
  ECHOTTS_ACOUSTIC_URL  -> saved to ECHOTTS_ACOUSTIC_CKPT (default /app/models/acoustic.pt)
  ECHOTTS_VOCODER_URL   -> saved to ECHOTTS_VOCODER_CKPT  (default /app/models/vocoder.pt)

Used by the Dockerfile at build time and again at container start, so the
weights never need to be in git.
"""
import os
import shutil
import urllib.request
from pathlib import Path

PAIRS = [("ECHOTTS_ACOUSTIC_URL", "ECHOTTS_ACOUSTIC_CKPT", "/app/models/acoustic.pt"),
         ("ECHOTTS_VOCODER_URL", "ECHOTTS_VOCODER_CKPT", "/app/models/vocoder.pt")]


def download(url: str, dest: Path) -> None:
    """Stream to a temp file, then rename, so a failed download never looks finished."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".part")
    with urllib.request.urlopen(url, timeout=60) as response, open(tmp, "wb") as f:
        shutil.copyfileobj(response, f)
    tmp.replace(dest)


def main() -> None:
    for url_var, path_var, default in PAIRS:
        url = os.environ.get(url_var)
        dest = Path(os.environ.get(path_var, default))
        if dest.exists():
            print(f"{dest} already present")
        elif url:
            print(f"downloading {url_var} -> {dest}")
            download(url, dest)
        else:
            print(f"{url_var} not set; skipping {dest}")


if __name__ == "__main__":
    main()
