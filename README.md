# EchoTTS

English text-to-speech engine built from scratch in PyTorch: a Tacotron 2-style
acoustic model, a HiFi-GAN-style vocoder, a FastAPI backend and a web interface.
No pretrained weights. Every model starts from random initialization.

```
text -> character IDs -> acoustic model -> mel spectrogram -> vocoder -> 22,050 Hz WAV
```

## Setup

Requires Python 3.10 or newer.

```bash
git clone https://github.com/isaad-ui/echotts.git
cd echotts
python -m venv .venv
# Windows: .venv\Scripts\activate    macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
pytest -q
```

## Layout

| Folder | Contents |
|---|---|
| `configs/` | `base.yaml`, every setting |
| `echotts/` | audio, text, data, models, training and synthesis code |
| `api/` | FastAPI app |
| `web/` | plain HTML/CSS/JS frontend |
| `notebooks/` | learning notebooks and the cloud GPU training notebook |
| `tests/` | pytest tests (CPU, seconds) |
| `data/`, `checkpoints/` | local only, ignored by git |

## Data

Training uses [LJSpeech](https://keithito.com/LJ-Speech-Dataset/) (public domain).
Place it at `data/LJSpeech-1.1/`. Heavy training runs on a cloud GPU, not locally.

## Status

- [x] Stage 0: project setup
- [ ] Stage 1: audio foundations
- [ ] Stage 2: deep learning foundations
- [ ] Stage 3: dataset pipeline
- [ ] Stage 4: acoustic model
- [ ] Stage 5: acoustic training
- [ ] Stage 6: vocoder
- [ ] Stage 7: evaluation
- [ ] Stage 8: backend API
- [ ] Stage 9: frontend and deployment
