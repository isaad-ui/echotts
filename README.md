# EchoTTS

English text-to-speech engine built from scratch in PyTorch: a Tacotron 2-style
acoustic model, a HiFi-GAN-style vocoder, a FastAPI backend and a web interface.
**No pretrained weights.** Every model starts from random initialization and is
trained on LJSpeech.

**Demo:** _add the link after deploying (see [DEPLOY.md](DEPLOY.md))_
**Sample:** _add a sample WAV link after training_

## Architecture

```
text ──clean──> character IDs ──acoustic model──> mel spectrogram ──vocoder──> 22,050 Hz WAV
```

| Part | What it does | Size |
|---|---|---|
| Text front-end (`echotts/text.py`) | lowercase; expand numbers, years, money and abbreviations; map characters to IDs | |
| Encoder | embedding (512), 3 conv blocks, bidirectional LSTM | |
| Location-sensitive attention | decides which character is being spoken at each step | |
| Decoder | prenet (dropout always on), 2 × 1024 LSTMs, outputs 80 mel values + a stop token per step | |
| Postnet | 5 conv layers that refine the mel | **28.1 M params (acoustic model in total)** |
| Vocoder generator | transposed-conv upsampling 8×8×2×2 = 256 samples per mel frame, plus residual blocks | **0.93 M params** |
| Discriminators (training only) | multi-period (2, 3, 5, 7, 11) and multi-scale | |

All settings are in [configs/base.yaml](configs/base.yaml).

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

## Usage

| Task | Command |
|---|---|
| Precompute mels | `python -m echotts.precompute_mels --data_dir data/LJSpeech-1.1` |
| 8-clip overfit check | `python -m echotts.overfit --data_dir data/LJSpeech-1.1` |
| Train acoustic model (GPU) | [notebooks/train_on_gpu.ipynb](notebooks/train_on_gpu.ipynb) or `python -m echotts.train_acoustic --checkpoint_dir ...` |
| Train vocoder (GPU) | same notebook, or `python -m echotts.train_vocoder --checkpoint_dir ...` |
| Synthesize | `python -m echotts.synthesize "Hello world." --acoustic a.pt --vocoder v.pt` |
| Evaluate | `python -m echotts.evaluate --acoustic a.pt --vocoder v.pt` |
| Export weights for deployment | `python -m echotts.export acoustic step_X.pt acoustic.pt` |
| Run the web app | `ECHOTTS_ACOUSTIC_CKPT=a.pt ECHOTTS_VOCODER_CKPT=v.pt uvicorn api.main:app --reload`, then open http://localhost:8000 |

## Evaluation

_Fill in from `outputs/eval/report.md` after training._

| Metric | Value |
|---|---|
| Word Error Rate (500 test sentences, wav2vec 2.0 ASR) | – |
| Real-time factor (CPU) | – |
| Failed sentences | – |
| Mean Opinion Score (3 listeners, 20 samples, 1–5) | – |

The speech recognizer is used **only** to grade the output. It is not part of the engine.

## Layout

| Folder | Contents |
|---|---|
| `configs/` | `base.yaml`, every setting |
| `echotts/` | audio, text, data, models, training, synthesis and evaluation code |
| `api/` | FastAPI app and model downloader |
| `web/` | plain HTML/CSS/JS frontend |
| `notebooks/` | learning notebooks and the cloud GPU training notebook |
| `tests/` | pytest tests (CPU, seconds) |
| `data/`, `checkpoints/`, `outputs/`, `runs/` | local only, ignored by git |

## Data

[LJSpeech 1.1](https://keithito.com/LJ-Speech-Dataset/) (public domain): 13,100 clips,
about 24 hours, one speaker. The split is fixed: 12,500 train, 100 validation, 500 test.

## Status

- [x] Stage 0: project setup
- [x] Stage 1: audio foundations
- [x] Stage 2: deep learning foundations
- [x] Stage 3: dataset pipeline
- [x] Stage 4: acoustic model (code + tests; the 8-clip overfit on real LJSpeech is still to run)
- [ ] Stage 5: acoustic training (script ready; run on GPU)
- [ ] Stage 6: vocoder training (script ready; run on GPU)
- [ ] Stage 7: evaluation (script ready; needs trained weights)
- [x] Stage 8: backend API
- [ ] Stage 9: frontend done; deployment needs trained weights
