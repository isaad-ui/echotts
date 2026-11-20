# EchoTTS project rules

## What this project is
An English text-to-speech engine built from scratch in PyTorch.
Pipeline: text -> character IDs -> acoustic model (Tacotron 2 style)
-> mel spectrogram -> vocoder (HiFi-GAN style) -> 22,050 Hz WAV.
This is a learning project. The owner is a second-year CS student
who must understand every component.

## Hard rules
- Never download or load pretrained TTS or vocoder weights.
  All models start from random initialization.
- Never copy code from existing TTS repositories. Write it fresh.
  Papers are references for ideas only.
- Only exception: a pretrained speech recognizer may be used in
  echotts/evaluate.py to compute Word Error Rate.
- Do not add libraries outside requirements.txt without asking.
- Do not start work on a later stage than the one requested.

## How to work
- Plan before coding. Show the plan and wait for approval.
- Build one small piece at a time, then test it.
- Every module gets pytest tests in tests/. Tests must check
  tensor shapes on a tiny fake batch so they run on CPU in seconds.
- All settings come from configs/base.yaml. No magic numbers
  inside model or training code.
- Training scripts must save checkpoints and resume from the
  latest one, because GPU sessions are cut off.
- Heavy training is NOT run on this machine. It runs in
  notebooks/train_on_gpu.ipynb on a cloud GPU. Locally, only run
  tests and tiny overfit checks.

## Code style
- Simple, readable code over clever code.
- Comment every class and non-obvious line in plain language.
- Write tensor shapes in comments, e.g. # (batch, time, 80)
- Type hints on function signatures.
- Short functions. One job each.

## Explaining
- After finishing a piece, explain what it does, why it exists,
  and the maths behind it, using a plain everyday analogy.
- If something is a trade-off, say what the options were.

## Git
- Small commits with clear messages, one per finished piece.
- Never commit data/, checkpoints/, or audio output files.

## Commands
- Install: pip install -r requirements.txt
- Tests: pytest -q
- API: uvicorn api.main:app --reload
