# Deploying EchoTTS

The server is one Docker container: CPU-only PyTorch, FastAPI and the web page.
The model weights are **not** in git. They are hosted as GitHub Release files and
downloaded by the container.

## What to expect

| Item | Estimate |
|---|---|
| Docker image (no weights) | about 1.0–1.2 GB (CPU PyTorch is most of it) |
| Weights | acoustic ≈ 113 MB, vocoder ≈ 4 MB (after `echotts.export`) |
| RAM while running | about 0.6–1 GB: PyTorch ≈ 300 MB, models ≈ 120 MB, plus activations during synthesis |
| Speed | a few seconds per sentence on 2 vCPUs; check the real-time factor from `echotts.evaluate` |

**Pick a host with at least 1 GB RAM, ideally 2 GB.** Free tiers with 512 MB
(for example Render or Koyeb free instances) will run out of memory.

These numbers are estimates. The image was not built on the development
machine. After your first `docker build`, record the real figures from
`docker images echotts` and `docker stats` here.

## 1. Export the weights

Training checkpoints include optimizer state, which roughly triples their size.
Strip it out:

```bash
python -m echotts.export acoustic /path/to/checkpoints/acoustic/step_000050000.pt acoustic.pt
python -m echotts.export vocoder  /path/to/checkpoints/vocoder/step_000200000.pt  vocoder.pt
```

Check them locally before uploading:

```bash
python -m echotts.synthesize "Hello world." --acoustic acoustic.pt --vocoder vocoder.pt --out hello.wav
```

## 2. Upload them as a GitHub Release

```bash
gh release create v1.0 acoustic.pt vocoder.pt --title "EchoTTS v1.0 weights" --notes "Trained from scratch on LJSpeech."
```

The download links look like this:
`https://github.com/<user>/echotts/releases/download/v1.0/acoustic.pt`

## 3. Test the image locally

```bash
docker build -t echotts \
  --build-arg ECHOTTS_ACOUSTIC_URL=https://github.com/<user>/echotts/releases/download/v1.0/acoustic.pt \
  --build-arg ECHOTTS_VOCODER_URL=https://github.com/<user>/echotts/releases/download/v1.0/vocoder.pt .
docker run --rm -p 8000:8000 echotts
```

Open http://localhost:8000. Type something, generate it, play it and download it.
Then check memory with `docker stats`.

## 4. Free hosting: Hugging Face Spaces (Docker)

The free "CPU basic" Space hardware (2 vCPU, 16 GB RAM at the time of writing)
is more than enough. Check the current limits before you start.

1. Create a Space at https://huggingface.co/new-space. Choose SDK **Docker**,
   template **Blank** and hardware **CPU basic**.
2. In **Settings → Variables**, add `ECHOTTS_ACOUSTIC_URL` and `ECHOTTS_VOCODER_URL`
   with your Release links.
3. Push the code to the Space. It is a git repo:
   ```bash
   git remote add space https://huggingface.co/spaces/<hf-user>/echotts
   git push space main
   ```
   The Space needs a `README.md` header telling it the port. Add this at the
   very top of the README on the Space branch:
   ```yaml
   ---
   title: EchoTTS
   sdk: docker
   app_port: 8000
   ---
   ```
4. Wait for the build to finish (see the **Logs** tab). Then open
   `https://<hf-user>-echotts.hf.space` on your phone, on mobile data.

Free Spaces go to sleep when unused. The first request after a sleep takes a
while because the container has to start again.

## Environment variables

| Variable | Purpose |
|---|---|
| `ECHOTTS_ACOUSTIC_URL` / `ECHOTTS_VOCODER_URL` | where to download the weights (build arg or runtime) |
| `ECHOTTS_ACOUSTIC_CKPT` / `ECHOTTS_VOCODER_CKPT` | where the weights live in the container (defaults `/app/models/*.pt`) |
| `ECHOTTS_CORS_ORIGINS` | comma-separated allowed origins, if the page is hosted elsewhere (default `*`) |
| `PORT` | listening port (default 8000) |
