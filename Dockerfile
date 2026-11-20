# EchoTTS inference server: CPU-only PyTorch + FastAPI + the web page.
#
# Build (weights downloaded into the image):
#   docker build -t echotts \
#     --build-arg ECHOTTS_ACOUSTIC_URL=https://github.com/<user>/echotts/releases/download/v1.0/acoustic.pt \
#     --build-arg ECHOTTS_VOCODER_URL=https://github.com/<user>/echotts/releases/download/v1.0/vocoder.pt .
# Run:
#   docker run -p 8000:8000 echotts
# Or skip the build args and pass -e ECHOTTS_ACOUSTIC_URL=... -e ECHOTTS_VOCODER_URL=... to `docker run`
# to download at start instead.

FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    ECHOTTS_ACOUSTIC_CKPT=/app/models/acoustic.pt \
    ECHOTTS_VOCODER_CKPT=/app/models/vocoder.pt \
    PORT=8000

WORKDIR /app

# CPU-only PyTorch is ~4x smaller than the default CUDA build.
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu
COPY requirements-serve.txt .
RUN pip install -r requirements-serve.txt

COPY configs/ configs/
COPY echotts/ echotts/
COPY api/ api/
COPY web/ web/

# Optional: bake the weights into the image at build time.
ARG ECHOTTS_ACOUSTIC_URL=""
ARG ECHOTTS_VOCODER_URL=""
RUN ECHOTTS_ACOUSTIC_URL="$ECHOTTS_ACOUSTIC_URL" ECHOTTS_VOCODER_URL="$ECHOTTS_VOCODER_URL" \
    python -m api.download_models

# Run as a non-root user.
RUN useradd --create-home echotts && chown -R echotts /app
USER echotts

EXPOSE 8000
# Download at start if the files are still missing, then serve. One worker: the model is
# loaded once per worker, and a lock serialises requests anyway.
CMD ["sh", "-c", "python -m api.download_models && uvicorn api.main:app --host 0.0.0.0 --port ${PORT} --workers 1"]
