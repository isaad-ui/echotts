"""EchoTTS HTTP API (FastAPI).

Endpoints:
  POST /api/tts     {"text": "..."} -> audio/wav
  GET  /api/health  status and model info
  GET  /            the web page in web/ (stage 9)

Checkpoints come from environment variables:
  ECHOTTS_ACOUSTIC_CKPT  (required to synthesise)
  ECHOTTS_VOCODER_CKPT   (optional; without it Griffin-Lim is used)
  ECHOTTS_CORS_ORIGINS   (optional, comma separated; default "*")

Run:  uvicorn api.main:app --reload
"""
import os
import threading
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import torch
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from echotts.audio import wav_to_bytes
from echotts.config import load_config
from echotts.synthesize import Synthesizer

WEB_DIR = Path(__file__).resolve().parent.parent / "web"


class TTSRequest(BaseModel):
    text: str


def error(status: int, message: str) -> JSONResponse:
    """Every error has the same simple shape: {"error": "..."}."""
    return JSONResponse(status_code=status, content={"error": message})


def load_synthesizer_from_env(cfg: dict[str, Any]) -> tuple[Synthesizer | None, dict[str, Any]]:
    """Load both models once, on CPU, from the env-var paths."""
    acoustic = os.environ.get("ECHOTTS_ACOUSTIC_CKPT")
    vocoder = os.environ.get("ECHOTTS_VOCODER_CKPT") or None
    if not acoustic or not Path(acoustic).exists():
        return None, {"acoustic": None, "vocoder": None}
    torch.set_grad_enabled(False)
    synth = Synthesizer.from_checkpoints(cfg, acoustic, vocoder if vocoder and Path(vocoder).exists() else None)
    return synth, {"acoustic": Path(acoustic).name,
                   "vocoder": Path(vocoder).name if synth.vocoder is not None else "griffin-lim"}


def create_app(synthesizer: Synthesizer | None = None, cfg: dict[str, Any] | None = None) -> FastAPI:
    """Build the app. Tests pass a tiny fake synthesizer; production loads from env vars."""
    cfg = cfg or (synthesizer.cfg if synthesizer else load_config())
    max_chars = cfg["inference"]["max_chars"]

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if app.state.synth is None:                            # load once at startup
            app.state.synth, app.state.model_info = load_synthesizer_from_env(cfg)
        yield

    app = FastAPI(title="EchoTTS", version="1.0", lifespan=lifespan)
    app.state.synth = synthesizer
    app.state.model_info = {"acoustic": "injected", "vocoder": "injected"} if synthesizer else {}
    app.state.lock = threading.Lock()                          # one synthesis at a time

    origins = [o.strip() for o in os.environ.get("ECHOTTS_CORS_ORIGINS", "*").split(",")]
    app.add_middleware(CORSMiddleware, allow_origins=origins, allow_methods=["GET", "POST"],
                       allow_headers=["Content-Type"])

    @app.exception_handler(RequestValidationError)
    async def bad_request(_: Request, exc: RequestValidationError) -> JSONResponse:
        return error(400, 'Send JSON like {"text": "Hello world."}')

    @app.get("/api/health")
    def health() -> dict[str, Any]:
        return {"status": "ok" if app.state.synth else "no_model", **app.state.model_info,
                "sample_rate": cfg["audio"]["sample_rate"], "max_chars": max_chars}

    # A plain `def` endpoint runs in a worker thread, so the server stays responsive while the
    # model works; the lock makes concurrent requests wait their turn instead of sharing the model.
    @app.post("/api/tts", responses={200: {"content": {"audio/wav": {}}}})
    def tts(req: TTSRequest) -> Response:
        text = req.text.strip()
        if not text:
            return error(400, "Text is empty. Type something to say.")
        if len(text) > max_chars:
            return error(400, f"Text is too long ({len(text)} characters). The limit is {max_chars}.")
        if app.state.synth is None:
            return error(503, "No model loaded. Set ECHOTTS_ACOUSTIC_CKPT and restart the server.")
        try:
            with app.state.lock:
                wav, _ = app.state.synth.synthesize(text)     # long text is split into sentences inside
        except ValueError:
            return error(400, "Text has no letters the model can say.")
        return Response(wav_to_bytes(wav, cfg["audio"]["sample_rate"]), media_type="audio/wav",
                        headers={"Content-Disposition": 'inline; filename="echotts.wav"'})

    if WEB_DIR.exists():                                       # serve the frontend from the same server
        app.mount("/", StaticFiles(directory=WEB_DIR, html=True), name="web")
    return app


app = create_app()
