"""API tests with a tiny untrained model: no real checkpoints needed."""
import io

import pytest
import soundfile as sf
import torch
from fastapi.testclient import TestClient

from api.main import create_app
from echotts.models.hifigan import Generator
from echotts.models.tacotron import Tacotron
from echotts.synthesize import Synthesizer


@pytest.fixture()
def client(tiny_cfg):
    torch.manual_seed(0)
    synth = Synthesizer(tiny_cfg, Tacotron(tiny_cfg), Generator(tiny_cfg))
    with TestClient(create_app(synth)) as c:
        yield c


def test_health(client) -> None:
    body = client.get("/api/health").json()
    assert body["status"] == "ok" and body["sample_rate"] == 22050 and body["max_chars"] == 300


def test_tts_returns_playable_wav(client) -> None:
    r = client.post("/api/tts", json={"text": "Hello world. How are you?"})
    assert r.status_code == 200 and r.headers["content-type"] == "audio/wav"
    data, sr = sf.read(io.BytesIO(r.content))
    assert sr == 22050 and len(data) > 0


@pytest.mark.parametrize("payload, fragment", [
    ({"text": ""}, "empty"),
    ({"text": "   "}, "empty"),
    ({"text": "a" * 301}, "too long"),
    ({"text": "### ***"}, "no letters"),
    ({"wrong": "field"}, "JSON"),
])
def test_bad_input_gives_clean_400(client, payload, fragment) -> None:
    r = client.post("/api/tts", json=payload)
    assert r.status_code == 400
    assert fragment in r.json()["error"]


def test_no_model_gives_503(tiny_cfg, monkeypatch) -> None:
    monkeypatch.delenv("ECHOTTS_ACOUSTIC_CKPT", raising=False)
    with TestClient(create_app(cfg=tiny_cfg)) as c:
        assert c.get("/api/health").json()["status"] == "no_model"
        assert c.post("/api/tts", json={"text": "hi"}).status_code == 503
