"""Tests automatiques (lancés en local et par GitHub Actions à chaque push)."""
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from api.main import create_app  # noqa: E402
from make_dummy_model import build  # noqa: E402


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("run")
    model_dir = build(tmp / "model")
    app = create_app(model_dir=str(model_dir), db_path=str(tmp / "monitoring.db"))
    with TestClient(app) as c:
        yield c


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_predict_positive(client):
    r = client.post("/predict", json={"text": "Ce film est magnifique, j'ai adoré"})
    assert r.status_code == 200
    body = r.json()
    assert body["label"] == "positif"
    assert 0.5 < body["confidence"] <= 1.0
    assert abs(sum(body["scores"].values()) - 1) < 1e-3
    assert body["latency_ms"] >= 0


def test_predict_negative(client):
    r = client.post("/predict", json={"text": "Un film ennuyeux et vraiment raté"})
    assert r.json()["label"] == "négatif"


@pytest.mark.parametrize("payload", [{"text": ""}, {"text": "   "}, {}, {"text": "x" * 6000}])
def test_invalid_input(client, payload):
    assert client.post("/predict", json=payload).status_code == 422


def test_long_text_is_truncated(client):
    r = client.post("/predict", json={"text": "bon " * 100})
    assert r.status_code == 200
    assert r.json()["truncated"] is True


def test_stats(client):
    s = client.get("/stats").json()
    assert s["total_requests"] >= 3
    assert "p95" in s["latency_ms"]
    assert "drift" in s


def test_ui_and_docs(client):
    assert client.get("/ui/").status_code == 200
    assert client.get("/docs").status_code == 200
    home = client.get("/")
    assert home.status_code == 200
    assert "Critique-mètre" in home.text


def test_drift_detects_short_texts(client):
    # Des phrases très courtes, différentes des critiques d'entraînement -> alerte
    for _ in range(25):
        client.post("/predict", json={"text": "film"})
    drift = client.get("/stats").json()["drift"]
    assert drift["status"].startswith("ALERTE")
