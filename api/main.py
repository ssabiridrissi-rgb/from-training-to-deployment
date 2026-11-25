"""API FastAPI + interface Gradio dans la même application.

Lancement : uvicorn api.main:create_app --factory --host 0.0.0.0 --port 7860

Routes :
    GET  /         -> interface web (app/web/index.html)
    GET  /health   -> l'API est-elle vivante ? quelle version du modèle ?
    POST /predict  -> {"text": "..."} -> sentiment + confiance + latence
    GET  /stats    -> indicateurs de monitoring (latence, confiance, drift)
    GET  /docs     -> documentation Swagger générée automatiquement
    /ui/           -> interface Gradio (alternative)
"""
from __future__ import annotations

import logging
import os
from pathlib import Path

import gradio as gr
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field, field_validator

from app.ui import build_ui

from .model import SentimentModel, ensure_model
from .monitoring import Monitor

log = logging.getLogger("sentiment-api")
WEB_PAGE = Path(__file__).resolve().parents[1] / "app" / "web" / "index.html"


class PredictRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=5000,
                      examples=["Ce film est magnifique, j'ai adoré !"])

    @field_validator("text")
    @classmethod
    def not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("le texte est vide")
        return v


class PredictResponse(BaseModel):
    label: str
    confidence: float
    scores: dict[str, float]
    uncertain: bool
    n_tokens: int
    truncated: bool
    latency_ms: float
    model_version: str


def create_app(model_dir: str | None = None, db_path: str | None = None) -> FastAPI:
    model_dir = model_dir or os.getenv("MODEL_DIR", "models")
    db_path = db_path or os.getenv("MONITOR_DB", "data/monitoring.db")

    model = SentimentModel(ensure_model(model_dir))
    monitor = Monitor(db_path, reference=model.reference)

    app = FastAPI(
        title="Sentiment API · DistilBERT",
        description="Prédit si une critique de film en français est positive ou négative "
                    "(DistilBERT multilingue fine-tuné sur Allociné). Projet *From Training to Deployment*.",
        version=model.version,
    )

    def run_prediction(text: str, source: str) -> dict:
        """Même fonction pour l'API et l'interface : même prétraitement, même logs."""
        try:
            res = model.predict(text)
        except Exception:
            monitor.log_error(source)
            log.exception("Erreur d'inférence")
            raise
        monitor.log(res, source)
        return res

    @app.get("/", include_in_schema=False)
    def home():
        return FileResponse(WEB_PAGE, media_type="text/html", headers={"Cache-Control": "no-cache"})

    @app.get("/health")
    def health():
        return {"status": "ok", "model_version": model.version, "labels": model.labels,
                "examples": model.examples}

    @app.post("/predict", response_model=PredictResponse)
    def predict(req: PredictRequest):
        try:
            return run_prediction(req.text, "api")
        except Exception:
            raise HTTPException(status_code=500, detail="Erreur pendant l'inférence")

    @app.get("/stats")
    def stats():
        return monitor.stats()

    ui = build_ui(lambda t: run_prediction(t, "ui"), monitor.stats, model.examples, model.version)
    return gr.mount_gradio_app(app, ui, path="/ui")
