"""Chargement du modèle ONNX + tokenizer, et prédiction.

Le code ne dépend pas d'un modèle précis : il lit seulement 3 fichiers
dans le dossier du modèle (le "contrat" avec la personne 3) :

    model.onnx      le réseau exporté (entrées: input_ids, attention_mask ; sortie: logits)
    tokenizer.json  le tokenizer qui transforme le texte en nombres
    config.json     {"labels": ["négatif", "positif"], "max_length": 512, "version": "...", ...}
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

REQUIRED_FILES = ("model.onnx", "tokenizer.json", "config.json")
LOW_CONFIDENCE = 0.60  # en dessous, on considère que le modèle "hésite"


def ensure_model(model_dir: str | Path) -> Path:
    """Vérifie que le modèle est là. Sinon, le télécharge depuis le Hugging Face Hub
    si la variable HF_MODEL_REPO est définie (option "registre de modèles")."""
    d = Path(model_dir)
    if all((d / f).exists() for f in REQUIRED_FILES):
        return d
    repo = os.getenv("HF_MODEL_REPO")
    if not repo:
        missing = [f for f in REQUIRED_FILES if not (d / f).exists()]
        raise FileNotFoundError(
            f"Fichiers manquants dans {d.resolve()}: {missing}. "
            "Lance scripts/export_model.py ou définis HF_MODEL_REPO."
        )
    from huggingface_hub import snapshot_download

    snapshot_download(
        repo_id=repo,
        revision=os.getenv("HF_MODEL_REVISION") or None,  # = version du modèle
        local_dir=d,
        allow_patterns=list(REQUIRED_FILES),
    )
    return d


def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max())
    return e / e.sum()


class SentimentModel:
    def __init__(self, model_dir: str | Path):
        d = Path(model_dir)
        cfg = json.loads((d / "config.json").read_text(encoding="utf-8"))
        self.labels: list[str] = cfg["labels"]
        self.max_length: int = int(cfg.get("max_length", 512))
        self.version: str = cfg.get("version", "inconnue")
        self.examples: list[str] = cfg.get("examples", [])
        self.reference: dict = cfg.get("reference", {})  # stats du jeu de test (pour le drift)

        self.tokenizer = Tokenizer.from_file(str(d / "tokenizer.json"))
        # Un seul texte à la fois : pas de padding à 512 (inutile, et beaucoup plus lent).
        # Le résultat est identique grâce à l'attention_mask.
        self.tokenizer.no_padding()
        self.tokenizer.enable_truncation(max_length=self.max_length)

        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(
            str(d / "model.onnx"), sess_options=opts, providers=["CPUExecutionProvider"]
        )
        self.input_names = {i.name for i in self.session.get_inputs()}
        # Petit "warm-up" : la 1re inférence est toujours plus lente
        self.predict("warm up")

    def predict(self, text: str) -> dict:
        t0 = time.perf_counter()

        # 1) Texte -> nombres (même tokenizer qu'à l'entraînement)
        enc = self.tokenizer.encode(text)
        feeds = {
            "input_ids": np.array([enc.ids], dtype=np.int64),
            "attention_mask": np.array([enc.attention_mask], dtype=np.int64),
            "token_type_ids": np.zeros((1, len(enc.ids)), dtype=np.int64),
        }
        feeds = {k: v for k, v in feeds.items() if k in self.input_names}

        # 2) Nombres -> scores bruts (logits) -> probabilités
        logits = self.session.run(None, feeds)[0][0]
        probs = _softmax(logits.astype(np.float64))
        best = int(probs.argmax())

        return {
            "label": self.labels[best],
            "confidence": round(float(probs[best]), 4),
            "scores": {lab: round(float(p), 4) for lab, p in zip(self.labels, probs)},
            "uncertain": bool(probs[best] < LOW_CONFIDENCE),
            "n_tokens": len(enc.ids),
            "truncated": len(enc.overflowing) > 0,
            "latency_ms": round((time.perf_counter() - t0) * 1000, 2),
            "model_version": self.version,
        }
