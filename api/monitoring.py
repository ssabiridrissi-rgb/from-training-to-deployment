"""Monitoring minimal : chaque requête est enregistrée dans SQLite.

On ne stocke PAS le texte des utilisateurs (vie privée), seulement des
métadonnées : label, confiance, latence, nombre de tokens.
"""
from __future__ import annotations

import sqlite3
import time
from collections import Counter
from contextlib import closing
from pathlib import Path

import numpy as np

from .model import LOW_CONFIDENCE

DRIFT_MIN_REQUESTS = 20   # pas d'alerte drift avant d'avoir assez de requêtes
DRIFT_WINDOW = 50         # on compare les 50 dernières requêtes à la référence
DRIFT_CONF_DROP = 0.10    # alerte si la confiance moyenne baisse de plus de 10 points
DRIFT_LEN_RATIO = (0.5, 2.0)  # alerte si les textes sont 2x plus courts / plus longs qu'à l'entraînement


class Monitor:
    def __init__(self, db_path: str | Path, reference: dict | None = None):
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self.db_path = str(db_path)
        self.reference = reference or {}
        with closing(self._conn()) as c, c:
            c.execute(
                """CREATE TABLE IF NOT EXISTS requests (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ts REAL, source TEXT, status TEXT,
                    label TEXT, confidence REAL, latency_ms REAL,
                    n_tokens INTEGER, truncated INTEGER, model_version TEXT)"""
            )

    def _conn(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path, timeout=5)

    def log(self, res: dict, source: str) -> None:
        with closing(self._conn()) as c, c:
            c.execute(
                "INSERT INTO requests (ts, source, status, label, confidence, latency_ms,"
                " n_tokens, truncated, model_version) VALUES (?,?,?,?,?,?,?,?,?)",
                (time.time(), source, "ok", res["label"], res["confidence"], res["latency_ms"],
                 res["n_tokens"], int(res["truncated"]), res["model_version"]),
            )

    def log_error(self, source: str) -> None:
        with closing(self._conn()) as c, c:
            c.execute("INSERT INTO requests (ts, source, status) VALUES (?,?,?)",
                      (time.time(), source, "error"))

    def stats(self) -> dict:
        with closing(self._conn()) as c:
            rows = c.execute(
                "SELECT status, label, confidence, latency_ms, n_tokens FROM requests ORDER BY id"
            ).fetchall()

        ok = [r for r in rows if r[0] == "ok"]
        out: dict = {
            "total_requests": len(rows),
            "errors": len(rows) - len(ok),
            "error_rate": round((len(rows) - len(ok)) / len(rows), 4) if rows else 0.0,
        }
        if not ok:
            return out | {"drift": {"status": "pas assez de données"}}

        lat = np.array([r[3] for r in ok])
        conf = np.array([r[2] for r in ok])
        toks = np.array([r[4] for r in ok])
        out |= {
            "latency_ms": {"p50": round(float(np.percentile(lat, 50)), 2),
                           "p95": round(float(np.percentile(lat, 95)), 2)},
            "label_distribution": dict(Counter(r[1] for r in ok)),
            "mean_confidence": round(float(conf.mean()), 4),
            "low_confidence_rate": round(float((conf < LOW_CONFIDENCE).mean()), 4),
            "mean_tokens": round(float(toks.mean()), 1),
            "recent": {"latency_ms": lat[-DRIFT_WINDOW:].tolist(),
                       "confidence": conf[-DRIFT_WINDOW:].tolist()},
        }
        out["drift"] = self._drift(conf[-DRIFT_WINDOW:], toks[-DRIFT_WINDOW:])
        return out

    def _drift(self, conf: np.ndarray, toks: np.ndarray) -> dict:
        """Détection de drift SANS labels : on compare ce qu'on voit en production
        aux statistiques du jeu de test (fournies dans config.json -> "reference")."""
        ref_conf = self.reference.get("mean_confidence")
        ref_toks = self.reference.get("mean_tokens")
        if ref_conf is None and ref_toks is None:
            return {"status": "pas de référence (ajouter 'reference' dans config.json)"}
        if len(conf) < DRIFT_MIN_REQUESTS:
            return {"status": f"pas assez de données ({len(conf)}/{DRIFT_MIN_REQUESTS})"}

        info: dict = {}
        alerts: list[str] = []
        # 1) Le modèle est-il moins sûr de lui qu'au moment du test ?
        if ref_conf is not None:
            drop = ref_conf - float(conf.mean())
            info |= {"reference_mean_confidence": ref_conf,
                     "current_mean_confidence": round(float(conf.mean()), 4),
                     "confidence_drop": round(drop, 4)}
            if drop > DRIFT_CONF_DROP:
                alerts.append("confiance en baisse")
        # 2) Les textes reçus ressemblent-ils à ceux d'Allociné (longueur) ?
        if ref_toks is not None:
            ratio = float(toks.mean()) / ref_toks
            info |= {"reference_mean_tokens": ref_toks,
                     "current_mean_tokens": round(float(toks.mean()), 1),
                     "length_ratio": round(ratio, 2)}
            if ratio < DRIFT_LEN_RATIO[0] or ratio > DRIFT_LEN_RATIO[1]:
                alerts.append("textes " + ("beaucoup plus courts" if ratio < 1 else "beaucoup plus longs")
                              + " que les critiques d'entraînement")
        info["alerts"] = alerts
        info["status"] = ("ALERTE drift possible : " + ", ".join(alerts)) if alerts else "ok"
        return info
