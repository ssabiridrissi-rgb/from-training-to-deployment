"""Interface Gradio : un onglet pour tester le modèle, un onglet monitoring."""
from __future__ import annotations

from typing import Callable

import gradio as gr
import pandas as pd


def build_ui(predict_fn: Callable[[str], dict], stats_fn: Callable[[], dict],
             examples: list[str], version: str) -> gr.Blocks:

    def analyse(text: str):
        if not text or not text.strip():
            raise gr.Error("Écris un avis avant de lancer l'analyse.")
        r = predict_fn(text)
        details = (
            f"**Latence :** {r['latency_ms']} ms · **Tokens :** {r['n_tokens']}"
            f"{' (texte tronqué)' if r['truncated'] else ''} · **Modèle :** {r['model_version']}"
        )
        if r["uncertain"]:
            details += "\n\n⚠️ **Le modèle hésite** (confiance < 60 %) : avis ambigu, ironique ou hors sujet ?"
        return r["scores"], details

    def refresh():
        s = stats_fn()
        if s["total_requests"] == 0:
            empty = pd.DataFrame({"requete": [], "valeur": []})
            return "Aucune requête pour l'instant.", empty, empty
        lines = [
            "| Indicateur | Valeur |", "|---|---|",
            f"| Requêtes | {s['total_requests']} |",
            f"| Taux d'erreur | {s['error_rate']:.1%} |",
        ]
        if "latency_ms" in s:
            lines += [
                f"| Latence p50 / p95 | {s['latency_ms']['p50']} / {s['latency_ms']['p95']} ms |",
                f"| Confiance moyenne | {s['mean_confidence']:.1%} |",
                f"| Prédictions incertaines | {s['low_confidence_rate']:.1%} |",
                f"| Répartition | {s['label_distribution']} |",
            ]
        lines.append(f"| Data drift | {s['drift']['status']} |")
        rec = s.get("recent", {"latency_ms": [], "confidence": []})
        lat = pd.DataFrame({"requete": range(1, len(rec["latency_ms"]) + 1), "valeur": rec["latency_ms"]})
        conf = pd.DataFrame({"requete": range(1, len(rec["confidence"]) + 1), "valeur": rec["confidence"]})
        return "\n".join(lines), lat, conf

    with gr.Blocks(title="Analyse de sentiment · DistilBERT") as demo:
        gr.Markdown(
            "# 💬 Analyse de sentiment avec DistilBERT\n"
            f"Écris une critique de film en français, le modèle dit si elle est **positive** ou **négative**. · version `{version}`"
        )
        with gr.Tab("Analyser une critique"):
            txt = gr.Textbox(label="Critique de film", lines=4, placeholder="Ex : Un scénario captivant et des acteurs excellents.")
            btn = gr.Button("Analyser", variant="primary")
            out_scores = gr.Label(label="Sentiment prédit", num_top_classes=2)
            out_details = gr.Markdown()
            if examples:
                gr.Examples(examples=[[e] for e in examples], inputs=txt, label="Exemples")
            btn.click(analyse, txt, [out_scores, out_details])
            txt.submit(analyse, txt, [out_scores, out_details])

        with gr.Tab("Monitoring"):
            ref_btn = gr.Button("Rafraîchir")
            summary = gr.Markdown()
            with gr.Row():
                lat_plot = gr.LinePlot(x="requete", y="valeur", title="Latence (ms) — 50 dernières requêtes")
                conf_plot = gr.LinePlot(x="requete", y="valeur", title="Confiance — 50 dernières requêtes")
            ref_btn.click(refresh, None, [summary, lat_plot, conf_plot])

    return demo
