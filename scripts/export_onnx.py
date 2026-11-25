"""Exporte un modèle de sentiment Hugging Face vers le format attendu par l'API.

Modèle provisoire (en attendant celui du groupe) : DistilBERT multilingue cased déjà
entraîné en sentiment (positive / neutral / negative). Même architecture et même tokenizer
que notre modèle final, donc l'export se passe exactement pareil :
    python scripts/export_onnx.py

Avec le modèle fine-tuné sur Allociné par la personne 2 :
    python scripts/export_onnx.py --checkpoint chemin/vers/checkpoint --version v1.0 --quantize

Autre modèle provisoire possible, entraîné en français (Allociné + Amazon), 5 classes d'étoiles :
    python scripts/export_onnx.py --checkpoint cmarkea/distilcamembert-base-sentiment

À lancer sur Colab ou sur ton Mac (besoin de torch + transformers + onnx + onnxruntime) :
    pip install -r requirements-export.txt

Produit dans models/ :
    model.onnx, tokenizer.json, config.json   (+ vérifie que ONNX donne les mêmes résultats que PyTorch)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

PLACEHOLDER = "lxyuan/distilbert-base-multilingual-cased-sentiments-student"
DEFAULT_EXAMPLES = [
    "Ce film est magnifique, j'ai adoré ! Les acteurs sont excellents.",
    "Une histoire décevante et un film ennuyeux. J'ai failli partir avant la fin.",
    "Pas mal du tout, je ne m'attendais pas à passer un aussi bon moment.",
    "Génial, encore un remake inutile. Bravo, vraiment.",
    "Le début est lent, mais la deuxième moitié rattrape largement le tout.",
]
# Statistiques de la personne 1 sur le train nettoyé (tokenizer multilingue cased)
TRAIN_MEAN_TOKENS = 147.45


class LogitsOnly(torch.nn.Module):
    """Petit wrapper pour que le graphe ONNX ne sorte que les logits."""

    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, input_ids, attention_mask):
        return self.m(input_ids=input_ids, attention_mask=attention_mask).logits


# Dossier de sortie fixe : c'est là que l'API va chercher le modèle.
# (Pas d'argument --out : un chemin venant de la ligne de commande = risque de path traversal.)
MODELS_DIR = Path(__file__).resolve().parent.parent / "models"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default=PLACEHOLDER, help="nom HF ou dossier local")
    p.add_argument("--version", default="v0-placeholder")
    p.add_argument("--labels", default=None, help='ex: "négatif,positif" (sinon id2label du modèle)')
    p.add_argument("--max-length", type=int, default=512, help="doit être celui de l'entraînement")
    p.add_argument("--quantize", action="store_true", help="quantization dynamique INT8")
    a = p.parse_args()

    out = MODELS_DIR
    out.mkdir(parents=True, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(a.checkpoint)
    model = AutoModelForSequenceClassification.from_pretrained(a.checkpoint).eval()

    # 1) Tokenizer -> tokenizer.json (lu par la librairie légère `tokenizers` côté API)
    tok.backend_tokenizer.save(str(out / "tokenizer.json"))

    # 2) Export ONNX
    dummy = tok("export example", return_tensors="pt")
    export_kwargs = {
        "input_names": ["input_ids", "attention_mask"],
        "output_names": ["logits"],
        "dynamic_axes": {"input_ids": {0: "batch", 1: "seq"},
                         "attention_mask": {0: "batch", 1: "seq"},
                         "logits": {0: "batch"}},
        "opset_version": 17,
    }
    onnx_path = out / "model.onnx"
    wrapped = LogitsOnly(model)
    args = (dummy["input_ids"], dummy["attention_mask"])
    try:
        torch.onnx.export(wrapped, args, str(onnx_path), dynamo=False, **export_kwargs)
    except TypeError:  # anciennes versions de torch sans l'argument `dynamo`
        torch.onnx.export(wrapped, args, str(onnx_path), **export_kwargs)

    # 3) Quantization INT8 (optionnelle) : modèle ~4x plus petit
    if a.quantize:
        from onnxruntime.quantization import QuantType, quantize_dynamic

        fp32 = out / "model_fp32.onnx"
        onnx_path.rename(fp32)
        quantize_dynamic(str(fp32), str(onnx_path), weight_type=QuantType.QInt8)
        print(f"Taille FP32 : {fp32.stat().st_size / 1e6:.1f} Mo")
        fp32.unlink()
    print(f"Taille model.onnx : {onnx_path.stat().st_size / 1e6:.1f} Mo")

    # 4) Vérification : PyTorch et ONNX doivent donner (presque) la même chose
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    for text in DEFAULT_EXAMPLES[:3]:
        enc = tok(text, return_tensors="pt")
        with torch.no_grad():
            ref = model(**enc).logits.numpy()
        got = sess.run(None, {"input_ids": enc["input_ids"].numpy(),
                              "attention_mask": enc["attention_mask"].numpy()})[0]
        same = ref.argmax() == got.argmax()
        print(f"[{'OK' if same else 'DIFF'}] écart max {np.abs(ref - got).max():.4f} · {text[:50]}")

    # 5) config.json
    if a.labels:
        labels = a.labels.split(",")
    else:
        labels = [model.config.id2label[i] for i in range(model.config.num_labels)]
        # Si l'entraînement n'a pas défini id2label : LABEL_0 / LABEL_1 -> nos classes Allociné
        if labels == ["LABEL_0", "LABEL_1"]:
            labels = ["négatif", "positif"]
    cfg = {
        "labels": [l.strip() for l in labels],
        "max_length": a.max_length,
        "version": a.version,
        "examples": DEFAULT_EXAMPLES,
        # Sert à détecter le drift. La personne 3 ajoute "mean_confidence" = confiance
        # moyenne du modèle sur le jeu de test.
        "reference": {"mean_tokens": TRAIN_MEAN_TOKENS},
    }
    (out / "config.json").write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Modèle prêt dans {out.resolve()}")


if __name__ == "__main__":
    main()
