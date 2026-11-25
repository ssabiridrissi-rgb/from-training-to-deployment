"""Met en ligne l'application sur un Space Hugging Face (SDK Docker).

Attention : depuis 2026, créer un Space Docker demande un compte Hugging Face PRO.
Pour la démo, on utilise Docker en local + un tunnel Cloudflare (voir docs/deploiement.md).

Un Space attend un Dockerfile et un README.md à sa racine. Ce script prépare donc un
dossier temporaire avec la bonne forme, puis l'envoie :

    Space/
    ├── Dockerfile          <- docker/Dockerfile
    ├── README.md           <- configuration du Space (généré ici)
    ├── requirements-api.txt
    ├── api/  app/
    └── models/             <- le modèle (sauf avec --no-model)

Utilisation :
    pip install huggingface_hub
    export HF_TOKEN=hf_xxx          (token avec le droit "write")
    python scripts/deploy_space.py --space TON_PSEUDO/TON_SPACE

    --no-model : n'envoie pas le modèle. Le Space le télécharge alors au démarrage
                 depuis le dépôt de modèles indiqué dans la variable HF_MODEL_REPO du Space.
"""
from __future__ import annotations

import argparse
import os
import shutil
import tempfile
from pathlib import Path

from huggingface_hub import HfApi

ROOT = Path(__file__).resolve().parents[1]
MODEL_FILES = ("model.onnx", "tokenizer.json", "config.json")

SPACE_README = """---
title: Analyse de sentiment Allociné
emoji: 🎬
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
---

# From Training to Deployment

Critique de film en français → sentiment positif ou négatif.
DistilBERT multilingue fine-tuné sur Allociné, servi avec FastAPI + ONNX Runtime + Gradio.

- Interface : `/ui/`
- API : `POST /predict` · documentation : `/docs` · monitoring : `/stats`
"""


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--space", default=os.getenv("HF_SPACE"), help="ex: pseudo/sentiment-allocine")
    p.add_argument("--no-model", action="store_true")
    a = p.parse_args()
    if not a.space:
        raise SystemExit("Indique le Space : --space pseudo/nom (ou variable HF_SPACE)")
    token = os.getenv("HF_TOKEN")
    if not token:
        raise SystemExit("Variable HF_TOKEN manquante (token Hugging Face avec le droit write)")

    with tempfile.TemporaryDirectory() as tmp:
        dst = Path(tmp)
        shutil.copy(ROOT / "docker" / "Dockerfile", dst / "Dockerfile")
        shutil.copy(ROOT / "requirements-api.txt", dst / "requirements-api.txt")
        (dst / "README.md").write_text(SPACE_README, encoding="utf-8")
        ignore = shutil.ignore_patterns("__pycache__", "*.pyc")
        shutil.copytree(ROOT / "api", dst / "api", ignore=ignore)
        shutil.copytree(ROOT / "app", dst / "app", ignore=ignore)

        models = dst / "models"
        models.mkdir()
        (models / ".gitkeep").touch()
        if not a.no_model:
            missing = [f for f in MODEL_FILES if not (ROOT / "models" / f).exists()]
            if missing:
                raise SystemExit(f"Fichiers manquants dans models/: {missing}. "
                                 "Lance d'abord scripts/export_onnx.py (ou utilise --no-model).")
            for f in MODEL_FILES:
                shutil.copy(ROOT / "models" / f, models / f)
            size = (models / "model.onnx").stat().st_size / 1e6
            print(f"Envoi du modèle ({size:.0f} Mo), ça peut prendre quelques minutes...")

        api = HfApi(token=token)
        api.create_repo(a.space, repo_type="space", space_sdk="docker", exist_ok=True)
        commit = api.upload_folder(folder_path=str(dst), repo_id=a.space, repo_type="space",
                                   commit_message=f"Deploy {os.getenv('GITHUB_SHA', 'local')[:7]}")
        print("Envoyé :", commit.commit_url)
        url = "https://" + a.space.replace("/", "-").replace("_", "-").replace(".", "-").lower() + ".hf.space"
        print(f"Application : {url}  (build : quelques minutes)")


if __name__ == "__main__":
    main()
