# Image de production : pas de PyTorch, seulement ONNX Runtime (image beaucoup plus légère)
# Build depuis la racine du repo :  docker build -f docker/Dockerfile -t sentiment-api .
FROM python:3.12-slim

# Hugging Face Spaces lance le conteneur avec l'utilisateur 1000
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    MODEL_DIR=/home/user/app/models \
    MONITOR_DB=/home/user/app/data/monitoring.db
WORKDIR /home/user/app

# 1) Dépendances d'abord (mises en cache tant que requirements-api.txt ne change pas)
COPY --chown=user requirements-api.txt .
RUN pip install --no-cache-dir --user -r requirements-api.txt

# 2) Puis le code et le modèle
COPY --chown=user api ./api
COPY --chown=user app ./app
COPY --chown=user models ./models

EXPOSE 7860
HEALTHCHECK --interval=30s --timeout=5s --start-period=90s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:7860/health')" || exit 1

CMD ["uvicorn", "api.main:create_app", "--factory", "--host", "0.0.0.0", "--port", "7860"]
