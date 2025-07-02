# Étape 1 : base Python avec CUDA compatible (si tu veux le GPU)
FROM python:3.12-slim

# Étape 2 : variables d’environnement
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    OLLAMA_HOST=http://host.docker.internal:11434

# Étape 3 : création d’un utilisateur propre et répertoires
WORKDIR /app
COPY . /app

# Étape 4 : installation de uv (gestionnaire moderne)
RUN pip install --upgrade pip && \
    pip install uv

# Étape 5 : installation des dépendances via uv
RUN uv pip install --system .
# Étape 6 : port Gradio (et données localisées dans ./data)
EXPOSE 7860

# Étape 7 : point d’entrée
CMD ["python", "gradio_pizza.py"]
