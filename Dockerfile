# ============================================================
#  KIbalione8 – IA Agent (ERT, Voice, RAG, Geosciences)
# ============================================================

# Image de base compatible GPU/CPU
FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

# Empêche Python de générer des fichiers .pyc et force le buffer stdout
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Configure le répertoire de travail
WORKDIR /app

# Installe dépendances système de base
RUN apt-get update && apt-get install -y \
    git wget curl build-essential ffmpeg libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Copie les fichiers de dépendances
COPY requirements.txt /app/

# Installe Python et les dépendances IA
RUN apt-get install -y python3 python3-pip && \
    pip install --no-cache-dir -r requirements.txt

# Copie tout ton code dans le conteneur
COPY . /app/

# Expose le port de ton interface web (FastAPI / Streamlit)
EXPOSE 7860

# Définis la commande de lancement par défaut
# ⚠️ adapte la ligne selon ton script principal :
# Exemple Streamlit :
# CMD ["streamlit", "run", "ERT.py", "--server.port=7860", "--server.address=0.0.0.0"]

# Exemple FastAPI :
CMD ["python3", "kibalione8.py"]
