#!/bin/bash
# Script de téléchargement rapide du modèle Qwen2.5-1.5B-Instruct

MODEL_NAME="Qwen/Qwen2.5-1.5B-Instruct"
CACHE_DIR="/root/.cache/huggingface/hub"
MODEL_DIR="$CACHE_DIR/models--Qwen--Qwen2.5-1.5B-Instruct"

echo "🔄 Téléchargement du modèle $MODEL_NAME avec aria2..."

# Créer le répertoire si nécessaire
mkdir -p "$MODEL_DIR"

# Télécharger le modèle avec aria2 (beaucoup plus rapide)
cd "$CACHE_DIR" || exit 1

# Utiliser huggingface-cli avec hf-transfer activé
export HF_HUB_ENABLE_HF_TRANSFER=1
# HF_TOKEN doit être défini dans l'environnement ou ~/.bashrc
# export HF_TOKEN="your_token_here"

/root/miniconda3/envs/ert_clean/bin/huggingface-cli download "$MODEL_NAME" \
    --local-dir "$MODEL_DIR" \
    --local-dir-use-symlinks False \
    --resume-download

echo "✅ Téléchargement terminé!"