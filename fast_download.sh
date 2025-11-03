#!/bin/bash
# Téléchargement ultra-rapide du modèle Qwen2.5-1.5B-Instruct avec aria2

MODEL_ID="Qwen/Qwen2.5-1.5B-Instruct"
# HF_TOKEN doit être défini dans l'environnement ou ~/.bashrc
# Exemple: export HF_TOKEN="your_token_here"
if [ -z "$HF_TOKEN" ]; then
    echo "❌ Erreur: HF_TOKEN non défini. Définissez-le avec: export HF_TOKEN='your_token'"
    exit 1
fi
CACHE_DIR="/root/.cache/huggingface/hub/models--Qwen--Qwen2.5-1.5B-Instruct"

echo "🚀 Téléchargement ultra-rapide de $MODEL_ID avec aria2..."

# Créer le répertoire
mkdir -p "$CACHE_DIR"

# Télécharger tous les fichiers avec aria2 (16 connexions simultanées)
aria2c -x 16 -s 16 -k 1M --continue=true --auto-file-renaming=false \
  --header="Authorization: Bearer $HF_TOKEN" \
  "https://huggingface.co/$MODEL_ID/resolve/main/model.safetensors" \
  -d "$CACHE_DIR" -o model.safetensors

aria2c -x 16 -s 16 -k 1M --continue=true --auto-file-renaming=false \
  --header="Authorization: Bearer $HF_TOKEN" \
  "https://huggingface.co/$MODEL_ID/resolve/main/config.json" \
  -d "$CACHE_DIR" -o config.json

aria2c -x 16 -s 16 -k 1M --continue=true --auto-file-renaming=false \
  --header="Authorization: Bearer $HF_TOKEN" \
  "https://huggingface.co/$MODEL_ID/resolve/main/generation_config.json" \
  -d "$CACHE_DIR" -o generation_config.json

aria2c -x 16 -s 16 -k 1M --continue=true --auto-file-renaming=false \
  --header="Authorization: Bearer $HF_TOKEN" \
  "https://huggingface.co/$MODEL_ID/resolve/main/tokenizer.json" \
  -d "$CACHE_DIR" -o tokenizer.json

aria2c -x 16 -s 16 -k 1M --continue=true --auto-file-renaming=false \
  --header="Authorization: Bearer $HF_TOKEN" \
  "https://huggingface.co/$MODEL_ID/resolve/main/tokenizer_config.json" \
  -d "$CACHE_DIR" -o tokenizer_config.json

aria2c -x 16 -s 16 -k 1M --continue=true --auto-file-renaming=false \
  --header="Authorization: Bearer $HF_TOKEN" \
  "https://huggingface.co/$MODEL_ID/resolve/main/special_tokens_map.json" \
  -d "$CACHE_DIR" -o special_tokens_map.json

echo "✅ Téléchargement terminé! Modèle prêt à utiliser."