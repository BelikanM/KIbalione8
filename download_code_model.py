#!/usr/bin/env python3
"""
Script simplifié pour télécharger DeepSeek-Coder-1.3B
Télécharge uniquement sans charger en mémoire
"""

import os
from huggingface_hub import snapshot_download

def download_model():
    """Télécharge le modèle sans le charger"""
    model_name = "deepseek-ai/deepseek-coder-1.3b-instruct"
    cache_dir = "/root/.cache/huggingface/code_models"
    
    print("=" * 60)
    print("  TÉLÉCHARGEMENT DEEPSEEK-CODER-1.3B")
    print("=" * 60)
    print(f"\n📦 Modèle: {model_name}")
    print(f"📂 Cache: {cache_dir}")
    print(f"📏 Taille: ~1.3 GB\n")
    
    try:
        print("🚀 Téléchargement en cours...\n")
        
        path = snapshot_download(
            repo_id=model_name,
            cache_dir=cache_dir,
            resume_download=True,
            local_files_only=False
        )
        
        print(f"\n✅ ✨ Téléchargement terminé ! ✨")
        print(f"📍 Emplacement: {path}\n")
        
        # Afficher la taille
        total_size = 0
        for root, dirs, files in os.walk(cache_dir):
            for file in files:
                filepath = os.path.join(root, file)
                total_size += os.path.getsize(filepath)
        
        size_mb = total_size / (1024 * 1024)
        size_gb = size_mb / 1024
        print(f"📊 Taille: {size_gb:.2f} GB ({size_mb:.0f} MB)")
        print(f"\n🚀 Le modèle sera automatiquement utilisé en Mode Code Expert de Kibali\n")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Erreur: {e}\n")
        return False

if __name__ == '__main__':
    success = download_model()
    exit(0 if success else 1)
