#!/usr/bin/env python3
"""
Script de téléchargement et vérification des modèles IA
Pour KIbalione8 - Système d'analyse ERT avancé
"""

import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple
import json

def print_section(title: str):
    """Affiche une section"""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}\n")

def check_disk_space() -> Tuple[int, int]:
    """Vérifie l'espace disque disponible"""
    import shutil
    stat = shutil.disk_usage(Path.home())
    free_gb = stat.free / (1024**3)
    total_gb = stat.total / (1024**3)
    return free_gb, total_gb

def download_embedding_models():
    """Télécharge les modèles d'embedding"""
    print_section("1. Modèles d'Embedding")
    
    models = [
        "sentence-transformers/all-MiniLM-L6-v2",
        "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
    ]
    
    try:
        from sentence_transformers import SentenceTransformer
        
        for model_name in models:
            print(f"📥 Téléchargement: {model_name}...")
            model = SentenceTransformer(model_name)
            print(f"✅ {model_name} téléchargé")
            
    except Exception as e:
        print(f"❌ Erreur: {e}")
        return False
    
    return True

def download_whisper_models():
    """Télécharge les modèles Whisper"""
    print_section("2. Modèles Whisper (Speech-to-Text)")
    
    models = ["tiny", "base", "small"]  # Modèles légers
    
    try:
        import whisper
        
        for model_name in models:
            print(f"📥 Téléchargement Whisper '{model_name}'...")
            model = whisper.load_model(model_name)
            print(f"✅ Whisper '{model_name}' téléchargé")
            
    except Exception as e:
        print(f"❌ Erreur: {e}")
        print("⚠️  Installation de Whisper requise: pip install openai-whisper")
        return False
    
    return True

def download_tts_models():
    """Télécharge les modèles TTS"""
    print_section("3. Modèles TTS (Text-to-Speech)")
    
    try:
        from TTS.api import TTS
        
        # Modèle français léger
        model_name = "tts_models/fr/mai/tacotron2-DDC"
        
        print(f"📥 Téléchargement TTS: {model_name}...")
        tts = TTS(model_name)
        print(f"✅ TTS '{model_name}' téléchargé")
        
    except Exception as e:
        print(f"❌ Erreur: {e}")
        print("⚠️  Installation de TTS requise: pip install TTS")
        return False
    
    return True

def verify_llm_access():
    """Vérifie l'accès aux modèles LLM via HuggingFace"""
    print_section("4. Modèles LLM (Large Language Models)")
    
    from dotenv import load_dotenv
    load_dotenv()
    
    hf_token = os.getenv("HF_TOKEN")
    
    if not hf_token:
        print("❌ HF_TOKEN non configuré dans .env")
        print("   Obtenez un token: https://huggingface.co/settings/tokens")
        return False
    
    print(f"✅ Token HuggingFace configuré: {hf_token[:10]}...")
    
    # Test d'accès
    try:
        from huggingface_hub import HfApi
        api = HfApi(token=hf_token)
        
        # Vérifier l'accès à un modèle public
        test_model = "Qwen/Qwen2.5-7B-Instruct"
        print(f"🔍 Vérification accès à {test_model}...")
        
        model_info = api.model_info(test_model)
        print(f"✅ Accès confirmé: {test_model}")
        print(f"   Taille: ~{model_info.safetensors['total']/1e9:.1f}GB")
        
    except Exception as e:
        print(f"⚠️  Impossible de vérifier l'accès: {e}")
        print("   Le modèle sera téléchargé au premier usage")
    
    return True

def verify_tavily_access():
    """Vérifie l'accès à Tavily API"""
    print_section("5. Tavily API (Recherche Web)")
    
    from dotenv import load_dotenv
    load_dotenv()
    
    tavily_key = os.getenv("TAVILY_API_KEY")
    
    if not tavily_key:
        print("❌ TAVILY_API_KEY non configuré dans .env")
        print("   Obtenez une clé: https://tavily.com")
        return False
    
    print(f"✅ Clé Tavily configurée: {tavily_key[:10]}...")
    
    # Test de connexion
    try:
        from tavily import TavilyClient
        client = TavilyClient(api_key=tavily_key)
        
        print("🔍 Test de recherche...")
        result = client.search("test", max_results=1)
        print("✅ Connexion Tavily fonctionnelle")
        
    except Exception as e:
        print(f"⚠️  Erreur de connexion: {e}")
        print("   Vérifiez votre clé API")
        return False
    
    return True

def check_dependencies():
    """Vérifie les dépendances critiques"""
    print_section("6. Vérification des Dépendances")
    
    critical_packages = {
        "torch": "PyTorch",
        "transformers": "Transformers",
        "langchain": "LangChain",
        "streamlit": "Streamlit",
        "numpy": "NumPy",
        "pandas": "Pandas",
        "matplotlib": "Matplotlib",
        "faiss": "FAISS (CPU)",
        "sentence_transformers": "Sentence Transformers",
    }
    
    optional_packages = {
        "whisper": "Whisper (Voice)",
        "TTS": "Coqui TTS (Voice)",
        "pygimli": "PyGIMLi (Geophysics)",
        "pyres": "PyRes (ERT)",
    }
    
    print("📦 Packages critiques:")
    critical_ok = True
    for package, name in critical_packages.items():
        try:
            __import__(package)
            print(f"  ✅ {name}")
        except ImportError:
            print(f"  ❌ {name} - MANQUANT")
            critical_ok = False
    
    print("\n📦 Packages optionnels:")
    for package, name in optional_packages.items():
        try:
            __import__(package)
            print(f"  ✅ {name}")
        except ImportError:
            print(f"  ⚠️  {name} - Non installé (optionnel)")
    
    return critical_ok

def create_model_registry():
    """Crée un registre des modèles disponibles"""
    print_section("7. Création du Registre des Modèles")
    
    registry = {
        "embeddings": {
            "all-MiniLM-L6-v2": {
                "model_id": "sentence-transformers/all-MiniLM-L6-v2",
                "size_mb": 90,
                "languages": ["en"],
                "use_case": "embeddings_fast"
            },
            "paraphrase-multilingual": {
                "model_id": "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
                "size_mb": 420,
                "languages": ["multilingual"],
                "use_case": "embeddings_multilingual"
            }
        },
        "llm": {
            "qwen-7b": {
                "model_id": "Qwen/Qwen2.5-7B-Instruct",
                "size_gb": 4.2,
                "context_length": 8192,
                "use_case": "general_purpose"
            },
            "gemma-2b": {
                "model_id": "google/gemma-2-2b-it",
                "size_gb": 2.5,
                "context_length": 8192,
                "use_case": "lightweight"
            },
            "deepseek-v3": {
                "model_id": "deepseek-ai/DeepSeek-V3-0324",
                "size_gb": 14.0,
                "context_length": 32768,
                "use_case": "advanced_reasoning"
            }
        },
        "voice": {
            "whisper-base": {
                "model_id": "openai/whisper-base",
                "size_mb": 150,
                "use_case": "speech_to_text"
            },
            "tts-fr": {
                "model_id": "tts_models/fr/mai/tacotron2-DDC",
                "size_mb": 250,
                "use_case": "text_to_speech_french"
            }
        }
    }
    
    registry_path = Path("local_models_paths.json")
    with open(registry_path, 'w', encoding='utf-8') as f:
        json.dump(registry, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Registre créé: {registry_path}")
    print(f"   - {len(registry['embeddings'])} modèles d'embedding")
    print(f"   - {len(registry['llm'])} modèles LLM")
    print(f"   - {len(registry['voice'])} modèles vocaux")
    
    return True

def display_summary(results: Dict[str, bool]):
    """Affiche un résumé de l'installation"""
    print_section("Résumé de l'Installation")
    
    total = len(results)
    success = sum(results.values())
    
    print(f"📊 Taux de succès: {success}/{total} ({success/total*100:.0f}%)")
    print("")
    
    for step, status in results.items():
        icon = "✅" if status else "❌"
        print(f"  {icon} {step}")
    
    print("")
    
    if success == total:
        print("🎉 Installation complète réussie!")
        print("")
        print("📋 Prochaines étapes:")
        print("  1. Configurez vos tokens API dans .env")
        print("  2. Lancez l'application: streamlit run kibalione8.py")
    else:
        print("⚠️  Installation partielle. Vérifiez les erreurs ci-dessus.")
        print("")
        print("💡 Conseils:")
        print("  - Installez les packages manquants: pip install <package>")
        print("  - Vérifiez votre connexion internet")
        print("  - Configurez correctement le fichier .env")

def main():
    """Fonction principale"""
    print("🚀 KIbalione8 - Téléchargement et Vérification des Modèles")
    
    # Vérifier l'espace disque
    free_gb, total_gb = check_disk_space()
    print(f"💾 Espace disque: {free_gb:.1f}GB libres / {total_gb:.1f}GB total")
    
    if free_gb < 15:
        print("⚠️  Espace disque faible! Au moins 15GB recommandés.")
        response = input("Continuer quand même? (y/N): ")
        if response.lower() != 'y':
            print("❌ Installation annulée")
            return
    
    # Exécuter les étapes
    results = {}
    
    results["Dépendances"] = check_dependencies()
    
    if results["Dépendances"]:
        results["Embeddings"] = download_embedding_models()
        results["Whisper"] = download_whisper_models()
        results["TTS"] = download_tts_models()
        results["LLM Access"] = verify_llm_access()
        results["Tavily API"] = verify_tavily_access()
        results["Registre"] = create_model_registry()
    else:
        print("❌ Dépendances critiques manquantes. Installation arrêtée.")
        print("💡 Exécutez d'abord: pip install -r requirements_complete.txt")
        return
    
    # Afficher le résumé
    display_summary(results)

if __name__ == "__main__":
    main()
