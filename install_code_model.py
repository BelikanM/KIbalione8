#!/usr/bin/env python3
"""
Script CLI pour installer le modèle de code DeepSeek-Coder-1.3B
Utilisé par Kibali en Mode Code Expert
"""

import os
import sys

def install_code_model():
    """Installe DeepSeek-Coder-1.3B-Instruct"""
    try:
        print("🚀 Installation de DeepSeek-Coder-1.3B-Instruct...")
        print("📦 Téléchargement en cours (~1.3GB)...\n")
        
        # Désactiver TensorFlow pour éviter les imports lents
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
        os.environ['TRANSFORMERS_NO_TF'] = '1'
        
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch
        
        model_name = "deepseek-ai/deepseek-coder-1.3b-instruct"
        cache_dir = "/root/.cache/huggingface/code_models"
        
        print(f"📍 Modèle: {model_name}")
        print(f"📂 Cache: {cache_dir}\n")
        
        # Vérifier si déjà installé
        if os.path.exists(cache_dir) and os.listdir(cache_dir):
            print("⚠️  Le modèle semble déjà installé dans le cache.")
            response = input("Voulez-vous le retélécharger ? (o/N): ")
            if response.lower() not in ['o', 'oui', 'y', 'yes']:
                print("✅ Installation annulée")
                return True
        
        # Créer le dossier cache
        os.makedirs(cache_dir, exist_ok=True)
        
        # Télécharger tokenizer
        print("1️⃣ Téléchargement du tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            model_name, 
            cache_dir=cache_dir,
            trust_remote_code=True
        )
        print("   ✅ Tokenizer téléchargé\n")
        
        # Télécharger modèle
        print("2️⃣ Téléchargement du modèle (cela peut prendre 5-10 minutes)...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"   🖥️  Device: {device}")
        
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
            trust_remote_code=False,  # Changed: no custom code
            low_cpu_mem_usage=True,
            use_safetensors=True  # Use safetensors for faster loading
        )
        print("   ✅ Modèle téléchargé\n")
        
        # Test rapide
        print("3️⃣ Test du modèle...")
        test_prompt = "def fibonacci(n):"
        inputs = tokenizer(test_prompt, return_tensors="pt")
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=50,
                temperature=0.7,
                do_sample=True
            )
        
        generated_code = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"   Prompt: {test_prompt}")
        print(f"   Généré: {generated_code[:100]}...")
        print("   ✅ Test réussi\n")
        
        # Afficher la taille
        total_size = 0
        for root, dirs, files in os.walk(cache_dir):
            for file in files:
                filepath = os.path.join(root, file)
                total_size += os.path.getsize(filepath)
        
        size_mb = total_size / (1024 * 1024)
        print(f"📊 Taille totale du cache: {size_mb:.1f} MB")
        print(f"📍 Emplacement: {cache_dir}")
        print("\n✅ ✨ Installation terminée avec succès ! ✨")
        print("\n🚀 Utilisation:")
        print("   1. Ouvrez Kibali dans Streamlit")
        print("   2. Sélectionnez 'Mode Code Expert'")
        print("   3. Le modèle sera automatiquement utilisé pour générer du code\n")
        
        return True
        
    except ImportError as e:
        print(f"\n❌ Erreur: Module manquant")
        print(f"   {e}")
        print("\n💡 Solution:")
        print("   pip install transformers torch accelerate")
        return False
        
    except Exception as e:
        print(f"\n❌ Erreur lors de l'installation:")
        print(f"   {e}")
        print("\n💡 Suggestions:")
        print("   • Vérifiez votre connexion Internet")
        print("   • Vérifiez que vous avez ~2GB d'espace disque libre")
        print("   • Réessayez avec: python install_code_model.py")
        return False

if __name__ == '__main__':
    print("=" * 60)
    print("  INSTALLATION DEEPSEEK-CODER-1.3B POUR KIBALI")
    print("=" * 60)
    print()
    
    success = install_code_model()
    
    sys.exit(0 if success else 1)
