#!/usr/bin/env python3
"""
Installation des modèles vocaux pour Kibali
Whisper (transcription) + Coqui TTS (synthèse)
"""

import subprocess
import sys
import os

def run_command(cmd, description):
    """Exécute une commande shell"""
    print(f"\n{'='*60}")
    print(f"📦 {description}")
    print(f"{'='*60}")
    
    try:
        result = subprocess.run(
            cmd,
            shell=True,
            check=True,
            text=True,
            capture_output=False
        )
        print(f"✅ {description} - SUCCÈS")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} - ERREUR: {e}")
        return False

def install_voice_dependencies():
    """Installe les dépendances pour le système vocal"""
    
    print("""
╔═══════════════════════════════════════════════════════════╗
║        INSTALLATION SYSTÈME VOCAL KIBALI                  ║
║  Whisper (transcription) + Coqui TTS (synthèse)          ║
║  Taille totale: ~1.5GB                                   ║
╚═══════════════════════════════════════════════════════════╝
    """)
    
    # 1. Mise à jour pip
    run_command(
        f"{sys.executable} -m pip install --upgrade pip",
        "Mise à jour de pip"
    )
    
    # 2. Installation de Whisper
    print("\n🎤 WHISPER - Transcription Speech-to-Text")
    print("   Taille: ~150MB (modèle base)")
    
    success = run_command(
        f"{sys.executable} -m pip install -U openai-whisper",
        "Installation de Whisper"
    )
    
    if not success:
        print("⚠️ Essai avec la version GitHub...")
        run_command(
            f"{sys.executable} -m pip install git+https://github.com/openai/whisper.git",
            "Installation Whisper (GitHub)"
        )
    
    # 3. Installation de Coqui TTS
    print("\n🔊 COQUI TTS - Synthèse Text-to-Speech")
    print("   Taille: ~500MB-1GB (modèle français)")
    
    run_command(
        f"{sys.executable} -m pip install TTS",
        "Installation de Coqui TTS"
    )
    
    # 4. Dépendances audio
    print("\n🎵 DÉPENDANCES AUDIO")
    
    dependencies = [
        "soundfile",
        "sounddevice",
        "librosa",
        "pyaudio"
    ]
    
    for dep in dependencies:
        run_command(
            f"{sys.executable} -m pip install {dep}",
            f"Installation de {dep}"
        )
    
    # 5. Installation des dépendances système (si nécessaire)
    print("\n🔧 DÉPENDANCES SYSTÈME")
    
    system_deps = [
        "sudo apt-get update -qq",
        "sudo apt-get install -y -qq ffmpeg libsndfile1 portaudio19-dev"
    ]
    
    for cmd in system_deps:
        run_command(cmd, f"Installation système: {cmd.split()[-1]}")
    
    print("\n" + "="*60)
    print("✅ INSTALLATION TERMINÉE!")
    print("="*60)
    
    return True

def download_models():
    """Télécharge les modèles vocaux"""
    
    print("\n" + "="*60)
    print("📥 TÉLÉCHARGEMENT DES MODÈLES VOCAUX")
    print("="*60)
    
    # Créer le dossier de cache
    cache_dir = "/root/.cache/voice_models"
    os.makedirs(cache_dir, exist_ok=True)
    os.makedirs(os.path.join(cache_dir, "whisper"), exist_ok=True)
    
    # 1. Whisper
    print("\n1️⃣ Téléchargement de Whisper 'base' (~150MB)")
    
    try:
        import whisper
        model = whisper.load_model(
            "base",
            download_root=os.path.join(cache_dir, "whisper")
        )
        print("✅ Whisper 'base' téléchargé et vérifié")
        del model
    except Exception as e:
        print(f"❌ Erreur Whisper: {e}")
    
    # 2. Coqui TTS
    print("\n2️⃣ Téléchargement de Coqui TTS français (~500MB)")
    
    try:
        from TTS.api import TTS
        
        # Modèle français rapide et de qualité
        print("   Modèle: tts_models/fr/mai/tacotron2-DDC")
        tts = TTS(
            model_name="tts_models/fr/mai/tacotron2-DDC",
            progress_bar=True,
            gpu=False
        )
        
        # Test rapide
        test_text = "Test de synthèse vocale."
        test_file = "/tmp/test_tts.wav"
        tts.tts_to_file(text=test_text, file_path=test_file)
        
        if os.path.exists(test_file):
            os.remove(test_file)
            print("✅ Coqui TTS téléchargé et vérifié")
        
        del tts
        
    except Exception as e:
        print(f"❌ Erreur TTS: {e}")
        print("⚠️ Essai d'un modèle alternatif...")
        
        try:
            # Fallback: modèle anglais plus léger
            tts = TTS(
                model_name="tts_models/en/ljspeech/tacotron2-DDC",
                progress_bar=True
            )
            print("✅ TTS alternatif (anglais) installé")
            del tts
        except Exception as e2:
            print(f"❌ Erreur TTS alternatif: {e2}")
    
    print("\n" + "="*60)
    print("✅ TÉLÉCHARGEMENT DES MODÈLES TERMINÉ!")
    print(f"📁 Cache: {cache_dir}")
    print("📊 Taille totale: ~650MB-1.2GB")
    print("="*60)

def verify_installation():
    """Vérifie que tout fonctionne"""
    
    print("\n" + "="*60)
    print("🔍 VÉRIFICATION DE L'INSTALLATION")
    print("="*60)
    
    errors = []
    
    # Test Whisper
    print("\n1️⃣ Test Whisper...")
    try:
        import whisper
        print("   ✅ Whisper importé")
    except ImportError as e:
        print(f"   ❌ Whisper: {e}")
        errors.append("Whisper")
    
    # Test TTS
    print("\n2️⃣ Test Coqui TTS...")
    try:
        from TTS.api import TTS
        print("   ✅ TTS importé")
    except ImportError as e:
        print(f"   ❌ TTS: {e}")
        errors.append("TTS")
    
    # Test audio
    print("\n3️⃣ Test audio...")
    try:
        import soundfile
        import sounddevice
        print("   ✅ Modules audio importés")
    except ImportError as e:
        print(f"   ❌ Audio: {e}")
        errors.append("Audio")
    
    # Test VoiceAgent
    print("\n4️⃣ Test VoiceAgent...")
    try:
        from voice_agent import VoiceAgent
        agent = VoiceAgent()
        print("   ✅ VoiceAgent importé")
    except Exception as e:
        print(f"   ❌ VoiceAgent: {e}")
        errors.append("VoiceAgent")
    
    # Résultat final
    print("\n" + "="*60)
    if not errors:
        print("✅ INSTALLATION COMPLÈTE ET FONCTIONNELLE!")
        print("="*60)
        print("\n💡 Vous pouvez maintenant utiliser:")
        print("   - Transcription vocale avec Whisper")
        print("   - Synthèse vocale avec Coqui TTS")
        print("   - Interface vocale dans Kibali")
        print("\n🚀 Lancez: streamlit run ERT.py")
        return True
    else:
        print(f"❌ ERREURS DÉTECTÉES: {', '.join(errors)}")
        print("="*60)
        print("\n⚠️ Réexécutez le script ou installez manuellement:")
        for err in errors:
            if err == "Whisper":
                print("   pip install openai-whisper")
            elif err == "TTS":
                print("   pip install TTS")
            elif err == "Audio":
                print("   pip install soundfile sounddevice")
        return False

if __name__ == '__main__':
    print("\n🎤 Installation du Système Vocal Kibali\n")
    
    # 1. Installer les dépendances
    print("Étape 1: Installation des dépendances...")
    install_voice_dependencies()
    
    # 2. Télécharger les modèles
    print("\nÉtape 2: Téléchargement des modèles...")
    download_models()
    
    # 3. Vérifier
    print("\nÉtape 3: Vérification...")
    verify_installation()
    
    print("\n" + "="*60)
    print("✅ INSTALLATION TERMINÉE!")
    print("="*60)
