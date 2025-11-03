#!/usr/bin/env python3
"""
Test rapide du système vocal Kibali
Vérifie que Whisper et Coqui TTS fonctionnent
"""

import sys
import os

def test_imports():
    """Test des imports de base"""
    print("="*60)
    print("🔍 TEST 1: Imports des modules")
    print("="*60)
    
    tests = {
        "whisper": lambda: __import__("whisper"),
        "TTS": lambda: __import__("TTS"),
        "soundfile": lambda: __import__("soundfile"),
        "sounddevice": lambda: __import__("sounddevice"),
        "numpy": lambda: __import__("numpy"),
        "voice_agent": lambda: __import__("voice_agent")
    }
    
    results = {}
    for name, test_func in tests.items():
        try:
            test_func()
            results[name] = "✅"
            print(f"  {name}: ✅")
        except ImportError as e:
            results[name] = f"❌ {e}"
            print(f"  {name}: ❌ {e}")
    
    return all("✅" in v for v in results.values())

def test_whisper_model():
    """Test du modèle Whisper"""
    print("\n" + "="*60)
    print("🎤 TEST 2: Modèle Whisper (transcription)")
    print("="*60)
    
    try:
        import whisper
        print("  Chargement du modèle 'base'...")
        model = whisper.load_model("base")
        print("  ✅ Modèle Whisper chargé!")
        
        # Test simple
        import numpy as np
        test_audio = np.random.randn(16000)  # 1 seconde de bruit
        print("  Test de transcription...")
        # Note: résultat vide attendu (bruit aléatoire)
        result = model.transcribe(test_audio, language="fr", fp16=False)
        print(f"  ✅ Transcription OK (résultat: '{result['text']}')")
        
        return True
    except Exception as e:
        print(f"  ❌ Erreur: {e}")
        return False

def test_tts_model():
    """Test du modèle TTS"""
    print("\n" + "="*60)
    print("🔊 TEST 3: Modèle Coqui TTS (synthèse vocale)")
    print("="*60)
    
    try:
        from TTS.api import TTS
        print("  Chargement du modèle TTS français...")
        
        # Essayer le modèle français
        try:
            tts = TTS(
                model_name="tts_models/fr/mai/tacotron2-DDC",
                progress_bar=False,
                gpu=False
            )
            model_name = "français (mai/tacotron2-DDC)"
        except Exception as e:
            print(f"  ⚠️ Modèle français non disponible: {e}")
            print("  Essai du modèle anglais...")
            tts = TTS(
                model_name="tts_models/en/ljspeech/tacotron2-DDC",
                progress_bar=False,
                gpu=False
            )
            model_name = "anglais (ljspeech/tacotron2-DDC)"
        
        print(f"  ✅ Modèle TTS chargé: {model_name}")
        
        # Test de synthèse
        test_text = "Test de synthèse vocale."
        output_file = "/tmp/test_tts_kibali.wav"
        
        print(f"  Synthèse de: '{test_text}'...")
        tts.tts_to_file(text=test_text, file_path=output_file)
        
        if os.path.exists(output_file):
            size = os.path.getsize(output_file)
            print(f"  ✅ Audio généré: {output_file} ({size} bytes)")
            os.remove(output_file)
            return True
        else:
            print("  ❌ Fichier audio non généré")
            return False
            
    except Exception as e:
        print(f"  ❌ Erreur: {e}")
        return False

def test_audio_devices():
    """Test des périphériques audio"""
    print("\n" + "="*60)
    print("🎵 TEST 4: Périphériques audio")
    print("="*60)
    
    try:
        import sounddevice as sd
        devices = sd.query_devices()
        
        print(f"  Périphériques trouvés: {len(devices)}")
        
        # Trouver le périphérique par défaut
        default_in = sd.query_devices(kind='input')
        default_out = sd.query_devices(kind='output')
        
        print(f"\n  Entrée par défaut (microphone):")
        print(f"    Nom: {default_in['name']}")
        print(f"    Canaux: {default_in['max_input_channels']}")
        
        print(f"\n  Sortie par défaut (haut-parleurs):")
        print(f"    Nom: {default_out['name']}")
        print(f"    Canaux: {default_out['max_output_channels']}")
        
        print("\n  ✅ Périphériques audio OK")
        return True
        
    except Exception as e:
        print(f"  ❌ Erreur: {e}")
        print("  💡 Vérifiez l'installation de portaudio:")
        print("     sudo apt-get install portaudio19-dev")
        return False

def test_voice_agent():
    """Test de la classe VoiceAgent"""
    print("\n" + "="*60)
    print("🤖 TEST 5: VoiceAgent (intégration)")
    print("="*60)
    
    try:
        from voice_agent import VoiceAgent
        
        print("  Initialisation de VoiceAgent...")
        agent = VoiceAgent(whisper_model="base")
        print("  ✅ VoiceAgent initialisé")
        
        print("\n  Chargement des modèles...")
        success = agent.load_models(load_whisper=True, load_tts=True)
        
        if success:
            print("  ✅ Modèles chargés dans VoiceAgent")
            
            # Test de synthèse simple
            print("\n  Test de synthèse vocale...")
            test_text = "Bonjour, je suis Kibali."
            audio_path = agent.synthesize_speech(
                test_text,
                output_path="/tmp/test_voice_agent.wav",
                play=False
            )
            
            if audio_path and os.path.exists(audio_path):
                print(f"  ✅ Audio généré: {audio_path}")
                os.remove(audio_path)
                return True
            else:
                print("  ❌ Échec génération audio")
                return False
        else:
            print("  ❌ Échec chargement des modèles")
            return False
            
    except Exception as e:
        print(f"  ❌ Erreur: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Fonction principale"""
    print("\n" + "="*60)
    print("🎤 TEST DU SYSTÈME VOCAL KIBALI")
    print("="*60 + "\n")
    
    results = []
    
    # Test 1: Imports
    results.append(("Imports", test_imports()))
    
    # Test 2: Whisper
    results.append(("Whisper", test_whisper_model()))
    
    # Test 3: TTS
    results.append(("TTS", test_tts_model()))
    
    # Test 4: Audio devices
    results.append(("Audio", test_audio_devices()))
    
    # Test 5: VoiceAgent
    results.append(("VoiceAgent", test_voice_agent()))
    
    # Résumé
    print("\n" + "="*60)
    print("📊 RÉSUMÉ DES TESTS")
    print("="*60)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {test_name:20s}: {status}")
    
    success_count = sum(1 for _, r in results if r)
    total_count = len(results)
    
    print(f"\nScore: {success_count}/{total_count}")
    
    if success_count == total_count:
        print("\n🎉 TOUS LES TESTS RÉUSSIS!")
        print("✅ Le système vocal est prêt à l'emploi")
        print("\n💡 Prochaine étape:")
        print("   streamlit run ERT.py")
        print("   → Activer le mode vocal dans la sidebar")
        return 0
    else:
        print("\n⚠️ CERTAINS TESTS ONT ÉCHOUÉ")
        print("💡 Solutions:")
        print("   1. Installer les dépendances:")
        print("      python install_voice_models.py")
        print("   2. Vérifier les dépendances système:")
        print("      sudo apt-get install ffmpeg portaudio19-dev libsndfile1")
        print("   3. Relancer les tests:")
        print("      python test_voice_system.py")
        return 1

if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)
