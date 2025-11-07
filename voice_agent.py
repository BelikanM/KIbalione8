"""
Système Vocal pour Kibali - Voice Agent
Transcription (Whisper) + Synthèse vocale (Coqui TTS)
"""

import os
import numpy as np
import soundfile as sf
from typing import Optional, Tuple
import tempfile
import time

# Import optionnel de sounddevice (nécessite PortAudio)
try:
    import sounddevice as sd
    SOUNDDEVICE_AVAILABLE = True
except (OSError, ImportError) as e:
    print(f"⚠️  sounddevice non disponible: {e}")
    print("💡 Pour activer l'enregistrement audio, installez: sudo apt-get install portaudio19-dev")
    sd = None
    SOUNDDEVICE_AVAILABLE = False

class VoiceAgent:
    """Agent vocal pour Kibali Analyst avec Whisper + Coqui TTS"""
    
    def __init__(self, 
                 whisper_model: str = "base",
                 tts_model: str = "tts_models/fr/mai/tacotron2-DDC",
                 cache_dir: str = None):
        """
        Args:
            whisper_model: 'tiny' (~1GB), 'base' (~1.5GB), 'small' (~2GB)
            tts_model: Modèle Coqui TTS français
            cache_dir: Dossier de cache des modèles (défaut: ~/.cache/voice_models)
        """
        self.whisper_model_name = whisper_model
        self.tts_model_name = tts_model
        # Utiliser le dossier home de l'utilisateur par défaut
        if cache_dir is None:
            cache_dir = os.path.expanduser("~/.cache/voice_models")
        self.cache_dir = cache_dir
        
        self.whisper = None
        self.tts = None
        self.is_ready = False
        
        # Créer le dossier de cache
        os.makedirs(cache_dir, exist_ok=True)
    
    def load_models(self, load_whisper: bool = True, load_tts: bool = True):
        """Charge les modèles vocaux"""
        try:
            # 1. Whisper pour transcription
            if load_whisper:
                print("🎤 Chargement de Whisper (transcription)...")
                import whisper
                
                self.whisper = whisper.load_model(
                    self.whisper_model_name,
                    download_root=os.path.join(self.cache_dir, "whisper")
                )
                print(f"✅ Whisper '{self.whisper_model_name}' chargé")
            
            # 2. Coqui TTS pour synthèse vocale
            if load_tts:
                print("🔊 Chargement de Coqui TTS (synthèse vocale)...")
                import torch
                from TTS.api import TTS
                
                # Correction pour PyTorch 2.6+ : permettre le chargement des modèles TTS
                original_load = torch.load
                def patched_load(*args, **kwargs):
                    # Forcer weights_only=False pour la compatibilité TTS
                    kwargs['weights_only'] = False
                    return original_load(*args, **kwargs)
                
                # Patch temporairement torch.load
                torch.load = patched_load
                
                try:
                    # Modèle français haute qualité
                    self.tts = TTS(
                        model_name=self.tts_model_name,
                        progress_bar=True,
                        gpu=False  # CPU pour compatibilité
                    )
                    print(f"✅ TTS '{self.tts_model_name}' chargé")
                finally:
                    # Restaurer torch.load original
                    torch.load = original_load
            
            self.is_ready = True
            return True
            
        except Exception as e:
            print(f"❌ Erreur chargement modèles: {e}")
            return False
    
    def transcribe_audio(self, 
                        audio_path: str = None,
                        audio_array: np.ndarray = None,
                        language: str = "fr") -> Optional[str]:
        """
        Transcrit un audio en texte avec Whisper
        
        Args:
            audio_path: Chemin vers fichier audio
            audio_array: Array numpy (alternative)
            language: Code langue ('fr', 'en', etc.)
        
        Returns:
            str: Texte transcrit ou None si erreur
        """
        if self.whisper is None:
            print("❌ Whisper non chargé")
            return None
        
        try:
            # Si audio_array fourni, sauvegarder temporairement
            if audio_array is not None:
                with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
                    sf.write(f.name, audio_array, 16000)
                    audio_path = f.name
            
            if not audio_path or not os.path.exists(audio_path):
                print("❌ Fichier audio introuvable")
                return None
            
            # Transcription avec Whisper
            result = self.whisper.transcribe(
                audio_path,
                language=language,
                fp16=False,  # CPU compatible
                verbose=False
            )
            
            text = result["text"].strip()
            
            # Nettoyer le fichier temporaire
            if audio_array is not None:
                try:
                    os.unlink(audio_path)
                except:
                    pass
            
            return text
            
        except Exception as e:
            print(f"❌ Erreur transcription: {e}")
            return None
    
    def record_audio(self, duration: int = 5, sample_rate: int = 16000) -> np.ndarray:
        """
        Enregistre de l'audio depuis le microphone
        
        Args:
            duration: Durée en secondes
            sample_rate: Fréquence d'échantillonnage
        
        Returns:
            np.ndarray: Audio enregistré
        """
        if not SOUNDDEVICE_AVAILABLE:
            print("❌ sounddevice non disponible - Enregistrement audio désactivé")
            print("💡 Pour activer: sudo apt-get install portaudio19-dev && pip install sounddevice")
            return np.array([])
        
        print(f"🎤 Enregistrement ({duration}s)...")
        
        try:
            # Vérifier qu'un périphérique audio est disponible
            devices = sd.query_devices()
            if not devices:
                print("❌ Aucun périphérique audio détecté")
                return np.array([])
            
            # Trouver le périphérique d'entrée par défaut
            default_input = sd.default.device[0]
            if default_input is None or default_input < 0:
                print("❌ Pas de périphérique d'entrée audio configuré")
                print("💡 Configurez un microphone ou désactivez le mode vocal")
                return np.array([])
            
            audio = sd.rec(
                int(duration * sample_rate),
                samplerate=sample_rate,
                channels=1,
                dtype='float32',
                device=default_input
            )
            sd.wait()  # Attendre la fin
            
            print("✅ Enregistrement terminé")
            return audio.flatten()
            
        except Exception as e:
            print(f"❌ Erreur enregistrement: {e}")
            print("💡 Le mode vocal nécessite un microphone fonctionnel")
            return np.array([])
    
    def synthesize_speech(self, 
                         text: str,
                         output_path: str = None,
                         play: bool = False) -> Optional[str]:
        """
        Synthétise de la parole à partir de texte
        
        Args:
            text: Texte à synthétiser
            output_path: Chemin de sortie (optionnel)
            play: Jouer directement l'audio
        
        Returns:
            str: Chemin du fichier audio généré
        """
        if self.tts is None:
            print("❌ TTS non chargé")
            return None
        
        try:
            # Générer un nom de fichier si non fourni
            if output_path is None:
                timestamp = int(time.time())
                output_path = f"/tmp/kibali_voice_{timestamp}.wav"
            
            # Synthèse avec Coqui TTS
            print("🔊 Génération de la parole...")
            self.tts.tts_to_file(
                text=text,
                file_path=output_path
            )
            
            print(f"✅ Audio généré: {output_path}")
            
            # Jouer l'audio si demandé
            if play:
                self.play_audio(output_path)
            
            return output_path
            
        except Exception as e:
            print(f"❌ Erreur synthèse: {e}")
            return None
    
    def play_audio(self, audio_path: str):
        """Joue un fichier audio"""
        if not SOUNDDEVICE_AVAILABLE:
            print("❌ sounddevice non disponible (PortAudio manquant)")
            return
            
        try:
            data, sample_rate = sf.read(audio_path)
            sd.play(data, sample_rate)
            sd.wait()
            print("✅ Lecture terminée")
        except Exception as e:
            print(f"❌ Erreur lecture: {e}")
    
    def voice_conversation(self, 
                          callback_function,
                          record_duration: int = 5,
                          auto_play: bool = True) -> Tuple[str, str, str]:
        """
        Conversation vocale complète: enregistrement → transcription → réponse → synthèse
        
        Args:
            callback_function: Fonction qui prend le texte transcrit et retourne la réponse
            record_duration: Durée d'enregistrement
            auto_play: Jouer automatiquement la réponse
        
        Returns:
            (transcription, réponse_texte, audio_path)
        """
        # 1. Enregistrer
        print("\n🎤 === CONVERSATION VOCALE ===")
        audio = self.record_audio(duration=record_duration)
        
        if len(audio) == 0:
            return "", "", ""
        
        # 2. Transcrire
        print("\n📝 Transcription...")
        transcription = self.transcribe_audio(audio_array=audio)
        
        if not transcription:
            return "", "", ""
        
        print(f"✅ Vous: {transcription}")
        
        # 3. Obtenir la réponse (callback)
        print("\n🤖 Kibali réfléchit...")
        response_text = callback_function(transcription)
        
        if not response_text:
            return transcription, "", ""
        
        print(f"✅ Kibali: {response_text[:100]}...")
        
        # 4. Synthétiser la réponse
        print("\n🔊 Synthèse vocale...")
        audio_path = self.synthesize_speech(
            response_text,
            play=auto_play
        )
        
        print("\n✅ === CONVERSATION TERMINÉE ===\n")
        
        return transcription, response_text, audio_path or ""


class StreamingVoiceAgent(VoiceAgent):
    """Version améliorée avec streaming audio pour fluidité"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.is_recording = False
        self.recorded_chunks = []
        self.stream = None
    
    def start_recording_stream(self, callback=None):
        """Démarre l'enregistrement en streaming"""
        if not SOUNDDEVICE_AVAILABLE:
            print("❌ sounddevice non disponible - Enregistrement audio désactivé")
            print("💡 Pour activer: sudo apt-get install portaudio19-dev && pip install sounddevice")
            return
        
        try:
            # Vérifier qu'un périphérique audio est disponible
            devices = sd.query_devices()
            if not devices:
                print("❌ Aucun périphérique audio détecté")
                return
            
            # Trouver le périphérique d'entrée par défaut
            default_input = sd.default.device[0]
            if default_input is None or default_input < 0:
                print("❌ Pas de périphérique d'entrée audio configuré")
                return
                
            self.is_recording = True
            self.recorded_chunks = []
            
            def audio_callback(indata, frames, time_info, status):
                if status:
                    print(f"⚠️ {status}")
                if self.is_recording:
                    self.recorded_chunks.append(indata.copy())
                    if callback:
                        callback(indata)
            
            self.stream = sd.InputStream(
                callback=audio_callback,
                channels=1,
                samplerate=16000,
                dtype='float32',
                device=default_input
            )
            self.stream.start()
            print("🎤 Enregistrement streaming démarré")
            
        except Exception as e:
            print(f"❌ Erreur démarrage enregistrement: {e}")
            print("💡 Le mode vocal nécessite un microphone fonctionnel")
            self.is_recording = False
    
    def stop_recording_stream(self) -> np.ndarray:
        """Arrête l'enregistrement et retourne l'audio"""
        if not SOUNDDEVICE_AVAILABLE or self.stream is None:
            return np.array([])
            
        self.is_recording = False
        if hasattr(self, 'stream'):
            self.stream.stop()
            self.stream.close()
        
        if self.recorded_chunks:
            audio = np.concatenate(self.recorded_chunks, axis=0)
            print(f"✅ Audio capturé: {len(audio)} samples")
            return audio.flatten()
        
        return np.array([])
    
    def synthesize_streaming(self, text: str, chunk_size: int = 50):
        """
        Synthèse vocale par morceaux pour plus de fluidité
        (Simulation - TTS complet pour l'instant)
        """
        # Pour l'instant, utiliser la synthèse normale
        # TODO: Implémenter streaming réel si Coqui TTS le supporte
        return self.synthesize_speech(text, play=True)


def download_voice_models():
    """Télécharge les modèles vocaux optimaux"""
    print("📦 Téléchargement des modèles vocaux...")
    
    # 1. Whisper
    print("\n1️⃣ Whisper (transcription)")
    print("   Modèle: base (~150MB)")
    import whisper
    cache_dir = os.path.expanduser("~/.cache/voice_models/whisper")
    whisper.load_model("base", download_root=cache_dir)
    print("   ✅ Whisper téléchargé")
    
    # 2. Coqui TTS
    print("\n2️⃣ Coqui TTS (synthèse vocale)")
    print("   Modèle: French Tacotron2 (~500MB)")
    from TTS.api import TTS
    
    # Lister les modèles disponibles
    tts = TTS(model_name="tts_models/fr/mai/tacotron2-DDC")
    print("   ✅ Coqui TTS téléchargé")
    
    print("\n✅ Tous les modèles vocaux sont prêts!")
    print("📊 Taille totale: ~650MB")
    
    return True


if __name__ == '__main__':
    # Test du système vocal
    print("=== TEST VOICE AGENT ===\n")
    
    # Option 1: Télécharger les modèles
    choice = input("Télécharger les modèles? (o/N): ")
    if choice.lower() in ['o', 'oui', 'y', 'yes']:
        download_voice_models()
    
    # Option 2: Test rapide
    print("\n=== TEST TRANSCRIPTION ===")
    agent = VoiceAgent(whisper_model="base")
    
    if agent.load_models(load_whisper=True, load_tts=False):
        print("\n🎤 Enregistrez un message de 5 secondes...")
        time.sleep(1)
        
        audio = agent.record_audio(duration=5)
        
        if len(audio) > 0:
            text = agent.transcribe_audio(audio_array=audio)
            print(f"\n✅ Transcription: {text}")
    
    print("\n=== TEST SYNTHÈSE VOCALE ===")
    agent2 = VoiceAgent()
    
    if agent2.load_models(load_whisper=False, load_tts=True):
        text_to_speak = "Bonjour, je suis Kibali Analyst, votre assistant vocal intelligent."
        audio_file = agent2.synthesize_speech(text_to_speak, play=True)
        print(f"\n✅ Audio généré: {audio_file}")
