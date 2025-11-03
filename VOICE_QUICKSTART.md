# 🎤 GUIDE DE DÉMARRAGE RAPIDE - Système Vocal Kibali

## ⚡ Installation Express (5 minutes)

### 1️⃣ Installer les dépendances

```bash
cd /root/RAG_ChatBot
python install_voice_models.py
```

**Ce script va:**
- ✅ Installer Whisper (transcription vocale)
- ✅ Installer Coqui TTS (synthèse vocale)
- ✅ Télécharger les modèles (~1.5GB)
- ✅ Configurer l'audio système
- ✅ Tester le fonctionnement

**Durée:** 5-15 minutes selon votre connexion

---

### 2️⃣ Tester l'installation

```bash
python test_voice_system.py
```

**Résultat attendu:**
```
✅ Imports: PASS
✅ Whisper: PASS
✅ TTS: PASS
✅ Audio: PASS
✅ VoiceAgent: PASS

Score: 5/5
🎉 TOUS LES TESTS RÉUSSIS!
```

---

### 3️⃣ Lancer Kibali

```bash
streamlit run ERT.py
```

---

### 4️⃣ Activer le mode vocal

Dans l'interface Streamlit:

1. **Ouvrir la sidebar** (panneau gauche)
2. **Trouver "🎤 Interface Vocale"**
3. **Cocher "Activer le mode vocal"**
4. **Attendre le chargement** (~10-30s)
5. **Statut: 🟢 "Vocal: Actif"**

---

## 🎙️ Utilisation

### Enregistrer une question

```
1. Cliquer "🎤 Enregistrer Question"
2. Parler pendant 5 secondes
3. Attendre la transcription
4. Kibali répond par texte ET voix!
```

### Répéter la dernière réponse

```
1. Cliquer "🔊 Répéter Dernière Réponse"
2. La réponse est lue vocalement
3. Option: télécharger l'audio (.wav)
```

---

## ⚙️ Configuration

### Durée d'enregistrement

Dans **"⚙️ Options vocales"**:
- **Slider "Durée d'enregistrement"**: 3-30 secondes
- **Défaut:** 5 secondes
- **Recommandé:** 5-10s pour questions courtes

### Lecture automatique

- **ON (défaut):** Kibali parle automatiquement
- **OFF:** Lecture manuelle uniquement

### Langue de transcription

- **fr (défaut):** Français
- **en:** Anglais
- **es:** Espagnol
- **de:** Allemand

---

## 🔥 Exemples

### Exemple 1: Question simple

```
🎤 VOUS: "Qui est Nyundu Francis Arnaud?"

📝 Transcription: "Qui est Nyundu Francis Arnaud?"

🔊 KIBALI: "Nyundu Francis Arnaud est le directeur général
           de Kibali Mining Company, basée en République
           Démocratique du Congo..."

[Audio joué automatiquement]
```

### Exemple 2: Analyse technique

```
🎤 VOUS: "Analyse les profondeurs d'eau salée dans resistivity.npy"

📝 Transcription: "Analyse les profondeurs d'eau salée dans resistivity.npy"

💻 KIBALI: [Génère du code Python]
           [Exécute l'analyse]
           
🔊 KIBALI: "J'ai identifié trois zones d'eau salée aux profondeurs
           de 12, 24 et 45 mètres avec des résistivités
           inférieures à 10 Ohm-mètre."

[Audio + code visible]
```

### Exemple 3: Conversation continue

```
🎤 "Génère un rapport ERT"
🔊 "Rapport généré en PDF de 15 pages..."

🎤 "Résume-le en 2 minutes"
🔊 "Voici le résumé: [résumé vocal 500 mots]"

🎤 "Envoie-le par email"
🔊 "Fonction email à implémenter..."
```

---

## 🐛 Problèmes fréquents

### Erreur: "Modèles non chargés"

**Solution:**
```bash
python install_voice_models.py
```

### Erreur: "Microphone non trouvé"

**Solution Ubuntu/Linux:**
```bash
sudo apt-get install portaudio19-dev
pip install --upgrade sounddevice
```

**Vérifier:**
```python
python -c "import sounddevice; print(sounddevice.query_devices())"
```

### Erreur: "TTS ne fonctionne pas"

**Solution - Modèle alternatif:**
```python
# Dans voice_agent.py, ligne 16:
tts_model="tts_models/en/ljspeech/tacotron2-DDC"  # Anglais
```

### Transcription vide

**Causes:**
- Volume microphone trop bas
- Bruit de fond excessif
- Durée trop courte

**Solutions:**
- Augmenter durée à 10s
- Parler plus fort
- Réduire bruit ambiant
- Changer la langue de détection

---

## 📊 Comparaison

| Fonctionnalité | ChatGPT Voice | Kibali Voice |
|----------------|---------------|--------------|
| **Prix** | $20/mois | ✅ GRATUIT |
| **Confidentialité** | ❌ Cloud | ✅ 100% Local |
| **Latence** | ~8-15s | ~10-23s |
| **Langues** | ~50 | ✅ 99+ |
| **Offline** | ❌ Non | ✅ Oui |
| **Personnalisation** | ❌ Limitée | ✅ Complète |

**Verdict:** Kibali = Meilleur rapport qualité/prix/confidentialité! 🏆

---

## 📚 Ressources

- **Documentation complète:** `VOICE_SYSTEM_DOCS.md`
- **Code source:** `voice_agent.py`
- **Tests:** `test_voice_system.py`
- **Installation:** `install_voice_models.py`
- **Dépendances:** `requirements_voice.txt`

---

## 🚀 Prêt à l'emploi!

```bash
# Installation complète en 3 commandes:
python install_voice_models.py
python test_voice_system.py
streamlit run ERT.py

# Activer le mode vocal dans la sidebar
# 🎤 Commencer à parler!
```

**Profitez d'une expérience vocale fluide et gratuite!** 🎉
