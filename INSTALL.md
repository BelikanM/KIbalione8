# 🚀 Guide d'Installation KIbalione8

## Installation Rapide (Recommandée)

### Option 1: Installation avec environnement existant (gestmodo)

```bash
# 1. Se placer dans le dossier
cd /home/belikan/KIbalione8

# 2. Activer l'environnement
# (Si besoin: conda activate gestmodo)

# 3. Installer les packages manquants essentiels
~/miniconda3/envs/gestmodo/bin/pip install -r requirements.txt

# 4. Configurer les tokens API
cp .env.example .env
nano .env  # Ajouter vos tokens HF_TOKEN et TAVILY_API_KEY

# 5. Tester l'installation
~/miniconda3/envs/gestmodo/bin/python download_all_models.py

# 6. Lancer l'application
~/miniconda3/envs/gestmodo/bin/streamlit run kibalione8.py
```

### Option 2: Installation automatique avec script

```bash
cd /home/belikan/KIbalione8

# Installation rapide (sans TTS lourd)
./install_fast.sh

# OU Installation complète avec conda
./install_with_conda.sh
```

## Packages Déjà Installés ✅

Dans l'environnement `gestmodo`, vous avez déjà:

- ✅ PyTorch 2.5.1 (CUDA 12.1)
- ✅ Transformers 4.57.1
- ✅ LangChain 1.0.3
- ✅ Streamlit 1.51.0
- ✅ Sentence Transformers 5.1.2
- ✅ FAISS-CPU 1.12.0
- ✅ PyGIMLi 1.5.4
- ✅ PyRes 1.5
- ✅ Tavily 0.7.12
- ✅ NumPy 1.26.4
- ✅ Pandas 2.3.3
- ✅ OpenCV 4.12.0

## Packages à Installer

### Essentiels (installation rapide ~2 minutes)

```bash
pip install \
    openai-whisper \
    soundfile \
    librosa \
    open3d \
    pymupdf \
    reportlab \
    scikit-image \
    imageio \
    rich \
    tqdm
```

### Optionnels (selon besoins)

```bash
# Synthèse vocale (lourd ~1GB, peut causer conflits)
pip install TTS

# Audio avancé
pip install sounddevice pydub

# 3D avancé
pip install pyvista trimesh

# GIS complet
pip install geopandas osmium
```

## Vérification

```bash
python -c "
import torch, transformers, langchain, streamlit
import whisper, cv2, open3d, pygimli
print('✅ Tous les packages critiques sont installés!')
"
```

## Configuration API

Créez/modifiez `.env`:

```bash
# HuggingFace Token (obligatoire)
HF_TOKEN=hf_votre_token_ici

# Tavily API Key (obligatoire pour recherche web)
TAVILY_API_KEY=tvly-votre_cle_ici

# OpenAI (optionnel)
OPENAI_API_KEY=sk-votre_cle_ici
```

Obtenez vos tokens:
- HuggingFace: https://huggingface.co/settings/tokens
- Tavily: https://tavily.com

## Téléchargement des Modèles

```bash
# Télécharger et vérifier tous les modèles
python download_all_models.py
```

Modèles téléchargés automatiquement:
- Whisper base (~150MB)
- Sentence Transformers (~90MB)
- Embeddings multilingues (~420MB)

Modèles LLM (téléchargés au premier usage):
- Qwen 2.5 7B (~4GB) - Recommandé
- Gemma 2B (~2.5GB) - Léger
- DeepSeek V3 (~14GB) - Puissant

## Problèmes Courants

### Conflit NumPy/Pandas avec TTS

**Solution**: Ne pas installer TTS si non nécessaire. Whisper (transcription) fonctionne sans TTS.

```bash
# Désinstaller TTS si problème
pip uninstall tts -y
```

### PyGIMLi ne s'installe pas

**Solution**: Utiliser conda obligatoirement

```bash
conda install -c gimli -c conda-forge pygimli
```

### Erreur CUDA

**Solution**: Utiliser PyTorch CPU

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

## Structure des Dossiers

```
KIbalione8/
├── kibalione8.py              # Application principale
├── ERT_final_20251103_200808.py  # ERT avancé
├── voice_agent.py             # Système vocal
├── resistivity_color_mapper.py # Base matériaux
├── requirements.txt           # Dépendances principales
├── requirements_fast.txt      # Installation rapide
├── requirements_complete.txt  # Installation complète
├── install_fast.sh           # Script installation rapide
├── install_with_conda.sh     # Script installation conda
├── download_all_models.py    # Téléchargement modèles
└── .env                      # Configuration (à créer)
```

## Commandes Utiles

```bash
# Lancer application principale
streamlit run kibalione8.py

# Lancer analyse ERT
streamlit run ERT_final_20251103_200808.py

# Tester système vocal
python test_voice_system.py

# Mettre à jour base vectorielle
python update_vectordb.py

# Vérifier installation
python -c "from voice_agent import VoiceAgent; print('✅ OK')"
```

## Support

- Documentation: README.md, VOICE_SYSTEM_DOCS.md, IA_SPECIALISTS_README.md
- Issues GitHub
- Logs: `./logs/`

## Estimation Taille

- **Installation minimale**: ~3GB (sans TTS, sans LLM)
- **Installation standard**: ~8GB (avec Whisper, embeddings, Qwen 7B)
- **Installation complète**: ~20GB (avec TTS, tous les LLM, 3D avancé)

---

**✨ Installation terminée ? Lancez: `streamlit run kibalione8.py`**
