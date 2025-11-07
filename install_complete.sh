#!/bin/bash
# ========================================
# Installation complète KIbalione8
# Système d'analyse ERT avec IA avancée
# ========================================

set -e  # Arrêter en cas d'erreur

echo "🚀 Installation complète de KIbalione8"
echo "======================================"

# Vérification Python
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 n'est pas installé"
    exit 1
fi

PYTHON_VERSION=$(python3 --version | cut -d' ' -f2 | cut -d'.' -f1,2)
echo "✅ Python $PYTHON_VERSION détecté"

# Vérification conda
if ! command -v conda &> /dev/null; then
    echo "⚠️  Conda non détecté. Installation de Miniconda recommandée."
    echo "   Téléchargez: https://docs.conda.io/en/latest/miniconda.html"
else
    echo "✅ Conda détecté"
fi

# ========================================
# ÉTAPE 1: Dépendances système
# ========================================
echo ""
echo "📦 Étape 1/7: Installation des dépendances système"
echo "------------------------------------------------"

if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    echo "Détection: Linux"
    
    if command -v apt-get &> /dev/null; then
        echo "Installation via apt-get..."
        sudo apt-get update
        sudo apt-get install -y \
            ffmpeg \
            libsndfile1 \
            portaudio19-dev \
            libopenblas-dev \
            liblapack-dev \
            gfortran \
            libsuitesparse-dev \
            libvtk9-dev \
            python3-dev \
            build-essential \
            git \
            wget \
            curl \
            cmake
        echo "✅ Dépendances système installées"
    else
        echo "⚠️  apt-get non disponible. Installation manuelle requise."
    fi
elif [[ "$OSTYPE" == "darwin"* ]]; then
    echo "Détection: macOS"
    
    if command -v brew &> /dev/null; then
        echo "Installation via Homebrew..."
        brew install ffmpeg portaudio libsndfile openblas lapack
        echo "✅ Dépendances système installées"
    else
        echo "⚠️  Homebrew non installé. Téléchargez: https://brew.sh"
    fi
else
    echo "⚠️  Système d'exploitation non reconnu: $OSTYPE"
fi

# ========================================
# ÉTAPE 2: Environnement conda
# ========================================
echo ""
echo "🐍 Étape 2/7: Configuration environnement conda"
echo "------------------------------------------------"

if command -v conda &> /dev/null; then
    ENV_NAME="kibalione8"
    
    # Vérifier si l'environnement existe
    if conda env list | grep -q "^$ENV_NAME "; then
        echo "⚠️  L'environnement '$ENV_NAME' existe déjà"
        read -p "Voulez-vous le recréer? (y/N): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            conda env remove -n $ENV_NAME -y
            conda create -n $ENV_NAME python=3.10 -y
            echo "✅ Environnement recréé"
        fi
    else
        conda create -n $ENV_NAME python=3.10 -y
        echo "✅ Environnement créé"
    fi
    
    echo "🔄 Activation de l'environnement..."
    eval "$(conda shell.bash hook)"
    conda activate $ENV_NAME
    echo "✅ Environnement '$ENV_NAME' activé"
else
    echo "⚠️  Conda non disponible, utilisation de l'environnement Python global"
fi

# ========================================
# ÉTAPE 3: PyTorch (CPU optimized)
# ========================================
echo ""
echo "🔥 Étape 3/7: Installation PyTorch (CPU)"
echo "------------------------------------------------"

pip install --upgrade pip setuptools wheel

echo "Installation de PyTorch CPU..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

echo "✅ PyTorch installé"

# ========================================
# ÉTAPE 4: Dépendances Python essentielles
# ========================================
echo ""
echo "📚 Étape 4/7: Installation des dépendances Python"
echo "------------------------------------------------"

echo "Installation des packages essentiels..."
pip install -r requirements_complete.txt

echo "✅ Dépendances Python installées"

# ========================================
# ÉTAPE 5: PyGIMLi (Geophysics)
# ========================================
echo ""
echo "🌍 Étape 5/7: Installation PyGIMLi (Geophysics)"
echo "------------------------------------------------"

if command -v conda &> /dev/null; then
    echo "Installation de PyGIMLi via conda..."
    conda install -c gimli -c conda-forge pygimli -y
    echo "✅ PyGIMLi installé"
else
    echo "⚠️  PyGIMLi nécessite conda pour l'installation"
    echo "   Alternative: pip install pygimli (peut échouer)"
    pip install pygimli || echo "❌ Échec installation PyGIMLi"
fi

# ========================================
# ÉTAPE 6: Configuration
# ========================================
echo ""
echo "⚙️  Étape 6/7: Configuration"
echo "------------------------------------------------"

# Créer le fichier .env s'il n'existe pas
if [ ! -f .env ]; then
    echo "Création du fichier .env..."
    cat > .env << 'EOF'
# Configuration KIbalione8

# HuggingFace Token (requis pour modèles)
# Obtenez votre token: https://huggingface.co/settings/tokens
HF_TOKEN=hf_votre_token_ici

# Tavily API Key (requis pour recherche web)
# Obtenez votre clé: https://tavily.com
TAVILY_API_KEY=tvly-votre_cle_ici

# OpenAI API Key (optionnel)
OPENAI_API_KEY=sk-votre_cle_ici

# Anthropic API Key (optionnel)
ANTHROPIC_API_KEY=sk-ant-votre_cle_ici

# Configuration Cache
CACHE_DIR=~/.cache/kibalione8
HF_HOME=~/.cache/huggingface

# Configuration Modèles
DEFAULT_LLM_MODEL=Qwen/Qwen2.5-7B-Instruct
DEFAULT_EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
DEFAULT_WHISPER_MODEL=base
DEFAULT_TTS_MODEL=tts_models/fr/mai/tacotron2-DDC
EOF
    echo "✅ Fichier .env créé (configurez vos tokens!)"
else
    echo "✅ Fichier .env existe déjà"
fi

# Créer les dossiers nécessaires
echo "Création des dossiers..."
mkdir -p data/uploads
mkdir -p data/examples
mkdir -p logs
mkdir -p chat_vectordb
mkdir -p vectordb
mkdir -p generated
mkdir -p submodels
mkdir -p local_models
mkdir -p pdfs
mkdir -p graphs
mkdir -p maps

echo "✅ Dossiers créés"

# ========================================
# ÉTAPE 7: Téléchargement des modèles
# ========================================
echo ""
echo "📥 Étape 7/7: Téléchargement des modèles"
echo "------------------------------------------------"

echo "⚠️  Les modèles seront téléchargés automatiquement au premier usage"
echo ""
echo "Modèles qui seront téléchargés:"
echo "  - Embeddings: sentence-transformers/all-MiniLM-L6-v2 (~90MB)"
echo "  - Whisper: base (~150MB)"
echo "  - TTS: tts_models/fr/mai/tacotron2-DDC (~250MB)"
echo "  - LLM: Qwen/Qwen2.5-7B-Instruct (~4GB) - sur demande"
echo ""
echo "Taille totale estimée: ~5-10GB"
echo ""

read -p "Voulez-vous pré-télécharger les modèles maintenant? (y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Téléchargement des modèles d'embedding..."
    python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"
    
    echo "Téléchargement de Whisper base..."
    python -c "import whisper; whisper.load_model('base')"
    
    echo "✅ Modèles pré-téléchargés"
else
    echo "⏭️  Téléchargement des modèles reporté au premier usage"
fi

# ========================================
# FINALISATION
# ========================================
echo ""
echo "🎉 Installation terminée!"
echo "========================"
echo ""
echo "📋 Prochaines étapes:"
echo ""
echo "1. Configurez vos tokens API dans le fichier .env:"
echo "   nano .env"
echo ""
echo "2. Activez l'environnement conda:"
echo "   conda activate kibalione8"
echo ""
echo "3. Lancez l'application:"
echo "   streamlit run kibalione8.py"
echo ""
echo "4. Ou lancez l'analyse ERT:"
echo "   streamlit run ERT_final_20251103_200808.py"
echo ""
echo "📚 Documentation:"
echo "   - README.md"
echo "   - VOICE_SYSTEM_DOCS.md"
echo "   - IA_SPECIALISTS_README.md"
echo ""
echo "🆘 En cas de problème:"
echo "   - Vérifiez les logs dans ./logs/"
echo "   - Consultez les issues GitHub"
echo "   - Vérifiez votre configuration .env"
echo ""
echo "✨ Bon usage de KIbalione8!"
