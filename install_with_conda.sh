#!/bin/bash
# ========================================
# Installation optimisée avec CONDA
# KIbalione8 - Méthode observée dans les fichiers
# ========================================

set -e

echo "🚀 Installation KIbalione8 avec CONDA (méthode optimisée)"
echo "=========================================================="

# ========================================
# Vérification conda
# ========================================
if ! command -v conda &> /dev/null; then
    echo "❌ Conda non trouvé!"
    echo "Installez Miniconda d'abord: https://docs.conda.io/en/latest/miniconda.html"
    exit 1
fi

echo "✅ Conda détecté: $(conda --version)"

# ========================================
# Utilisation de l'environnement gestmodo existant
# ========================================
ENV_NAME="gestmodo"

echo ""
echo "🔍 Vérification environnement '$ENV_NAME'..."

# Activer l'environnement
eval "$(conda shell.bash hook)"
conda activate $ENV_NAME

echo "✅ Environnement '$ENV_NAME' activé"

# ========================================
# Installation avec conda (priorité)
# ========================================
echo ""
echo "📦 Installation avec conda (packages optimisés)..."
echo "---------------------------------------------------"

# Packages disponibles via conda-forge (plus rapides et optimisés)
echo "Installation des packages conda-forge..."
conda install -c conda-forge -y \
    numpy \
    pandas \
    scipy \
    scikit-learn \
    matplotlib \
    networkx \
    shapely \
    opencv \
    librosa \
    pyyaml \
    tqdm \
    click \
    rich

echo "✅ Packages conda-forge installés"

# ========================================
# PyGIMLi via conda (méthode officielle)
# ========================================
echo ""
echo "🌍 Installation PyGIMLi (geophysics) via conda..."
echo "--------------------------------------------------"

conda install -c gimli -c conda-forge pygimli -y

echo "✅ PyGIMLi installé"

# ========================================
# Packages spécifiques via pip (non disponibles sur conda)
# ========================================
echo ""
echo "📦 Installation packages pip (spécialisés)..."
echo "----------------------------------------------"

# Audio processing
echo "Installation audio processing..."
pip install --no-cache-dir \
    openai-whisper \
    soundfile \
    sounddevice

# 3D processing
echo "Installation 3D processing..."
pip install --no-cache-dir \
    open3d

# PDF processing
echo "Installation PDF processing..."
pip install --no-cache-dir \
    pymupdf \
    reportlab

# Image processing additionnels
echo "Installation scikit-image..."
pip install --no-cache-dir \
    scikit-image \
    imageio

echo "✅ Packages pip installés"

# ========================================
# TTS (optionnel - peut causer conflits)
# ========================================
echo ""
echo "🔊 Installation TTS (synthèse vocale)..."
echo "-----------------------------------------"

read -p "Installer Coqui TTS? (Peut causer des conflits) (y/N): " -n 1 -r
echo

if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Installation TTS (peut prendre du temps)..."
    
    # Installer d'abord les dépendances manquantes
    pip install --no-cache-dir \
        anyascii \
        bangla \
        bnnumerizer \
        bnunicodenormalizer \
        coqpit \
        einops \
        encodec \
        g2pkk \
        hangul-romanize \
        jamo \
        jieba \
        nltk \
        num2words \
        pypinyin \
        pysbd \
        trainer \
        umap-learn \
        unidecode
    
    # Installer TTS
    pip install --no-cache-dir TTS
    
    # Installer gruut et spacy séparément
    pip install --no-cache-dir "gruut[de,es,fr]==2.2.3"
    pip install --no-cache-dir spacy
    
    echo "✅ TTS installé (avec dépendances)"
else
    echo "⏭️  TTS non installé (seulement transcription Whisper disponible)"
fi

# ========================================
# Vérification finale
# ========================================
echo ""
echo "🔍 Vérification finale..."
echo "-------------------------"

python -c "
import sys

# Packages critiques
packages = {
    'torch': 'PyTorch',
    'transformers': 'Transformers',
    'langchain': 'LangChain',
    'streamlit': 'Streamlit',
    'whisper': 'Whisper',
    'soundfile': 'SoundFile',
    'cv2': 'OpenCV',
    'skimage': 'Scikit-Image',
    'open3d': 'Open3D',
    'fitz': 'PyMuPDF',
    'reportlab': 'ReportLab',
    'shapely': 'Shapely',
    'networkx': 'NetworkX',
    'pygimli': 'PyGIMLi',
    'pyres': 'PyRes',
    'tavily': 'Tavily',
    'numpy': 'NumPy',
    'pandas': 'Pandas',
    'scipy': 'SciPy',
    'sklearn': 'Scikit-Learn',
    'matplotlib': 'Matplotlib',
}

print('\\n📦 Vérification des packages:')
print('='*60)

success = 0
total = len(packages)

for package, name in packages.items():
    try:
        mod = __import__(package)
        version = getattr(mod, '__version__', 'N/A')
        print(f'✅ {name:20s} ({version})')
        success += 1
    except ImportError:
        print(f'❌ {name:20s} - MANQUANT')

print('='*60)
print(f'\\n📊 Résultat: {success}/{total} packages ({success/total*100:.0f}%)')

# Optionnel TTS
try:
    import TTS
    print('\\n🔊 Bonus: TTS installé (synthèse vocale)')
except ImportError:
    print('\\n⚠️  TTS non installé (seulement transcription)')

if success >= total * 0.9:  # 90% de succès
    print('\\n🎉 Installation conda réussie!')
    sys.exit(0)
else:
    print('\\n⚠️  Installation incomplète')
    sys.exit(1)
"

echo ""
echo "=========================================================="
echo "✅ Installation avec CONDA terminée!"
echo "=========================================================="
echo ""
echo "📋 Prochaines étapes:"
echo "  1. Configurez .env avec vos tokens:"
echo "     cp .env.example .env"
echo "     nano .env"
echo ""
echo "  2. Téléchargez les modèles:"
echo "     python download_all_models.py"
echo ""
echo "  3. Lancez l'application:"
echo "     streamlit run kibalione8.py"
echo ""
echo "💡 Environnement actif: $ENV_NAME"
echo "   Activez avec: conda activate $ENV_NAME"
