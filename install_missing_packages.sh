#!/bin/bash
# ========================================
# Installation des packages manquants pour KIbalione8
# À exécuter dans l'environnement gestmodo
# ========================================

set -e

echo "🚀 Installation des packages manquants pour KIbalione8"
echo "======================================================="

# Activer l'environnement conda
eval "$(~/miniconda3/bin/conda shell.bash hook)"
conda activate gestmodo

echo "✅ Environnement gestmodo activé"

# ========================================
# 1. Packages Voice (Whisper + TTS)
# ========================================
echo ""
echo "🎤 1/5: Installation des packages vocaux..."
echo "--------------------------------------------"

echo "Installation de Whisper..."
pip install openai-whisper

echo "Installation de Coqui TTS..."
pip install TTS

echo "Installation des packages audio..."
pip install soundfile sounddevice librosa pydub noisereduce

echo "✅ Packages vocaux installés"

# ========================================
# 2. Packages 3D & Geometry
# ========================================
echo ""
echo "🎨 2/5: Installation des packages 3D et géométrie..."
echo "-----------------------------------------------------"

echo "Installation de Open3D..."
pip install open3d

echo "Installation de trimesh..."
pip install trimesh

echo "Installation de pyvista..."
pip install pyvista

echo "✅ Packages 3D installés"

# ========================================
# 3. Packages Geospatial
# ========================================
echo ""
echo "🌍 3/5: Installation des packages géospatiaux..."
echo "-------------------------------------------------"

echo "Installation d'osmium..."
# osmium nécessite des dépendances système
# sudo apt-get install -y libosmium-dev
pip install osmium || echo "⚠️  osmium installation échouée (dépendances système manquantes)"

echo "Installation de shapely..."
pip install shapely

echo "Installation de geopandas..."
pip install geopandas

echo "✅ Packages géospatiaux installés"

# ========================================
# 4. Packages PDF & Documents
# ========================================
echo ""
echo "📄 4/5: Installation des packages PDF..."
echo "-----------------------------------------"

echo "Installation de pymupdf (fitz)..."
pip install pymupdf

echo "Installation de reportlab..."
pip install reportlab

echo "Installation de weasyprint..."
pip install weasyprint

echo "Installation de python-docx..."
pip install python-docx

echo "✅ Packages PDF installés"

# ========================================
# 5. Packages Image Processing
# ========================================
echo ""
echo "🖼️  5/5: Installation des packages traitement d'images..."
echo "----------------------------------------------------------"

echo "Installation de opencv-python..."
pip install opencv-python

echo "Installation de scikit-image..."
pip install scikit-image

echo "Installation d'imageio..."
pip install imageio

echo "✅ Packages traitement d'images installés"

# ========================================
# Packages additionnels utiles
# ========================================
echo ""
echo "📦 Installation de packages additionnels..."
echo "--------------------------------------------"

pip install \
    networkx \
    rich \
    colorama \
    tqdm \
    click \
    pyyaml \
    toml

echo "✅ Packages additionnels installés"

# ========================================
# Vérification finale
# ========================================
echo ""
echo "🔍 Vérification des installations..."
echo "-------------------------------------"

python -c "
import sys

packages = {
    'torch': 'PyTorch',
    'transformers': 'Transformers',
    'langchain': 'LangChain',
    'streamlit': 'Streamlit',
    'whisper': 'Whisper',
    'TTS': 'Coqui TTS',
    'soundfile': 'SoundFile',
    'sounddevice': 'SoundDevice',
    'open3d': 'Open3D',
    'pyvista': 'PyVista',
    'shapely': 'Shapely',
    'fitz': 'PyMuPDF',
    'cv2': 'OpenCV',
    'PIL': 'Pillow',
    'pygimli': 'PyGIMLi',
    'pyres': 'PyRes',
    'tavily': 'Tavily',
    'sentence_transformers': 'Sentence Transformers',
    'faiss': 'FAISS',
}

print('\\n📦 Vérification des packages:')
print('='*50)

missing = []
for package, name in packages.items():
    try:
        __import__(package)
        print(f'✅ {name}')
    except ImportError:
        print(f'❌ {name} - MANQUANT')
        missing.append(name)

print('='*50)

if missing:
    print(f'\\n⚠️  {len(missing)} packages manquants: {', '.join(missing)}')
    sys.exit(1)
else:
    print('\\n🎉 Tous les packages sont installés!')
    sys.exit(0)
"

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Installation complète réussie!"
    echo ""
    echo "📋 Prochaines étapes:"
    echo "  1. Configurez vos tokens API dans .env"
    echo "  2. Téléchargez les modèles: python download_all_models.py"
    echo "  3. Lancez l'application: streamlit run kibalione8.py"
else
    echo ""
    echo "⚠️  Certains packages n'ont pas pu être installés"
    echo "Consultez les messages d'erreur ci-dessus"
fi
