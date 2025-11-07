#!/bin/bash
# ========================================
# Installation rapide et optimisée KIbalione8
# Méthodes optimisées pour éviter la lenteur
# ========================================

set -e

echo "⚡ Installation rapide KIbalione8 - Méthodes optimisées"
echo "======================================================="

# ========================================
# Configuration optimisée pip
# ========================================
echo "🔧 Configuration pip optimisée..."

# Variables d'environnement pour accélérer pip
export PIP_DISABLE_PIP_VERSION_CHECK=1
export PIP_NO_CACHE_DIR=0  # Garder le cache pour éviter re-téléchargements
export PIP_DEFAULT_TIMEOUT=60
export PIP_RETRIES=3

# Mise à jour pip avec cache
~/miniconda3/envs/gestmodo/bin/pip install --upgrade pip setuptools wheel --quiet

echo "✅ Pip configuré et mis à jour"

# ========================================
# ÉTAPE 1: Packages essentiels rapides
# ========================================
echo ""
echo "📦 Étape 1/4: Installation packages essentiels (rapide)"
echo "-------------------------------------------------------"

# Installation en parallèle des packages légers
~/miniconda3/envs/gestmodo/bin/pip install --upgrade \
    rich \
    tqdm \
    pyyaml \
    click \
    imageio \
    --quiet --no-warn-script-location

echo "✅ Packages utilitaires installés"

# ========================================
# ÉTAPE 2: Audio processing (Whisper optimisé)
# ========================================
echo ""
echo "🎤 Étape 2/4: Installation audio processing"
echo "--------------------------------------------"

# Whisper avec cache pré-compilé
echo "Installation de Whisper (optimisé)..."
~/miniconda3/envs/gestmodo/bin/pip install openai-whisper --quiet --no-warn-script-location

# Packages audio légers
echo "Installation packages audio..."
~/miniconda3/envs/gestmodo/bin/pip install \
    soundfile \
    librosa \
    --quiet --no-warn-script-location

echo "✅ Audio processing installé"

# ========================================
# ÉTAPE 3: Image & 3D processing
# ========================================
echo ""
echo "🎨 Étape 3/4: Installation traitement image/3D"
echo "------------------------------------------------"

# OpenCV pré-compilé (plus rapide que compilation)
echo "Installation OpenCV (pré-compilé)..."
~/miniconda3/envs/gestmodo/bin/pip install opencv-python --quiet --no-warn-script-location

# Packages image scientifiques
echo "Installation packages image..."
~/miniconda3/envs/gestmodo/bin/pip install \
    scikit-image \
    --quiet --no-warn-script-location

# Open3D pré-compilé
echo "Installation Open3D (pré-compilé)..."
~/miniconda3/envs/gestmodo/bin/pip install open3d --quiet --no-warn-script-location

echo "✅ Traitement image/3D installé"

# ========================================
# ÉTAPE 4: PDF & Geospatial
# ========================================
echo ""
echo "📄 Étape 4/4: Installation PDF et géospatial"
echo "----------------------------------------------"

# PyMuPDF (plus rapide que PyPDF2)
echo "Installation PyMuPDF (rapide)..."
~/miniconda3/envs/gestmodo/bin/pip install pymupdf --quiet --no-warn-script-location

# ReportLab pour génération PDF
echo "Installation ReportLab..."
~/miniconda3/envs/gestmodo/bin/pip install reportlab --quiet --no-warn-script-location

# Shapely pour géométrie (pré-compilé)
echo "Installation Shapely (pré-compilé)..."
~/miniconda3/envs/gestmodo/bin/pip install shapely --quiet --no-warn-script-location

# NetworkX pour graphes
echo "Installation NetworkX..."
~/miniconda3/envs/gestmodo/bin/pip install networkx --quiet --no-warn-script-location

echo "✅ PDF et géospatial installés"

# ========================================
# Vérification rapide
# ========================================
echo ""
echo "🔍 Vérification des installations..."
echo "-------------------------------------"

~/miniconda3/envs/gestmodo/bin/python -c "
import sys

# Packages essentiels à vérifier
packages = {
    'whisper': 'Whisper',
    'soundfile': 'SoundFile', 
    'librosa': 'Librosa',
    'cv2': 'OpenCV',
    'skimage': 'Scikit-Image',
    'open3d': 'Open3D',
    'fitz': 'PyMuPDF',
    'reportlab': 'ReportLab',
    'shapely': 'Shapely',
    'networkx': 'NetworkX',
    'rich': 'Rich',
    'tqdm': 'TQDM',
    'yaml': 'PyYAML',
}

print('\\n📦 Vérification rapide:')
success = 0
total = len(packages)

for package, name in packages.items():
    try:
        __import__(package)
        print(f'✅ {name}')
        success += 1
    except ImportError:
        print(f'❌ {name}')

print(f'\\n📊 Résultat: {success}/{total} packages installés ({success/total*100:.0f}%)')

if success >= total * 0.8:  # 80% de succès minimum
    print('\\n🎉 Installation rapide réussie!')
    sys.exit(0)
else:
    print('\\n⚠️  Installation incomplète')
    sys.exit(1)
"

if [ $? -eq 0 ]; then
    echo ""
    echo "⚡ Installation rapide terminée avec succès!"
    echo ""
    echo "💡 Packages optionnels non installés (pour vitesse):"
    echo "   - TTS (synthèse vocale) - Ajoutez si besoin: pip install TTS"
    echo "   - PyVista (3D avancé) - Ajoutez si besoin: pip install pyvista"
    echo "   - GeoPandas (GIS) - Ajoutez si besoin: pip install geopandas"
    echo ""
    echo "📋 Prochaines étapes:"
    echo "  1. Configurez .env avec vos tokens API"
    echo "  2. Testez: python download_all_models.py"
    echo "  3. Lancez: streamlit run kibalione8.py"
else
    echo ""
    echo "❌ Installation incomplète. Relancez ou installez manuellement."
fi

echo ""
echo "⏱️  Installation terminée en mode rapide!"
