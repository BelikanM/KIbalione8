#!/usr/bin/env python3
"""
Script pour vérifier et installer tous les modules requis par ERT.py
"""

import subprocess
import sys

# Mapping des imports vers les packages pip
PACKAGE_MAP = {
    'PIL': 'pillow',
    'cv2': 'opencv-python',
    'bs4': 'beautifulsoup4',
    'skimage': 'scikit-image',
    'sklearn': 'scikit-learn',
    'fitz': 'pymupdf',
    'dotenv': 'python-dotenv',
}

# Modules à vérifier (extraits de ERT.py)
REQUIRED_MODULES = [
    'numpy', 'pandas', 'matplotlib', 'scipy',
    'PIL', 'cv2', 'open3d',
    'torch', 'transformers', 'sentence_transformers',
    'langchain', 'langchain_community', 'langchain_core', 
    'langchain_huggingface', 'langchain_tavily', 'langchain_text_splitters',
    'huggingface_hub', 'safetensors',
    'streamlit', 'networkx', 'shapely',
    'fitz', 'osmium', 'bs4',
    'skimage', 'sklearn', 'imageio',
    'whisper', 'gtts', 'pytesseract', 'speech_recognition',
    'ultralytics', 'pdf2image',
    'tavily', 'requests', 'dotenv',
    'faiss', 'pygimli', 'pyres',
]

def get_pip_package_name(module_name):
    """Retourne le nom du package pip pour un module"""
    return PACKAGE_MAP.get(module_name, module_name.replace('_', '-'))

def check_module(module_name):
    """Vérifie si un module est installé"""
    try:
        __import__(module_name)
        return True
    except ImportError:
        return False

def main():
    print("🔍 Vérification de tous les modules requis par ERT.py")
    print("="*60)
    
    missing = []
    installed = []
    
    for module in REQUIRED_MODULES:
        if check_module(module):
            installed.append(module)
            print(f"✅ {module}")
        else:
            missing.append(module)
            print(f"❌ {module}")
    
    print("="*60)
    print(f"\n📊 Résultat: {len(installed)}/{len(REQUIRED_MODULES)} modules installés")
    
    if missing:
        print(f"\n📦 {len(missing)} modules manquants à installer:")
        
        # Convertir les noms de modules en noms de packages pip
        packages_to_install = []
        for module in missing:
            pkg = get_pip_package_name(module)
            packages_to_install.append(pkg)
            print(f"  - {module} → {pkg}")
        
        print(f"\n💡 Commande d'installation:")
        cmd = f"pip install {' '.join(packages_to_install)}"
        print(f"  {cmd}")
        
        # Demander confirmation
        response = input("\n❓ Installer maintenant ? (y/N): ")
        if response.lower() == 'y':
            print("\n🚀 Installation en cours...")
            try:
                subprocess.check_call([
                    sys.executable, '-m', 'pip', 'install', '--no-cache-dir'
                ] + packages_to_install)
                print("\n✅ Installation terminée!")
                
                # Revérifier
                print("\n🔍 Revérification...")
                still_missing = [m for m in missing if not check_module(m)]
                if still_missing:
                    print(f"⚠️  {len(still_missing)} modules toujours manquants:")
                    for m in still_missing:
                        print(f"  - {m}")
                else:
                    print("🎉 Tous les modules sont maintenant installés!")
                    
            except subprocess.CalledProcessError as e:
                print(f"\n❌ Erreur lors de l'installation: {e}")
                return 1
        else:
            print("\n⏭️  Installation annulée")
            return 1
    else:
        print("\n🎉 Tous les modules requis sont déjà installés!")
        return 0

if __name__ == "__main__":
    sys.exit(main())
