#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Wrapper de lancement pour ERT.py
Force l'utilisation de l'environnement gestmodo (Python 3.10)
Bloque le lancement avec Python 3.13 ou autre
"""

import sys
import os
import subprocess
from pathlib import Path

# Définir les versions Python autorisées
REQUIRED_PYTHON_VERSION = (3, 10)
GESTMODO_PYTHON = Path.home() / "miniconda3" / "envs" / "gestmodo" / "bin" / "python"
GESTMODO_STREAMLIT = Path.home() / "miniconda3" / "envs" / "gestmodo" / "bin" / "streamlit"

def get_python_version():
    """Retourne la version Python courante"""
    return sys.version_info[:2]

def check_environment():
    """Vérifie que nous sommes dans le bon environnement"""
    current_version = get_python_version()
    
    print(f"🔍 Version Python actuelle: {current_version[0]}.{current_version[1]}")
    print(f"📦 Environnement requis: gestmodo (Python {REQUIRED_PYTHON_VERSION[0]}.{REQUIRED_PYTHON_VERSION[1]})")
    
    # Vérifier si nous sommes dans gestmodo
    current_env = os.environ.get('CONDA_DEFAULT_ENV', 'unknown')
    print(f"🌍 Environnement conda actuel: {current_env}")
    
    if current_version != REQUIRED_PYTHON_VERSION:
        print(f"\n❌ ERREUR: Version Python incorrecte!")
        print(f"   Python {current_version[0]}.{current_version[1]} détecté")
        print(f"   Python {REQUIRED_PYTHON_VERSION[0]}.{REQUIRED_PYTHON_VERSION[1]} requis (gestmodo)")
        print(f"\n🔄 Redémarrage avec l'environnement correct...\n")
        return False
    
    if current_env != 'gestmodo':
        print(f"\n⚠️  Environnement conda incorrect: {current_env}")
        print(f"   Environnement 'gestmodo' requis")
        print(f"\n🔄 Redémarrage avec l'environnement correct...\n")
        return False
    
    print(f"✅ Environnement correct détecté!\n")
    return True

def launch_with_gestmodo():
    """Lance l'application avec l'environnement gestmodo"""
    if not GESTMODO_PYTHON.exists():
        print(f"❌ Erreur: Python gestmodo non trouvé!")
        print(f"   Chemin: {GESTMODO_PYTHON}")
        sys.exit(1)
    
    if not GESTMODO_STREAMLIT.exists():
        print(f"❌ Erreur: Streamlit non installé dans gestmodo!")
        print(f"   Installez avec: conda activate gestmodo && pip install streamlit")
        sys.exit(1)
    
    # Arrêter les instances existantes
    print("🔄 Arrêt des instances Streamlit existantes...")
    subprocess.run(["pkill", "-9", "-f", "streamlit"], stderr=subprocess.DEVNULL)
    
    # Chemin vers ERT.py
    ert_path = Path(__file__).parent / "ERT.py"
    
    # Lancer avec gestmodo
    print(f"🚀 Lancement avec gestmodo...")
    print(f"   Python: {GESTMODO_PYTHON}")
    print(f"   Script: {ert_path}")
    print()
    
    try:
        subprocess.run([
            str(GESTMODO_STREAMLIT),
            "run",
            str(ert_path),
            "--server.port", "8503"
        ])
    except KeyboardInterrupt:
        print("\n🛑 Arrêt de l'application")

def main():
    """Point d'entrée principal"""
    print("=" * 50)
    print("  🎯 Lanceur ERT - Environnement gestmodo")
    print("=" * 50)
    print()
    
    # Vérifier l'environnement
    if check_environment():
        # Si nous sommes déjà dans gestmodo, lancer directement
        print("✅ Environnement correct, lancement de Streamlit...")
        os.system(f"cd {Path(__file__).parent} && streamlit run ERT.py --server.port 8503")
    else:
        # Sinon, relancer avec gestmodo
        launch_with_gestmodo()

if __name__ == "__main__":
    main()
