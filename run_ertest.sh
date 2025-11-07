#!/bin/bash
# ========================================
# Script de lancement ERTest.py
# Force l'utilisation de l'environnement gestmodo (Python 3.10)
# ========================================

# Couleurs pour les messages
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}  Lancement de l'application ERTest${NC}"
echo -e "${GREEN}========================================${NC}"

# Vérification de l'environnement Python actuel
CURRENT_PYTHON=$(python --version 2>&1)
echo -e "${YELLOW}Python actuel: $CURRENT_PYTHON${NC}"

# Définir le chemin de l'environnement gestmodo
GESTMODO_PYTHON="$HOME/miniconda3/envs/gestmodo/bin/python"

# Vérifier que l'environnement gestmodo existe
if [ ! -f "$GESTMODO_PYTHON" ]; then
    echo -e "${RED}❌ Erreur: Environnement gestmodo non trouvé!${NC}"
    echo -e "${RED}   Chemin attendu: $GESTMODO_PYTHON${NC}"
    exit 1
fi

# Vérifier que streamlit est installé dans gestmodo
if ! $GESTMODO_PYTHON -m streamlit --version &>/dev/null; then
    echo -e "${YELLOW}⚠️  Streamlit non trouvé dans gestmodo, installation...${NC}"
    $GESTMODO_PYTHON -m pip install streamlit -q
fi

# Arrêter toutes les instances Streamlit en cours
echo -e "${YELLOW}🔄 Arrêt des instances Streamlit existantes...${NC}"
pkill -9 -f streamlit 2>/dev/null || true
sleep 2

# Aller dans le répertoire du projet
cd /home/belikan/KIbalione8 || exit 1

# Vérifier la version de Python dans gestmodo
GESTMODO_VERSION=$($GESTMODO_PYTHON --version 2>&1)
echo -e "${GREEN}✅ Utilisation de: $GESTMODO_VERSION${NC}"
echo -e "${GREEN}✅ Environnement: gestmodo${NC}"
echo ""

# Lancer l'application avec l'environnement gestmodo
echo -e "${GREEN}🚀 Démarrage de l'application ERTest...${NC}"
echo -e "${GREEN}📍 URL: http://localhost:8504${NC}"
echo ""
$GESTMODO_PYTHON -m streamlit run ERTest.py --server.port 8504

# Si le script se termine (Ctrl+C), nettoyer
echo -e "${YELLOW}🛑 Arrêt de l'application${NC}"
