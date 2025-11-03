#!/bin/bash
# Script de surveillance CPU pour Streamlit

echo "🔍 Surveillance CPU de Streamlit ERT.py"
echo "======================================"
echo ""

# Trouver le PID du processus streamlit
STREAMLIT_PID=$(pgrep -f "streamlit run ERT.py" | head -1)

if [ -z "$STREAMLIT_PID" ]; then
    echo "❌ Streamlit n'est pas en cours d'exécution"
    exit 1
fi

echo "📊 PID Streamlit: $STREAMLIT_PID"
echo ""
echo "Appuyez sur Ctrl+C pour arrêter la surveillance"
echo ""

# Surveillance continue
while true; do
    # Récupérer CPU et mémoire
    CPU=$(ps -p $STREAMLIT_PID -o %cpu= 2>/dev/null)
    MEM=$(ps -p $STREAMLIT_PID -o %mem= 2>/dev/null)
    RSS=$(ps -p $STREAMLIT_PID -o rss= 2>/dev/null)
    
    if [ -z "$CPU" ]; then
        echo "❌ Processus terminé"
        exit 1
    fi
    
    # Convertir RSS en MB
    RSS_MB=$((RSS / 1024))
    
    # Afficher avec couleur selon CPU
    TIMESTAMP=$(date '+%H:%M:%S')
    
    # Déterminer emoji selon charge CPU
    if (( $(echo "$CPU < 30" | bc -l) )); then
        EMOJI="✅"
        STATUS="OPTIMAL"
    elif (( $(echo "$CPU < 60" | bc -l) )); then
        EMOJI="⚠️"
        STATUS="MODÉRÉ"
    else
        EMOJI="🔥"
        STATUS="ÉLEVÉ"
    fi
    
    printf "\r[%s] %s CPU: %5.1f%% (%s) | RAM: %5.1f%% (%d MB)" \
           "$TIMESTAMP" "$EMOJI" "$CPU" "$STATUS" "$MEM" "$RSS_MB"
    
    sleep 2
done
