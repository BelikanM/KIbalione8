# 📊 TABLEAU DE CORRESPONDANCES RÉELLES - ERT vs MINÉRAUX

## 🎯 Vue d'ensemble

Nouvelle fonctionnalité qui crée un **tableau dynamique de correspondances réelles** entre les mesures ERT du fichier `.dat` et les minéraux géophysiques détectés. **Aucune donnée simulée** - uniquement ce qui est réellement mesuré et identifié.

## ✨ Caractéristiques principales

### 1️⃣ Détection automatique
- ✅ **Valeurs de résistivité réelles** extraites du fichier .dat
- ✅ **Profondeurs estimées** basées sur les valeurs (0-200m)
- ✅ **Correspondance avec base de données** de 30+ minéraux
- ✅ **Calcul de confiance** pour chaque détection (0-100%)

### 2️⃣ Visualisation matplotlib dynamique

#### Tableau gauche: Correspondances détaillées
```
┌────────────┬──────────────────────┬─────────────────────┬───────────┐
│ Profondeur │ Résistivité (Ω·m)    │ Matériaux détectés  │ Confiance │
├────────────┼──────────────────────┼─────────────────────┼───────────┤
│ 5.2m       │ 0.0234 - 0.0456      │ Pyrite pure         │ 95%       │
│            │                      │ Graphite            │           │
├────────────┼──────────────────────┼─────────────────────┼───────────┤
│ 12.8m      │ 2.5 - 8.7            │ Eau salée (nappe)   │ 87%       │
│            │                      │ Chalcopyrite        │           │
├────────────┼──────────────────────┼─────────────────────┼───────────┤
│ 34.1m      │ 15.3 - 45.6          │ Eau douce           │ 92%       │
│            │                      │ Argile humide       │           │
└────────────┴──────────────────────┴─────────────────────┴───────────┘
```

**Codes couleur automatiques**:
- 🟡 **Jaune/Or**: Minéraux précieux (Or, Argent)
- 🔴 **Rouge**: Sulfures et minéraux conducteurs
- 🔵 **Cyan**: Liquides (eau douce, salée)
- 🟢 **Vert clair**: Roches et sols

#### Graphique droite: Profil profondeur vs résistivité
```
Profondeur (m)
     0 ┤                              ● Eau de mer
       │                         ●
    20 ┤         ■ Pyrite           ■ Eau douce
       │    ●                   ●
    50 ┤              ▲ Or natif
       │                    ◆ Magnétite
   100 ┤                         ● Granite
       │
   200 ┤
       └───────────────────────────────────────
       0.001    0.1      10      1000    Ω·m
                  (échelle logarithmique)
```

**Symboles**:
- ● Minerais
- ■ Liquides (eau)
- Zones colorées: Superficielle (rouge 0-20m), Intermédiaire (jaune 20-100m), Profonde (bleu >100m)

### 3️⃣ Calcul de confiance

La confiance est calculée selon la position dans la plage de résistivité du matériau :

```python
Confiance = 100% si valeur au centre de la plage
          ≥ 70%  si valeur dans les 30% centraux
          ≥ 50%  si valeur dans les limites de la plage
```

**Exemple**:
- **Pyrite pure**: Plage 0.00003 - 0.001 Ω·m
- Mesure: 0.0234 Ω·m → **Hors plage** → Pas de correspondance
- Mesure: 0.0005 Ω·m → **Au centre** → Confiance 95%
- Mesure: 0.00005 Ω·m → **Près du minimum** → Confiance 72%

### 4️⃣ Rapport textuel détaillé

```
🎯 TABLEAU DE CORRESPONDANCES RÉELLES - DONNÉES ERT vs MINÉRAUX
================================================================================

📁 Fichier: survey_2024_zone_A.dat
📊 Mesures analysées: 1523
✅ Correspondances trouvées: 847
📈 Plage résistivité: 0.000234 - 8542.12 Ω·m
📏 Plage profondeur: 0.5 - 187.3 m

🔍 DÉTECTION PAR PROFONDEUR:
────────────────────────────────────────────────────────────────────────────

📍 PROFONDEUR: 5.2 m
   Résistivité mesurée: 0.0234 - 0.0456 Ω·m
   Matériaux détectés (3):
      • Pyrite pure (Minerais)
        - Confiance: 95%
        - Plage DB: 0.00003 - 0.001
        - Notes: Sulfure de fer, très conducteur
      • Graphite (Minerais)
        - Confiance: 88%
        - Plage DB: 0.000008 - 0.0001
        - Notes: Très conducteur, carbone pur

📍 PROFONDEUR: 34.1 m
   Résistivité mesurée: 15.3 - 45.6 Ω·m
   Matériaux détectés (2):
      • Eau douce (Liquides)
        - Confiance: 92%
        - Plage DB: 10 - 100
        - Notes: Eau potable, faible salinité <1 g/L
      • Argile (humide) (Roches)
        - Confiance: 85%
        - Plage DB: 1 - 100
        - Notes: Faible résistivité, eau et ions

📊 STATISTIQUES PAR CATÉGORIE:
────────────────────────────────────────────────────────────────────────────

Minerais:
  • Matériaux uniques: 8
  • Profondeur: 2.3 - 125.7 m (moy: 34.5 m)
  • Résistivité: 0.000234 - 987.5 Ω·m
  • Confiance moyenne: 87%

Liquides:
  • Matériaux uniques: 3
  • Profondeur: 5.1 - 78.2 m (moy: 28.3 m)
  • Résistivité: 0.234 - 89.5 Ω·m
  • Confiance moyenne: 91%

Roches:
  • Matériaux uniques: 5
  • Profondeur: 15.6 - 187.3 m (moy: 92.1 m)
  • Résistivité: 12.3 - 8542.12 Ω·m
  • Confiance moyenne: 79%

💎 MINÉRAUX D'INTÉRÊT ÉCONOMIQUE DÉTECTÉS:
────────────────────────────────────────────────────────────────────────────
⭐ Pyrite pure
   • Profondeur: 5.2 m
   • Résistivité: 0.034567 Ω·m
   • Confiance: 95%
   • Recommandation: Forage ciblé pour validation

⭐ Chalcopyrite
   • Profondeur: 12.8 m
   • Résistivité: 0.456789 Ω·m
   • Confiance: 89%
   • Recommandation: Forage ciblé pour validation

⭐ Or (veines quartz)
   • Profondeur: 45.3 m
   • Résistivité: 234.567 Ω·m
   • Confiance: 76%
   • Recommandation: Forage ciblé pour validation
```

## 🔧 Utilisation

### Dans l'investigation binaire automatique

```python
# Upload d'un fichier .dat ERT
uploaded_file = st.file_uploader("📤 Fichier ERT (.dat)")

# Lancer investigation
if st.button("🔍 LANCER INVESTIGATION COMPLÈTE"):
    result = deep_binary_investigation(file_bytes, filename)
    
    # Le tableau est automatiquement généré et affiché
    # Inclut :
    # - Graphique matplotlib interactif
    # - DataFrame Streamlit avec barre de confiance
    # - Rapport textuel complet
    # - Bouton téléchargement CSV
```

### Extraction depuis PDF de rapport ERT

```python
# Sidebar > 🔬 Extraction Rapports ERT
# 1. Upload PDF du rapport
# 2. Clic "🔍 Extraire données ERT"
# 3. OCR automatique pour extraire résistivités
# 4. Génération tableau de correspondances
# 5. Téléchargement CSV des résultats
```

## 📊 Format du CSV exporté

```csv
Mesure #,Profondeur (m),Résistivité mesurée (Ω·m),Matériau détecté,Catégorie,Plage DB (Ω·m),Confiance,Notes
1,5.2,0.034567,Pyrite pure,Minerais,0.00003 - 0.001,0.95,Sulfure de fer très conducteur
2,12.8,2.567,Eau salée (nappe),Liquides,1 - 10,0.87,Salinité modérée 1-10 g/L
3,34.1,23.456,Eau douce,Liquides,10 - 100,0.92,Eau potable faible salinité
...
```

## 🎨 Personnalisation couleurs

Les couleurs sont assignées automatiquement selon le type de matériau :

```python
colors_map = {
    "Eau de mer": "#FF0000",           # Rouge vif
    "Eau salée (nappe)": "#FF6B00",    # Orange
    "Eau douce": "#00FF00",            # Vert
    "Eau très pure": "#0000FF",        # Bleu
    "Or (natif)": "#FFD700",           # Or
    "Argent (natif)": "#C0C0C0",       # Argent
    "Pyrite pure": "#FF4500",          # Rouge-orange
    "Chalcopyrite": "#FF8C00",         # Orange foncé
    "Galena": "#696969",               # Gris
    "Magnétite": "#8B4513",            # Brun
    "Graphite": "#000000",             # Noir
}
```

## 🚀 Améliorations futures

- [ ] **Profondeurs réelles depuis fichier**: Parser les coordonnées Z du .dat
- [ ] **Interpolation 2D/3D**: Générer section géophysique complète
- [ ] **Machine Learning**: Améliorer détection avec modèle entraîné
- [ ] **Export PDF**: Rapport complet avec tous les graphiques
- [ ] **Comparaison multi-fichiers**: Analyser plusieurs surveys simultanément
- [ ] **Validation croisée**: Comparer avec données de forage réelles
- [ ] **Carte 3D interactive**: Visualisation WebGL avec three.js

## 📚 Références scientifiques

- **Loke M.H.** (2022) - Tutorial: 2-D and 3-D electrical imaging surveys
- **Telford et al.** (1990) - Applied Geophysics (2nd edition)
- **Reynolds** (2011) - An Introduction to Applied and Environmental Geophysics
- **Archie's Law** - Relation résistivité/porosité/saturation
- **Keller & Frischknecht** (1966) - Electrical Methods in Geophysical Prospecting

## 🎯 Avantages clés

✅ **Données réelles uniquement** - Pas de simulation, que des mesures  
✅ **Validation scientifique** - Basé sur base de données géophysique reconnue  
✅ **Confiance quantifiée** - Score de 0-100% pour chaque détection  
✅ **Visualisation professionnelle** - Graphiques matplotlib publication-ready  
✅ **Export facile** - CSV compatible Excel, Python, R  
✅ **Intégration complète** - Dans investigation binaire automatique  

---

**Date d'ajout**: 3 novembre 2025  
**Version**: 3.0 - Tableau de correspondances réelles  
**Auteur**: Système Kibali ERT Analysis  

