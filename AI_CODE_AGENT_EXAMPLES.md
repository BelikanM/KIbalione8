# 🤖 AI Code Agent - Exemples d'Utilisation

## Vue d'ensemble

Kibali peut maintenant **générer et exécuter du code automatiquement** pour accomplir des tâches complexes, exactement comme GitHub Copilot !

## 🎯 Exemples Concrets

### 1. Analyse Géophysique

```
Utilisateur: "Kibali, analyse les profondeurs où se trouve l'eau salée dans resistivity.npy"

Kibali:
🤖 ACTION EXÉCUTÉE AVEC SUCCÈS

📋 Tâche: analyze sur resistivity.npy

📊 Résultats:
📊 Données chargées: shape=(50, 100), dtype=float64

🌊 RÉSULTATS ANALYSE EAU SALÉE:
Seuil résistivité: 10 Ω·m
Nombre de points détectés: 342

Profondeurs trouvées:
  - Profondeur 15: 28 points
  - Profondeur 20: 45 points
  - Profondeur 25: 67 points
  - Profondeur 30: 89 points
  - Profondeur 35: 113 points

Statistiques:
  Profondeur min: 15
  Profondeur max: 35
  Profondeur moyenne: 27.84
```

### 2. Recherche d'Anomalies

```
Utilisateur: "Va chercher les anomalies de résistivité dans data_ert.npy et dis-moi ce que tu trouves"

Kibali:
🤖 ACTION EXÉCUTÉE AVEC SUCCÈS

📊 Résultats:
🔍 RECHERCHE DANS data_ert.npy
Shape: (100, 200), Type: float64

📊 Statistiques globales:
  Min: 2.34 Ω·m
  Max: 456.78 Ω·m
  Moyenne: 87.45 Ω·m
  Médiane: 72.30 Ω·m

⚡ Anomalies détectées:
  Basse résistivité (< 12.67): 1247 points
  Haute résistivité (> 162.23): 892 points
```

### 3. Génération de Rapport

```
Utilisateur: "Crée un rapport ERT complet pour le fichier survey_2024.npy"

Kibali:
🤖 ACTION EXÉCUTÉE AVEC SUCCÈS

📊 Résultats:
============================================================
RAPPORT D'ANALYSE ERT
============================================================
Fichier: survey_2024.npy
Date: 2025-11-04 02:15:30

DONNÉES:
- Shape: (75, 150)
- Type: float64
- Taille: 87.89 KB

STATISTIQUES:
- Min: 5.23 Ω·m
- Max: 289.45 Ω·m  
- Moyenne: 94.67 Ω·m
- Écart-type: 45.12 Ω·m

INTERPRÉTATION:

- Très basse résistivité (< 10 Ω·m): 345 points
  → Eau salée, argile saturée
  
- Basse résistivité (10-50 Ω·m): 1023 points
  → Argile, sable humide
  
- Résistivité moyenne (50-200 Ω·m): 7854 points
  → Sol mixte, sable sec
  
- Haute résistivité (> 200 Ω·m): 2028 points
  → Roche, gravier sec

============================================================

💾 Rapport sauvegardé: rapport_ert_20251104_021530.txt
```

### 4. Traitement de Fichier Binaire

```
Utilisateur: "Traite le fichier calibration_image_sample_data_20x128x128x3_float32.npy et extrais les statistiques"

Kibali:
🤖 ACTION EXÉCUTÉE AVEC SUCCÈS

📊 Résultats:
📊 Données chargées: shape=(20, 128, 128, 3), dtype=float32
Type: Images (4D array)

Statistiques par canal:
  Canal 0 (R): min=0.12, max=0.98, moyenne=0.54
  Canal 1 (G): min=0.08, max=0.95, moyenne=0.51
  Canal 2 (B): min=0.15, max=0.92, moyenne=0.56

Qualité des images:
  - Contraste moyen: 0.43
  - Luminosité moyenne: 0.54
  - Images valides: 20/20
```

## 🔧 Types d'Actions Supportées

### 1. **analyze** - Analyse de données
**Mots-clés**: analyse, analyser, examine, étudie, vérifie

**Exemples**:
- "Analyse la profondeur de l'aquifère dans ert_data.npy"
- "Examine les valeurs de résistivité dans survey.bin"
- "Étudie la distribution des données dans results.npz"

### 2. **search** - Recherche de patterns
**Mots-clés**: cherche, trouve, recherche, localise, détecte

**Exemples**:
- "Cherche les zones conductrices dans data.npy"
- "Trouve les anomalies thermiques dans temperature.bin"
- "Détecte les variations brusques dans timeseries.dat"

### 3. **create** - Génération de contenu
**Mots-clés**: crée, génère, fabrique, construis, produis

**Exemples**:
- "Crée un rapport complet pour analyse.npy"
- "Génère un graphique de résistivité pour ert.bin"
- "Produis une synthèse des données de survey.npz"

### 4. **process** - Traitement de données
**Mots-clés**: traite, transforme, convertis, calcule, extrait

**Exemples**:
- "Traite les données brutes de raw_data.bin"
- "Convertis le fichier numpy en CSV"
- "Calcule la moyenne mobile sur timeseries.npy"

### 5. **visualize** - Visualisation
**Mots-clés**: affiche, montre, visualise, dessine, trace

**Exemples**:
- "Affiche un heatmap de resistivity.npy"
- "Trace un profil 2D de la section ERT"
- "Visualise la distribution spatiale des données"

## 📂 Formats de Fichiers Supportés

✅ **NumPy**: `.npy`, `.npz`
✅ **Binaire**: `.bin`, `.dat`
✅ **Texte**: `.txt`, `.csv`
✅ **JSON**: `.json`
✅ **PDF**: `.pdf` (extraction)

## 🎨 Syntaxe des Commandes

### Structure recommandée
```
[Action] [Détails] dans/pour [Fichier] [Paramètres optionnels]
```

### Exemples structurés

**Simple**:
```
"Analyse data.npy"
```

**Avec détails**:
```
"Analyse les profondeurs dans survey_2024.npy"
```

**Avec paramètres**:
```
"Cherche les zones d'eau salée (résistivité < 10) dans ert_results.npy"
```

**Complexe**:
```
"Crée un rapport ERT complet avec graphiques pour le fichier geo_survey_site_A.npy et sauvegarde-le en PDF"
```

## ⚙️ Paramètres Détectés Automatiquement

### Géophysique
- `profondeur` → `depth_analysis: true`
- `eau salée` → `water_type: saline`
- `eau douce` → `water_type: fresh`
- `résistivité` → `resistivity: true`
- `ert`, `géophysique` → `geophysics: true`

### Statistiques
- `moyenne`, `médiane`, `écart-type` → stats détaillées
- `minimum`, `maximum` → extrema analysis
- `distribution` → histogramme

### Visualisation
- `graphique`, `plot` → génération de figures
- `heatmap`, `carte` → visualisation 2D
- `profil`, `section` → coupe transversale

## 🛡️ Sécurité & Limites

### Sandbox d'exécution
- ✅ Code exécuté dans subprocess isolé
- ✅ Timeout de 30 secondes par défaut
- ✅ Fichiers temporaires auto-nettoyés
- ✅ Capture stderr pour debugging

### Limitations
- ⚠️ Fichiers doivent exister dans le workspace
- ⚠️ Permissions de lecture nécessaires
- ⚠️ Taille maximale ~500MB recommandée
- ⚠️ Pas d'accès réseau depuis le code généré

## 💡 Conseils d'Utilisation

### ✅ Bonnes pratiques

1. **Soyez spécifique**
   ```
   ❌ "Analyse le fichier"
   ✅ "Analyse les profondeurs d'eau salée dans resistivity_survey.npy"
   ```

2. **Mentionnez le fichier complet**
   ```
   ❌ "Cherche dans data"
   ✅ "Cherche dans data_ert_2024.npy"
   ```

3. **Indiquez le type de résultat souhaité**
   ```
   ❌ "Traite survey.npy"
   ✅ "Traite survey.npy et crée un rapport avec statistiques"
   ```

### ❌ À éviter

- Commandes trop vagues
- Fichiers sans extension
- Actions ambiguës
- Multiples fichiers sans clarification

## 🔍 Inspection du Code Généré

Le code Python généré est toujours **visible** dans un expander :

```python
# Exemple de code généré automatiquement
import numpy as np
import os

file_path = "resistivity.npy"
data = np.load(file_path)

# Analyser les profondeurs d'eau salée
resistivity_threshold = 10  # Ω·m
saline_locations = np.where(data < resistivity_threshold)
depths = saline_locations[0]

print(f"Profondeurs détectées: {np.unique(depths)}")
```

Vous pouvez :
- ✅ Voir exactement ce que Kibali exécute
- ✅ Copier le code pour réutilisation
- ✅ Modifier et exécuter manuellement
- ✅ Apprendre des exemples générés

## 📊 Historique d'Exécution

L'agent garde un historique de toutes les exécutions :
- Code généré
- Succès/échec
- Sortie standard (stdout)
- Erreurs (stderr)

Accessible via `st.session_state.code_agent.execution_history`

## 🚀 Cas d'Usage Avancés

### 1. Pipeline d'analyse complet
```
"Analyse survey.npy, trouve les anomalies, et crée un rapport PDF complet"
```

Kibali va :
1. Charger les données
2. Calculer les statistiques
3. Détecter les anomalies
4. Générer le rapport
5. Sauvegarder en PDF

### 2. Comparaison de fichiers
```
"Compare les résistivités entre site_A.npy et site_B.npy"
```

### 3. Validation de qualité
```
"Vérifie la qualité des données dans raw_acquisition.bin et signale les erreurs"
```

### 4. Extraction ciblée
```
"Extrais seulement les profondeurs entre 20 et 50 mètres de ert_deep.npy"
```

## 🎓 Apprentissage Continu

Le système peut être étendu avec :
- Nouveaux templates de code
- Modèles de ML spécialisés
- Intégration d'outils externes
- Génération de visualisations avancées

---

**Version**: 1.0  
**Date**: 4 novembre 2025  
**Auteur**: GitHub Copilot  
**Contexte**: Kibali AI Code Agent - Exécution autonome de tâches
