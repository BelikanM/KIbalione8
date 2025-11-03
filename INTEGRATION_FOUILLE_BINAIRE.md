# 🔍 Système de Fouille Intelligente de Fichiers Binaires

## 📊 Vue d'ensemble

Kibali AI intègre maintenant un **système de fouille intelligente** inspiré de l'agent VSCode avec todo list, permettant une analyse approfondie des fichiers binaires en combinant plusieurs sources de connaissances.

## 🎯 Objectif

Analyser scientifiquement les fichiers binaires uploadés en combinant :
- ✅ **Hex + ASCII Dump** (analyse brute)
- ✅ **Base Vectorielle RAG** (connaissances documentaires)
- ✅ **Base ERT spécialisée** (géophysique)
- ✅ **Recherche Web** (contexte externe)
- ✅ **Synthèse IA** (interprétation intelligente)

## 🚀 Fonctionnalités

### 1️⃣ Phase 1: Extraction Hex + ASCII
```
📜 Dump hexadécimal complet (100 premières lignes)
🔢 Extraction automatique des nombres
📊 Statistiques: Range, Moyenne, Médiane, Écart-type
```

**Exemple:**
```
00000000 7F 45 4C 46 02 01 01 00 00 00 00 00 00 00 00 00 |.ELF............|
00000010 02 00 3E 00 01 00 00 00 50 10 40 00 00 00 00 00 |..>.....P.@.....|
```

### 2️⃣ Phase 2: Analyses Techniques
```
📊 Entropie: Mesure de randomisation (0-8 bits)
🎯 Patterns: Détection format (ELF, JPEG, PDF, etc.)
📋 Métadonnées: Extraction headers, signatures
🗜️ Compression: Ratio de compression
📈 Fréquences: Distribution des bytes
```

**Indicateurs:**
- **Entropie < 3**: Fichier structuré (texte, code)
- **Entropie 3-6**: Données compressées
- **Entropie > 7**: Fichier chiffré ou très compressé

### 3️⃣ Phase 3: Fouille Base Vectorielle RAG

Le système construit automatiquement des **requêtes intelligentes** basées sur les patterns détectés :

| Pattern Détecté | Requêtes RAG Générées |
|----------------|----------------------|
| **ELF/Executable** | "analyse fichier exécutable binaire ELF format Linux sécurité" |
| **JPEG/PNG** | "format image JPEG PNG métadonnées EXIF analyse forensique" |
| **PDF** | "structure PDF analyse document métadonnées forensique" |
| **Données ERT** | "ERT electrical resistivity tomography geophysics data interpretation" |
| **Haute entropie** | "fichier chiffré crypté haute entropie analyse cryptographique" |

**Résultats:**
- Recherche dans tous les PDFs indexés
- Extraction de connaissances pertinentes
- Contextualisation scientifique

### 4️⃣ Phase 4: Fouille Spécialisée ERT

Détection automatique de données géophysiques ERT :

**Critères de détection:**
```python
✅ Plage de résistivité: 0.1 - 10,000 Ohm.m
✅ Distribution log-normale (test Shapiro-Wilk)
✅ Nombre de mesures suffisant (> 50)
✅ Coefficient de variation élevé (hétérogénéité)
```

**Si données ERT détectées:**
- Clustering automatique (K-means)
- Classification géologique
- Requêtes spécialisées dans base ERT
- Recherche de matériaux correspondants

### 5️⃣ Phase 5: Recherche Web Contextualisée

Construction de requêtes web basées sur **tous les indices** collectés :

```
"analyse {type_fichier} fichier binaire format {nom_fichier}"
```

**Sources:**
- DuckDuckGo (privé, sans tracking)
- Résultats filtrés et pertinents
- Extraction de contexte scientifique

### 6️⃣ Phase 6: Synthèse IA Multi-Sources

Utilisation du **modèle Qwen2.5-1.5B** pour synthétiser :
- Toutes les analyses techniques
- Connaissances RAG extraites
- Détection ERT
- Résultats web

**Prompt de synthèse:**
```
Fichier: {nom} ({taille} bytes)
Type: {pattern_détecté}
Entropie: {entropie}
Connaissances RAG: {extraits}
Détection ERT: {analyse_ert}
Web: {résultats_web}

→ Interprétation scientifique complète
```

**Optimisations:**
- GPU/CPU auto-détection
- max_new_tokens=1000 pour synthèse complète
- temperature=0.7 pour créativité maîtrisée
- torch.inference_mode() pour performance

### 7️⃣ Phase 7: Recommandations Actionnables

Le système génère des **actions concrètes** basées sur les découvertes :

| Détection | Recommandations |
|-----------|----------------|
| **Données ERT** | ✅ Utiliser PyGIMLI pour inversion<br>✅ Visualiser avec AI_Plot_Generator<br>✅ Calculer résistivité apparente |
| **Entropie élevée** | 🔒 Fichier potentiellement chiffré<br>🔍 Analyser avec outils cryptographiques |
| **Fichier exécutable** | ⚠️ Analyser avec reverse engineering<br>🛡️ Scanner avec antivirus |

## 🎓 Utilisation

### Interface Streamlit

1. **Upload fichier binaire**
   ```
   Types supportés: .bin, .dat, .raw, .safetensors, .pt, .ckpt
   ```

2. **Cliquer sur "🔬 LANCER INVESTIGATION COMPLÈTE"**
   - 7 phases exécutées automatiquement
   - Progression affichée en temps réel
   - Rapport complet généré

3. **Consulter le rapport**
   - Affichage dans expander
   - Téléchargement en .txt possible
   - Toutes les sources détaillées

### Via l'Agent LangChain

L'agent peut utiliser l'outil **Deep_Binary_Investigation** automatiquement :

```python
Agent: Deep_Binary_Investigation
Input: nom_fichier.dat
Output: Rapport complet 7 phases
```

**Déclenchement automatique:**
- User: "Analyse ce fichier binaire en profondeur"
- User: "Que contient ce fichier .dat ?"
- User: "Fouille toutes les sources pour ce fichier"

## 📊 Exemple de Rapport Complet

```
🔬 RAPPORT D'INVESTIGATION BINAIRE APPROFONDIE
================================================================================

1️⃣ PHASE 1: EXTRACTION HEX + ASCII
────────────────────────────────────────────────────────────────────────────────
📜 Dump hexadécimal (2048 bytes):
00000000 7F 45 4C 46 02 01 01 00 00 00 00 00 00 00 00 00 |.ELF............|
...

🔢 Nombres extraits: 450 valeurs
   • Range: 1.250 - 8950.300
   • Moyenne: 125.450 ± 234.890
   • Médiane: 89.300

2️⃣ PHASE 2: ANALYSES TECHNIQUES
────────────────────────────────────────────────────────────────────────────────
📊 Entropie: 5.234 / 8 (modérément aléatoire)
🎯 Patterns: Format ERT DAT détecté (Syscal)
📋 Métadonnées: Header Syscal Pro, 4 électrodes, 128 mesures
🗜️ Compression: Ratio 1.23 (peu compressible)
📈 Fréquences: Distribution bimodale (2 pics)

3️⃣ PHASE 3: FOUILLE BASE VECTORIELLE RAG
────────────────────────────────────────────────────────────────────────────────
🔍 Requête: 'ERT electrical resistivity tomography geophysics'
   Résultat: La tomographie de résistivité électrique (ERT) est une méthode...
   
🔍 Requête: 'résistivité électrique inversion subsurface'
   Résultat: L'inversion permet de calculer la distribution 2D/3D de...

4️⃣ PHASE 4: FOUILLE SPÉCIALISÉE ERT
────────────────────────────────────────────────────────────────────────────────
🔍 ANALYSE SPÉCIALISÉE ERT (Résistivité Électrique)
==================================================

📊 Valeurs résistivité: 1.250 - 8950.300 ✅ Plage typique ERT

📈 Statistiques:
 • Moyenne: 125.450
 • Écart-type: 234.890
 • Coefficient de variation: 1.871
 • Médiane: 89.300

📊 Distribution: Log-normale (p=0.234) ✅ Typique ERT

🎯 Clustering résistivité (5 groupes):
 • Groupe 1: 2.500 Ohm.m (85 valeurs)  → Argile saturée
 • Groupe 2: 25.300 Ohm.m (120 valeurs) → Sol sableux
 • Groupe 3: 150.700 Ohm.m (95 valeurs) → Roche altérée
 • Groupe 4: 450.200 Ohm.m (75 valeurs) → Roche saine
 • Groupe 5: 2500.100 Ohm.m (75 valeurs) → Cavité/Air

📚 CONNAISSANCES ERT DE LA BASE:
🔍 résistivité 125.4 Ohm.m interprétation géologique:
   Les valeurs autour de 125 Ohm.m sont typiques de...

5️⃣ PHASE 5: RECHERCHE WEB INTELLIGENTE
────────────────────────────────────────────────────────────────────────────────
🌐 Recherche: 'analyse Format ERT DAT fichier binaire mesure_terrain.dat'
La tomographie électrique ERT permet d'investiguer le sous-sol...
Format DAT Syscal: colonnes A, B, M, N, Resistivity, IP...

6️⃣ PHASE 6: SYNTHÈSE MULTI-SOURCES
────────────────────────────────────────────────────────────────────────────────
🤖 SYNTHÈSE IA:
Ce fichier est un ensemble de mesures ERT (Electrical Resistivity Tomography)
acquises avec un système Syscal Pro. Les 450 mesures couvrent une plage de
résistivité de 1.25 à 8950 Ohm.m, typique d'un milieu hétérogène avec:

- Argiles saturées (2-10 Ohm.m) en profondeur
- Sols sableux (20-50 Ohm.m) en surface
- Roches altérées (100-300 Ohm.m) en zone intermédiaire
- Roches saines (400-1000 Ohm.m) en profondeur
- Possibles cavités (>2000 Ohm.m) localisées

La distribution log-normale confirme un contexte géologique naturel.
Le coefficient de variation élevé (1.87) indique une forte hétérogénéité,
suggérant une zone fracturée ou karstique.

INTERPRÉTATION GÉOLOGIQUE: Site probablement calcaire avec circulation
d'eau souterraine, présence de fractures et possibles cavités karstiques.

7️⃣ PHASE 7: RECOMMANDATIONS
────────────────────────────────────────────────────────────────────────────────
✅ Données ERT détectées → Utiliser PyGIMLI pour inversion
✅ Visualiser avec matplotlib/seaborn (utiliser AI_Plot_Generator)
✅ Calculer résistivité apparente avec mathematical_calculator

================================================================================
✅ INVESTIGATION TERMINÉE - Rapport complet généré
```

## 🔧 Configuration Technique

### Fonction Principale

```python
def deep_binary_investigation(file_bytes: bytes, file_name: str = "unknown") -> str:
    """
    🔍 FOUILLE INTELLIGENTE DE FICHIER BINAIRE
    Combine Hex+ASCII Dump + Base Vectorielle RAG + Base ERT
    """
    # 1. Extraction Hex+ASCII
    hex_dump = hex_ascii_view(file_bytes, max_lines=100)
    numbers = extract_numbers(file_bytes)
    
    # 2. Analyses techniques
    entropy = entropy_analysis(file_bytes)
    patterns = pattern_recognition(file_bytes)
    metadata = metadata_extraction(file_bytes)
    
    # 3. Fouille RAG (requêtes intelligentes)
    for query in rag_queries:
        result = search_vectorstore(query)
    
    # 4. Fouille ERT (si applicable)
    if is_ert_data(numbers):
        ert_analysis = ert_data_detection(file_bytes, numbers)
    
    # 5. Recherche Web
    web_result = web_search_enhanced(context_query)
    
    # 6. Synthèse IA
    synthesis = model.generate(combined_context)
    
    # 7. Recommandations
    recommendations = generate_recommendations(all_findings)
    
    return full_report
```

### Intégration LangChain

```python
Tool(
    name="Deep_Binary_Investigation",
    func=lambda file_name: deep_binary_investigation(
        file_bytes, 
        file_name
    ),
    description="🔍 FOUILLE INTELLIGENTE fichiers binaires uploadés"
)
```

### Fichiers Concernés

```
ERT.py
├── Lignes 963-1182: deep_binary_investigation() (219 lignes)
├── Lignes 1692-1721: Interface Streamlit avec bouton
├── Lignes 3496: Outil Deep_Binary_Investigation
└── Lignes 3524: Documentation dans prompt agent
```

## 📈 Performance

| Métrique | Valeur |
|----------|--------|
| **Temps d'analyse** | 5-15 secondes (selon taille) |
| **Requêtes RAG** | 3-6 requêtes intelligentes |
| **Requêtes Web** | 1 requête contextualisée |
| **Tokens synthèse** | ~1000 tokens générés |
| **GPU/CPU** | Auto-détection optimisée |

## 🎯 Avantages vs Analyse Simple

| Critère | Analyse Simple | Fouille Intelligente |
|---------|---------------|---------------------|
| **Sources** | 1 (Hex dump) | 5 (Hex + RAG + ERT + Web + IA) |
| **Contexte** | Aucun | Enrichi documentaire + scientifique |
| **Interprétation** | Manuelle | Automatique + Synthèse IA |
| **Recommandations** | Aucune | Actionnables et précises |
| **Rapport** | Basique | Complet 7 phases |
| **Temps** | Instantané | 5-15 sec |

## 🔮 Évolutions Futures

### Prévues
- 🔍 **Fouille récursive**: Analyse de fichiers imbriqués (archives, containers)
- 🧬 **Détection signatures**: Base de données de patterns malveillants
- 📊 **Visualisation interactive**: Graphiques exploratoires automatiques
- 🤖 **Auto-apprentissage**: Fine-tuning du modèle sur patterns découverts

### En Recherche
- 🌐 **Fouille distribuée**: Analyse parallèle multi-sources
- 🔐 **Cryptanalyse automatique**: Détection algorithmes de chiffrement
- 📡 **Corrélation temporelle**: Analyse de séries de fichiers
- 🧠 **Mémoire de session**: Apprentissage continu sur types rencontrés

---

✅ **Version actuelle:** 1.0 avec fouille multi-sources  
📅 **Dernière mise à jour:** 3 novembre 2025  
🔧 **Fichier:** `/root/RAG_ChatBot/ERT.py` (4940 lignes)  
🎯 **Inspiration:** Agent VSCode avec todo list multi-tâches
