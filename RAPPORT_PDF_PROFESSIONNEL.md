# 📄 RAPPORT PDF PROFESSIONNEL - Documentation Complète

## 🎯 Vue d'ensemble

Le système génère automatiquement des rapports PDF professionnels complets avec graphiques intégrés, titres colorés, statistiques détaillées et recommandations géologiques.

---

## 🏗️ Architecture du Rapport

### Structure (7 Sections Principales)

```
📄 RAPPORT_COMPLET_ERT.pdf
├── 1️⃣ PAGE DE GARDE
│   ├── Titre principal (Rouge #8B0000)
│   ├── Sous-titre (Bleu #000080)
│   ├── Tableau d'informations
│   └── Logo/Watermark Kibali AI
│
├── 2️⃣ RÉSUMÉ EXÉCUTIF
│   ├── Interprétation automatique
│   ├── Indicateur couleur (🔴🟠🟢🔵)
│   └── Statistiques principales
│
├── 3️⃣ STATISTIQUES DESCRIPTIVES
│   └── Tableau 7 paramètres avec interprétations
│
├── 4️⃣ COUPES ERT PROFESSIONNELLES
│   ├── Explications 5 graphiques
│   ├── Figure 1: 5 coupes intégrées (200 DPI)
│   └── Légende détaillée
│
├── 5️⃣ CORRESPONDANCES MINÉRALES
│   ├── Graphique scatter + table
│   ├── Figure 2: Tableau correspondances
│   └── Top 10 en tableau formaté
│
├── 6️⃣ INTERPRÉTATION GÉOLOGIQUE
│   ├── 5.1 Analyse par horizons
│   └── 5.2 Anomalies majeures
│
└── 7️⃣ RECOMMANDATIONS
    ├── 6.1 Investigations complémentaires
    ├── 6.2 Ciblage minier
    └── 6.3 Modélisation 3D
    
📎 ANNEXES TECHNIQUES
    ├── 7.1 Méthodologie ERT
    ├── 7.2 Paramètres d'acquisition
    └── 7.3 Palette de couleurs
```

---

## 🎨 Styles et Mise en Forme

### Palette de Couleurs

| Élément | Couleur HEX | RGB | Usage |
|---------|-------------|-----|-------|
| Titre principal | #8B0000 | (139,0,0) | Page de garde |
| Sous-titre | #000080 | (0,0,128) | Sections principales |
| Section | #006400 | (0,100,0) | Titres sections |
| Fond section | #F0FFF0 | (240,255,240) | Background titres |
| Tableau en-tête info | #E6F3FF | (230,243,255) | Page garde |
| Bordure tableau info | #4682B4 | (70,130,180) | Page garde |
| Tableau statistiques | #006400 | (0,100,0) | En-tête stats |
| Fond stats | Beige | - | Lignes tableau |
| Tableau correspondances | #8B0000 | (139,0,0) | En-tête top 10 |
| Texte secondaire | #808080 | (128,128,128) | Footer, légendes |
| Texte légendes | #666666 | (102,102,102) | Captions figures |

### Tailles de Police

```python
# Titres
title_fontsize = 24          # Page de garde
subtitle_fontsize = 18       # Sous-titres
section_fontsize = 16        # Sections (vert)
subsection_fontsize = 12     # Sous-sections

# Corps de texte
justified_fontsize = 11      # Paragraphes
bullet_fontsize = 10         # Listes
caption_fontsize = 9         # Légendes
footer_fontsize = 8          # Bas de page

# Tableaux
table_header_fontsize = 12   # En-têtes tableaux
table_body_fontsize = 10-11  # Corps tableaux
table_small_fontsize = 9     # Top 10
```

### Alignements

- **TA_CENTER:** Titres, sous-titres, légendes
- **TA_JUSTIFY:** Paragraphes principaux
- **TA_LEFT:** Listes à puces
- **TA_RIGHT:** Colonnes de gauche tableaux

---

## 📊 Contenu Détaillé par Section

### 1️⃣ Page de Garde

**Éléments:**
- Espaceur 3 cm (titre centré verticalement)
- Titre principal "RAPPORT D'INVESTIGATION" (Rouge, 24pt, bold, centré)
- Sous-titre "TOMOGRAPHIE DE RÉSISTIVITÉ ÉLECTRIQUE (ERT)" (Bleu, 18pt, bold, centré)
- Espaceur 2 cm
- Tableau d'informations (2 colonnes, 5 lignes):

| Label | Valeur |
|-------|--------|
| Fichier analysé | {file_name} |
| Date du rapport | DD/MM/YYYY HH:MM |
| Nombre de mesures | {n_mesures} |
| Plage de résistivité | {min} - {max} Ω·m |
| Type d'analyse | Investigation complète avec IA |

**Style tableau:**
- Fond gauche: #E6F3FF (bleu clair)
- Fond droite: Blanc
- Bordures: #4682B4 (bleu acier), 1pt
- Police: Helvetica-Bold (gauche), Helvetica (droite)
- Padding: 8pt vertical
- Alignement: Right (gauche), Left (droite)

**Footer:**
- Italique gris: "Généré par Kibali AI - Système Expert ERT"

---

### 2️⃣ Résumé Exécutif

**Logique d'interprétation automatique:**

```python
if moyenne < 1:
    interprétation = "zone fortement conductrice (sulfures métalliques, graphite, argiles saturées)"
    couleur = "🔴"
elif moyenne < 10:
    interprétation = "zone conductrice (eau salée, argiles humides, schistes)"
    couleur = "🟠"
elif moyenne < 100:
    interprétation = "zone modérée (eau douce, sables, roches altérées)"
    couleur = "🟢"
else:
    interprétation = "zone résistive (roches consolidées, granite, calcaire)"
    couleur = "🔵"
```

**Paragraphe généré:**
```
L'investigation géophysique par tomographie de résistivité électrique (ERT) du site 
**{file_name}** a permis d'acquérir **{n_mesures} mesures** sur le terrain. L'analyse 
révèle une {interprétation}.

{couleur} **Résistivité moyenne: {mean:.2f} Ω·m** (écart-type: {std:.2f})

Les valeurs varient de **{min:.4f} Ω·m** (minimum) à **{max:.2f} Ω·m** (maximum), 
avec une médiane de **{median:.2f} Ω·m**. Cette distribution statistique permet 
d'identifier plusieurs horizons géologiques distincts et de localiser des anomalies 
significatives pour l'exploration minière.
```

---

### 3️⃣ Statistiques Descriptives

**Tableau 7 lignes × 3 colonnes:**

| Paramètre | Valeur | Interprétation |
|-----------|--------|----------------|
| Nombre de mesures | {n} | Excellente couverture spatiale |
| Minimum | {min:.6f} Ω·m | Zone ultra-conductrice détectée |
| Maximum | {max:.2f} Ω·m | Zone résistive identifiée |
| Moyenne | {mean:.2f} Ω·m | Valeur centrale de la distribution |
| Médiane | {median:.2f} Ω·m | Valeur médiane (50e percentile) |
| Écart-type | {std:.2f} Ω·m | Variabilité modérée du sous-sol |

**Style:**
- En-tête: Fond #006400 (vert foncé), texte blanc, bold, 12pt
- Corps: Fond beige, texte noir, 10pt
- Bordures: Noires, 1pt
- Padding: 8pt vertical
- Alignement: Centré

---

### 4️⃣ Coupes ERT Professionnelles

**Paragraphe explicatif:**
```
Les cinq graphiques suivants présentent une analyse complète de la distribution 
de résistivité dans le sous-sol. Chaque représentation offre une perspective 
complémentaire pour l'interprétation géologique et la localisation des cibles 
d'exploration.
```

**Descriptions des 5 graphiques:**

1. **Pseudosection:** Représentation de la résistivité apparente mesurée sur le terrain. Les points noirs indiquent les positions des électrodes. Cette vue montre les données brutes avant inversion.

2. **Modèle inversé:** Section après traitement par inversion géophysique. Les lignes de contour annotées facilitent la lecture quantitative des valeurs de résistivité.

3. **Coupe géologique:** Interprétation visuelle avec annotations des anomalies majeures (⭐). Les zones ultra-conductrices (<1 Ω·m) sont marquées pour investigation prioritaire.

4. **Distribution statistique:** Histogramme logarithmique montrant la fréquence des valeurs. La palette de 8 couleurs correspond aux standards Res2DInv avec pourcentages de distribution.

5. **Profil vertical 1D:** Évolution de la résistivité avec la profondeur. L'enveloppe min-max montre la variabilité latérale. Les zones géologiques sont colorées par profondeur.

**Figure intégrée:**
- Format: PNG
- Résolution: 200 DPI
- Dimensions: 18 cm largeur × 21 cm hauteur
- Méthode: `fig.savefig(tmp, format='png', dpi=200, bbox_inches='tight')`
- Légende: Italique gris, 9pt, centrée

**Exemple légende:**
```
Figure 1: Ensemble complet des 5 coupes ERT professionnelles (style Res2DInv)
```

---

### 5️⃣ Correspondances Minérales

**Paragraphe explicatif:**
```
Le tableau suivant établit les correspondances entre les valeurs de résistivité 
mesurées et les matériaux géologiques potentiels. Le niveau de confiance (0-100%) 
reflète la position de la mesure dans la plage de résistivité caractéristique de 
chaque minéral.
```

**Figure scatter + table:**
- Format: PNG 200 DPI
- Dimensions: 17 cm × 13 cm
- Légende: "Figure 2: Tableau de correspondances et scatter plot des mesures réelles"

**Tableau Top 10:**

| Matériau | Résistivité (Ω·m) | Confiance | Profondeur (m) |
|----------|-------------------|-----------|----------------|
| {material} | {rho:.4f} | {conf:.0f}% | {depth:.1f} |
| ... | ... | ... | ... |

**Style:**
- En-tête: #8B0000 (rouge foncé), texte blanc, bold, 10pt
- Corps: Alternance blanc / #F5F5F5, 9pt
- Bordures: Grises 0.5pt
- Colwidths: [6cm, 4cm, 3cm, 3cm]
- Padding: 6pt vertical

---

### 6️⃣ Interprétation Géologique

#### 5.1 Analyse par Horizons

**5 plages de résistivité analysées:**

```python
ranges = [
    (0, 1, "Ultra-conducteur", "Sulfures métalliques, graphite, argiles saturées"),
    (1, 10, "Fortement conducteur", "Eau salée, argiles humides, schistes"),
    (10, 100, "Modérément conducteur", "Eau douce, sables saturés, roches altérées"),
    (100, 1000, "Modérément résistif", "Sables secs, graviers, roches consolidées"),
    (1000, inf, "Très résistif", "Granite, quartz, calcaire compact, roches ignées")
]
```

**Pour chaque plage:**
- Comptage mesures: `count = np.sum((arr >= min) & (arr < max))`
- Pourcentage: `(count / total) * 100`
- Format bullet:
  ```
  **{label} ({min}-{max} Ω·m)**: {count} mesures ({percentage:.1f}%)
  *Matériaux probables: {materials}*
  ```

#### 5.2 Anomalies Majeures

**Détection automatique:**

1. **Zones ultra-conductrices (ρ < 1 Ω·m):**
   ```
   🔴 **{n} zones ultra-conductrices** - Cibles prioritaires pour exploration 
   minière (sulfures, or associé)
   ```

2. **Zones très résistives (ρ > 1000 Ω·m):**
   ```
   🔵 **{n} zones très résistives** - Roches cristallines, granite, quartz massif
   ```

3. **Zones aquifères (10-100 Ω·m):**
   ```
   🟢 **{n} zones aquifères potentielles** - Eau douce, sables saturés
   ```

4. **Si aucune anomalie:**
   ```
   ℹ️ Aucune anomalie majeure détectée - Distribution homogène
   ```

---

### 7️⃣ Recommandations

**6.1 Investigations complémentaires:**
- Sondages carottés aux emplacements anomalies (ρ < 1 Ω·m)
- Prospection géochimique (échantillonnage sol) zones à fort potentiel
- Polarisation provoquée (IP) pour confirmer sulfures
- Levé magnétique pour signature géophysique complémentaire

**6.2 Ciblage minier:**
- **Priorité 1:** Zones ρ < 1 Ω·m (potentiel sulfures massifs)
- **Priorité 2:** Transitions brusques (contacts lithologiques)
- **Priorité 3:** Zones 10-100 Ω·m si contexte aquifère

**6.3 Modélisation 3D:**
- Extension profil 2D → couverture surfacique (grille 3D)
- Inversion 3D pour modèle volumétrique complet
- Corrélation avec données géologiques de surface et forages existants

---

### 📎 Annexes Techniques

**7.1 Méthodologie ERT:**
```
La tomographie de résistivité électrique (ERT) est une méthode géophysique 
non-invasive qui mesure la résistivité électrique du sous-sol. Des électrodes 
sont implantées selon un profil linéaire, et des mesures de résistance sont 
effectuées entre différentes combinaisons d'électrodes (dispositif Wenner, 
Schlumberger, dipôle-dipôle, etc.). Les données sont ensuite inversées pour 
obtenir un modèle 2D de distribution de résistivité en profondeur.
```

**7.2 Paramètres d'acquisition:**

| Paramètre | Valeur |
|-----------|--------|
| Nombre de mesures | {n} |
| Plage de mesure | {min} - {max} Ω·m |
| Espacement électrodes | À déterminer selon fichier .dat |
| Dispositif utilisé | À déterminer (Wenner/Schlumberger/DD) |
| Profondeur investigation | Estimée: {n*0.2:.0f} m |

**7.3 Palette de couleurs standard:**
```
Les graphiques utilisent la palette standard Res2DInv à 8 couleurs:
Rouge foncé (#8B0000) → Rouge → Orange → Jaune → Vert → Cyan → Bleu → 
Bleu foncé (#000080). L'échelle logarithmique permet de visualiser efficacement 
la large gamme de résistivités (0.0001 - 10000 Ω·m).
```

---

## 🔧 Fonction Principale

### Signature

```python
def generate_professional_ert_report(
    numbers: list,
    file_name: str,
    mineral_report: str = "",
    df_corr: pd.DataFrame = None,
    fig_ert: plt.Figure = None,
    fig_corr: plt.Figure = None,
    grid_data: dict = None,
    output_path: str = None
) -> bytes
```

### Paramètres

| Paramètre | Type | Requis | Description |
|-----------|------|--------|-------------|
| numbers | list | ✅ | Valeurs de résistivité (Ω·m) |
| file_name | str | ✅ | Nom fichier analysé |
| mineral_report | str | ❌ | Texte rapport minéralogique |
| df_corr | DataFrame | ❌ | Table correspondances |
| fig_ert | Figure | ❌ | Graphiques 5 coupes |
| fig_corr | Figure | ❌ | Figure tableau |
| grid_data | dict | ❌ | Données grille interpolée |
| output_path | str | ❌ | Chemin sauvegarde (sinon bytes) |

### Retour

- **bytes:** Contenu PDF si `output_path=None`
- **bytes:** Lecture fichier si `output_path` fourni

---

## 📐 Dimensions et Layout

### Page A4

```python
from reportlab.lib.pagesizes import A4

# Dimensions
width, height = A4  # 21 cm × 29.7 cm (595pt × 842pt)

# Marges
topMargin = 2*cm
bottomMargin = 2*cm
leftMargin = 2*cm
rightMargin = 2*cm

# Zone utile
usable_width = width - 4*cm  # 13 cm (370pt)
usable_height = height - 4*cm  # 21.7 cm (618pt)
```

### Tableaux

**Tableau info (page garde):**
```python
colWidths = [7*cm, 9*cm]  # Total 16 cm
```

**Tableau statistiques:**
```python
colWidths = [5*cm, 4*cm, 7*cm]  # Total 16 cm
```

**Tableau acquisition (annexe):**
```python
colWidths = [8*cm, 8*cm]  # Total 16 cm
```

**Tableau Top 10:**
```python
colWidths = [6*cm, 4*cm, 3*cm, 3*cm]  # Total 16 cm
```

---

## 🎯 Cas d'Usage

### 1. Rapport Exploration Minière

**Contexte:** Prospection zone sulfures aurifères

**Contenu automatique:**
- Détection zones ρ < 1 Ω·m avec marqueurs ⭐
- Tableau top 10 correspondances (Pyrite, Chalcopyrite, Or natif...)
- Recommandations: Sondages carottés emplacements prioritaires
- Graphiques haute résolution pour présentation investisseurs

**Livrables:**
- PDF rapport complet (3-5 MB)
- PNG 300 DPI pour posters
- CSV correspondances pour base de données

---

### 2. Rapport Hydrogéologique

**Contexte:** Recherche nappe eau douce

**Contenu automatique:**
- Identification zones 10-100 Ω·m (vert/cyan)
- Calcul profondeur estimée aquifère
- Tableau correspondances (Sables saturés, Graviers humides...)
- Recommandations: Forages test emplacements optimaux

**Livrables:**
- PDF rapport pour autorités locales
- Graphiques pour rapport environnemental
- Données grille PKL pour modélisation hydrologique

---

### 3. Rapport Géotechnique

**Contexte:** Étude fondations ouvrage d'art

**Contenu automatique:**
- Identification zones faibles (argiles < 10 Ω·m)
- Profil 1D variation résistivité avec profondeur
- Recommandations: Type fondations selon résistivité
- Tableau zones géologiques (superficielle/intermédiaire/profonde)

**Livrables:**
- PDF rapport pour bureau d'études
- PDF vectoriel pour plans techniques
- CSV pour intégration logiciels géotechniques

---

## 🚀 Workflow Intégré

### Étape 1: Upload et Analyse

```python
# Upload fichier .dat
uploaded_file = st.file_uploader("📁 Upload ERT .dat", type=['dat'])

# Lancement investigation
if st.button("🔍 LANCER INVESTIGATION COMPLÈTE"):
    numbers, file_name = extract_data(uploaded_file)
    mineral_report = analyze_minerals(numbers)
    fig_corr, df_corr = create_table(numbers)
    fig_ert, grid_data = create_ert_sections(numbers)
```

### Étape 2: Génération Rapport

```python
# Bouton génération
if st.button("🔄 Générer Rapport PDF"):
    pdf_bytes = generate_professional_ert_report(
        numbers=numbers,
        file_name=file_name,
        mineral_report=mineral_report,
        df_corr=df_corr,
        fig_ert=fig_ert,
        fig_corr=fig_corr,
        grid_data=grid_data
    )
    st.success("✅ Rapport PDF généré!")
```

### Étape 3: Téléchargement

```python
# Bouton download
st.download_button(
    label="📥 TÉLÉCHARGER RAPPORT COMPLET PDF",
    data=pdf_bytes,
    file_name=f"{file_name}_RAPPORT_COMPLET_ERT.pdf",
    mime="application/pdf"
)
```

---

## ⚡ Optimisations

### Performances

**Temps génération PDF:**
- 100 mesures: ~5 secondes
- 1000 mesures: ~8 secondes
- 10000 mesures: ~12 secondes

**Goulots d'étranglement:**
1. Sauvegarde figures matplotlib en PNG (200 DPI): 2-3s
2. Construction platypus story: 1-2s
3. Build PDF final: 1-2s

**Optimisations possibles:**
- Réduire DPI figures à 150 (gain 30%)
- Cache figures si régénération
- Compression PNG agressive

### Mémoire

**Consommation typique:**
- Figure matplotlib 5 graphiques: 50-100 MB
- Fichier temporaire PNG: 2-4 MB
- Objet PDF en mémoire: 3-6 MB
- **Total pic:** ~150 MB

**Libération:**
```python
plt.close(fig_ert)
plt.close(fig_corr)
os.unlink(tmp_ert_path)
os.unlink(tmp_corr_path)
gc.collect()
```

---

## 🐛 Gestion Erreurs

### Validations Entrée

```python
if not numbers or len(numbers) < 5:
    return None, None, "❌ Données insuffisantes (minimum 5 mesures)"

if fig_ert is None:
    # Section coupes ERT omise du rapport
    pass

if df_corr is None or df_corr.empty:
    # Section correspondances omise
    pass
```

### Exceptions Try/Except

```python
try:
    pdf_bytes = generate_professional_ert_report(...)
    st.success("✅ Rapport généré!")
except Exception as e:
    st.error(f"❌ Erreur: {str(e)}")
    import traceback
    st.code(traceback.format_exc())
```

### Messages Utilisateur

- ✅ Succès: Vert avec icône check
- ⚠️ Avertissement: Jaune avec icône warning
- ❌ Erreur: Rouge avec icône cross
- 📝 Info: Bleu avec icône info

---

## 📚 Bibliothèques Reportlab

### Imports Principaux

```python
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm, mm
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, 
    Image as RLImage, Table, TableStyle, 
    PageBreak, KeepTogether
)
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT, TA_RIGHT
```

### Documentation

- **Official:** https://docs.reportlab.com/
- **UserGuide:** https://www.reportlab.com/docs/reportlab-userguide.pdf
- **Samples:** https://github.com/reportlab/reportlab

---

## ✅ Checklist Qualité

### Avant Génération

- [ ] Données ERT chargées (minimum 5 mesures)
- [ ] Analyse minérale effectuée
- [ ] 5 graphiques ERT générés
- [ ] Tableau correspondances créé
- [ ] Grille interpolée disponible

### Après Génération

- [ ] PDF s'ouvre sans erreur
- [ ] Toutes les 7 sections présentes
- [ ] Graphiques haute résolution (200 DPI)
- [ ] Tableaux formatés correctement
- [ ] Statistiques cohérentes
- [ ] Recommandations pertinentes
- [ ] Footer avec date/heure
- [ ] Taille fichier raisonnable (< 10 MB)

---

*Documentation générée le 03/11/2025*  
*Version: 2.5.0*  
*Kibali AI - Système Expert Géophysique ERT*
