# 📖 Mode Documentation - Génération Automatique de PDF

## Vue d'ensemble

Le **Mode Documentation** de Kibali génère automatiquement des PDFs professionnels pour les réponses longues (>1500 mots).

## 🎯 Déclenchement Automatique

### Conditions
- **Mode sélectionné**: `doc` (Documentation)
- **Longueur**: Réponse ≥ 1500 mots
- **Format**: PDF A4 professionnel

### Exemple
```
Utilisateur: "Écris une dissertation complète sur l'intelligence artificielle éthique"

Kibali (Mode doc):
1. Génère la réponse complète (3000+ mots)
2. Crée automatiquement un PDF formaté
3. Affiche un bouton de téléchargement
```

## 📄 Format PDF

### Structure
- **Page de titre**: Titre + métadonnées (date, générateur)
- **Corps**: Texte formaté avec styles hiérarchiques
- **Pied de page**: Statistiques (nombre de mots)

### Styles appliqués
- **Titres H1** (`# Titre`): Police 18pt, gras, nouvelle page
- **Titres H2** (`## Sous-titre`): Police 14pt, gras
- **Titres H3** (`### Section`): Police 12pt, gras
- **Corps de texte**: Police 11pt, justifié, Helvetica
- **Listes** (`- item`): Puces automatiques
- **Citations** (`> texte`): Italique, indenté, gris
- **Gras** (`**texte**`): Bold

### Mise en page
- **Format**: A4 (21 × 29,7 cm)
- **Marges**: 2cm gauche/droite, 2.5cm haut, 2cm bas
- **Interligne**: 16pt
- **Alignement**: Texte justifié

## 🚀 Utilisation

### 1. Activer le Mode Doc

Dans l'onglet **💬 Chat**:
1. Sélectionner **"📖 Mode Documentation"**
2. Poser votre question

### 2. Types de requêtes idéales

```markdown
✅ "Écris une dissertation de 5000 mots sur le réchauffement climatique"
✅ "Fais un livre de 30 pages sur la blockchain"
✅ "Rédige un rapport complet sur l'économie circulaire"
✅ "Crée un manuel détaillé sur Python pour débutants"
✅ "Analyse approfondie de la révolution française"
```

### 3. Réception du PDF

Après génération:
```
📄 PDF généré: doc_intelligence_artificielle_20251104_001530.pdf
[📥 Télécharger le PDF]  ← Bouton cliquable
📊 3245 mots | Format: A4 | Police: Helvetica
```

## 📂 Stockage

### Emplacement
```
~/RAG_ChatBot/generated/documents/
├── doc_dissertation_ia_20251104_120530.pdf
├── doc_rapport_climat_20251104_143215.pdf
└── doc_livre_blockchain_20251104_160845.pdf
```

### Nom de fichier
```
doc_[titre_court]_[YYYYMMDD_HHMMSS].pdf
```

**Exemple**: `doc_ethique_IA_20251104_153045.pdf`

## 🎨 Exemple de Contenu Markdown → PDF

### Input (Markdown)
```markdown
# Intelligence Artificielle Éthique

## Introduction

L'intelligence artificielle soulève des **questions éthiques** majeures.

### Enjeux principaux

- Biais algorithmiques
- Transparence des décisions
- Protection de la vie privée

> "L'IA doit servir l'humanité, pas la dominer" - Expert IA

## Développement

**Cadre éthique**: Les principes suivants doivent guider...
```

### Output (PDF)
- **Titre principal**: Grande police, centré, nouvelle page
- **Sous-titres**: Hiérarchie claire (H2 > H3)
- **Texte**: Justifié, lisible
- **Listes**: Puces alignées
- **Citations**: Indentées, italiques
- **Mots-clés**: En gras

## ⚙️ Configuration Technique

### Fonction principale
```python
generate_pdf_from_text(
    text: str,      # Contenu markdown
    title: str,     # Titre du document
    output_path: str # Chemin de sortie
) -> bool
```

### Dépendances
- `reportlab` (génération PDF)
- `datetime` (timestamp)
- `os` (gestion fichiers)

### Seuils
```python
WORD_THRESHOLD_PDF = 1500  # Auto-génération PDF
WORD_THRESHOLD_INFO = 2000 # Message informatif
```

## 📊 Statistiques incluses

Chaque PDF contient:
- **Nombre de mots**: Comptage automatique
- **Date de génération**: Format DD/MM/YYYY HH:MM
- **Générateur**: "Généré par Kibali (Mode Documentation)"
- **Titre**: Sujet de la question

## 🔧 Dépannage

### PDF non généré
**Symptôme**: Message "⚠️ Génération PDF échouée"

**Causes possibles**:
1. Permissions insuffisantes sur le dossier `generated/documents/`
2. ReportLab non installé
3. Espace disque insuffisant

**Solutions**:
```bash
# Vérifier ReportLab
pip install reportlab

# Créer le dossier
mkdir -p ~/RAG_ChatBot/generated/documents

# Vérifier permissions
ls -ld ~/RAG_ChatBot/generated/documents
```

### Caractères spéciaux mal affichés
**Cause**: Encodage UTF-8 non supporté par la police

**Solution**: Les caractères XML (<, >, &) sont automatiquement échappés

### PDF vide ou corrompu
**Cause**: Erreur dans le parsing Markdown

**Solution**: Vérifier que le texte utilise la syntaxe Markdown standard

## 📈 Améliorations Futures

- [ ] Table des matières automatique
- [ ] Numérotation des pages
- [ ] En-têtes/pieds de page personnalisés
- [ ] Choix de police (Serif/Sans-serif)
- [ ] Export en DOCX (Word)
- [ ] Insertion d'images inline
- [ ] Graphiques et tableaux
- [ ] Annotations et commentaires

## 💡 Conseils d'utilisation

### Pour des PDFs optimaux

1. **Structurer avec Markdown**:
   ```
   # Titre principal
   ## Chapitres
   ### Sections
   Texte courant
   ```

2. **Utiliser les listes**:
   ```
   - Point 1
   - Point 2
   ```

3. **Citer les sources**:
   ```
   > Citation importante
   ```

4. **Mettre en valeur**:
   ```
   **Mots-clés** importants
   ```

### Requêtes efficaces

❌ **Éviter**: "Parle-moi de l'IA"  
✅ **Préférer**: "Rédige une analyse complète de 3000 mots sur l'impact de l'IA dans l'éducation, avec introduction, développement (3 parties) et conclusion"

## 🎓 Cas d'usage

### Académique
- Dissertations universitaires
- Rapports de recherche
- Mémoires (sections)
- Revues de littérature

### Professionnel
- Livres blancs (white papers)
- Rapports d'activité
- Analyses de marché
- Guides utilisateur

### Personnel
- Livres auto-publiés
- Essais longs
- Documentation projet
- Tutoriels détaillés

---

**Version**: 1.0  
**Date**: 4 novembre 2025  
**Auteur**: GitHub Copilot  
**Contexte**: Kibali Mode Documentation avec auto-génération PDF
