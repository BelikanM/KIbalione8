# 🚀 Guide de Lancement - Application ERT

## ✅ Méthode Recommandée : Script Automatique

Le script `launch_ert.sh` **force automatiquement** l'utilisation de l'environnement gestmodo (Python 3.10) avec toutes les dépendances installées.

### Lancement Simple

```bash
./launch_ert.sh
```

ou

```bash
bash /home/belikan/KIbalione8/launch_ert.sh
```

### ⚡ Ce que fait le script automatiquement :

1. ✅ **Vérifie la version Python** (3.10 requis)
2. ✅ **Force l'environnement gestmodo** (même si vous êtes dans base/3.13)
3. ✅ **Arrête les instances existantes** de Streamlit
4. ✅ **Installe Streamlit** si manquant dans gestmodo
5. ✅ **Lance l'application** sur le port 8503

### 📊 Sortie Attendue

```
========================================
  Lancement de l'application ERT
========================================
Python actuel: Python 3.13.9
🔄 Arrêt des instances Streamlit existantes...
✅ Utilisation de: Python 3.10.19
✅ Environnement: gestmodo

🚀 Démarrage de l'application ERT...

  You can now view your Streamlit app in your browser.

  Local URL: http://localhost:8503
  Network URL: http://172.20.31.35:8503

✅ Agents LangChain 1.0+ importés avec succès
✅ Fichier .env chargé depuis /home/belikan/KIbalione8/.env
✅ PyGIMLI disponible pour analyses ERT avancées
✅ hf_transfer activé pour téléchargements accélérés
✅ aria2c détecté - téléchargements multi-connexions activés (16 connexions)
```

---

## 🔧 Méthodes Alternatives

### Option 1 : Script Python avec Vérification

```bash
python3 start_ert.py
```

Ce script vérifie l'environnement et relance automatiquement avec gestmodo si nécessaire.

### Option 2 : Lancement Manuel (Ancien)

**⚠️ Non recommandé** - Peut utiliser le mauvais environnement !

```bash
streamlit run ERT.py --server.port 8503
```

### Option 3 : Script run_ert.sh (Original)

```bash
bash run_ert.sh
```

Plus simple mais moins de vérifications.

---

## 🐛 Dépannage

### Problème : "No module named 'shapely'" ou autres imports

**Cause** : Lancement avec Python 3.13 (base) au lieu de Python 3.10 (gestmodo)

**Solution** : Utilisez `./launch_ert.sh` qui force gestmodo

### Problème : "streamlit: command not found"

**Cause** : Streamlit pas dans l'environnement actuel

**Solution** : Le script `launch_ert.sh` l'installe automatiquement

### Problème : Port 8503 déjà utilisé

**Solution** : Le script tue automatiquement les instances existantes, ou changez le port :

```bash
# Éditer launch_ert.sh, ligne 55 :
$GESTMODO_PYTHON -m streamlit run ERT.py --server.port 8504
```

---

## 📦 Vérification de l'Environnement

### Voir les environnements conda disponibles

```bash
conda env list
```

### Vérifier Python dans gestmodo

```bash
~/miniconda3/envs/gestmodo/bin/python --version
# Attendu : Python 3.10.19
```

### Vérifier Streamlit dans gestmodo

```bash
~/miniconda3/envs/gestmodo/bin/python -m streamlit --version
# Attendu : Streamlit, version 1.51.0
```

### Lister les packages installés dans gestmodo

```bash
~/miniconda3/envs/gestmodo/bin/pip list
```

---

## 🎯 Fonctionnalités de l'Application

Une fois lancée sur http://localhost:8503 :

- ✅ **Chat AI** avec Kibali (assistant intelligent)
- ✅ **Upload de fichiers** (drag & drop style ChatGPT)
- ✅ **7 Outils autonomes** :
  - Génération de coupes de résistivité (fichiers .dat)
  - Analyse statistique
  - Recherche web contextuelle
  - Visualisation de données
  - Extraction de données
  - Cartographie colorimétrique
- ✅ **Analyse ERT avancée** avec PyGIMLi
- ✅ **Téléchargements accélérés** (aria2c 16 connexions + hf_transfer)
- ✅ **Cache intelligent** des modèles HuggingFace

---

## 📝 Structure des Scripts

```
KIbalione8/
├── launch_ert.sh        # 🌟 Script principal recommandé (force gestmodo)
├── start_ert.py         # Script Python avec vérifications
├── run_ert.sh           # Script simple original
├── ERT.py               # Application principale (9939 lignes)
└── LANCEMENT.md         # Ce fichier
```

---

## ⚡ Résumé Rapide

**Pour lancer l'application :**
```bash
./launch_ert.sh
```

**Pour arrêter :**
```bash
Ctrl+C dans le terminal
```

**Pour tuer toutes les instances :**
```bash
pkill -9 -f streamlit
```

---

*Dernière mise à jour : 6 novembre 2025*
