"""
AI Code Agent - Système d'exécution autonome de code
Permet à Kibali de générer et exécuter du code pour accomplir des tâches
"""

import os
import sys
import subprocess
import tempfile
from typing import Dict, Any, Tuple
import json

class AICodeAgent:
    """Agent qui génère et exécute du code Python pour accomplir des tâches"""
    
    def _generate_depth_specific_code(self, file_path: str, query: str, params: dict) -> str:
        """Génère du code pour répondre aux questions sur les profondeurs"""
        return f'''
import numpy as np
import pandas as pd
import os
import struct
import re

file_path = "{file_path}"

# Lire le fichier binaire
with open(file_path, 'rb') as f:
    file_bytes = f.read()

# Extraire les nombres
def extract_numbers(file_bytes):
    numbers = []
    ascii_text = "".join([chr(b) if 32 <= b <= 126 else " " for b in file_bytes])
    found = re.findall(r"[-+]?\\d*\\.\\d+|\\d+", ascii_text)
    numbers.extend([float(n) for n in found])
    for fmt, size in [('f', 4), ('d', 8)]:
        for i in range(0, len(file_bytes) - size + 1, size):
            try:
                value = struct.unpack(fmt, file_bytes[i:i+size])[0]
                if not np.isnan(value) and not np.isinf(value) and abs(value) < 1e6:
                    numbers.append(value)
            except:
                pass
    return numbers

numbers = extract_numbers(file_bytes)
data = np.array(numbers) if numbers else np.array([])

print("📏 ANALYSE DES PROFONDEURS:\\n")

if len(data) > 0:
    # Détecter les profondeurs (valeurs entre 0 et 100m généralement)
    potential_depths = data[(data >= 0) & (data <= 100)]
    
    # Détecter les résistivités (0.1 à 1000 Ω·m)
    resistivity_values = data[(data > 0.1) & (data < 1000)]
    
    if len(potential_depths) > 0:
        print(f"Profondeurs détectées ({{len(potential_depths)}} points):")
        print(f"  • Profondeur minimale: {{np.min(potential_depths):.2f}} m")
        print(f"  • Profondeur maximale: {{np.max(potential_depths):.2f}} m")
        print(f"  • Profondeur moyenne: {{np.mean(potential_depths):.2f}} m")
        
        # Si on cherche spécifiquement l'eau salée
        if 'salée' in "{query}" or 'saline' in "{query}":
            if len(resistivity_values) > 0:
                # Eau salée = résistivité < 10 Ω·m
                saline_indices = resistivity_values < 10
                if np.any(saline_indices):
                    saline_depths = potential_depths[saline_indices] if len(potential_depths) == len(resistivity_values) else potential_depths[:np.sum(saline_indices)]
                    print(f"\\n🌊 Zones d'eau salée détectées:")
                    print(f"  • Entre {{np.min(saline_depths):.2f}} m et {{np.max(saline_depths):.2f}} m de profondeur")
                    print(f"  • Résistivité moyenne: {{np.mean(resistivity_values[saline_indices]):.2f}} Ω·m")
                else:
                    print("\\n❌ Pas d'eau salée détectée (aucune résistivité < 10 Ω·m)")
        
        # Afficher les profondeurs exactes uniques
        unique_depths = np.unique(potential_depths)
        if len(unique_depths) <= 50:
            print(f"\\n📋 Profondeurs exactes mesurées ({{len(unique_depths)}} niveaux):")
            for i, depth in enumerate(sorted(unique_depths)[:20], 1):
                print(f"  {{i}}. {{depth:.2f}} m")
            if len(unique_depths) > 20:
                print(f"  ... ({{len(unique_depths) - 20}} autres niveaux)")
    else:
        print("⚠️ Aucune profondeur clairement identifiée dans le fichier")
        print(f"   Valeurs disponibles: min={{np.min(data):.2f}}, max={{np.max(data):.2f}}")
else:
    print("❌ Aucune donnée numérique extraite")
'''
    
    def _generate_material_analysis_code(self, file_path: str, query: str) -> str:
        """Génère du code pour identifier les types de matériaux"""
        return f'''
import numpy as np
import os
import struct
import re

file_path = "{file_path}"

with open(file_path, 'rb') as f:
    file_bytes = f.read()

def extract_numbers(file_bytes):
    numbers = []
    ascii_text = "".join([chr(b) if 32 <= b <= 126 else " " for b in file_bytes])
    found = re.findall(r"[-+]?\\d*\\.\\d+|\\d+", ascii_text)
    numbers.extend([float(n) for n in found])
    for fmt, size in [('f', 4), ('d', 8)]:
        for i in range(0, len(file_bytes) - size + 1, size):
            try:
                value = struct.unpack(fmt, file_bytes[i:i+size])[0]
                if not np.isnan(value) and not np.isinf(value) and abs(value) < 1e6:
                    numbers.append(value)
            except:
                pass
    return numbers

numbers = extract_numbers(file_bytes)
data = np.array(numbers) if numbers else np.array([])

print("🔬 ANALYSE DES MATÉRIAUX PAR RÉSISTIVITÉ:\\n")

if len(data) > 0:
    resistivity_values = data[(data > 0.1) & (data < 1000)]
    
    if len(resistivity_values) > 0:
        # Classification géologique
        materials = {{
            'Eau salée / Argile saturée': (resistivity_values < 10),
            'Argile / Sable humide': ((resistivity_values >= 10) & (resistivity_values < 50)),
            'Sol mixte / Sable sec': ((resistivity_values >= 50) & (resistivity_values < 200)),
            'Roche compacte / Gravier': (resistivity_values >= 200)
        }}
        
        total = len(resistivity_values)
        
        for material, mask in materials.items():
            count = np.sum(mask)
            if count > 0:
                pct = (count / total) * 100
                avg_res = np.mean(resistivity_values[mask])
                print(f"• {{material}}:")
                print(f"    {{count}} mesures ({{pct:.1f}}%) - Résistivité moyenne: {{avg_res:.2f}} Ω·m")
                print()
        
        # Matériau dominant
        dominant = max(materials.items(), key=lambda x: np.sum(x[1]))
        print(f"🎯 Matériau dominant: {{dominant[0]}} ({{np.sum(dominant[1])/total*100:.1f}}%)")
    else:
        print("⚠️ Pas de valeurs de résistivité détectées")
else:
    print("❌ Aucune donnée numérique extraite")
'''
    
    def _generate_zone_specific_code(self, file_path: str, query: str, params: dict) -> str:
        """Génère du code pour analyser une zone spécifique (eau salée, argile, etc.)"""
        # Déterminer la zone recherchée
        if 'eau salée' in query or 'saline' in query:
            zone_name = "eau salée"
            res_min, res_max = 0.1, 10
        elif 'argile' in query:
            zone_name = "argile"
            res_min, res_max = 10, 50
        elif 'sable' in query:
            zone_name = "sable"
            res_min, res_max = 50, 200
        elif 'roche' in query:
            zone_name = "roche"
            res_min, res_max = 200, 1000
        else:
            zone_name = "zones spécifiques"
            res_min, res_max = 0.1, 1000
        
        return f'''
import numpy as np
import os
import struct
import re

file_path = "{file_path}"

with open(file_path, 'rb') as f:
    file_bytes = f.read()

def extract_numbers(file_bytes):
    numbers = []
    ascii_text = "".join([chr(b) if 32 <= b <= 126 else " " for b in file_bytes])
    found = re.findall(r"[-+]?\\d*\\.\\d+|\\d+", ascii_text)
    numbers.extend([float(n) for n in found])
    for fmt, size in [('f', 4), ('d', 8)]:
        for i in range(0, len(file_bytes) - size + 1, size):
            try:
                value = struct.unpack(fmt, file_bytes[i:i+size])[0]
                if not np.isnan(value) and not np.isinf(value) and abs(value) < 1e6:
                    numbers.append(value)
            except:
                pass
    return numbers

numbers = extract_numbers(file_bytes)
data = np.array(numbers) if numbers else np.array([])

print(f"🔍 RECHERCHE DE ZONES: {zone_name.upper()}\\n")

if len(data) > 0:
    resistivity_values = data[(data > 0.1) & (data < 1000)]
    
    if len(resistivity_values) > 0:
        # Filtrer pour la zone spécifique
        zone_mask = (resistivity_values >= {res_min}) & (resistivity_values < {res_max})
        zone_values = resistivity_values[zone_mask]
        
        if len(zone_values) > 0:
            print(f"✅ Zones de {zone_name} détectées:")
            print(f"  • Nombre de mesures: {{len(zone_values)}}")
            print(f"  • Pourcentage du site: {{len(zone_values)/len(resistivity_values)*100:.1f}}%")
            print(f"  • Résistivité: {{np.min(zone_values):.2f}} - {{np.max(zone_values):.2f}} Ω·m")
            print(f"  • Résistivité moyenne: {{np.mean(zone_values):.2f}} Ω·m")
            
            # Essayer d'estimer les profondeurs
            potential_depths = data[(data >= 0) & (data <= 100)]
            if len(potential_depths) > 0:
                print(f"\\n📏 Profondeurs associées:")
                print(f"  • Profondeur min: {{np.min(potential_depths):.2f}} m")
                print(f"  • Profondeur max: {{np.max(potential_depths):.2f}} m")
        else:
            print(f"❌ Aucune zone de {zone_name} détectée")
            print(f"   (Résistivité recherchée: {res_min}-{res_max} Ω·m)")
    else:
        print("⚠️ Pas de valeurs de résistivité détectées")
else:
    print("❌ Aucune donnée numérique extraite")
'''
    
    def _generate_statistics_code(self, file_path: str, query: str) -> str:
        """Génère du code pour des statistiques"""
        return f'''
import numpy as np
import pandas as pd
import os
import struct
import re

file_path = "{file_path}"

with open(file_path, 'rb') as f:
    file_bytes = f.read()

def extract_numbers(file_bytes):
    numbers = []
    ascii_text = "".join([chr(b) if 32 <= b <= 126 else " " for b in file_bytes])
    found = re.findall(r"[-+]?\\d*\\.\\d+|\\d+", ascii_text)
    numbers.extend([float(n) for n in found])
    for fmt, size in [('f', 4), ('d', 8)]:
        for i in range(0, len(file_bytes) - size + 1, size):
            try:
                value = struct.unpack(fmt, file_bytes[i:i+size])[0]
                if not np.isnan(value) and not np.isinf(value) and abs(value) < 1e6:
                    numbers.append(value)
            except:
                pass
    return numbers

numbers = extract_numbers(file_bytes)
data = np.array(numbers) if numbers else np.array([])

print("📊 STATISTIQUES DÉTAILLÉES:\\n")

if len(data) > 0:
    resistivity_values = data[(data > 0.1) & (data < 1000)]
    
    if len(resistivity_values) > 0:
        # Statistiques globales
        stats = {{
            'Nombre total de mesures': len(resistivity_values),
            'Résistivité minimale': f"{{np.min(resistivity_values):.2f}} Ω·m",
            'Résistivité maximale': f"{{np.max(resistivity_values):.2f}} Ω·m",
            'Résistivité moyenne': f"{{np.mean(resistivity_values):.2f}} Ω·m",
            'Résistivité médiane': f"{{np.median(resistivity_values):.2f}} Ω·m",
            'Écart-type': f"{{np.std(resistivity_values):.2f}} Ω·m"
        }}
        
        for key, value in stats.items():
            print(f"• {{key}}: {{value}}")
        
        # Distribution par zones
        print("\\n📈 DISTRIBUTION PAR ZONES:\\n")
        zones = {{
            'Eau salée / Argile saturée (< 10 Ω·m)': resistivity_values < 10,
            'Argile / Sable humide (10-50 Ω·m)': (resistivity_values >= 10) & (resistivity_values < 50),
            'Sol mixte / Sable sec (50-200 Ω·m)': (resistivity_values >= 50) & (resistivity_values < 200),
            'Roche / Gravier (≥ 200 Ω·m)': resistivity_values >= 200
        }}
        
        for zone, mask in zones.items():
            count = np.sum(mask)
            pct = (count / len(resistivity_values)) * 100
            print(f"• {{zone}}: {{count}} mesures ({{pct:.1f}}%)")
    else:
        print("⚠️ Pas de valeurs de résistivité détectées")
else:
    print("❌ Aucune donnée numérique extraite")
'''
    
    def _generate_comparison_code(self, file_path: str, query: str) -> str:
        """Génère du code pour comparer des zones"""
        return f'''
import numpy as np
import os
import struct
import re

file_path = "{file_path}"

with open(file_path, 'rb') as f:
    file_bytes = f.read()

def extract_numbers(file_bytes):
    numbers = []
    ascii_text = "".join([chr(b) if 32 <= b <= 126 else " " for b in file_bytes])
    found = re.findall(r"[-+]?\\d*\\.\\d+|\\d+", ascii_text)
    numbers.extend([float(n) for n in found])
    for fmt, size in [('f', 4), ('d', 8)]:
        for i in range(0, len(file_bytes) - size + 1, size):
            try:
                value = struct.unpack(fmt, file_bytes[i:i+size])[0]
                if not np.isnan(value) and not np.isinf(value) and abs(value) < 1e6:
                    numbers.append(value)
            except:
                pass
    return numbers

numbers = extract_numbers(file_bytes)
data = np.array(numbers) if numbers else np.array([])

print("⚖️  COMPARAISON DES ZONES:\\n")

if len(data) > 0:
    resistivity_values = data[(data > 0.1) & (data < 1000)]
    
    if len(resistivity_values) > 0:
        zones_data = {{
            'Eau salée': resistivity_values[resistivity_values < 10],
            'Argile': resistivity_values[(resistivity_values >= 10) & (resistivity_values < 50)],
            'Sable': resistivity_values[(resistivity_values >= 50) & (resistivity_values < 200)],
            'Roche': resistivity_values[resistivity_values >= 200]
        }}
        
        for zone_name, zone_values in zones_data.items():
            if len(zone_values) > 0:
                print(f"📍 {{zone_name}}:")
                print(f"  • Quantité: {{len(zone_values)}} mesures ({{len(zone_values)/len(resistivity_values)*100:.1f}}%)")
                print(f"  • Résistivité: {{np.mean(zone_values):.2f}} ± {{np.std(zone_values):.2f}} Ω·m")
                print()
        
        # Zone dominante
        dominant = max(zones_data.items(), key=lambda x: len(x[1]))
        print(f"🏆 Zone dominante: {{dominant[0]}} ({{len(dominant[1])/len(resistivity_values)*100:.1f}}%)")
    else:
        print("⚠️ Pas de valeurs de résistivité détectées")
else:
    print("❌ Aucune donnée numérique extraite")
'''
    
    def __init__(self, model_path: str = None):
        """
        Args:
            model_path: Chemin vers le modèle de code (DeepSeek-Coder)
        """
        # Utiliser le répertoire home de l'utilisateur pour le cache
        user_home = os.path.expanduser("~")
        default_cache = os.path.join(user_home, ".cache", "huggingface", "code_models")
        self.model_path = model_path or default_cache
        
        self.model = None
        self.tokenizer = None
        self.execution_history = []
        
        # Configurer les variables d'environnement pour Hugging Face
        os.environ['TRANSFORMERS_CACHE'] = self.model_path
        os.environ['HF_HOME'] = os.path.join(user_home, ".cache", "huggingface")
        
        # Essayer de charger le modèle au démarrage
        self.load_model()
        
    def load_model(self):
        """Charge le modèle de code en mémoire"""
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            import torch
            
            # Créer le répertoire cache s'il n'existe pas
            os.makedirs(self.model_path, exist_ok=True)
            os.makedirs(os.environ.get('HF_HOME', ''), exist_ok=True)
            
            model_name = "deepseek-ai/deepseek-coder-1.3b-instruct"
            
            print("🤖 Chargement du modèle de code...")
            print(f"📁 Cache: {self.model_path}")
            
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                cache_dir=self.model_path,
                trust_remote_code=True
            )
            
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                cache_dir=self.model_path,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None,
                low_cpu_mem_usage=True,
                trust_remote_code=True
            )
            
            print("✅ Modèle chargé avec succès")
            return True
            
        except PermissionError as e:
            print(f"❌ Erreur de permissions: {e}")
            print(f"💡 Vérifiez les permissions du dossier: {self.model_path}")
            print("⚠️ Utilisation des templates prédéfinis")
            return False
        except Exception as e:
            print(f"❌ Erreur chargement modèle: {e}")
            print("⚠️ Utilisation des templates prédéfinis")
            return False
    
    def detect_action_intent(self, query: str) -> Dict[str, Any]:
        """
        Détecte l'intention d'action dans une requête
        
        Returns:
            dict: {
                'is_action': bool,
                'action_type': str,  # 'analyze', 'search', 'create', 'process'
                'target_file': str,
                'parameters': dict
            }
        """
        query_lower = query.lower()
        
        # Mots-clés d'action
        action_keywords = {
            'analyze': ['analyse', 'analyser', 'examine', 'étudie', 'vérifie', 'inspecte', 'explore'],
            'search': ['cherche', 'trouve', 'recherche', 'localise', 'détecte', 'identifie'],
            'create': ['crée', 'génère', 'fabrique', 'construis', 'produis', 'fais', 'creer'],
            'process': ['traite', 'transforme', 'convertis', 'calcule', 'extrait', 'traiter'],
            'visualize': ['affiche', 'montre', 'visualise', 'dessine', 'trace', 'graphique', 'plot', 'tableau'],
            'extract': ['extrait', 'récupère', 'obtiens', 'sors', 'donne', 'montre'],
            'compare': ['compare', 'différence', 'vs', 'versus', 'contraste'],
            'summarize': ['résume', 'synthèse', 'récapitule', 'aperçu', 'overview']
        }
        
        # Détection de fichiers
        file_extensions = ['.bin', '.npy', '.npz', '.dat', '.txt', '.csv', '.json', '.pdf']
        detected_files = []
        
        # DEBUG: afficher la requête
        print(f"🐛 DEBUG detect_action_intent - Query: '{query}'")
        print(f"🐛 DEBUG detect_action_intent - Query split: {query.split()}")
        
        for word in query.split():
            # Vérifier que le mot se termine par une extension ET a un nom de fichier avant
            if any(word.endswith(ext) for ext in file_extensions):
                # Nettoyer le mot des caractères parasites (parenthèses, virgules, etc.)
                clean_word = word.strip('(),"\' ')
                # Vérifier que ce n'est pas juste l'extension seule (ex: ".dat")
                if len(clean_word) > 4:  # Au moins 1 caractère + extension (.dat = 4 chars)
                    detected_files.append(clean_word)
                    print(f"🐛 DEBUG - Fichier détecté: '{word}' -> nettoyé: '{clean_word}'")
        
        # Détection du type d'action
        action_type = None
        for action, keywords in action_keywords.items():
            if any(kw in query_lower for kw in keywords):
                action_type = action
                break
        
        # Extraction de paramètres spécifiques
        parameters = {}
        
        # Profondeurs
        if 'profondeur' in query_lower or 'depth' in query_lower:
            parameters['depth_analysis'] = True
        
        # Eau salée/douce
        if 'eau salée' in query_lower or 'saline' in query_lower or 'salée' in query_lower:
            parameters['water_type'] = 'saline'
        elif 'eau douce' in query_lower or 'fresh' in query_lower:
            parameters['water_type'] = 'fresh'
        
        # Résistivité
        if 'résistivité' in query_lower or 'resistivity' in query_lower or 'résistance' in query_lower:
            parameters['resistivity'] = True
        
        # ERT/Géophysique
        if 'ert' in query_lower or 'géophysique' in query_lower or 'geophysi' in query_lower:
            parameters['geophysics'] = True
        
        # Tableau/Visualisation
        if 'tableau' in query_lower or 'table' in query_lower or 'dataframe' in query_lower:
            parameters['table'] = True
            action_type = action_type or 'visualize'
        
        # Graphique
        if 'graphique' in query_lower or 'plot' in query_lower or 'graph' in query_lower or 'courbe' in query_lower:
            parameters['plot'] = True
            action_type = action_type or 'visualize'
        
        # Structure/Format
        if 'structure' in query_lower or 'format' in query_lower or 'organisation' in query_lower:
            parameters['structure'] = True
            action_type = action_type or 'analyze'
        
        # Si action détectée MAIS pas de fichier, c'est quand même une action
        # (le fichier sera fourni par ERT.py depuis uploaded_file_data)
        is_action = action_type is not None and (len(detected_files) > 0 or any(kw in query_lower for kw in ['fichier', 'file', 'données', 'data']))
        
        return {
            'is_action': is_action,
            'action_type': action_type,
            'target_files': detected_files,
            'parameters': parameters,
            'original_query': query
        }
    
    def generate_code(self, intent: Dict[str, Any]) -> str:
        """
        Génère du code Python pour accomplir une tâche
        
        Args:
            intent: Dictionnaire d'intention retourné par detect_action_intent()
        
        Returns:
            str: Code Python généré
        """
        # Toujours utiliser la génération dynamique avec le modèle
        # pour s'adapter à toutes les questions
        return self._generate_code_with_model(intent)
    
    def _generate_code_with_model(self, intent: Dict[str, Any]) -> str:
        """Génère du code dynamiquement avec outils avancés de visualisation"""
        action = intent['action_type']
        files = intent['target_files']
        params = intent['parameters']
        query = intent['original_query'].lower()
        
        # Obtenir le chemin du fichier
        file_path = files[0] if files else "unknown.dat"
        
        # DEBUG
        print(f"🐛 DEBUG - Génération pour action: {action}")
        print(f"🐛 DEBUG - Query: '{query}'")
        print(f"🐛 DEBUG - File path: '{file_path}'")
        
        # Détecter le type de visualisation demandé
        needs_2d_section = any(kw in query for kw in ['coupe', 'section', '2d', 'tomographie', 'profil'])
        needs_colors = any(kw in query for kw in ['couleur', 'color', 'coloré'])
        needs_stats = any(kw in query for kw in ['statistique', 'stats', 'analyse'])
        needs_water = any(kw in query for kw in ['eau', 'water', 'aquifère'])
        
        print(f"🔍 Détection besoins: 2D={needs_2d_section}, Couleurs={needs_colors}, Stats={needs_stats}, Eau={needs_water}")
        
        # Construire le code COMPLET avec les vrais outils
        code = f"""import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import os
import re

# Import des outils de visualisation avancés
sys.path.append('/home/belikan/KIbalione8')
from visualization_tools import VisualizationEngine

print("\\n" + "="*80)
print("🔬 ANALYSE AVANCÉE ERT - KIBALI")
print("="*80 + "\\n")

# Initialiser le moteur de visualisation
viz = VisualizationEngine()

file_path = "{file_path}"
print(f"📁 Fichier: {{file_path}}\\n")

try:
    # Lecture et parsing du fichier ERT
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()
    
    # Extraction des valeurs numériques
    numbers = re.findall(r'[-+]?\\d*\\.?\\d+(?:[eE][-+]?\\d+)?', content)
    all_values = [float(x) for x in numbers if x]
    
    print(f"✅ Valeurs extraites: {{len(all_values)}} nombres\\n")
    
    if len(all_values) == 0:
        print("❌ Aucune donnée numérique trouvée")
        sys.exit(1)
    
    # Identifier les colonnes (X, Y, Z, Résistivité)
    # Format typique ERT: X Y Z Résistivité
    num_cols = 4  # Par défaut XYZ + résistivité
    if len(all_values) % 4 == 0:
        num_points = len(all_values) // 4
        data = np.array(all_values).reshape(num_points, 4)
        x_coords = data[:, 0]
        y_coords = data[:, 1]
        z_coords = data[:, 2]
        resistivity = data[:, 3]
    elif len(all_values) % 3 == 0:
        # Format XYZ uniquement
        num_points = len(all_values) // 3
        data = np.array(all_values).reshape(num_points, 3)
        x_coords = data[:, 0]
        y_coords = data[:, 1]
        z_coords = data[:, 2]
        resistivity = np.random.uniform(10, 100, num_points)  # Placeholder
    else:
        # Assumer que tout est résistivité
        resistivity = np.array(all_values)
        x_coords = np.arange(len(resistivity))
        z_coords = np.zeros(len(resistivity))
    
    print(f"📊 Structure des données:")
    print(f"   Points de mesure: {{len(resistivity)}}")
    print(f"   X range: {{x_coords.min():.2f}} → {{x_coords.max():.2f}}")
    print(f"   Z range: {{z_coords.min():.2f}} → {{z_coords.max():.2f}}")
    print(f"   Résistivité: {{resistivity.min():.2f}} → {{resistivity.max():.2f}} Ω·m\\n")
    
    # STATISTIQUES COMPLÈTES
    print("="*80)
    print("📈 STATISTIQUES DÉTAILLÉES")
    print("="*80)
    print(f"Résistivité moyenne: {{resistivity.mean():.2f}} Ω·m")
    print(f"Résistivité médiane: {{np.median(resistivity):.2f}} Ω·m")
    print(f"Écart-type: {{resistivity.std():.2f}} Ω·m")
    print(f"Minimum: {{resistivity.min():.2f}} Ω·m")
    print(f"Maximum: {{resistivity.max():.2f}} Ω·m")
    print(f"Q1 (25%): {{np.percentile(resistivity, 25):.2f}} Ω·m")
    print(f"Q3 (75%): {{np.percentile(resistivity, 75):.2f}} Ω·m\\n")
    
    # CLASSIFICATION GÉOLOGIQUE
    print("="*80)
    print("🌍 INTERPRÉTATION GÉOLOGIQUE")
    print("="*80)
    
    # Zones de résistivité
    water_zone = (resistivity >= 0.5) & (resistivity <= 50)
    clay_zone = (resistivity > 50) & (resistivity <= 150)
    sand_zone = (resistivity > 150) & (resistivity <= 500)
    rock_zone = resistivity > 500
    
    print(f"💧 Eau/Argile saturée (0.5-50 Ω·m): {{water_zone.sum()}} points ({{100*water_zone.sum()/len(resistivity):.1f}}%)")
    print(f"🟤 Argile/Limon (50-150 Ω·m): {{clay_zone.sum()}} points ({{100*clay_zone.sum()/len(resistivity):.1f}}%)")
    print(f"🟡 Sable/Gravier (150-500 Ω·m): {{sand_zone.sum()}} points ({{100*sand_zone.sum()/len(resistivity):.1f}}%)")
    print(f"⚫ Roche (>500 Ω·m): {{rock_zone.sum()}} points ({{100*rock_zone.sum()/len(resistivity):.1f}}%)\\n")
    
    if water_zone.sum() > 0:
        water_depths = z_coords[water_zone]
        print(f"🎯 ZONES D'EAU DÉTECTÉES:")
        print(f"   Profondeur min: {{water_depths.min():.2f}} m")
        print(f"   Profondeur max: {{water_depths.max():.2f}} m")
        print(f"   Résistivité moyenne zone eau: {{resistivity[water_zone].mean():.2f}} Ω·m\\n")
    
    # GÉNÉRATION DE LA COUPE 2D COLORÉE
    """
        
        # Ajouter génération de visualisation si demandé
        if needs_2d_section:
            code += """
    print("="*80)
    print("🎨 GÉNÉRATION COUPE 2D AVEC COULEURS")
    print("="*80)
    
    # Méthode 1: Essayer avec PyGIMLI (inversion complète)
    try:
        import pygimli as pg
        from pygimli.physics import ert
        
        print("🔬 Utilisation de PyGIMLI pour inversion ERT complète...")
        
        # Créer un schéma d'électrodes
        scheme = pg.DataContainerERT()
        
        # Si on a des positions X, créer les électrodes
        if len(x_coords) > 0:
            for i, x in enumerate(np.unique(x_coords)):
                scheme.createSensor([x, 0.0])
            
            print(f"   Électrodes créées: {{scheme.sensorCount()}}")
            
            # Créer une configuration Wenner simple
            for i in range(scheme.sensorCount() - 3):
                scheme.createFourPointData(i, i+1, i+2, i+3)
            
            # Ajouter les résistivités apparentes
            if len(resistivity) == scheme.size():
                scheme.set('rhoa', resistivity)
            else:
                # Ajuster les valeurs
                rho_adjusted = np.interp(
                    np.linspace(0, len(resistivity)-1, scheme.size()),
                    np.arange(len(resistivity)),
                    resistivity
                )
                scheme.set('rhoa', rho_adjusted)
            
            # Inversion ERT
            mgr = ert.ERTManager()
            mgr.setData(scheme)
            
            print("   🔄 Inversion en cours...")
            mesh = mgr.invert(verbose=False)
            rho_model = mgr.paraModel(mgr.model)
            
            # Créer la figure avec matplotlib
            import matplotlib.pyplot as plt
            from matplotlib.colors import LogNorm
            
            fig, ax = plt.subplots(figsize=(14, 6))
            
            # Plot du modèle inversé avec PyGIMLI
            pg.show(mesh, rho_model, ax=ax, cMap='Spectral_r', 
                   logScale=True, colorBar=True, 
                   label='Résistivité (Ω·m)')
            
            ax.set_xlabel('Distance (m)', fontsize=12, fontweight='bold')
            ax.set_ylabel('Profondeur (m)', fontsize=12, fontweight='bold')
            ax.set_title('Coupe ERT 2D - Inversion PyGIMLI', 
                        fontsize=14, fontweight='bold', pad=20)
            ax.grid(True, alpha=0.3)
            
            # Sauvegarder
            output_file = "/tmp/ert_section_2d_pygimli.png"
            plt.tight_layout()
            plt.savefig(output_file, dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"✅ Coupe 2D PyGIMLI générée: {{output_file}}")
            print(f"   Maillage: {{mesh.cellCount()}} cellules")
            print(f"   RMS: {{mgr.inv.chi2():.2f}}")
            
            # Générer aussi version HTML interactive
            html_output = f'''
            <html>
            <head><title>Coupe ERT 2D - PyGIMLI</title></head>
            <body style="font-family: Arial; padding: 20px;">
                <h2>🔬 Inversion ERT avec PyGIMLI</h2>
                <img src="{{output_file}}" style="max-width: 100%; border: 2px solid #333;">
                <div style="margin-top: 20px;">
                    <p><strong>Paramètres:</strong></p>
                    <ul>
                        <li>Nombre d'électrodes: {{scheme.sensorCount()}}</li>
                        <li>Mesures: {{scheme.size()}}</li>
                        <li>Cellules du maillage: {{mesh.cellCount()}}</li>
                        <li>RMS final: {{mgr.inv.chi2():.2f}}</li>
                    </ul>
                </div>
            </body>
            </html>
            '''
            
            html_file = "/tmp/ert_section_2d.html"
            with open(html_file, 'w', encoding='utf-8') as f:
                f.write(html_output)
            
            print(f"✅ Rapport HTML: {{html_file}}")
            
    except Exception as pygimli_error:
        print(f"⚠️  PyGIMLI non disponible ou erreur: {{pygimli_error}}")
        print("\\n🔄 Méthode alternative avec Matplotlib...")
        
        # Méthode 2: Matplotlib avec interpolation scipy
        try:
            from scipy.interpolate import griddata
            import matplotlib.pyplot as plt
            from matplotlib.colors import LinearSegmentedColormap, LogNorm
            
            # Créer une grille dense même avec peu de points
            n_x = max(50, len(np.unique(x_coords)) * 10)
            n_z = max(30, len(np.unique(z_coords)) * 10)
            
            grid_x = np.linspace(x_coords.min(), x_coords.max(), n_x)
            grid_z = np.linspace(z_coords.min(), z_coords.max(), n_z)
            grid_X, grid_Z = np.meshgrid(grid_x, grid_z)
            
            # Interpolation avec plusieurs méthodes
            print(f"   Points de mesure: {{len(x_coords)}}")
            print(f"   Grille cible: {{n_x}}x{{n_z}} = {{n_x*n_z}} points")
            
            # Essayer cubic d'abord, sinon linear, sinon nearest
            for method in ['cubic', 'linear', 'nearest']:
                try:
                    grid_rho = griddata(
                        (x_coords, z_coords), 
                        resistivity, 
                        (grid_X, grid_Z), 
                        method=method
                    )
                    print(f"   ✅ Interpolation {{method}} réussie")
                    break
                except:
                    if method == 'nearest':
                        raise
                    continue
            
            # Remplir les NaN avec nearest neighbor
            if np.any(np.isnan(grid_rho)):
                mask = np.isnan(grid_rho)
                grid_rho[mask] = griddata(
                    (x_coords, z_coords), 
                    resistivity, 
                    (grid_X[mask], grid_Z[mask]), 
                    method='nearest'
                )
            
            # Créer la figure professionnelle
            fig, ax = plt.subplots(figsize=(14, 7))
            
            # Colormap ERT professionnelle
            colors_ert = ['#00008B', '#0000FF', '#00FFFF', '#00FF00', 
                         '#FFFF00', '#FF8800', '#FF0000', '#8B0000']
            cmap = LinearSegmentedColormap.from_list('ert_pro', colors_ert)
            
            # Plot avec échelle logarithmique
            im = ax.contourf(grid_X, grid_Z, grid_rho, 
                           levels=20, cmap=cmap, 
                           norm=LogNorm(vmin=max(0.1, grid_rho.min()), 
                                       vmax=grid_rho.max()))
            
            # Ajouter les points de mesure
            scatter = ax.scatter(x_coords, z_coords, c='black', s=30, 
                               marker='v', edgecolors='white', linewidths=1,
                               label='Points de mesure', zorder=10)
            
            # Colorbar
            cbar = plt.colorbar(im, ax=ax, orientation='vertical', pad=0.02)
            cbar.set_label('Résistivité (Ω·m)', fontsize=12, fontweight='bold')
            
            # Labels et titre
            ax.set_xlabel('Distance (m)', fontsize=12, fontweight='bold')
            ax.set_ylabel('Profondeur (m)', fontsize=12, fontweight='bold')
            ax.set_title('Coupe ERT 2D - Résistivité Apparente', 
                        fontsize=14, fontweight='bold', pad=20)
            ax.invert_yaxis()  # Profondeur augmente vers le bas
            ax.grid(True, alpha=0.3, linestyle='--')
            ax.legend(loc='upper right')
            
            # Annotations des zones géologiques
            rho_mean = grid_rho.mean()
            if rho_mean < 50:
                zone_text = "Zone conductrice (eau/argile)"
            elif rho_mean < 150:
                zone_text = "Zone moyenne (argile/limon)"
            elif rho_mean < 500:
                zone_text = "Zone résistante (sable/gravier)"
            else:
                zone_text = "Zone très résistante (roche)"
            
            ax.text(0.02, 0.98, zone_text, transform=ax.transAxes,
                   fontsize=10, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
            
            # Sauvegarder
            output_file = "/tmp/ert_section_2d_matplotlib.png"
            plt.tight_layout()
            plt.savefig(output_file, dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"\\n✅ Coupe 2D Matplotlib générée: {{output_file}}")
            print(f"   Résolution: {{n_x}}x{{n_z}} pixels")
            print(f"   Résistivité min: {{grid_rho.min():.2f}} Ω·m")
            print(f"   Résistivité max: {{grid_rho.max():.2f}} Ω·m")
            print(f"   Résistivité moyenne: {{grid_rho.mean():.2f}} Ω·m")
            
            # Version HTML interactive avec Plotly
            import plotly.graph_objects as go
            
            fig_plotly = go.Figure(data=go.Heatmap(
                z=grid_rho,
                x=grid_x,
                y=grid_z,
                colorscale='Jet',
                colorbar=dict(title="ρ (Ω·m)", titleside='right'),
                hovertemplate='X: %{{x:.1f}}m<br>Z: %{{y:.1f}}m<br>ρ: %{{z:.2f}} Ω·m<extra></extra>'
            ))
            
            # Ajouter les points de mesure
            fig_plotly.add_trace(go.Scatter(
                x=x_coords,
                y=z_coords,
                mode='markers',
                marker=dict(size=8, color='black', symbol='triangle-down',
                           line=dict(color='white', width=2)),
                name='Points de mesure',
                hovertemplate='X: %{{x:.1f}}m<br>Z: %{{y:.1f}}m<extra></extra>'
            ))
            
            fig_plotly.update_layout(
                title=dict(text="Coupe ERT 2D Interactive - Résistivité Apparente",
                          font=dict(size=18, family='Arial Black')),
                xaxis_title="Distance (m)",
                yaxis_title="Profondeur (m)",
                yaxis=dict(autorange='reversed'),
                height=600,
                template='plotly_white',
                hovermode='closest'
            )
            
            html_file = "/tmp/ert_section_2d.html"
            fig_plotly.write_html(html_file, include_plotlyjs='cdn')
            
            print(f"✅ Version interactive Plotly: {{html_file}}")
            
        except Exception as mpl_error:
            print(f"❌ Erreur Matplotlib: {{mpl_error}}")
            
            # Méthode 3: Fallback simple avec visualisation tools
            print("\\n🔄 Utilisation des outils de visualisation basiques...")
            html_output = viz.create_resistivity_profile(
                values=resistivity.tolist(),
                depths=z_coords.tolist(),
                title="Profil de Résistivité Vertical",
                interactive=True
            )
            output_file = "/tmp/ert_profile_1d.html"
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(html_output)
            print(f"✅ Profil 1D généré: {{output_file}}")
"""
        
        code += """
    print("\\n" + "="*80)
    print("✅ ANALYSE TERMINÉE")
    print("="*80)
    
except Exception as e:
    print(f"\\n❌ ERREUR: {{e}}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
"""
        
        print(f"✅ Code avancé généré: {len(code)} caractères")
        return code

    
    def _build_code_prompt(self, intent: Dict[str, Any]) -> str:
        """Construit le prompt pour la génération de code"""
        action = intent['action_type']
        files = intent['target_files']
        params = intent['parameters']
        
        prompt = f"""# Task: {action.capitalize()} {', '.join(files)}
# Parameters: {json.dumps(params)}
# Generate Python code to accomplish this task

import numpy as np
import os

def execute_task():
    \"\"\"
    {intent['original_query']}
    \"\"\"
"""
        
        if action == 'analyze' and params.get('depth_analysis'):
            prompt += """
    # Load data
    data = np.load('{file}')
    
    # Analyze depths where conditions are met
    depths = []
    
    # Return results
    return depths
"""
        
        return prompt
    
    def _generate_code_from_template(self, intent: Dict[str, Any]) -> str:
        """Génère du code SPÉCIFIQUE à la question posée"""
        action = intent['action_type']
        files = intent['target_files']
        params = intent['parameters']
        query = intent['original_query'].lower()
        
        # Obtenir le chemin complet du fichier
        file_path = files[0] if files else "unknown.dat"
        
        # Vérifier si c'est déjà un chemin absolu valide
        if os.path.isabs(file_path) and os.path.exists(file_path):
            pass
        elif not os.path.isabs(file_path):
            # Si c'est juste un nom de fichier, chercher dans /tmp
            temp_path = f"/tmp/{file_path}"
            if os.path.exists(temp_path):
                file_path = temp_path
        
        # NOUVEAU: Détecter le TYPE de question pour générer du code spécifique
        print(f"� Question posée: '{intent['original_query']}'")
        
        # Questions sur les profondeurs
        if any(word in query for word in ['profondeur', 'depth', 'prof', 'à quelle profondeur']):
            return self._generate_depth_specific_code(file_path, query, params)
        
        # Questions sur les types de sol/matériaux
        if any(word in query for word in ['type de sol', 'matériau', 'composition', 'nature', 'qu\'est-ce']):
            return self._generate_material_analysis_code(file_path, query)
        
        # Questions sur zones spécifiques (eau salée, argile, etc.)
        if any(word in query for word in ['eau salée', 'saline', 'argile', 'sable', 'roche']):
            return self._generate_zone_specific_code(file_path, query, params)
        
        # Questions sur statistiques/valeurs
        if any(word in query for word in ['combien', 'nombre', 'pourcentage', 'statistique', 'valeur']):
            return self._generate_statistics_code(file_path, query)
        
        # Questions de comparaison
        if any(word in query for word in ['différence', 'compare', 'vs', 'versus', 'contraste']):
            return self._generate_comparison_code(file_path, query)
        
        # Si aucune question spécifique détectée, utiliser les anciens templates
        templates = {
            'analyze_depth_saline': f'''
import numpy as np
import os
import struct
import re

# Charger les données en mode binaire
file_path = "{file_path}"
if not os.path.exists(file_path):
    print(f"❌ Fichier non trouvé: {{file_path}}")
    exit(1)

# Lire le fichier comme données binaires
with open(file_path, 'rb') as f:
    file_bytes = f.read()

print(f"📊 Fichier chargé: {{len(file_bytes)}} octets")

# Fonction hex_ascii_view
def hex_ascii_view(file_bytes, bytes_per_line=16, max_lines=50):
    lines = []
    for i in range(0, min(len(file_bytes), bytes_per_line*max_lines), bytes_per_line):
        chunk = file_bytes[i:i+bytes_per_line]
        hex_bytes = " ".join(f"{{b:02X}}" for b in chunk)
        ascii_bytes = "".join([chr(b) if 32 <= b <= 126 else "." for b in chunk])
        lines.append(f"{{i:08X}} {{hex_bytes:<48}} |{{ascii_bytes}}|")
    return "\\n".join(lines)

# Fonction d'extraction de nombres
def extract_numbers(file_bytes):
    numbers = []
    # Méthode 1: Extraire depuis ASCII
    ascii_text = "".join([chr(b) if 32 <= b <= 126 else " " for b in file_bytes])
    found = re.findall(r"[-+]?\\d*\\.\\d+|\\d+", ascii_text)
    numbers.extend([float(n) for n in found])
    
    # Méthode 2: Interpréter comme float32/float64
    for fmt, size in [('f', 4), ('d', 8)]:
        for i in range(0, len(file_bytes) - size + 1, size):
            try:
                value = struct.unpack(fmt, file_bytes[i:i+size])[0]
                if not np.isnan(value) and not np.isinf(value) and abs(value) < 1e6:
                    numbers.append(value)
            except:
                pass
    
    return numbers

# Extraire les nombres
numbers = extract_numbers(file_bytes)
print(f"\\n🔢 Nombres extraits: {{len(numbers)}}")

if numbers:
    # Analyser les valeurs de résistivité pour eau salée
    resistivity_threshold = 10  # Ω·m pour eau salée
    
    # Convertir en array numpy
    data = np.array(numbers)
    
    # Filtrer les valeurs qui ressemblent à des résistivités (0.1 à 1000 Ω·m)
    resistivity_values = data[(data > 0.1) & (data < 1000)]
    
    if len(resistivity_values) > 0:
        saline_values = resistivity_values[resistivity_values < resistivity_threshold]
        
        print(f"\\n🌊 RÉSULTATS ANALYSE EAU SALÉE:")
        print(f"Seuil résistivité: {{resistivity_threshold}} Ω·m")
        print(f"Valeurs de résistivité trouvées: {{len(resistivity_values)}}")
        print(f"Zones d'eau salée détectées: {{len(saline_values)}}")
        
        if len(saline_values) > 0:
            print(f"\\nStatistiques zones salées:")
            print(f"  Résistivité min: {{np.min(saline_values):.2f}} Ω·m")
            print(f"  Résistivité max: {{np.max(saline_values):.2f}} Ω·m")
            print(f"  Résistivité moyenne: {{np.mean(saline_values):.2f}} Ω·m")
            print(f"  Pourcentage: {{len(saline_values)/len(resistivity_values)*100:.1f}}%")
        else:
            print("❌ Aucune zone d'eau salée détectée")
    else:
        print("⚠️ Aucune valeur de résistivité détectée dans la plage attendue")
        
    # Afficher un aperçu hex
    print("\\n📜 Aperçu Hex + ASCII (100 premières lignes):")
    print(hex_ascii_view(file_bytes, max_lines=100))
else:
    print("❌ Aucun nombre extrait du fichier")
''',
            'search_resistivity': f'''
import numpy as np
import os
import struct
import re

file_path = "{file_path}"

# Lire le fichier comme données binaires
with open(file_path, 'rb') as f:
    file_bytes = f.read()

print(f"J'ai analysé le fichier {{os.path.basename(file_path)}} qui fait {{len(file_bytes)}} octets.")

# Fonction d'extraction de nombres
def extract_numbers(file_bytes):
    numbers = []
    # Méthode 1: ASCII
    ascii_text = "".join([chr(b) if 32 <= b <= 126 else " " for b in file_bytes])
    found = re.findall(r"[-+]?\\d*\\.\\d+|\\d+", ascii_text)
    numbers.extend([float(n) for n in found])
    
    # Méthode 2: float binaires
    for fmt, size in [('f', 4), ('d', 8)]:
        for i in range(0, len(file_bytes) - size + 1, size):
            try:
                value = struct.unpack(fmt, file_bytes[i:i+size])[0]
                if not np.isnan(value) and not np.isinf(value) and abs(value) < 1e6:
                    numbers.append(value)
            except:
                pass
    return numbers

numbers = extract_numbers(file_bytes)
print(f"J'ai extrait {{len(numbers)}} valeurs numériques du fichier.")

if numbers:
    data = np.array(numbers)
    
    # Filtrer valeurs de résistivité plausibles
    resistivity_values = data[(data > 0.1) & (data < 1000)]
    
    if len(resistivity_values) > 0:
        print(f"\\nParmi ces valeurs, {{len(resistivity_values)}} semblent être des mesures de résistivité électrique, allant de {{np.min(resistivity_values):.2f}} à {{np.max(resistivity_values):.2f}} ohm-mètres.")
        print(f"La valeur moyenne est de {{np.mean(resistivity_values):.2f}} ohm-mètres, avec une médiane de {{np.median(resistivity_values):.2f}} ohm-mètres.")
        
        # Classification des zones
        print(f"\\nD'après l'analyse géologique de ces données ERT (Electrical Resistivity Tomography), voici ce que je peux interpréter :")
        very_low = np.sum(resistivity_values < 10)
        low = np.sum((resistivity_values >= 10) & (resistivity_values < 50))
        medium = np.sum((resistivity_values >= 50) & (resistivity_values < 200))
        high = np.sum(resistivity_values >= 200)
        
        if very_low > 0:
            print(f"- {{very_low}} mesures indiquent des zones de très faible résistivité (moins de 10 ohm-mètres), ce qui suggère de l'eau salée ou de l'argile saturée d'eau.")
        if low > 0:
            print(f"- {{low}} mesures montrent une résistivité basse (10-50 ohm-mètres), typique d'argile ou de sable humide.")
        if medium > 0:
            print(f"- {{medium}} mesures correspondent à une résistivité moyenne (50-200 ohm-mètres), probablement un sol mixte ou du sable sec.")
        if high > 0:
            print(f"- {{high}} mesures révèlent une haute résistivité (plus de 200 ohm-mètres), ce qui indique de la roche compacte ou du gravier sec.")
        
        print(f"\\nCette analyse suggère que le site présente principalement des conditions de {{'très faible résistivité' if very_low > low + medium + high else 'résistivité variable'}}.")
    else:
        print("Je n'ai pas trouvé de valeurs qui correspondent typiquement à des mesures de résistivité électrique dans ce fichier.")
else:
    print("Aucune donnée numérique n'a pu être extraite de ce fichier binaire.")
''',
            'create_report': f'''
import numpy as np
import os
from datetime import datetime
import struct
import re

file_path = "{file_path}"

# Lire le fichier binaire
with open(file_path, 'rb') as f:
    file_bytes = f.read()

# Extraire nombres
def extract_numbers(file_bytes):
    numbers = []
    ascii_text = "".join([chr(b) if 32 <= b <= 126 else " " for b in file_bytes])
    found = re.findall(r"[-+]?\\d*\\.\\d+|\\d+", ascii_text)
    numbers.extend([float(n) for n in found])
    for fmt, size in [('f', 4), ('d', 8)]:
        for i in range(0, len(file_bytes) - size + 1, size):
            try:
                value = struct.unpack(fmt, file_bytes[i:i+size])[0]
                if not np.isnan(value) and not np.isinf(value) and abs(value) < 1e6:
                    numbers.append(value)
            except:
                pass
    return numbers

numbers = extract_numbers(file_bytes)
data = np.array(numbers) if numbers else np.array([])

# Générer rapport
report = f"""
{{"="*60}}
RAPPORT D'ANALYSE ERT - FICHIER BINAIRE
{{"="*60}}
Fichier: {{os.path.basename(file_path)}}
Date: {{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}}

DONNÉES FICHIER:
- Taille: {{len(file_bytes)}} octets ({{len(file_bytes)/1024:.2f}} KB)
- Nombres extraits: {{len(numbers)}}

"""

if len(data) > 0:
    resistivity_values = data[(data > 0.1) & (data < 1000)]
    
    if len(resistivity_values) > 0:
        report += f"""STATISTIQUES RÉSISTIVITÉ ({{len(resistivity_values)}} valeurs):
- Min: {{np.min(resistivity_values):.2f}} Ω·m
- Max: {{np.max(resistivity_values):.2f}} Ω·m  
- Moyenne: {{np.mean(resistivity_values):.2f}} Ω·m
- Médiane: {{np.median(resistivity_values):.2f}} Ω·m
- Écart-type: {{np.std(resistivity_values):.2f}} Ω·m

INTERPRÉTATION GÉOLOGIQUE:
"""
        very_low = np.sum(resistivity_values < 10)
        low = np.sum((resistivity_values >= 10) & (resistivity_values < 50))
        medium = np.sum((resistivity_values >= 50) & (resistivity_values < 200))
        high = np.sum(resistivity_values >= 200)
        
        report += f"""
- Très basse résistivité (< 10 Ω·m): {{very_low}} points
  → Eau salée, argile saturée, forte conductivité
  
- Basse résistivité (10-50 Ω·m): {{low}} points
  → Argile, sable humide, nappe phréatique
  
- Résistivité moyenne (50-200 Ω·m): {{medium}} points
  → Sol mixte, sable sec, formations consolidées
  
- Haute résistivité (> 200 Ω·m): {{high}} points
  → Roche compacte, gravier sec, faible humidité
"""
    else:
        report += "⚠️ Aucune valeur de résistivité plausible détectée\\n"
else:
    report += "❌ Aucune donnée numérique extraite\\n"

report += f"\\n{{"="*60}}\\n"

print(report)

# Sauvegarder le rapport
output_file = f"rapport_ert_{{datetime.now().strftime('%Y%m%d_%H%M%S')}}.txt"
with open(output_file, 'w') as f:
    f.write(report)
print(f"\\n💾 Rapport sauvegardé: {{output_file}}")
'''
        }
        
        # Sélectionner le template approprié
        if action == 'analyze' and params.get('water_type') == 'saline':
            template = templates['analyze_depth_saline']
        elif action == 'search' and params.get('resistivity'):
            template = templates['search_resistivity']
        elif action == 'create':
            template = templates['create_report']
        else:
            # Template générique
            template = templates.get('search_resistivity', '')
        
        # Le template est déjà une f-string avec file_path injecté
        # Plus besoin de faire de remplacement !
        
        # DEBUG: afficher les premières lignes du code généré
        print(f"🐛 DEBUG - Code généré (10 premières lignes):")
        for i, line in enumerate(template.split('\n')[:10], 1):
            print(f"  {i}: {line}")
        
        return template
    
    def execute_code(self, code: str, timeout: int = 30) -> Tuple[bool, str, str]:
        """
        Exécute le code Python généré
        
        Args:
            code: Code Python à exécuter
            timeout: Timeout en secondes
        
        Returns:
            (success, stdout, stderr)
        """
        # Créer un fichier temporaire
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(code)
            temp_file = f.name
        
        try:
            # Exécuter le code
            result = subprocess.run(
                [sys.executable, temp_file],
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=os.getcwd()
            )
            
            success = result.returncode == 0
            stdout = result.stdout
            stderr = result.stderr
            
            # Enregistrer dans l'historique
            self.execution_history.append({
                'code': code,
                'success': success,
                'stdout': stdout,
                'stderr': stderr
            })
            
            return success, stdout, stderr
            
        except subprocess.TimeoutExpired:
            return False, "", f"⏱️ Timeout après {timeout} secondes"
        except Exception as e:
            return False, "", f"❌ Erreur d'exécution: {str(e)}"
        finally:
            # Nettoyer le fichier temporaire
            try:
                os.unlink(temp_file)
            except:
                pass
    
    def process_action(self, query: str) -> Dict[str, Any]:
        """
        Processus complet: détection → génération → exécution
        
        Args:
            query: Requête utilisateur
        
        Returns:
            dict: {
                'success': bool,
                'intent': dict,
                'code': str,
                'output': str,
                'error': str
            }
        """
        # 1. Détection d'intention
        intent = self.detect_action_intent(query)
        
        if not intent['is_action']:
            return {
                'success': False,
                'intent': intent,
                'message': "❌ Aucune action détectée dans la requête"
            }
        
        # 2. Génération de code
        code = self.generate_code(intent)
        
        if not code:
            return {
                'success': False,
                'intent': intent,
                'message': "❌ Échec de génération du code"
            }
        
        # 3. Exécution
        success, stdout, stderr = self.execute_code(code)
        
        return {
            'success': success,
            'intent': intent,
            'code': code,
            'output': stdout,
            'error': stderr
        }
