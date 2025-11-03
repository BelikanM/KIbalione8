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
    
    def __init__(self, model_path: str = None):
        """
        Args:
            model_path: Chemin vers le modèle de code (DeepSeek-Coder)
        """
        self.model_path = model_path or "/root/.cache/huggingface/code_models"
        self.model = None
        self.tokenizer = None
        self.execution_history = []
        
    def load_model(self):
        """Charge le modèle de code en mémoire"""
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            import torch
            
            model_name = "deepseek-ai/deepseek-coder-1.3b-instruct"
            
            print("🤖 Chargement du modèle de code...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                cache_dir=self.model_path
            )
            
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                cache_dir=self.model_path,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None,
                low_cpu_mem_usage=True
            )
            
            print("✅ Modèle chargé avec succès")
            return True
            
        except Exception as e:
            print(f"❌ Erreur chargement modèle: {e}")
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
            'analyze': ['analyse', 'analyser', 'examine', 'étudie', 'vérifie'],
            'search': ['cherche', 'trouve', 'recherche', 'localise', 'détecte'],
            'create': ['crée', 'génère', 'fabrique', 'construis', 'produis'],
            'process': ['traite', 'transforme', 'convertis', 'calcule', 'extrait'],
            'visualize': ['affiche', 'montre', 'visualise', 'dessine', 'trace']
        }
        
        # Détection de fichiers
        file_extensions = ['.bin', '.npy', '.npz', '.dat', '.txt', '.csv', '.json', '.pdf']
        detected_files = []
        
        for word in query.split():
            if any(word.endswith(ext) for ext in file_extensions):
                detected_files.append(word)
        
        # Détection du type d'action
        action_type = None
        for action, keywords in action_keywords.items():
            if any(kw in query_lower for kw in keywords):
                action_type = action
                break
        
        # Extraction de paramètres spécifiques
        parameters = {}
        
        # Profondeurs
        if 'profondeur' in query_lower:
            parameters['depth_analysis'] = True
        
        # Eau salée/douce
        if 'eau salée' in query_lower or 'saline' in query_lower:
            parameters['water_type'] = 'saline'
        elif 'eau douce' in query_lower:
            parameters['water_type'] = 'fresh'
        
        # Résistivité
        if 'résistivité' in query_lower or 'resistivity' in query_lower:
            parameters['resistivity'] = True
        
        # ERT/Géophysique
        if 'ert' in query_lower or 'géophysique' in query_lower:
            parameters['geophysics'] = True
        
        is_action = action_type is not None and len(detected_files) > 0
        
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
        # Construire le prompt pour le modèle
        prompt = self._build_code_prompt(intent)
        
        # Si le modèle n'est pas chargé, utiliser des templates
        if self.model is None:
            return self._generate_code_from_template(intent)
        
        # Générer avec le modèle
        try:
            import torch
            
            inputs = self.tokenizer(prompt, return_tensors="pt")
            if torch.cuda.is_available():
                inputs = {k: v.cuda() for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=512,
                    temperature=0.2,  # Plus déterministe
                    do_sample=True,
                    top_p=0.95,
                    stop_strings=["```", "###"]
                )
            
            generated_code = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extraire seulement le code
            if "```python" in generated_code:
                code = generated_code.split("```python")[1].split("```")[0]
            else:
                code = generated_code.split(prompt)[1] if prompt in generated_code else generated_code
            
            return code.strip()
            
        except Exception as e:
            print(f"⚠️ Erreur génération avec modèle: {e}. Utilisation template.")
            return self._generate_code_from_template(intent)
    
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
        """Génère du code à partir de templates prédéfinis"""
        action = intent['action_type']
        files = intent['target_files']
        params = intent['parameters']
        
        templates = {
            'analyze_depth_saline': '''
import numpy as np
import os

# Charger les données
file_path = "{file}"
if not os.path.exists(file_path):
    print(f"❌ Fichier non trouvé: {{file_path}}")
    exit(1)

data = np.load(file_path)
print(f"📊 Données chargées: shape={{data.shape}}, dtype={{data.dtype}}")

# Analyser les profondeurs d'eau salée
# Résistivité < 10 Ω·m = eau salée typiquement
if len(data.shape) >= 2:
    # Données 2D/3D
    resistivity_threshold = 10  # Ω·m pour eau salée
    
    saline_locations = np.where(data < resistivity_threshold)
    depths = saline_locations[0] if len(saline_locations) > 0 else []
    
    print(f"\\n🌊 RÉSULTATS ANALYSE EAU SALÉE:")
    print(f"Seuil résistivité: {{resistivity_threshold}} Ω·m")
    print(f"Nombre de points détectés: {{len(depths)}}")
    
    if len(depths) > 0:
        unique_depths = np.unique(depths)
        print(f"\\nProfondeurs trouvées:")
        for depth in unique_depths:
            count = np.sum(depths == depth)
            print(f"  - Profondeur {{depth}}: {{count}} points")
            
        print(f"\\nStatistiques:")
        print(f"  Profondeur min: {{np.min(unique_depths)}}")
        print(f"  Profondeur max: {{np.max(unique_depths)}}")
        print(f"  Profondeur moyenne: {{np.mean(unique_depths):.2f}}")
    else:
        print("❌ Aucune zone d'eau salée détectée")
else:
    print("⚠️ Format de données non supporté pour cette analyse")
''',
            'search_resistivity': '''
import numpy as np
import os

file_path = "{file}"
data = np.load(file_path)

print(f"🔍 RECHERCHE DANS {{file_path}}")
print(f"Shape: {{data.shape}}, Type: {{data.dtype}}")

# Statistiques de résistivité
print(f"\\n📊 Statistiques globales:")
print(f"  Min: {{np.min(data):.2f}} Ω·m")
print(f"  Max: {{np.max(data):.2f}} Ω·m")
print(f"  Moyenne: {{np.mean(data):.2f}} Ω·m")
print(f"  Médiane: {{np.median(data):.2f}} Ω·m")

# Détection d'anomalies
mean = np.mean(data)
std = np.std(data)
anomalies_low = np.where(data < mean - 2*std)
anomalies_high = np.where(data > mean + 2*std)

print(f"\\n⚡ Anomalies détectées:")
print(f"  Basse résistivité (< {{mean - 2*std:.2f}}): {{len(anomalies_low[0])}} points")
print(f"  Haute résistivité (> {{mean + 2*std:.2f}}): {{len(anomalies_high[0])}} points")
''',
            'create_report': '''
import numpy as np
import os
from datetime import datetime

file_path = "{file}"
data = np.load(file_path)

# Générer rapport
report = f"""
{'='*60}
RAPPORT D'ANALYSE ERT
{'='*60}
Fichier: {{file_path}}
Date: {{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}}

DONNÉES:
- Shape: {{data.shape}}
- Type: {{data.dtype}}
- Taille: {{data.nbytes / 1024:.2f}} KB

STATISTIQUES:
- Min: {{np.min(data):.2f}} Ω·m
- Max: {{np.max(data):.2f}} Ω·m  
- Moyenne: {{np.mean(data):.2f}} Ω·m
- Écart-type: {{np.std(data):.2f}} Ω·m

INTERPRÉTATION:
"""

# Classification des zones
very_low = np.sum(data < 10)
low = np.sum((data >= 10) & (data < 50))
medium = np.sum((data >= 50) & (data < 200))
high = np.sum(data >= 200)

report += f"""
- Très basse résistivité (< 10 Ω·m): {{very_low}} points
  → Eau salée, argile saturée
  
- Basse résistivité (10-50 Ω·m): {{low}} points
  → Argile, sable humide
  
- Résistivité moyenne (50-200 Ω·m): {{medium}} points
  → Sol mixte, sable sec
  
- Haute résistivité (> 200 Ω·m): {{high}} points
  → Roche, gravier sec
  
{'='*60}
"""

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
        
        # Remplacer le placeholder du fichier
        if files:
            template = template.replace('{file}', files[0])
        
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
