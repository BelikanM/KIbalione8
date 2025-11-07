"""
Agent de Génération de Graphiques Scientifiques
Modèle spécialisé 1-2GB pour créer des visualisations ERT professionnelles
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
# import seaborn as sns  # Non nécessaire
from scipy.interpolate import griddata
from scipy import stats
import re
import json
from typing import Dict, List, Optional, Tuple, Any
import io
import base64
from datetime import datetime

# Ajouter le chemin pour les outils de visualisation
sys.path.append('/home/belikan/KIbalione8')
from visualization_tools import VisualizationEngine

class GraphGenerationAgent:
    """
    Agent IA spécialisé dans la génération de graphiques scientifiques
    Utilise Qwen2.5-0.5B-Instruct (500MB - ultra rapide) pour comprendre les demandes
    """
    
    def __init__(self, model_name: str = "Qwen/Qwen2.5-0.5B-Instruct", device: str = "cpu"):
        """
        Initialiser l'agent de génération de graphiques
        
        Args:
            model_name: Modèle HuggingFace (par défaut Qwen2.5-0.5B)
            device: "cuda" ou "cpu"
        """
        self.device = device
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        self.viz_engine = VisualizationEngine()
        
        print(f"🎨 Initialisation de l'agent de génération de graphiques...")
        print(f"📦 Modèle: {model_name}")
        
        self._load_model()
    
    def _load_model(self):
        """Charger le modèle de langage pour comprendre les demandes"""
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            import torch
            
            cache_dir = os.path.expanduser("~/.cache/huggingface/graph_models")
            os.makedirs(cache_dir, exist_ok=True)
            
            print(f"📁 Cache: {cache_dir}")
            
            # Vérifier si modèle en cache
            if os.path.exists(os.path.join(cache_dir, self.model_name.replace('/', '_'))):
                print("📦 Modèle trouvé en cache - chargement rapide")
            
            # Charger tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                cache_dir=cache_dir,
                trust_remote_code=True
            )
            
            # Charger modèle
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                cache_dir=cache_dir,
                trust_remote_code=True,
                torch_dtype=torch.float32,
                low_cpu_mem_usage=True
            )
            
            if self.device == "cuda" and torch.cuda.is_available():
                self.model = self.model.cuda()
                print("⚡ GPU CUDA activé")
            else:
                print("🖥️ CPU utilisé")
            
            self.model.eval()
            print("✅ Modèle chargé avec succès")
            
        except Exception as e:
            print(f"⚠️ Erreur chargement modèle: {e}")
            print("📊 Utilisation en mode fallback (sans LLM)")
    
    def understand_request(self, user_query: str, file_context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Comprendre la demande de l'utilisateur et extraire les informations
        
        Args:
            user_query: Demande de l'utilisateur
            file_context: Contexte du fichier (données, stats, etc.)
        
        Returns:
            Dict avec type_graph, params, data_needed
        """
        query_lower = user_query.lower()
        
        # Détection du type de graphique demandé
        graph_type = "unknown"
        if any(kw in query_lower for kw in ["3d", "volume", "volumique", "cube", "sous-sol", "profondeur 3d"]):
            graph_type = "3d_volume"
        elif any(kw in query_lower for kw in ["coupe", "section", "2d", "tomographie"]):
            graph_type = "2d_section"
        elif any(kw in query_lower for kw in ["profil", "vertical", "1d", "sondage"]):
            graph_type = "profile_1d"
        elif any(kw in query_lower for kw in ["histogramme", "distribution", "histogram"]):
            graph_type = "histogram"
        elif any(kw in query_lower for kw in ["scatter", "nuage", "points"]):
            graph_type = "scatter"
        elif any(kw in query_lower for kw in ["statistique", "stats", "tableau"]):
            graph_type = "statistics_table"
        elif any(kw in query_lower for kw in ["courbe", "ligne", "évolution"]):
            graph_type = "line_plot"
        
        # Détection des paramètres
        needs_colors = any(kw in query_lower for kw in ["couleur", "color", "coloré"])
        needs_legend = any(kw in query_lower for kw in ["légende", "legend"])
        needs_grid = any(kw in query_lower for kw in ["grille", "grid"])
        interactive = not any(kw in query_lower for kw in ["png", "image", "static"])
        
        return {
            "graph_type": graph_type,
            "needs_colors": needs_colors,
            "needs_legend": needs_legend,
            "needs_grid": needs_grid,
            "interactive": interactive,
            "file_context": file_context,
            "query": user_query
        }
    
    def generate_explanation(self, graph_info: Dict[str, Any], max_tokens: int = 800) -> str:
        """
        Générer une explication détaillée du graphique avec le LLM
        
        Args:
            graph_info: Informations sur le graphique généré
            max_tokens: Nombre maximum de tokens pour l'explication (800 = ~600 mots)
        
        Returns:
            Explication en markdown
        """
        if self.model is None or self.tokenizer is None:
            return self._generate_fallback_explanation(graph_info)
        
        try:
            import torch
            
            # Construire le prompt pour l'explication
            prompt = f"""Tu es un expert en géophysique ERT. Explique ce graphique de façon détaillée et pédagogique.

GRAPHIQUE GÉNÉRÉ:
Type: {graph_info['graph_type']}
Données: {graph_info.get('data_summary', 'N/A')}
Statistiques: {graph_info.get('statistics', 'N/A')}

Donne une explication complète avec:
1. Description du graphique (que voit-on?)
2. Interprétation géologique (que signifie-t-on?)
3. Points clés à retenir (3-5 bullet points)
4. Recommandations d'analyse (prochaines étapes)

Explication:"""
            
            # Tokenizer
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                max_length=512,
                truncation=True
            )
            
            if self.device == "cuda":
                inputs = {k: v.cuda() for k, v in inputs.items()}
            
            # Génération avec 1000 tokens pour explications complètes et structurées
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=1000,  # 1000 tokens pour réponses détaillées
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    repetition_penalty=1.1,  # Éviter les répétitions
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            # Décoder
            explanation = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extraire seulement la partie après "Explication:"
            if "Explication:" in explanation:
                explanation = explanation.split("Explication:")[-1].strip()
            
            return explanation
            
        except Exception as e:
            print(f"⚠️ Erreur génération explication: {e}")
            return self._generate_fallback_explanation(graph_info)
    
    def _generate_fallback_explanation(self, graph_info: Dict[str, Any]) -> str:
        """Génération d'explication basique sans LLM"""
        graph_type = graph_info.get('graph_type', 'unknown')
        
        explanations = {
            "2d_section": """
## 📊 Coupe 2D de Résistivité

Cette coupe 2D montre la distribution spatiale de la résistivité électrique du sous-sol.

**Interprétation:**
- 🔴 **Zones rouges** (haute résistivité): Roches compactes, sables secs
- 🔵 **Zones bleues** (basse résistivité): Eau, argiles saturées
- 🟡 **Zones jaunes** (résistivité moyenne): Formations intermédiaires

**Points clés:**
• La résistivité varie de {min_value} à {max_value} Ω·m
• Les zones conductrices peuvent indiquer la présence d'eau
• L'interpolation permet de visualiser la continuité des formations
""",
            "profile_1d": """
## 📈 Profil Vertical de Résistivité

Ce profil montre l'évolution de la résistivité en fonction de la profondeur.

**Interprétation:**
- Les variations abruptes indiquent des changements de lithologie
- Les zones de faible résistivité peuvent correspondre à des aquifères
- La tendance générale révèle la structure stratifiée du sous-sol

**Recommandations:**
• Analyser les discontinuités majeures
• Corréler avec les données géologiques locales
• Vérifier la cohérence avec les forages existants
""",
            "histogram": """
## 📊 Histogramme de Distribution

Cet histogramme montre la répartition statistique des valeurs de résistivité.

**Analyse statistique:**
- La distribution révèle les matériaux dominants
- Les pics multiples indiquent plusieurs formations distinctes
- L'asymétrie peut révéler des anomalies géologiques

**Utilité:**
• Identification des populations statistiques
• Détection des valeurs aberrantes
• Classification automatique des formations
"""
        }
        
        return explanations.get(graph_type, "Graphique généré avec succès.")
    
    def create_2d_section(
        self,
        x_coords: np.ndarray,
        z_coords: np.ndarray,
        resistivity: np.ndarray,
        title: str = "Coupe ERT 2D",
        output_path: str = "/tmp/ert_section_2d.html"
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Créer une coupe 2D professionnelle avec Plotly
        
        Returns:
            (file_path, graph_info)
        """
        print(f"🎨 Génération coupe 2D: {len(resistivity)} points")
        
        # Créer grille interpolée
        n_x, n_z = 60, 40
        grid_x = np.linspace(x_coords.min(), x_coords.max(), n_x)
        grid_z = np.linspace(z_coords.min(), z_coords.max(), n_z)
        X, Z = np.meshgrid(grid_x, grid_z)
        
        # Interpolation
        try:
            rho_grid = griddata(
                (x_coords, z_coords),
                resistivity,
                (X, Z),
                method='cubic',
                fill_value=np.nan
            )
            
            # Remplir NaN avec nearest
            if np.any(np.isnan(rho_grid)):
                mask = np.isnan(rho_grid)
                rho_grid[mask] = griddata(
                    (x_coords, z_coords),
                    resistivity,
                    (X[mask], Z[mask]),
                    method='nearest'
                )
        except:
            rho_grid = griddata(
                (x_coords, z_coords),
                resistivity,
                (X, Z),
                method='nearest'
            )
        
        # Créer figure Plotly
        fig = go.Figure()
        
        # Heatmap
        fig.add_trace(go.Heatmap(
            z=rho_grid,
            x=grid_x,
            y=grid_z,
            colorscale='Jet',
            colorbar=dict(
                title=dict(text="ρ (Ω·m)", side='right'),
                thickness=20,
                len=0.7
            ),
            hovertemplate='X: %{x:.1f}m<br>Z: %{y:.1f}m<br>ρ: %{z:.2f} Ω·m<extra></extra>'
        ))
        
        # Points de mesure
        fig.add_trace(go.Scatter(
            x=x_coords,
            y=z_coords,
            mode='markers',
            marker=dict(
                size=6,
                color='black',
                symbol='triangle-down',
                line=dict(color='white', width=1)
            ),
            name='Points de mesure',
            hovertemplate='Mesure<br>X: %{x:.1f}m<br>Z: %{y:.1f}m<extra></extra>'
        ))
        
        # Layout
        fig.update_layout(
            title=dict(
                text=title,
                font=dict(size=20, family='Arial Black'),
                x=0.5,
                xanchor='center'
            ),
            xaxis_title="Distance (m)",
            yaxis_title="Profondeur (m)",
            yaxis=dict(autorange='reversed'),
            height=600,
            template='plotly_white',
            hovermode='closest',
            showlegend=True,
            legend=dict(x=1.02, y=1)
        )
        
        # Ajouter bouton de téléchargement
        fig.write_html(
            output_path,
            include_plotlyjs='cdn',
            config={
                'toImageButtonOptions': {
                    'format': 'png',
                    'filename': 'coupe_ert_2d',
                    'height': 800,
                    'width': 1200,
                    'scale': 2
                },
                'displayModeBar': True,
                'modeBarButtonsToAdd': ['downloadImage']
            }
        )
        
        print(f"✅ Coupe 2D sauvegardée: {output_path}")
        
        # Informations du graphique
        graph_info = {
            "graph_type": "2d_section",
            "output_path": output_path,
            "data_summary": f"{len(resistivity)} mesures, grille {n_x}x{n_z}",
            "statistics": {
                "min": float(resistivity.min()),
                "max": float(resistivity.max()),
                "mean": float(resistivity.mean()),
                "median": float(np.median(resistivity))
            }
        }
        
        return output_path, graph_info
    
    def create_profile_1d(
        self,
        depths: np.ndarray,
        resistivity: np.ndarray,
        title: str = "Profil de Résistivité Vertical",
        output_path: str = "/tmp/ert_profile_1d.html"
    ) -> Tuple[str, Dict[str, Any]]:
        """Créer un profil vertical 1D interactif"""
        print(f"📈 Génération profil 1D: {len(resistivity)} points")
        
        fig = go.Figure()
        
        # Ligne principale
        fig.add_trace(go.Scatter(
            x=resistivity,
            y=depths,
            mode='lines+markers',
            name='Résistivité',
            line=dict(color='royalblue', width=3),
            marker=dict(
                size=8,
                color=resistivity,
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="ρ (Ω·m)", x=1.15)
            ),
            hovertemplate='ρ: %{x:.2f} Ω·m<br>Profondeur: %{y:.1f}m<extra></extra>'
        ))
        
        # Zones géologiques colorées
        water_mask = (resistivity >= 0.5) & (resistivity <= 50)
        if water_mask.any():
            fig.add_trace(go.Scatter(
                x=resistivity[water_mask],
                y=depths[water_mask],
                mode='markers',
                name='Eau/Argile',
                marker=dict(size=12, color='cyan', symbol='square'),
                hovertemplate='Zone conductrice<br>%{x:.2f} Ω·m<extra></extra>'
            ))
        
        fig.update_layout(
            title=dict(text=title, font=dict(size=20, family='Arial Black')),
            xaxis_title="Résistivité (Ω·m)",
            yaxis_title="Profondeur (m)",
            yaxis=dict(autorange='reversed'),
            height=700,
            template='plotly_white',
            hovermode='closest',
            showlegend=True
        )
        
        fig.write_html(
            output_path,
            include_plotlyjs='cdn',
            config={
                'toImageButtonOptions': {
                    'format': 'png',
                    'filename': 'profil_ert_1d',
                    'height': 900,
                    'width': 800,
                    'scale': 2
                },
                'displayModeBar': True
            }
        )
        
        print(f"✅ Profil 1D sauvegardé: {output_path}")
        
        graph_info = {
            "graph_type": "profile_1d",
            "output_path": output_path,
            "data_summary": f"{len(resistivity)} mesures de profondeur",
            "statistics": {
                "depth_range": f"{depths.min():.1f} - {depths.max():.1f} m",
                "rho_range": f"{resistivity.min():.2f} - {resistivity.max():.2f} Ω·m"
            }
        }
        
        return output_path, graph_info
    
    def create_statistics_table(
        self,
        resistivity: np.ndarray,
        output_path: str = "/tmp/ert_statistics.html"
    ) -> Tuple[str, Dict[str, Any]]:
        """Créer un tableau de statistiques détaillé"""
        print("📊 Génération tableau statistiques")
        
        # Calculer statistiques
        stats_data = {
            "Statistique": [
                "Nombre de mesures",
                "Minimum",
                "Maximum",
                "Moyenne",
                "Médiane",
                "Écart-type",
                "Quartile 1 (Q1)",
                "Quartile 3 (Q3)",
                "Intervalle interquartile (IQR)",
                "Coefficient de variation"
            ],
            "Valeur": [
                len(resistivity),
                f"{resistivity.min():.2f} Ω·m",
                f"{resistivity.max():.2f} Ω·m",
                f"{resistivity.mean():.2f} Ω·m",
                f"{np.median(resistivity):.2f} Ω·m",
                f"{resistivity.std():.2f} Ω·m",
                f"{np.percentile(resistivity, 25):.2f} Ω·m",
                f"{np.percentile(resistivity, 75):.2f} Ω·m",
                f"{np.percentile(resistivity, 75) - np.percentile(resistivity, 25):.2f} Ω·m",
                f"{(resistivity.std() / resistivity.mean() * 100):.1f}%"
            ]
        }
        
        df = pd.DataFrame(stats_data)
        
        # Créer table avec Plotly
        fig = go.Figure(data=[go.Table(
            header=dict(
                values=list(df.columns),
                fill_color='paleturquoise',
                align='left',
                font=dict(size=14, family='Arial Black')
            ),
            cells=dict(
                values=[df[col] for col in df.columns],
                fill_color='lavender',
                align='left',
                font=dict(size=12)
            )
        )])
        
        fig.update_layout(
            title="📊 Statistiques Descriptives - Résistivité ERT",
            height=500
        )
        
        fig.write_html(output_path, include_plotlyjs='cdn')
        
        print(f"✅ Tableau sauvegardé: {output_path}")
        
        graph_info = {
            "graph_type": "statistics_table",
            "output_path": output_path,
            "data_summary": f"Analyse statistique de {len(resistivity)} mesures"
        }
        
        return output_path, graph_info

    def generate_structured_report(
        self, 
        data: Dict[str, Any], 
        user_query: str,
        output_path: str = "/tmp/rapport_ert.html"
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Génère un rapport structuré complet avec tableaux, graphiques et explications
        
        Args:
            data: Données à analyser (x, z, resistivity, etc.)
            user_query: Question de l'utilisateur pour contextualiser
            output_path: Chemin de sauvegarde
        
        Returns:
            Tuple (chemin fichier, info graphique)
        """
        print(f"📋 Génération du rapport structuré...")
        
        x_coords = np.array(data.get('x', []))
        z_coords = np.array(data.get('z', []))
        resistivity = np.array(data.get('resistivity', []))
        
        if len(resistivity) == 0:
            print("❌ Pas de données de résistivité")
            return None, {}
        
        # Créer document HTML complet
        html_content = f"""
<!DOCTYPE html>
<html lang="fr">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Rapport d'Analyse ERT - Kibali</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        }}
        .container {{
            background: white;
            padding: 30px;
            border-radius: 15px;
            box-shadow: 0 10px 40px rgba(0,0,0,0.2);
        }}
        h1 {{
            color: #2c3e50;
            border-bottom: 4px solid #667eea;
            padding-bottom: 15px;
            font-size: 32px;
        }}
        h2 {{
            color: #34495e;
            margin-top: 30px;
            border-left: 5px solid #667eea;
            padding-left: 15px;
        }}
        .stat-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }}
        .stat-card {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 5px 15px rgba(0,0,0,0.1);
        }}
        .stat-card h3 {{
            margin: 0 0 10px 0;
            font-size: 14px;
            opacity: 0.9;
            text-transform: uppercase;
        }}
        .stat-card .value {{
            font-size: 28px;
            font-weight: bold;
            margin: 0;
        }}
        .download-btn {{
            background: #667eea;
            color: white;
            padding: 12px 24px;
            border: none;
            border-radius: 8px;
            cursor: pointer;
            font-size: 16px;
            margin: 10px 5px;
            transition: all 0.3s;
            display: inline-block;
            text-decoration: none;
        }}
        .download-btn:hover {{
            background: #764ba2;
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(0,0,0,0.2);
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        th {{
            background: #667eea;
            color: white;
            padding: 15px;
            text-align: left;
            font-weight: bold;
        }}
        td {{
            padding: 12px 15px;
            border-bottom: 1px solid #ddd;
        }}
        tr:hover {{
            background: #f5f5f5;
        }}
        .interpretation {{
            background: #f8f9fa;
            border-left: 5px solid #28a745;
            padding: 20px;
            margin: 20px 0;
            border-radius: 5px;
        }}
        .warning {{
            background: #fff3cd;
            border-left: 5px solid #ffc107;
            padding: 15px;
            margin: 15px 0;
            border-radius: 5px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🔬 Rapport d'Analyse ERT</h1>
        <p><strong>Date:</strong> {datetime.now().strftime('%d/%m/%Y %H:%M')}</p>
        <p><strong>Demande:</strong> {user_query}</p>
        
        <h2>📊 Statistiques Globales</h2>
        <div class="stat-grid">
            <div class="stat-card">
                <h3>Points de Mesure</h3>
                <p class="value">{len(resistivity)}</p>
            </div>
            <div class="stat-card">
                <h3>Résistivité Min</h3>
                <p class="value">{resistivity.min():.2f} Ω·m</p>
            </div>
            <div class="stat-card">
                <h3>Résistivité Max</h3>
                <p class="value">{resistivity.max():.2f} Ω·m</p>
            </div>
            <div class="stat-card">
                <h3>Moyenne</h3>
                <p class="value">{resistivity.mean():.2f} Ω·m</p>
            </div>
            <div class="stat-card">
                <h3>Médiane</h3>
                <p class="value">{np.median(resistivity):.2f} Ω·m</p>
            </div>
            <div class="stat-card">
                <h3>Écart-type</h3>
                <p class="value">{resistivity.std():.2f} Ω·m</p>
            </div>
        </div>
        
        <h2>🌍 Classification Géologique</h2>
        <table>
            <thead>
                <tr>
                    <th>Matériau</th>
                    <th>Résistivité Typique</th>
                    <th>Points Détectés</th>
                    <th>Pourcentage</th>
                    <th>Interprétation</th>
                </tr>
            </thead>
            <tbody>
"""
        
        # Classification par zones de résistivité
        zones = [
            ("💧 Eau/Argile saturée", "0.5-50 Ω·m", (resistivity >= 0.5) & (resistivity <= 50)),
            ("🟤 Argile/Limon", "50-150 Ω·m", (resistivity > 50) & (resistivity <= 150)),
            ("🟡 Sable/Gravier", "150-500 Ω·m", (resistivity > 150) & (resistivity <= 500)),
            ("⚫ Roche compacte", ">500 Ω·m", resistivity > 500),
        ]
        
        for material, range_val, mask in zones:
            count = mask.sum()
            percent = (count / len(resistivity) * 100) if len(resistivity) > 0 else 0
            
            if percent > 50:
                interpretation = "✅ Matériau dominant"
            elif percent > 20:
                interpretation = "⚠️ Présence significative"
            elif percent > 5:
                interpretation = "📍 Traces détectées"
            else:
                interpretation = "❌ Absent ou négligeable"
            
            html_content += f"""
                <tr>
                    <td><strong>{material}</strong></td>
                    <td>{range_val}</td>
                    <td>{count}</td>
                    <td>{percent:.1f}%</td>
                    <td>{interpretation}</td>
                </tr>
"""
        
        html_content += """
            </tbody>
        </table>
"""
        
        # Interprétation hydrogéologique
        water_zone = (resistivity >= 0.5) & (resistivity <= 50)
        if water_zone.sum() > 0:
            html_content += f"""
        <div class="interpretation">
            <h3>💧 Analyse Hydrogéologique</h3>
            <p><strong>{water_zone.sum()} points</strong> ({water_zone.sum()/len(resistivity)*100:.1f}%) 
            présentent des valeurs de résistivité compatibles avec la présence d'eau (0.5-50 Ω·m).</p>
            <ul>
                <li>Résistivité moyenne zone eau: <strong>{resistivity[water_zone].mean():.2f} Ω·m</strong></li>
                <li>Résistivité minimale: <strong>{resistivity[water_zone].min():.2f} Ω·m</strong></li>
                <li>Variation: <strong>{resistivity[water_zone].std():.2f} Ω·m</strong> (écart-type)</li>
            </ul>
            <p><strong>Interprétation:</strong> Les zones conductrices suggèrent la présence d'eau 
            ou d'argiles saturées. Un forage de reconnaissance est recommandé pour confirmer.</p>
        </div>
"""
        
        # Recommandations
        html_content += """
        <h2>🎯 Recommandations</h2>
        <div class="warning">
            <h4>⚠️ Points d'Attention</h4>
            <ul>
"""
        
        if resistivity.std() / resistivity.mean() > 2:
            html_content += "<li>Forte hétérogénéité détectée - analyses complémentaires recommandées</li>"
        
        if water_zone.sum() > len(resistivity) * 0.3:
            html_content += "<li>Zone conductrice étendue - potentiel aquifère à investiguer</li>"
        
        if (resistivity > 1000).sum() > 0:
            html_content += "<li>Valeurs de résistivité très élevées - vérifier calibration instrument</li>"
        
        html_content += """
            </ul>
        </div>
        
        <h2>📥 Téléchargements</h2>
        <button class="download-btn" onclick="downloadCSV()">📊 Télécharger CSV</button>
        <button class="download-btn" onclick="window.print()">🖨️ Imprimer/PDF</button>
        <button class="download-btn" onclick="downloadJSON()">📦 Données JSON</button>
        
        <script>
        function downloadCSV() {
            const data = """ + json.dumps({
                'X': x_coords.tolist() if len(x_coords) > 0 else [],
                'Z': z_coords.tolist() if len(z_coords) > 0 else [],
                'Resistivity': resistivity.tolist()
            }) + """;
            
            let csv = 'X,Z,Resistivity\\n';
            for (let i = 0; i < data.Resistivity.length; i++) {
                csv += `${data.X[i] || i},${data.Z[i] || 0},${data.Resistivity[i]}\\n`;
            }
            
            const blob = new Blob([csv], { type: 'text/csv' });
            const url = window.URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = 'donnees_ert.csv';
            a.click();
        }
        
        function downloadJSON() {
            const data = """ + json.dumps({
                'metadata': {
                    'date': datetime.now().isoformat(),
                    'query': user_query,
                    'n_points': len(resistivity)
                },
                'statistics': {
                    'min': float(resistivity.min()),
                    'max': float(resistivity.max()),
                    'mean': float(resistivity.mean()),
                    'median': float(np.median(resistivity)),
                    'std': float(resistivity.std())
                },
                'data': {
                    'x': x_coords.tolist() if len(x_coords) > 0 else [],
                    'z': z_coords.tolist() if len(z_coords) > 0 else [],
                    'resistivity': resistivity.tolist()
                }
            }, indent=2) + """;
            
            const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
            const url = window.URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = 'rapport_ert.json';
            a.click();
        }
        </script>
        
        <footer style="margin-top: 40px; padding-top: 20px; border-top: 2px solid #eee; text-align: center; color: #777;">
            <p>📍 Généré par <strong>Kibali ERT Analysis System</strong> | Agent de Génération de Graphiques v2.0</p>
            <p>🔬 Rapport professionnel d'analyse géophysique</p>
        </footer>
    </div>
</body>
</html>
"""
        
        # Sauvegarder
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"✅ Rapport structuré généré: {output_path}")
        
        graph_info = {
            "graph_type": "structured_report",
            "output_path": output_path,
            "data_summary": f"Rapport complet avec {len(resistivity)} mesures",
            "statistics": {
                "n_points": len(resistivity),
                "min": float(resistivity.min()),
                "max": float(resistivity.max()),
                "mean": float(resistivity.mean())
            }
        }
        
        return output_path, graph_info
    
    def create_3d_volume(
        self,
        x_coords: np.ndarray,
        y_coords: np.ndarray,
        z_coords: np.ndarray,
        resistivity: np.ndarray,
        title: str = "Volume 3D de Résistivité",
        output_path: str = "/tmp/ert_3d_volume.html"
    ) -> Tuple[str, Dict]:
        """
        Créer une visualisation 3D interactive du volume de résistivité
        
        Args:
            x_coords: Coordonnées X (distance horizontale)
            y_coords: Coordonnées Y (distance latérale)
            z_coords: Coordonnées Z (profondeur, valeurs négatives)
            resistivity: Valeurs de résistivité
            title: Titre du graphique
            output_path: Chemin de sauvegarde
        
        Returns:
            Tuple (chemin_fichier, info_graphique)
        """
        print(f"🎨 Génération volume 3D avec {len(resistivity)} points...")
        
        # Si données 1D, créer une grille 3D
        if len(resistivity.shape) == 1:
            # Créer une grille 3D à partir de données 1D
            n_x = int(np.cbrt(len(resistivity))) + 1
            n_y = n_x
            n_z = len(resistivity) // (n_x * n_y) + 1
            
            # Créer coordonnées si pas fournies
            if len(x_coords) != len(resistivity):
                x = np.linspace(0, 100, n_x)
                y = np.linspace(0, 100, n_y)
                z = np.linspace(0, -50, n_z)
                X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
                x_coords = X.flatten()
                y_coords = Y.flatten()
                z_coords = Z.flatten()
                
                # Répéter/tronquer résistivité pour matcher
                n_total = len(x_coords)
                if len(resistivity) < n_total:
                    resistivity = np.pad(resistivity, (0, n_total - len(resistivity)), mode='edge')
                else:
                    resistivity = resistivity[:n_total]
        
        # Créer figure Plotly 3D
        fig = go.Figure()
        
        # Volume avec isosurfaces
        fig.add_trace(go.Volume(
            x=x_coords.flatten(),
            y=y_coords.flatten(),
            z=z_coords.flatten(),
            value=resistivity.flatten(),
            isomin=resistivity.min(),
            isomax=resistivity.max(),
            opacity=0.15,
            surface_count=12,
            colorscale='Jet',
            colorbar=dict(
                title=dict(text="ρ (Ω·m)", side='right'),
                thickness=20,
                len=0.8,
                x=1.02
            ),
            hovertemplate='X: %{x:.1f}m<br>Y: %{y:.1f}m<br>Z: %{z:.1f}m<br>ρ: %{value:.2f} Ω·m<extra></extra>',
            caps=dict(x_show=True, y_show=True, z_show=True)
        ))
        
        # Ajout de coupes optionnelles
        # Coupe horizontale au milieu
        mid_z = z_coords.mean()
        mask_z = np.abs(z_coords - mid_z) < 5
        if mask_z.any():
            fig.add_trace(go.Scatter3d(
                x=x_coords[mask_z],
                y=y_coords[mask_z],
                z=z_coords[mask_z],
                mode='markers',
                marker=dict(
                    size=3,
                    color=resistivity[mask_z],
                    colorscale='Jet',
                    showscale=False,
                    opacity=0.6
                ),
                name=f'Coupe Z={mid_z:.1f}m',
                hovertemplate='ρ: %{marker.color:.2f} Ω·m<extra></extra>'
            ))
        
        # Layout 3D
        fig.update_layout(
            title=dict(
                text=f'<b>{title}</b>',
                x=0.5,
                xanchor='center',
                font=dict(size=20, color='#2c3e50')
            ),
            scene=dict(
                xaxis=dict(
                    title='Distance X (m)',
                    backgroundcolor="rgb(230, 230,230)",
                    gridcolor="white",
                    showbackground=True,
                    zerolinecolor="white"
                ),
                yaxis=dict(
                    title='Distance Y (m)',
                    backgroundcolor="rgb(230, 230,230)",
                    gridcolor="white",
                    showbackground=True,
                    zerolinecolor="white"
                ),
                zaxis=dict(
                    title='Profondeur Z (m)',
                    backgroundcolor="rgb(230, 230,230)",
                    gridcolor="white",
                    showbackground=True,
                    zerolinecolor="white"
                ),
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=1.2),
                    center=dict(x=0, y=0, z=-0.1)
                ),
                aspectmode='manual',
                aspectratio=dict(x=1, y=1, z=0.7)
            ),
            width=1000,
            height=800,
            showlegend=True,
            legend=dict(x=0.02, y=0.98),
            hovermode='closest',
            paper_bgcolor='#f8f9fa',
            font=dict(family='Arial', size=12)
        )
        
        # Sauvegarder
        fig.write_html(output_path)
        
        print(f"✅ Volume 3D sauvegardé: {output_path}")
        
        graph_info = {
            "graph_type": "3d_volume",
            "output_path": output_path,
            "data_summary": f"Volume 3D avec {len(resistivity)} points",
            "statistics": {
                "n_points": len(resistivity),
                "min": float(resistivity.min()),
                "max": float(resistivity.max()),
                "mean": float(resistivity.mean()),
                "std": float(resistivity.std())
            },
            "dimensions": {
                "x_range": (float(x_coords.min()), float(x_coords.max())),
                "y_range": (float(y_coords.min()), float(y_coords.max())),
                "z_range": (float(z_coords.min()), float(z_coords.max()))
            }
        }
        
        return output_path, graph_info


# Test rapide
if __name__ == "__main__":
    print("🧪 Test de l'agent de génération de graphiques")
    
    # Créer agent
    agent = GraphGenerationAgent()
    
    # Données de test
    x = np.linspace(0, 100, 50)
    z = np.linspace(0, 20, 50)
    rho = np.random.uniform(1, 100, 50)
    
    # Test profil 1D
    path, info = agent.create_profile_1d(z, rho, "Test Profil")
    print(f"✅ Test réussi: {path}")
    
    # Générer explication
    explanation = agent.generate_explanation(info, max_tokens=500)
    print(f"\n📝 Explication:\n{explanation}")
