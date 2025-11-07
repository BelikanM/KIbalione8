#!/usr/bin/env python3
"""
PARSEUR SPÉCIALISÉ - Fichiers .DAT format Survey-Point / Depth / Data
Structure typique des profils ERT avec points de mesure latéraux et profondeurs
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from scipy.interpolate import griddata
from typing import Dict, Tuple, Optional
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

class SurveyDepthDataParser:
    """
    Parseur intelligent pour fichiers .dat avec structure:
    survey-point | depth | data | project
    
    Permet de créer des coupes 2D et visualisations 3D automatiquement
    """
    
    def __init__(self):
        self.data = None
        self.structure = None
        self.interpolated_grid = None
    
    def detect_format(self, file_path: str) -> bool:
        """Détecte si le fichier correspond au format survey-point/depth/data"""
        try:
            # Tester plusieurs encodages
            for enc in ['utf-8', 'utf-16', 'latin1', 'cp1252', 'iso-8859-1']:
                try:
                    with open(file_path, 'r', encoding=enc, errors='ignore') as f:
                        content = f.read()

                    lines = content.split('\n')
                    lines = [line.strip() for line in lines if line.strip()]

                    if len(lines) < 5:  # Au moins header + 4 lignes de données
                        continue

                    # Tester différents délimiteurs
                    for sep in ['\t', ',', ';', '|', ' ', '  ', '   ']:
                        try:
                            # Analyser header
                            header_parts = [part.strip() for part in lines[0].split(sep) if part.strip()]
                            if len(header_parts) < 3:
                                continue

                            # Compter les lignes de données valides
                            valid_data_lines = 0
                            survey_candidates = []
                            depth_candidates = []
                            data_candidates = []

                            for line in lines[1:]:  # Ignorer header
                                parts = [part.strip() for part in line.split(sep) if part.strip()]
                                if len(parts) >= 3:
                                    valid_data_lines += 1

                                    # Analyser chaque colonne
                                    for i, part in enumerate(parts):
                                        try:
                                            val = float(part)

                                            # Étendre les listes si nécessaire
                                            while len(survey_candidates) <= i:
                                                survey_candidates.append([])
                                            while len(depth_candidates) <= i:
                                                depth_candidates.append([])
                                            while len(data_candidates) <= i:
                                                data_candidates.append([])

                                            # Classifier la valeur
                                            if 1 <= val <= 1000 and val == int(val) and val <= 50:  # Survey point (petits entiers)
                                                survey_candidates[i].append(val)
                                            elif val < 0:  # Depth (négatif)
                                                depth_candidates[i].append(val)
                                            elif 0.001 <= val <= 10000:  # Résistivité (large plage)
                                                data_candidates[i].append(val)
                                        except ValueError:
                                            pass  # Pas un nombre

                            # Vérifier si nous avons au moins 3 lignes de données valides
                            if valid_data_lines < 3:
                                continue

                            # Trouver les colonnes avec le plus de candidats valides
                            survey_col = None
                            depth_col = None
                            data_col = None

                            max_survey = max(len(candidates) for candidates in survey_candidates) if survey_candidates else 0
                            max_depth = max(len(candidates) for candidates in depth_candidates) if depth_candidates else 0
                            max_data = max(len(candidates) for candidates in data_candidates) if data_candidates else 0

                            if max_survey >= valid_data_lines * 0.8:  # 80% des lignes
                                survey_col = survey_candidates.index(max(survey_candidates, key=len))

                            if max_depth >= valid_data_lines * 0.8:
                                depth_col = depth_candidates.index(max(depth_candidates, key=len))

                            if max_data >= valid_data_lines * 0.8:
                                data_col = data_candidates.index(max(data_candidates, key=len))

                            # Vérifier que nous avons les 3 colonnes essentielles
                            if survey_col is not None and depth_col is not None and data_col is not None:
                                # Vérifier qu'elles sont différentes
                                if len({survey_col, depth_col, data_col}) == 3:
                                    print(f"✅ Format détecté avec séparateur '{sep}' et encodage '{enc}'")
                                    print(f"   - Lignes de données valides: {valid_data_lines}")
                                    print(f"   - Colonnes: survey={survey_col}, depth={depth_col}, data={data_col}")
                                    return True

                        except Exception as e:
                            continue

                except Exception as e:
                    continue

            return False

        except Exception as e:
            print(f"Erreur détection format: {e}")
            return False
    
    def load_data(self, file_path: str, encoding: str = 'utf-8') -> pd.DataFrame:
        """
        Charge un fichier survey-point/depth/data
        Détecte automatiquement le délimiteur et les colonnes
        TRAITE TOUTES LES DONNÉES JUSQU'À LA FIN
        """
        # Tester plusieurs encodages
        for enc in [encoding, 'utf-8', 'utf-16', 'latin1', 'cp1252', 'iso-8859-1']:
            try:
                # Lire tout le fichier d'abord pour analyser
                with open(file_path, 'r', encoding=enc, errors='ignore') as f:
                    content = f.read()

                # Séparer en lignes
                lines = content.split('\n')
                lines = [line.strip() for line in lines if line.strip()]  # Supprimer lignes vides

                if len(lines) < 2:
                    continue

                # Tester différents délimiteurs
                for sep in ['\t', ',', ';', '|', ' ', '  ', '   ']:
                    try:
                        # Tester avec la première ligne comme header
                        header_line = lines[0]
                        header_parts = [part.strip() for part in header_line.split(sep) if part.strip()]

                        if len(header_parts) < 3:
                            continue

                        # Analyser les premières lignes de données
                        data_lines = []
                        for line in lines[1:]:  # Commencer après le header
                            parts = [part.strip() for part in line.split(sep) if part.strip()]
                            if len(parts) >= 3:  # Au moins survey, depth, data
                                data_lines.append(parts)

                        if len(data_lines) < 3:  # Au moins 3 lignes de données
                            continue

                        # Créer DataFrame
                        max_cols = max(len(row) for row in data_lines)
                        df_data = []

                        for row in data_lines:
                            # Compléter avec None si nécessaire
                            while len(row) < max_cols:
                                row.append(None)
                            df_data.append(row)

                        df = pd.DataFrame(df_data, columns=[f'col_{i}' for i in range(max_cols)])

                        # Identifier colonnes par contenu
                        survey_col = None
                        depth_col = None
                        data_col = None
                        project_col = None

                        for i, col_name in enumerate(df.columns):
                            sample_values = df[col_name].dropna().head(10).astype(str)

                            # Vérifier si colonne contient des nombres
                            numeric_count = 0
                            for val in sample_values:
                                try:
                                    float(val)
                                    numeric_count += 1
                                except:
                                    pass

                            if numeric_count >= len(sample_values) * 0.8:  # 80% numérique
                                # Colonne numérique - analyser le type
                                numeric_vals = pd.to_numeric(sample_values, errors='coerce').dropna()

                                if len(numeric_vals) > 0:
                                    min_val = numeric_vals.min()
                                    max_val = numeric_vals.max()

                                    # Survey point: généralement petit entier positif
                                    if min_val >= 1 and max_val <= 1000 and numeric_vals.dtype == 'int64':
                                        if survey_col is None:
                                            survey_col = col_name

                                    # Depth: généralement négatif (profondeur)
                                    elif min_val < 0 and max_val <= 0:
                                        if depth_col is None:
                                            depth_col = col_name

                                    # Data: valeurs de résistivité (généralement 0.001 à 10000)
                                    elif 0.001 <= min_val and max_val <= 10000:
                                        if data_col is None:
                                            data_col = col_name

                            else:
                                # Colonne texte - probablement project
                                if project_col is None:
                                    project_col = col_name

                        # Vérifier que nous avons les colonnes essentielles
                        if survey_col and depth_col and data_col:
                            # Renommer colonnes
                            rename_map = {
                                survey_col: 'survey_point',
                                depth_col: 'depth',
                                data_col: 'data'
                            }
                            if project_col:
                                rename_map[project_col] = 'project'

                            df = df.rename(columns=rename_map)

                            # Convertir en numérique et nettoyer
                            df['survey_point'] = pd.to_numeric(df['survey_point'], errors='coerce')
                            df['depth'] = pd.to_numeric(df['depth'], errors='coerce')
                            df['data'] = pd.to_numeric(df['data'], errors='coerce')

                            # Supprimer les lignes avec NaN dans les colonnes essentielles
                            df = df.dropna(subset=['survey_point', 'depth', 'data'])

                            # Trier par survey_point puis par depth
                            df = df.sort_values(['survey_point', 'depth'])

                            # Réinitialiser l'index
                            df = df.reset_index(drop=True)

                            if len(df) > 0:
                                self.data = df

                                # Analyser structure complète
                                self.structure = {
                                    'num_points': len(df),
                                    'num_survey_points': df['survey_point'].nunique(),
                                    'depth_range': (df['depth'].min(), df['depth'].max()),
                                    'data_range': (df['data'].min(), df['data'].max()),
                                    'survey_points': sorted(df['survey_point'].unique()),
                                    'depths': sorted(df['depth'].unique()),
                                    'has_project': 'project' in df.columns,
                                    'encoding_used': enc,
                                    'delimiter_used': sep
                                }

                                print(f"✅ Chargé {len(df)} points de données valides")
                                print(f"   - Survey points: {self.structure['num_survey_points']}")
                                print(f"   - Plage profondeur: {self.structure['depth_range']}")
                                print(f"   - Plage données: {self.structure['data_range']}")

                                return df

                    except Exception as e:
                        print(f"Erreur avec séparateur '{sep}': {e}")
                        continue

            except Exception as e:
                print(f"Erreur avec encodage '{enc}': {e}")
                continue

        raise ValueError("Impossible de charger le fichier avec le format survey-point/depth/data")
    
    def create_2d_section(
        self, 
        interpolation_method: str = 'cubic',
        resolution: int = 100,
        title: str = "Coupe 2D - Profil de Résistivité"
    ) -> Tuple[go.Figure, Dict]:
        """
        Crée une coupe 2D interpolée avec Plotly
        
        Args:
            interpolation_method: 'linear', 'cubic', 'nearest'
            resolution: Résolution de la grille (100 = 100x100)
            title: Titre du graphique
        
        Returns:
            (figure_plotly, info_dict)
        """
        if self.data is None:
            raise ValueError("Aucune donnée chargée. Appelez load_data() d'abord.")
        
        df = self.data
        
        # Extraire données
        x = df['survey_point'].values
        y = df['depth'].values
        z = df['data'].values
        
        # Créer grille régulière
        xi = np.linspace(x.min(), x.max(), resolution)
        yi = np.linspace(y.min(), y.max(), resolution)
        Xi, Yi = np.meshgrid(xi, yi)
        
        # Interpolation
        Zi = griddata((x, y), z, (Xi, Yi), method=interpolation_method)
        
        # Sauvegarder grille
        self.interpolated_grid = {
            'x': xi,
            'y': yi,
            'z': Zi
        }
        
        # Créer figure Plotly
        fig = go.Figure()
        
        # Heatmap interpolée
        fig.add_trace(go.Heatmap(
            x=xi,
            y=yi,
            z=Zi,
            colorscale='Jet',
            colorbar=dict(
                title=dict(text="Résistivité (Ω·m)", side='right'),
                thickness=20,
                len=0.8
            ),
            hovertemplate='Point: %{x}<br>Profondeur: %{y:.1f}m<br>Valeur: %{z:.3f}<extra></extra>'
        ))
        
        # Points de mesure réels
        fig.add_trace(go.Scatter(
            x=x,
            y=y,
            mode='markers',
            marker=dict(
                size=4,
                color='white',
                line=dict(color='black', width=1)
            ),
            name='Points mesure',
            hovertemplate='Point: %{x}<br>Profondeur: %{y:.1f}m<extra></extra>'
        ))
        
        # Layout
        fig.update_layout(
            title=dict(
                text=f'<b>{title}</b>',
                x=0.5,
                xanchor='center',
                font=dict(size=18, color='#2c3e50')
            ),
            xaxis=dict(
                title='Point de Mesure (Survey Point)',
                showgrid=True,
                gridcolor='rgba(128,128,128,0.2)'
            ),
            yaxis=dict(
                title='Profondeur (m)',
                autorange='reversed',  # Profondeur vers le bas
                showgrid=True,
                gridcolor='rgba(128,128,128,0.2)'
            ),
            width=1000,
            height=600,
            showlegend=True,
            hovermode='closest',
            plot_bgcolor='#f8f9fa'
        )
        
        info = {
            'type': '2d_section',
            'num_points': len(df),
            'survey_range': (x.min(), x.max()),
            'depth_range': (y.min(), y.max()),
            'data_range': (z.min(), z.max()),
            'interpolation': interpolation_method,
            'resolution': resolution
        }
        
        return fig, info
    
    def create_3d_volume(self, title: str = "Volume 3D - Profil Résistivité") -> Tuple[go.Figure, Dict]:
        """
        Crée une visualisation 3D du volume de données
        """
        if self.data is None:
            raise ValueError("Aucune donnée chargée.")
        
        df = self.data
        
        fig = go.Figure()
        
        # Scatter 3D avec couleurs
        fig.add_trace(go.Scatter3d(
            x=df['survey_point'],
            y=df['depth'],
            z=df['data'],
            mode='markers',
            marker=dict(
                size=6,
                color=df['data'],
                colorscale='Jet',
                showscale=True,
                colorbar=dict(
                    title=dict(text="Valeur", side='right'),
                    thickness=15,
                    len=0.7
                )
            ),
            hovertemplate='Point: %{x}<br>Profondeur: %{y:.1f}m<br>Valeur: %{z:.3f}<extra></extra>'
        ))
        
        # Layout 3D
        fig.update_layout(
            title=dict(text=f'<b>{title}</b>', x=0.5, xanchor='center'),
            scene=dict(
                xaxis=dict(title='Survey Point', backgroundcolor='rgb(230,230,230)'),
                yaxis=dict(title='Profondeur (m)', backgroundcolor='rgb(230,230,230)', autorange='reversed'),
                zaxis=dict(title='Valeur Mesurée', backgroundcolor='rgb(230,230,230)'),
                camera=dict(eye=dict(x=1.5, y=1.5, z=1.2))
            ),
            width=1000,
            height=700
        )
        
        info = {
            'type': '3d_volume',
            'num_points': len(df),
            'dimensions': self.structure
        }
        
        return fig, info
    
    def create_vertical_profiles(self) -> Tuple[go.Figure, Dict]:
        """
        Crée des profils verticaux pour chaque point de mesure
        """
        if self.data is None:
            raise ValueError("Aucune donnée chargée.")
        
        df = self.data
        fig = go.Figure()
        
        # Créer un profil par point de mesure
        for survey_pt in sorted(df['survey_point'].unique()):
            subset = df[df['survey_point'] == survey_pt].sort_values('depth')
            
            fig.add_trace(go.Scatter(
                x=subset['data'],
                y=subset['depth'],
                mode='lines+markers',
                name=f'Point {int(survey_pt)}',
                hovertemplate=f'Point {int(survey_pt)}<br>Profondeur: %{{y:.1f}}m<br>Valeur: %{{x:.3f}}<extra></extra>'
            ))
        
        fig.update_layout(
            title='Profils Verticaux par Point de Mesure',
            xaxis_title='Valeur Mesurée',
            yaxis_title='Profondeur (m)',
            yaxis=dict(autorange='reversed'),
            width=900,
            height=600,
            showlegend=True,
            hovermode='closest'
        )
        
        info = {
            'type': 'vertical_profiles',
            'num_profiles': df['survey_point'].nunique()
        }
        
        return fig, info
    
    def create_contour_map(self, num_levels: int = 15) -> Tuple[go.Figure, Dict]:
        """
        Crée une carte de contours
        """
        if self.interpolated_grid is None:
            # Créer grille d'abord
            self.create_2d_section()
        
        xi = self.interpolated_grid['x']
        yi = self.interpolated_grid['y']
        Zi = self.interpolated_grid['z']
        
        fig = go.Figure()
        
        # Contours remplis
        fig.add_trace(go.Contour(
            x=xi,
            y=yi,
            z=Zi,
            colorscale='Jet',
            ncontours=num_levels,
            contours=dict(
                showlabels=True,
                labelfont=dict(size=10, color='white')
            ),
            colorbar=dict(
                title=dict(text="Valeur", side='right'),
                thickness=20
            )
        ))
        
        fig.update_layout(
            title='Carte de Contours - Isolignes de Résistivité',
            xaxis_title='Survey Point',
            yaxis_title='Profondeur (m)',
            yaxis=dict(autorange='reversed'),
            width=1000,
            height=600
        )
        
        info = {
            'type': 'contour_map',
            'num_levels': num_levels
        }
        
        return fig, info
    
    def create_water_zones_map(self, title: str = "Carte des Zones d'Eau - Analyse Hydrogéologique") -> Tuple[go.Figure, Dict]:
        """
        Crée une carte spécialisée pour identifier les zones d'eau
        Utilise les valeurs typiques de résistivité de l'eau
        """
        if self.data is None:
            raise ValueError("Aucune donnée chargée.")
        
        from water_resistivity_analyzer import WaterResistivityAnalyzer
        water_analyzer = WaterResistivityAnalyzer()
        
        df = self.data
        
        # Créer une classification par type d'eau
        water_classifications = []
        for rho in df['data']:
            classification = water_analyzer.classify_water_type(rho)
            water_classifications.append(classification['type'])
        
        df = df.copy()
        df['water_type'] = water_classifications
        
        # Couleurs spécifiques pour les types d'eau
        water_colors = {
            'eau_ultra_pure': 'darkblue',
            'eau_distillee': 'blue',
            'eau_pluie': 'lightblue',
            'eau_douce': 'cyan',
            'eau_saumatre': 'green',
            'eau_salee': 'yellow',
            'eau_brine': 'orange',
            'eau_thermale': 'red',
            'eau_polluee': 'purple',
            'eau_tres_pure': 'navy',
            'conducteur_fort': 'maroon'
        }
        
        fig = go.Figure()
        
        # Scatter plot coloré par type d'eau
        for water_type, color in water_colors.items():
            subset = df[df['water_type'] == water_type]
            if len(subset) > 0:
                classification = water_analyzer.classify_water_type(subset['data'].iloc[0])
                fig.add_trace(go.Scatter(
                    x=subset['survey_point'],
                    y=subset['depth'],
                    mode='markers',
                    marker=dict(
                        size=8,
                        color=color,
                        symbol='circle',
                        line=dict(color='black', width=1)
                    ),
                    name=f"{water_type.replace('_', ' ').title()}",
                    hovertemplate=(
                        f'Type: {classification["description"]}<br>' +
                        'Point: %{x}<br>' +
                        'Profondeur: %{y:.1f}m<br>' +
                        'Résistivité: %{customdata:.3f} Ω·m<extra></extra>'
                    ),
                    customdata=subset['data']
                ))
        
        # Lignes de séparation des types d'eau
        water_thresholds = [0.01, 0.1, 1, 10, 100, 1000, 10000]
        for threshold in water_thresholds:
            if df['data'].min() <= threshold <= df['data'].max():
                fig.add_hline(
                    y=threshold,
                    line_dash="dash",
                    line_color="gray",
                    opacity=0.5,
                    annotation_text=f"ρ = {threshold} Ω·m",
                    annotation_position="top right"
                )
        
        fig.update_layout(
            title=dict(text=f'<b>{title}</b>', x=0.5, xanchor='center'),
            xaxis=dict(title='Point de Mesure (Survey Point)'),
            yaxis=dict(
                title='Profondeur (m)',
                autorange='reversed'
            ),
            width=1000,
            height=700,
            showlegend=True,
            hovermode='closest'
        )
        
        info = {
            'type': 'water_zones_map',
            'num_water_types': len(df['water_type'].unique()),
            'water_types_detected': list(df['water_type'].unique()),
            'potential_aquifers': len(df[df['data'] > 10]),  # Eau douce
            'conductive_zones': len(df[df['data'] < 1])     # Eau salée/brine
        }
        
        return fig, info
    
    def generate_statistics_report(self) -> str:
        """Génère un rapport statistique détaillé avec analyse d'eau intégrée"""
        if self.data is None or self.structure is None:
            return "Aucune donnée chargée"
        
        df = self.data
        
        # Importer l'analyseur d'eau
        try:
            from water_resistivity_analyzer import WaterResistivityAnalyzer
            water_analyzer = WaterResistivityAnalyzer()
            water_analysis = water_analyzer.analyze_resistivity_profile(df['data'].values)
        except ImportError:
            water_analysis = None
        
        report = f"""
╔══════════════════════════════════════════════════════════════════════════╗
║           RAPPORT STATISTIQUE - Profil Survey-Point / Depth              ║
╚══════════════════════════════════════════════════════════════════════════╝

📊 DIMENSIONS
   Points de mesure total   : {self.structure['num_points']}
   Nombre survey points     : {self.structure['num_survey_points']}
   Plage horizontale        : {self.structure['survey_points'][0]} → {self.structure['survey_points'][-1]}
   Plage profondeur         : {self.structure['depth_range'][0]:.1f} → {self.structure['depth_range'][1]:.1f} m
   
📈 STATISTIQUES DES DONNÉES
   Minimum                  : {self.structure['data_range'][0]:.6f}
   Maximum                  : {self.structure['data_range'][1]:.6f}
   Moyenne                  : {df['data'].mean():.6f}
   Médiane                  : {df['data'].median():.6f}
   Écart-type               : {df['data'].std():.6f}
   
🌊 ANALYSE DES VALEURS D'EAU (RÉSISTIVITÉ)
"""
        
        if water_analysis:
            report += f"""   Type d'eau dominant      : {water_analysis['water_zones'][0]['classification']['type'].replace('_', ' ').title()}
   Zone d'eau potentielle   : {'OUI' if df['data'].mean() <= 10 else 'NON'} (ρ ≤ 10 Ω·m)
   Variabilité aquifère     : {'Forte' if water_analysis['statistics']['std'] > water_analysis['statistics']['mean'] * 0.3 else 'Faible'}
   """
        
        report += f"""
📍 DISTRIBUTION PAR SURVEY POINT
"""
        
        for sp in self.structure['survey_points']:
            subset = df[df['survey_point'] == sp]
            water_type = ""
            if water_analysis:
                sp_water = [z for z in water_analysis['water_zones'] if z['index'] in subset.index][:1]
                if sp_water:
                    water_type = f" ({sp_water[0]['classification']['type'].replace('_', ' ').title()})"
            report += f"   Point {int(sp):2d} : {len(subset):3d} mesures | Moyenne: {subset['data'].mean():.4f} Ω·m{water_type}\n"
        
        if water_analysis:
            report += f"""
🎯 INTERPRÉTATION HYDROGÉOLOGIQUE
"""
            for interp in water_analysis['interpretation']:
                report += f"   • {interp}\n"
        
        report += f"""
🎯 ANALYSE GÉOPHYSIQUE
   Type de données          : {"Résistivité apparente (Ω·m)" if df['data'].max() > 1 else "Conductivité normalisée"}
   Profil                   : {"Transect latéral" if self.structure['num_survey_points'] < 20 else "Profil long"}
   Couverture profondeur    : {abs(self.structure['depth_range'][1] - self.structure['depth_range'][0]):.1f} m
   
╚══════════════════════════════════════════════════════════════════════════╝
"""
        return report


# Fonction d'utilisation rapide
def parse_survey_depth_file(
    file_path: str,
    create_plots: bool = True,
    plot_types: list = ['2d', '3d', 'profiles', 'contour']
) -> Dict:
    """
    Parse un fichier survey-point/depth/data et crée les visualisations
    
    Args:
        file_path: Chemin du fichier
        create_plots: Créer les graphiques automatiquement
        plot_types: Types de plots à créer
    
    Returns:
        Dict avec data, figures, stats
    """
    parser = SurveyDepthDataParser()
    
    # Détecter format
    if not parser.detect_format(file_path):
        raise ValueError("Format survey-point/depth/data non détecté dans ce fichier")
    
    # Charger données
    df = parser.load_data(file_path)
    
    results = {
        'data': df,
        'structure': parser.structure,
        'stats_report': parser.generate_statistics_report(),
        'figures': {}
    }
    
    if create_plots:
        if '2d' in plot_types:
            fig_2d, info_2d = parser.create_2d_section()
            results['figures']['2d_section'] = fig_2d
            results['info_2d'] = info_2d
        
        if '3d' in plot_types:
            fig_3d, info_3d = parser.create_3d_volume()
            results['figures']['3d_volume'] = fig_3d
            results['info_3d'] = info_3d
        
        if 'profiles' in plot_types:
            fig_prof, info_prof = parser.create_vertical_profiles()
            results['figures']['vertical_profiles'] = fig_prof
            results['info_profiles'] = info_prof
        
        if 'contour' in plot_types:
            fig_cont, info_cont = parser.create_contour_map()
            results['figures']['contour_map'] = fig_cont
            results['info_contour'] = info_cont
    
    return results


# Test
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
        
        print("🔍 Analyse du fichier survey-point/depth/data...")
        results = parse_survey_depth_file(file_path)
        
        print("\n" + results['stats_report'])
        
        print("\n📊 Visualisations créées:")
        for name, fig in results['figures'].items():
            output = f"/tmp/{name}.html"
            fig.write_html(output)
            print(f"   ✅ {name}: {output}")
    
    else:
        print("Usage: python survey_depth_parser.py <fichier.dat>")
