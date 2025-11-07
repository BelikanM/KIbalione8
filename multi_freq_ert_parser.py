"""
Parseur ERT Multi-Fréquences et Multi-Fichiers avec PyGIMLi
===========================================================

Ce parseur gère les fichiers .dat ERT avec :
- Multi-fréquences (MHz) dans l'en-tête
- Multi-fichiers complémentaires
- Colonnes : project, survey_point, depth, frequency_MHz, resistivity
- Visualisations 3D professionnelles avec PyGIMLi et Matplotlib

Auteur: Kibali AI
Date: 2025-11-07
"""

import matplotlib
matplotlib.use('Agg')  # Backend non-interactif

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import Rectangle
from typing import List, Dict, Tuple, Optional
import os
import re
from datetime import datetime

# PyGIMLi pour visualisations professionnelles
import pygimli as pg
import pygimli.meshtools as mt
from pygimli.viewer.mpl import drawModel
from scipy.interpolate import griddata


class MultiFreqERTParser:
    """
    Parseur ERT avancé pour fichiers multi-fréquences complémentaires
    """
    
    def __init__(self):
        self.data = None
        self.frequencies = []
        self.projects = []
        self.survey_points = []
        self.depths = []
        self.metadata = {}
        
    def detect_frequencies(self, content: str) -> List[float]:
        """
        Détecte les fréquences dans l'en-tête du fichier
        
        Args:
            content: Contenu du fichier texte
            
        Returns:
            Liste des fréquences en MHz
        """
        lines = content.split('\n')
        
        # Chercher ligne contenant MHz dans les 10 premières lignes
        for line in lines[:10]:
            if 'MHz' in line or 'mhz' in line.lower():
                # Extraire toutes les fréquences
                freqs = []
                # Séparer par virgules et autres délimiteurs
                parts = re.split(r'[,\s\t]+', line)
                for part in parts:
                    if 'MHz' in part or 'mhz' in part.lower():
                        # Nettoyer les caractères spéciaux
                        clean_part = re.sub(r'[^\d\.]', '', part.replace('MHz', '').replace('mhz', ''))
                        if clean_part:
                            try:
                                freq = float(clean_part)
                                if freq > 0:  # Ignorer les valeurs invalides
                                    freqs.append(freq)
                            except ValueError:
                                continue
                
                if freqs:
                    return freqs
        
        # Si aucune fréquence trouvée, retourner [0] pour mono-fréquence
        return [0.0]
    
    def parse_file(self, file_path: str) -> pd.DataFrame:
        """
        Parse un seul fichier .dat
        
        Args:
            file_path: Chemin du fichier
            
        Returns:
            DataFrame avec colonnes: project, survey_point, depth, frequency_MHz, resistivity
        """
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            # Détecter fréquences
            frequencies = self.detect_frequencies(content)
            
            # Parser les données
            lines = content.split('\n')
            data_rows = []
            
            # Détecter le format du fichier
            # Format 1: project,survey_point,res1,res2,res3... (sans depth)
            # Format 2: survey-point,depth,data,project (avec depth)
            
            has_depth_column = False
            header_line = None
            
            # Chercher ligne d'en-tête
            for line in lines[:5]:
                if 'depth' in line.lower() or 'profondeur' in line.lower():
                    has_depth_column = True
                    header_line = line
                    break
            
            for line in lines:
                # Ignorer lignes vides et en-têtes
                if not line.strip():
                    continue
                if 'MHz' in line or 'mhz' in line.lower():
                    continue
                if 'survey-point' in line.lower() or 'depth' in line.lower():
                    continue
                # Ignorer lignes avec caractères binaires ou invalides
                if any(ord(c) < 32 and c not in '\t\n\r' for c in line[:50]):
                    continue
                
                # Parser ligne de données
                # Essayer différents délimiteurs
                parts = None
                for delimiter in [',', '\t', ';', ' ']:
                    test_parts = line.strip().split(delimiter)
                    test_parts = [p.strip() for p in test_parts if p.strip()]
                    if len(test_parts) >= 3:
                        parts = test_parts
                        break
                
                if not parts or len(parts) < 3:
                    continue
                
                try:
                    if has_depth_column:
                        # Format 2: survey-point, depth, data, project
                        survey_point = int(parts[0])
                        depth = float(parts[1])
                        resistivity = float(parts[2])
                        project = parts[3] if len(parts) > 3 else "Unknown"
                        freq = frequencies[0] if frequencies else 0.0
                        
                        data_rows.append({
                            'project': project,
                            'survey_point': survey_point,
                            'depth': depth,
                            'frequency_MHz': freq,
                            'resistivity': resistivity
                        })
                    else:
                        # Format 1: project, survey_point, res1, res2, res3...
                        project = parts[0]
                        survey_point = int(parts[1])
                        resistivities = [float(r) for r in parts[2:] if r]
                        
                        # IMPORTANT: Depth adaptatif (peut être positif ou négatif)
                        # Si le fichier ne contient pas de colonne depth explicite, utiliser une depth par défaut
                        # La depth sera utilisée telle quelle (positive ou négative selon les données)
                        depth = 2.0  # Profondeur par défaut de 2m (sera adaptée dans get_coordinates_corrected)
                        
                        # Jumeler chaque résistivité avec sa fréquence
                        for i, resistivity in enumerate(resistivities):
                            freq = frequencies[i] if i < len(frequencies) else 0.0
                            data_rows.append({
                                'project': project,
                                'survey_point': survey_point,
                                'depth': depth,
                                'frequency_MHz': freq,
                                'resistivity': resistivity
                            })
                
                except (ValueError, IndexError) as e:
                    # Ignorer lignes mal formées
                    continue
            
            # Créer DataFrame et le stocker dans self.data
            df_result = pd.DataFrame(data_rows)
            
            if not df_result.empty:
                self.data = df_result
                
                # Extraire métadonnées
                self.frequencies = sorted(self.data['frequency_MHz'].unique().tolist())
                self.projects = sorted(self.data['project'].unique().tolist())
                self.survey_points = sorted(self.data['survey_point'].unique().tolist())
                self.depths = sorted(self.data['depth'].unique().tolist())
                
                self.metadata = {
                    'num_files': 1,
                    'num_frequencies': len(self.frequencies),
                    'num_projects': len(self.projects),
                    'num_survey_points': len(self.survey_points),
                    'num_depths': len(self.depths),
                    'total_measurements': len(self.data),
                    'depth_range': (self.data['depth'].min(), self.data['depth'].max()),
                    'resistivity_range': (self.data['resistivity'].min(), self.data['resistivity'].max()),
                    'frequencies': self.frequencies,
                    'files': [os.path.basename(file_path)]
                }
            
            return df_result
        
        except Exception as e:
            print(f"❌ Erreur parsing {file_path}: {e}")
            return pd.DataFrame()
    
    def parse_multiple_files(self, file_paths: List[str]) -> pd.DataFrame:
        """
        Parse et fusionne plusieurs fichiers .dat complémentaires
        
        Args:
            file_paths: Liste des chemins de fichiers
            
        Returns:
            DataFrame fusionné
        """
        all_dfs = []
        
        for file_path in file_paths:
            df = self.parse_file(file_path)
            if not df.empty:
                df['source_file'] = os.path.basename(file_path)
                all_dfs.append(df)
        
        if not all_dfs:
            return pd.DataFrame()
        
        # Fusionner tous les DataFrames
        self.data = pd.concat(all_dfs, ignore_index=True)
        
        # Extraire métadonnées
        self.frequencies = sorted(self.data['frequency_MHz'].unique())
        self.projects = sorted(self.data['project'].unique())
        self.survey_points = sorted(self.data['survey_point'].unique())
        self.depths = sorted(self.data['depth'].unique())
        
        self.metadata = {
            'num_files': len(file_paths),
            'num_frequencies': len(self.frequencies),
            'num_projects': len(self.projects),
            'num_survey_points': len(self.survey_points),
            'num_depths': len(self.depths),
            'total_measurements': len(self.data),
            'depth_range': (self.data['depth'].min(), self.data['depth'].max()),
            'resistivity_range': (self.data['resistivity'].min(), self.data['resistivity'].max()),
            'frequencies': self.frequencies,
            'files': [os.path.basename(f) for f in file_paths]
        }
        
        return self.data
    
    def get_coordinates_corrected(self) -> pd.DataFrame:
        """
        Génère un DataFrame avec coordonnées correctes pour chaque point
        
        Returns:
            DataFrame avec colonnes: survey_point, depth, x, y, z, resistivity, frequency_MHz
        """
        if self.data is None:
            return pd.DataFrame()
        
        # Créer coordonnées spatiales
        coords_data = []
        
        for _, row in self.data.iterrows():
            # Coordonnées spatiales ERT - CONVENTION CORRECTE:
            # X = survey_point (position le long du profil: 1, 2, 3, ...)
            x = float(row['survey_point'])
            
            # Y = 0 (profil 2D, pas de coordonnée perpendiculaire)
            y = 0.0
            
            # Z = depth (GARDER LA VALEUR ORIGINALE - négative pour sous-sol)
            # La profondeur est déjà correcte dans les données (-2, -4, -100, etc.)
            z = float(row['depth'])
            
            coords_data.append({
                'survey_point': row['survey_point'],
                'depth': row['depth'],
                'x': x,
                'y': y,
                'z': z,
                'resistivity': row['resistivity'],
                'frequency_MHz': row['frequency_MHz'],
                'project': row['project'],
                'source_file': row.get('source_file', 'unknown')
            })
        
        return pd.DataFrame(coords_data)
    
    def classify_water_type(self, resistivity: float) -> dict:
        """
        Classifie le type d'eau selon la résistivité avec couleur associée
        
        Args:
            resistivity: Résistivité en Ω·m
            
        Returns:
            Dict avec type_eau, couleur, couleur_hex
        """
        if resistivity < 1.0:
            return {
                'type_eau': 'Eau de mer',
                'couleur': 'Rouge vif / Orange',
                'couleur_hex': '#FF4500',
                'rgb': (255, 69, 0)
            }
        elif resistivity < 10.0:
            return {
                'type_eau': 'Eau salée (nappe)',
                'couleur': 'Jaune / Orange',
                'couleur_hex': '#FFA500',
                'rgb': (255, 165, 0)
            }
        elif resistivity < 100.0:
            return {
                'type_eau': 'Eau douce',
                'couleur': 'Vert / Bleu clair',
                'couleur_hex': '#00CED1',
                'rgb': (0, 206, 209)
            }
        else:
            return {
                'type_eau': 'Eau très pure',
                'couleur': 'Bleu foncé',
                'couleur_hex': '#00008B',
                'rgb': (0, 0, 139)
            }
    
    def add_water_classification(self, df: pd.DataFrame = None) -> pd.DataFrame:
        """
        Ajoute les colonnes de classification d'eau au DataFrame
        
        Args:
            df: DataFrame à enrichir (utilise self.data si None)
            
        Returns:
            DataFrame enrichi avec colonnes type_eau, couleur, couleur_hex
        """
        if df is None:
            df = self.data.copy() if self.data is not None else pd.DataFrame()
        else:
            df = df.copy()
        
        if df.empty:
            return df
        
        # Appliquer classification pour chaque ligne
        classifications = df['resistivity'].apply(self.classify_water_type)
        
        df['type_eau'] = classifications.apply(lambda x: x['type_eau'])
        df['couleur'] = classifications.apply(lambda x: x['couleur'])
        df['couleur_hex'] = classifications.apply(lambda x: x['couleur_hex'])
        
        return df
    
    def create_pygimli_mesh(self, x_coords, z_coords):
        """
        Crée un maillage 2D PyGIMLi pour visualisations
        
        Args:
            x_coords: Coordonnées X (survey points)
            z_coords: Coordonnées Z (profondeurs)
            
        Returns:
            Maillage PyGIMLi
        """
        x_min, x_max = min(x_coords), max(x_coords)
        z_min, z_max = min(z_coords), max(z_coords)
        
        # Créer domaine
        world = mt.createWorld(
            start=[x_min - 0.5, z_min - 5],
            end=[x_max + 0.5, z_max + 5],
            worldMarker=True
        )
        
        # Créer maillage
        mesh = mt.createMesh(world, quality=34.0, area=2.0)
        
        return mesh
    
    def create_2d_section_by_frequency(self, frequency: float = None, output_path: str = None) -> Tuple[plt.Figure, Dict]:
        """
        Crée une coupe 2D PyGIMLi pour une fréquence spécifique
        
        Args:
            frequency: Fréquence en MHz (None = toutes)
            output_path: Chemin pour sauvegarder (optionnel)
            
        Returns:
            (Figure matplotlib, dict d'infos)
        """
        if self.data is None:
            return None, {}
        
        coords_df = self.get_coordinates_corrected()
        
        # Filtrer par fréquence si spécifiée
        if frequency is not None:
            coords_df = coords_df[coords_df['frequency_MHz'] == frequency]
        
        # Extraire coordonnées
        x = coords_df['x'].values.astype(float)
        z = coords_df['z'].values.astype(float)
        resistivity = coords_df['resistivity'].values
        
        # Créer figure
        fig, ax = plt.subplots(figsize=(14, 8), facecolor='white')
        
        # Déterminer échelle
        res_min, res_max = resistivity.min(), resistivity.max()
        if res_max / res_min > 10 and res_min > 0:
            norm = mcolors.LogNorm(vmin=res_min, vmax=res_max)
        else:
            norm = mcolors.Normalize(vmin=res_min, vmax=res_max)
        
        # Scatter plot
        scatter = ax.scatter(
            x, z,
            c=resistivity,
            s=100,
            cmap='Spectral_r',
            norm=norm,
            edgecolors='black',
            linewidths=0.5,
            alpha=0.9
        )
        
        # Colorbar
        cbar = plt.colorbar(scatter, ax=ax, pad=0.02)
        cbar.set_label('Résistivité (Ω·m)', fontsize=12, weight='bold')
        
        # Axes
        ax.set_xlabel('Survey Point', fontsize=12, weight='bold')
        ax.set_ylabel('Profondeur (m)', fontsize=12, weight='bold')
        
        freq_label = f"{frequency} MHz" if frequency else "Toutes fréquences"
        ax.set_title(f'Coupe ERT 2D - {freq_label}\n(PyGIMLi/Matplotlib)', 
                     fontsize=14, weight='bold', pad=20)
        
        # Inverser axe Y
        ax.invert_yaxis()
        
        # Grille
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_axisbelow(True)
        
        # Annotations
        project = self.data['project'].iloc[0] if 'project' in self.data.columns else "ERT"
        ax.text(0.02, 0.98, f"Projet: {project}\nPoints: {len(coords_df)}",
                transform=ax.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                fontsize=9)
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
        
        info = {
            'num_points': len(coords_df),
            'res_min': res_min,
            'res_max': res_max,
            'frequency': frequency
        }
        
        return fig, info
    
    def create_2d_contour_section(self, frequency: float = None, output_path: str = None) -> Tuple[plt.Figure, Dict]:
        """
        Crée une coupe 2D avec contours remplis et interpolation
        
        Args:
            frequency: Fréquence en MHz (None = toutes)
            output_path: Chemin pour sauvegarder
            
        Returns:
            (Figure matplotlib, dict d'infos)
        """
        if self.data is None:
            return None, {}
        
        coords_df = self.get_coordinates_corrected()
        
        if frequency is not None:
            coords_df = coords_df[coords_df['frequency_MHz'] == frequency]
        
        x = coords_df['x'].values.astype(float)
        z = coords_df['z'].values.astype(float)
        resistivity = coords_df['resistivity'].values
        
        fig, ax = plt.subplots(figsize=(14, 8), facecolor='white')
        
        # Créer grille d'interpolation
        x_min, x_max = x.min(), x.max()
        z_min, z_max = z.min(), z.max()
        
        xi = np.linspace(x_min - 0.5, x_max + 0.5, 200)
        zi = np.linspace(z_min - 5, z_max + 5, 200)
        Xi, Zi = np.meshgrid(xi, zi)
        
        # Interpolation robuste
        try:
            Ri = griddata((x, z), resistivity, (Xi, Zi), method='cubic')
        except:
            try:
                Ri = griddata((x, z), resistivity, (Xi, Zi), method='linear')
            except:
                Ri = griddata((x, z), resistivity, (Xi, Zi), method='nearest')
        
        # Contours remplis
        res_min, res_max = resistivity.min(), resistivity.max()
        levels = np.linspace(res_min, res_max, 20)
        
        contourf = ax.contourf(Xi, Zi, Ri, levels=levels, cmap='RdYlBu_r', extend='both')
        contour_lines = ax.contour(Xi, Zi, Ri, levels=10, colors='black', linewidths=0.5, alpha=0.4)
        ax.clabel(contour_lines, inline=True, fontsize=8, fmt='%.3f')
        
        # Points de mesure
        ax.scatter(x, z, c='white', s=30, edgecolors='black', linewidths=1, zorder=10, alpha=0.8)
        
        # Colorbar
        cbar = plt.colorbar(contourf, ax=ax, pad=0.02)
        cbar.set_label('Résistivité (Ω·m)', fontsize=12, weight='bold')
        
        # Axes
        ax.set_xlabel('Survey Point', fontsize=12, weight='bold')
        ax.set_ylabel('Profondeur (m)', fontsize=12, weight='bold')
        
        freq_label = f"{frequency} MHz" if frequency else "Toutes fréquences"
        ax.set_title(f'Coupe ERT 2D avec Contours - {freq_label}\n(Interpolation PyGIMLi/Matplotlib)', 
                     fontsize=14, weight='bold', pad=20)
        
        ax.invert_yaxis()
        
        # Annotations
        project = self.data['project'].iloc[0] if 'project' in self.data.columns else "ERT"
        ax.text(0.02, 0.98, f"Projet: {project}\nInterpolation: Cubique\nPoints: {len(coords_df)}",
                transform=ax.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.9),
                fontsize=9)
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
        
        info = {
            'num_points': len(coords_df),
            'res_min': res_min,
            'res_max': res_max,
            'frequency': frequency
        }
        
        return fig, info
    
    def create_3d_volume(self, output_path: str = None) -> Tuple[plt.Figure, Dict]:
        """
        Crée un volume 3D matplotlib avec toutes les fréquences
        
        Args:
            output_path: Chemin pour sauvegarder
            
        Returns:
            (Figure matplotlib 3D, dict d'infos)
        """
        if self.data is None:
            return None, {}
        
        coords_df = self.get_coordinates_corrected()
        
        x = coords_df['x'].values
        y = coords_df['y'].values
        z = coords_df['z'].values
        resistivity = coords_df['resistivity'].values
        
        # Créer figure 3D
        fig = plt.figure(figsize=(16, 12), facecolor='white')
        ax = fig.add_subplot(111, projection='3d')
        
        # Déterminer échelle couleurs
        res_min, res_max = resistivity.min(), resistivity.max()
        if res_max / res_min > 10 and res_min > 0:
            norm = mcolors.LogNorm(vmin=res_min, vmax=res_max)
        else:
            norm = mcolors.Normalize(vmin=res_min, vmax=res_max)
        
        # Scatter 3D
        scatter = ax.scatter(
            x, y, z,
            c=resistivity,
            s=80,
            cmap='Spectral_r',
            norm=norm,
            edgecolors='black',
            linewidths=0.5,
            alpha=0.8
        )
        
        # Colorbar
        cbar = plt.colorbar(scatter, ax=ax, pad=0.1, shrink=0.8)
        cbar.set_label('Résistivité (Ω·m)', fontsize=12, weight='bold')
        
        # Axes
        ax.set_xlabel('Survey Point (X)', fontsize=11, weight='bold')
        ax.set_ylabel('Y (m)', fontsize=11, weight='bold')
        ax.set_zlabel('Profondeur (Z en m)', fontsize=11, weight='bold')
        
        ax.set_title('Volume 3D ERT Multi-Fréquences\n(PyGIMLi/Matplotlib 3D)', 
                     fontsize=14, weight='bold', pad=20)
        
        # Inverser axe Z
        ax.invert_zaxis()
        
        # Vue optimale
        ax.view_init(elev=20, azim=45)
        
        # Grille
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
        
        info = {
            'num_points': len(coords_df),
            'res_min': res_min,
            'res_max': res_max,
            'x_range': (x.min(), x.max()),
            'y_range': (y.min(), y.max()),
            'z_range': (z.min(), z.max())
        }
        
        return fig, info
    
    def create_pygimli_mesh_section(self, output_path: str = None) -> Tuple[plt.Figure, Dict]:
        """
        Crée une coupe avec maillage PyGIMLi triangulaire
        
        Args:
            output_path: Chemin pour sauvegarder
            
        Returns:
            (Figure matplotlib, dict d'infos)
        """
        if self.data is None:
            return None, {}
        
        coords_df = self.get_coordinates_corrected()
        
        x = coords_df['x'].values.astype(float)
        z = coords_df['z'].values.astype(float)
        resistivity = coords_df['resistivity'].values
        
        # Créer maillage
        mesh = self.create_pygimli_mesh(x, z)
        
        # Interpoler sur maillage
        cell_centers = []
        for cell in mesh.cells():
            cell_centers.append(cell.center())
        cell_centers = np.array([[c.x(), c.y()] for c in cell_centers])
        
        cell_x = cell_centers[:, 0]
        cell_z = cell_centers[:, 1]
        cell_resistivity = griddata((x, z), resistivity, (cell_x, cell_z), 
                                   method='linear', fill_value=resistivity.mean())
        
        # Double subplot
        fig, axes = plt.subplots(2, 1, figsize=(14, 12), facecolor='white')
        
        # SUBPLOT 1: Colormap sans maillage
        ax1 = axes[0]
        pg.show(mesh, data=cell_resistivity, ax=ax1, cMap='Spectral_r', 
                colorBar=True, label='Résistivité (Ω·m)', showMesh=False,
                xlabel='Survey Point', ylabel='Profondeur (m)')
        ax1.scatter(x, z, c='black', s=40, marker='v', edgecolors='white', 
                   linewidths=1, zorder=10, label='Mesures')
        ax1.set_title('Modèle ERT PyGIMLi - Vue Colormap\n(Interpolation sur maillage triangulaire)',
                     fontsize=13, weight='bold', pad=15)
        ax1.legend(loc='upper right')
        ax1.invert_yaxis()
        
        # SUBPLOT 2: Avec maillage visible
        ax2 = axes[1]
        pg.show(mesh, data=cell_resistivity, ax=ax2, cMap='RdYlBu_r',
                colorBar=True, label='Résistivité (Ω·m)', showMesh=True,
                xlabel='Survey Point', ylabel='Profondeur (m)')
        ax2.scatter(x, z, c='red', s=50, marker='o', edgecolors='black',
                   linewidths=1.5, zorder=10, label='Points de mesure')
        ax2.set_title('Modèle ERT avec Maillage Visible\n(Affichage des éléments triangulaires)',
                     fontsize=13, weight='bold', pad=15)
        ax2.legend(loc='upper right')
        ax2.invert_yaxis()
        
        # Annotations
        project = self.data['project'].iloc[0] if 'project' in self.data.columns else "ERT"
        fig.text(0.02, 0.98, 
                f"Projet: {project}\nMaillage: {mesh.cellCount()} cellules\n"
                f"Mesures: {len(coords_df)} points\nOutil: PyGIMLi v{pg.__version__}",
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.3),
                fontsize=10, weight='bold')
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
        
        info = {
            'num_points': len(coords_df),
            'mesh_cells': mesh.cellCount(),
            'res_min': resistivity.min(),
            'res_max': resistivity.max()
        }
        
        return fig, info
    
    def generate_all_pygimli_sections(self, output_dir: str = 'ert_sections', prefix: str = 'coupe') -> Dict[str, str]:
        """
        Génère toutes les coupes PyGIMLi (3 formats + 3D)
        
        Args:
            output_dir: Répertoire de sortie
            prefix: Préfixe des fichiers
            
        Returns:
            Dict avec chemins des fichiers générés
        """
        os.makedirs(output_dir, exist_ok=True)
        
        print("\n" + "="*70)
        print("🚀 GÉNÉRATION COUPES ERT PYGIMLI - 4 FORMATS")
        print("="*70)
        
        outputs = {}
        
        # Format 1: Pseudo-section
        path1 = os.path.join(output_dir, f'{prefix}_format1_pseudo_section.png')
        fig1, info1 = self.create_2d_section_by_frequency(output_path=path1)
        if fig1:
            plt.close(fig1)
            outputs['pseudo_section'] = path1
            print(f"✅ Format 1 (Pseudo-section): {path1}")
        
        # Format 2: Contours
        path2 = os.path.join(output_dir, f'{prefix}_format2_contours.png')
        fig2, info2 = self.create_2d_contour_section(output_path=path2)
        if fig2:
            plt.close(fig2)
            outputs['contours'] = path2
            print(f"✅ Format 2 (Contours): {path2}")
        
        # Format 3: Maillage PyGIMLi
        path3 = os.path.join(output_dir, f'{prefix}_format3_pygimli_mesh.png')
        fig3, info3 = self.create_pygimli_mesh_section(output_path=path3)
        if fig3:
            plt.close(fig3)
            outputs['mesh'] = path3
            print(f"✅ Format 3 (Maillage): {path3}")
        
        # Format 4: Volume 3D
        path4 = os.path.join(output_dir, f'{prefix}_format4_3d_volume.png')
        fig4, info4 = self.create_3d_volume(output_path=path4)
        if fig4:
            plt.close(fig4)
            outputs['3d_volume'] = path4
            print(f"✅ Format 4 (Volume 3D): {path4}")
        
        print("="*70)
        print(f"✅ {len(outputs)} fichiers générés dans: {output_dir}/")
        print("="*70)
        
        return outputs
    
    def create_frequency_comparison(self) -> Tuple[plt.Figure, Dict]:
        """
        Compare les résistivités pour différentes fréquences
        
        Returns:
            Figure Plotly
        """
        if self.data is None:
            return None
        
        fig = go.Figure()
        
        for freq in self.frequencies:
            freq_data = self.data[self.data['frequency_MHz'] == freq]
            
            # Calculer profil moyen par profondeur
            depth_profile = freq_data.groupby('depth')['resistivity'].mean().reset_index()
            
            line = go.Scatter(
                x=depth_profile['resistivity'],
                y=-depth_profile['depth'],
                mode='lines+markers',
                name=f"{freq} MHz",
                line=dict(width=2),
                marker=dict(size=6)
            )
            
            fig.add_trace(line)
        
        fig.update_layout(
            title="Comparaison Multi-Fréquences",
            xaxis_title="Résistivité moyenne (Ω·m)",
            yaxis_title="Profondeur (m)",
            height=600,
            template='plotly_white',
            hovermode='x unified'
        )
        
        return fig
    
    def generate_statistics_report(self) -> str:
        """
        Génère un rapport statistique complet
        
        Returns:
            Rapport texte formaté
        """
        if self.data is None:
            return "❌ Aucune donnée chargée"
        
        report = f"""
╔══════════════════════════════════════════════════════════════╗
║           RAPPORT STATISTIQUE ERT MULTI-FRÉQUENCES           ║
╚══════════════════════════════════════════════════════════════╝

📁 FICHIERS ANALYSÉS
{'─' * 60}
Nombre de fichiers : {self.metadata['num_files']}
Fichiers : {', '.join(self.metadata['files'])}

📊 STRUCTURE DES DONNÉES
{'─' * 60}
Nombre total de mesures : {self.metadata['total_measurements']:,}
Nombre de fréquences : {self.metadata['num_frequencies']}
Fréquences (MHz) : {', '.join([f'{f:.1f}' for f in self.frequencies])}
Nombre de projets : {self.metadata['num_projects']}
Projets : {', '.join(self.projects)}
Nombre de survey points : {self.metadata['num_survey_points']}
Nombre de profondeurs : {self.metadata['num_depths']}

📏 GAMMES DE MESURES
{'─' * 60}
Profondeur min : {self.metadata['depth_range'][0]:.2f} m
Profondeur max : {self.metadata['depth_range'][1]:.2f} m
Résistivité min : {self.metadata['resistivity_range'][0]:.2f} Ω·m
Résistivité max : {self.metadata['resistivity_range'][1]:.2f} Ω·m

📈 STATISTIQUES PAR FRÉQUENCE
{'─' * 60}
"""
        
        for freq in self.frequencies:
            freq_data = self.data[self.data['frequency_MHz'] == freq]
            report += f"""
{freq} MHz:
  Mesures : {len(freq_data):,}
  Résistivité moyenne : {freq_data['resistivity'].mean():.2f} Ω·m
  Écart-type : {freq_data['resistivity'].std():.2f} Ω·m
  Médiane : {freq_data['resistivity'].median():.2f} Ω·m
"""
        
        report += f"""
{'─' * 60}
✅ Rapport généré le {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
        
        return report
    
    def export_to_excel(self, output_path: str) -> str:
        """
        Export les données vers Excel avec plusieurs feuilles
        
        Args:
            output_path: Chemin du fichier de sortie
            
        Returns:
            Message de statut
        """
        if self.data is None:
            return "❌ Aucune donnée à exporter"
        
        try:
            with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
                # Feuille principale
                self.data.to_excel(writer, sheet_name='Données Complètes', index=False)
                
                # Feuille coordonnées
                coords_df = self.get_coordinates_corrected()
                coords_df.to_excel(writer, sheet_name='Coordonnées', index=False)
                
                # Feuille statistiques par fréquence
                stats_data = []
                for freq in self.frequencies:
                    freq_data = self.data[self.data['frequency_MHz'] == freq]
                    stats_data.append({
                        'Fréquence (MHz)': freq,
                        'Nombre mesures': len(freq_data),
                        'Résistivité moyenne': freq_data['resistivity'].mean(),
                        'Écart-type': freq_data['resistivity'].std(),
                        'Min': freq_data['resistivity'].min(),
                        'Max': freq_data['resistivity'].max(),
                        'Médiane': freq_data['resistivity'].median()
                    })
                
                stats_df = pd.DataFrame(stats_data)
                stats_df.to_excel(writer, sheet_name='Statistiques', index=False)
            
            return f"✅ Export Excel réussi : {output_path}"
        
        except Exception as e:
            return f"❌ Erreur export Excel : {e}"


# Instance globale
multi_freq_parser = MultiFreqERTParser()
