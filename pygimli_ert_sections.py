#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module de génération de coupes ERT avec PyGIMLi
Génère des visualisations professionnelles en plusieurs formats
"""

import matplotlib
matplotlib.use('Agg')  # Backend non-interactif pour éviter problèmes Tk

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Rectangle
import pygimli as pg
import pygimli.meshtools as mt
from pygimli.viewer.mpl import drawModel, drawMesh
import os
from datetime import datetime


class PyGIMLiERTSections:
    """
    Générateur de coupes ERT professionnelles avec PyGIMLi
    Supporte 3 formats de sortie différents
    """
    
    def __init__(self):
        self.data = None
        self.mesh = None
        self.sensor_positions = None
        
    def load_data_from_parser(self, parser_data: pd.DataFrame):
        """
        Charge les données depuis MultiFreqERTParser
        
        Args:
            parser_data: DataFrame avec colonnes: survey_point, depth, resistivity
        """
        self.data = parser_data.copy()
        
        # Extraire positions des capteurs (survey points)
        survey_points = sorted(self.data['survey_point'].unique())
        depths = sorted(self.data['depth'].unique())
        
        print(f"📊 Données chargées:")
        print(f"   • Survey points: {survey_points}")
        print(f"   • Profondeurs: {min(depths):.1f}m à {max(depths):.1f}m")
        print(f"   • {len(self.data)} mesures")
        
    def create_mesh(self, x_coords, z_coords):
        """
        Crée un maillage 2D pour PyGIMLi
        
        Args:
            x_coords: Coordonnées X (survey points)
            z_coords: Coordonnées Z (profondeurs)
            
        Returns:
            Maillage PyGIMLi
        """
        # Définir les bornes du maillage
        x_min, x_max = min(x_coords), max(x_coords)
        z_min, z_max = min(z_coords), max(z_coords)
        
        # Créer une grille de points
        world = mt.createWorld(
            start=[x_min - 0.5, z_min - 5],
            end=[x_max + 0.5, z_max + 5],
            worldMarker=True
        )
        
        # Créer le maillage avec raffinement
        mesh = mt.createMesh(world, quality=34.0, area=2.0)
        
        print(f"🔷 Maillage créé: {mesh.cellCount()} cellules")
        
        return mesh
    
    def format1_pseudo_section(self, output_path='ert_pseudo_section.png', dpi=300):
        """
        FORMAT 1: Pseudo-section classique (style géophysique traditionnel)
        Affiche les données en fonction de la position et de la profondeur d'investigation
        
        Args:
            output_path: Chemin de sortie
            dpi: Résolution
        """
        if self.data is None:
            print("❌ Aucune donnée chargée")
            return None
        
        print("\n🎨 FORMAT 1: Génération pseudo-section classique...")
        
        fig, ax = plt.subplots(figsize=(14, 8), facecolor='white')
        
        # Préparer les données
        x = self.data['survey_point'].values
        z = self.data['depth'].values
        resistivity = self.data['resistivity'].values
        
        # Créer scatter plot avec colorbar logarithmique
        res_min, res_max = resistivity.min(), resistivity.max()
        
        # Utiliser échelle log si grande plage
        if res_max / res_min > 10:
            norm = mcolors.LogNorm(vmin=res_min, vmax=res_max)
        else:
            norm = mcolors.Normalize(vmin=res_min, vmax=res_max)
        
        scatter = ax.scatter(
            x, z, 
            c=resistivity, 
            s=100, 
            cmap='Spectral_r',  # Classique en géophysique
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
        ax.set_title('Pseudo-Section ERT - Résistivité Apparente\n(Format Classique Géophysique)', 
                     fontsize=14, weight='bold', pad=20)
        
        # Inverser axe Y (profondeur vers le bas)
        ax.invert_yaxis()
        
        # Grille
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_axisbelow(True)
        
        # Annotations
        project = self.data['project'].iloc[0] if 'project' in self.data.columns else "ERT Survey"
        ax.text(0.02, 0.98, f"Projet: {project}\nDate: {datetime.now().strftime('%Y-%m-%d')}",
                transform=ax.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                fontsize=9)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
        plt.close()
        
        print(f"✅ FORMAT 1 généré: {output_path}")
        return output_path
    
    def format2_filled_contour(self, output_path='ert_filled_contour.png', dpi=300):
        """
        FORMAT 2: Coupe avec contours remplis (style moderne)
        Interpolation avec contours colorés et lignes de niveau
        
        Args:
            output_path: Chemin de sortie
            dpi: Résolution
        """
        if self.data is None:
            print("❌ Aucune donnée chargée")
            return None
        
        print("\n🎨 FORMAT 2: Génération coupe avec contours remplis...")
        
        from scipy.interpolate import griddata
        
        fig, ax = plt.subplots(figsize=(14, 8), facecolor='white')
        
        # Préparer les données
        x = self.data['survey_point'].values
        z = self.data['depth'].values
        resistivity = self.data['resistivity'].values
        
        # Créer grille d'interpolation
        x_min, x_max = x.min(), x.max()
        z_min, z_max = z.min(), z.max()
        
        xi = np.linspace(x_min - 0.5, x_max + 0.5, 200)
        zi = np.linspace(z_min - 5, z_max + 5, 200)
        Xi, Zi = np.meshgrid(xi, zi)
        
        # Interpolation (gérer le cas où les points sont coplanaires)
        try:
            Ri = griddata((x, z), resistivity, (Xi, Zi), method='cubic')
        except Exception as e:
            print(f"⚠️  Interpolation cubique impossible (points coplanaires), utilisation de 'linear'")
            try:
                Ri = griddata((x, z), resistivity, (Xi, Zi), method='linear')
            except:
                print(f"⚠️  Interpolation linéaire impossible, utilisation de 'nearest'")
                Ri = griddata((x, z), resistivity, (Xi, Zi), method='nearest')
        
        # Contours remplis
        res_min, res_max = resistivity.min(), resistivity.max()
        levels = np.linspace(res_min, res_max, 20)
        
        contourf = ax.contourf(
            Xi, Zi, Ri,
            levels=levels,
            cmap='RdYlBu_r',
            extend='both'
        )
        
        # Lignes de contour
        contour_lines = ax.contour(
            Xi, Zi, Ri,
            levels=10,
            colors='black',
            linewidths=0.5,
            alpha=0.4
        )
        ax.clabel(contour_lines, inline=True, fontsize=8, fmt='%.3f')
        
        # Points de mesure
        ax.scatter(x, z, c='white', s=30, edgecolors='black', 
                   linewidths=1, zorder=10, alpha=0.8)
        
        # Colorbar
        cbar = plt.colorbar(contourf, ax=ax, pad=0.02)
        cbar.set_label('Résistivité (Ω·m)', fontsize=12, weight='bold')
        
        # Axes
        ax.set_xlabel('Survey Point', fontsize=12, weight='bold')
        ax.set_ylabel('Profondeur (m)', fontsize=12, weight='bold')
        ax.set_title('Coupe ERT 2D - Interpolation par Contours\n(Format Moderne avec Gradients)', 
                     fontsize=14, weight='bold', pad=20)
        
        # Inverser axe Y
        ax.invert_yaxis()
        
        # Annotations
        project = self.data['project'].iloc[0] if 'project' in self.data.columns else "ERT Survey"
        ax.text(0.02, 0.98, 
                f"Projet: {project}\nMéthode: Interpolation cubique\nNombre de mesures: {len(self.data)}",
                transform=ax.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.9),
                fontsize=9)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
        plt.close()
        
        print(f"✅ FORMAT 2 généré: {output_path}")
        return output_path
    
    def format3_pygimli_mesh(self, output_path='ert_pygimli_mesh.png', dpi=300):
        """
        FORMAT 3: Visualisation avec maillage PyGIMLi (style professionnel)
        Affiche le modèle avec maillage triangulaire et résistivités interpolées
        
        Args:
            output_path: Chemin de sortie
            dpi: Résolution
        """
        if self.data is None:
            print("❌ Aucune donnée chargée")
            return None
        
        print("\n🎨 FORMAT 3: Génération coupe avec maillage PyGIMLi...")
        
        fig, axes = plt.subplots(2, 1, figsize=(14, 12), facecolor='white')
        
        # Préparer les données
        x = self.data['survey_point'].values.astype(float)
        z = self.data['depth'].values.astype(float)
        resistivity = self.data['resistivity'].values
        
        # Créer maillage
        self.mesh = self.create_mesh(x, z)
        
        # Interpoler les résistivités sur le maillage
        from scipy.interpolate import griddata
        
        # Centres des cellules (utiliser l'API correcte de PyGIMLi)
        cell_centers = []
        for cell in self.mesh.cells():
            cell_centers.append(cell.center())
        cell_centers = np.array([[c.x(), c.y()] for c in cell_centers])
        
        cell_x = cell_centers[:, 0]
        cell_z = cell_centers[:, 1]
        
        # Interpoler
        cell_resistivity = griddata((x, z), resistivity, (cell_x, cell_z), method='linear', fill_value=resistivity.mean())
        
        # SUBPLOT 1: Modèle avec colormap
        ax1 = axes[0]
        
        pg.show(
            self.mesh, 
            data=cell_resistivity, 
            ax=ax1,
            cMap='Spectral_r',
            colorBar=True,
            label='Résistivité (Ω·m)',
            showMesh=False,
            xlabel='Survey Point',
            ylabel='Profondeur (m)'
        )
        
        # Ajouter points de mesure
        ax1.scatter(x, z, c='black', s=40, marker='v', 
                    edgecolors='white', linewidths=1, zorder=10, label='Mesures')
        
        ax1.set_title('Modèle ERT avec Maillage PyGIMLi - Vue Colormap\n(Interpolation sur maillage triangulaire)', 
                      fontsize=13, weight='bold', pad=15)
        ax1.legend(loc='upper right')
        ax1.invert_yaxis()
        
        # SUBPLOT 2: Maillage avec lignes
        ax2 = axes[1]
        
        pg.show(
            self.mesh,
            data=cell_resistivity,
            ax=ax2,
            cMap='RdYlBu_r',
            colorBar=True,
            label='Résistivité (Ω·m)',
            showMesh=True,
            xlabel='Survey Point',
            ylabel='Profondeur (m)'
        )
        
        # Ajouter points de mesure
        ax2.scatter(x, z, c='red', s=50, marker='o', 
                    edgecolors='black', linewidths=1.5, zorder=10, label='Points de mesure')
        
        ax2.set_title('Modèle ERT avec Maillage Visible\n(Affichage des éléments triangulaires)', 
                      fontsize=13, weight='bold', pad=15)
        ax2.legend(loc='upper right')
        ax2.invert_yaxis()
        
        # Annotations générales
        project = self.data['project'].iloc[0] if 'project' in self.data.columns else "ERT Survey"
        fig.text(0.02, 0.98, 
                 f"Projet: {project}\nMaillage: {self.mesh.cellCount()} cellules\n"
                 f"Mesures: {len(self.data)} points\nOutil: PyGIMLi v{pg.__version__}",
                 verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.3),
                 fontsize=10, weight='bold')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
        plt.close()
        
        print(f"✅ FORMAT 3 généré: {output_path}")
        return output_path
    
    def generate_all_formats(self, output_dir='ert_sections', prefix='coupe'):
        """
        Génère les 3 formats de coupes ERT
        
        Args:
            output_dir: Répertoire de sortie
            prefix: Préfixe des fichiers
            
        Returns:
            Liste des chemins générés
        """
        # Créer répertoire
        os.makedirs(output_dir, exist_ok=True)
        
        print("\n" + "="*70)
        print("🚀 GÉNÉRATION DES COUPES ERT - 3 FORMATS PROFESSIONNELS")
        print("="*70)
        
        outputs = []
        
        # Format 1
        path1 = os.path.join(output_dir, f'{prefix}_format1_pseudo_section.png')
        self.format1_pseudo_section(path1)
        outputs.append(path1)
        
        # Format 2
        path2 = os.path.join(output_dir, f'{prefix}_format2_filled_contour.png')
        self.format2_filled_contour(path2)
        outputs.append(path2)
        
        # Format 3
        path3 = os.path.join(output_dir, f'{prefix}_format3_pygimli_mesh.png')
        self.format3_pygimli_mesh(path3)
        outputs.append(path3)
        
        print("\n" + "="*70)
        print("✅ GÉNÉRATION TERMINÉE!")
        print("="*70)
        print(f"\n📁 Fichiers générés dans: {output_dir}/")
        for i, path in enumerate(outputs, 1):
            print(f"   {i}. {os.path.basename(path)}")
        
        return outputs


def main():
    """Test avec données exemple"""
    from multi_freq_ert_parser import MultiFreqERTParser
    
    print("╔════════════════════════════════════════════════════════════╗")
    print("║     TEST PYGIMLI - GÉNÉRATION COUPES ERT PROFESSIONNELLES ║")
    print("╚════════════════════════════════════════════════════════════╝\n")
    
    # Parser les données
    parser = MultiFreqERTParser()
    df = parser.parse_file('freq.dat')
    
    # Créer générateur PyGIMLi
    gimli_gen = PyGIMLiERTSections()
    gimli_gen.load_data_from_parser(df)
    
    # Générer tous les formats
    outputs = gimli_gen.generate_all_formats(
        output_dir='/tmp/ert_pygimli_sections',
        prefix='archange_ondimba'
    )
    
    print(f"\n🎉 {len(outputs)} coupes générées avec succès!")


if __name__ == '__main__':
    main()
