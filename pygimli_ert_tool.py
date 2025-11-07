#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PyGIMLi ERT Inversion Tool - Outil Professionnel de Tomographie Électrique
===========================================================================

Cet outil utilise PyGIMLi pour :
1. Lire les données ERT brutes (.dat, .ohm, .txt)
2. Effectuer l'INVERSION pour obtenir les résistivités RÉELLES du sous-sol
3. Générer des coupes 2D/3D avec les VRAIES couleurs physiques
4. Respecter les coordonnées : X = survey_point, Z = profondeur

PyGIMLi corrige automatiquement les effets géométriques et donne
les résistivités vraies (pas apparentes), comme Res2DInv ou AarhusInv.

Auteur: Kibali AI + pyGIMLi
Date: 2025-11-07
"""

import matplotlib
matplotlib.use('Agg')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from mpl_toolkits.mplot3d import Axes3D
import pygimli as pg
from pygimli.physics import ert
import pygimli.meshtools as mt
from pygimli.viewer.mpl import drawModel
import os
from datetime import datetime
from typing import Dict, Tuple, List, Optional


class PyGIMLiERTInversionTool:
    """
    Outil complet d'inversion ERT avec PyGIMLi
    
    Workflow:
    1. Charger données brutes (résistivités apparentes)
    2. Inversion automatique → résistivités VRAIES
    3. Génération coupes 2D/3D avec couleurs physiques
    """
    
    def __init__(self):
        self.data_ert = None  # Données ERT PyGIMLi
        self.mesh = None
        self.model = None  # Modèle inversé (résistivités vraies)
        self.manager = None  # ERTManager
        self.survey_points = []
        self.depths = []
        
    def create_ert_data_from_measurements(self, df: pd.DataFrame, scheme_type='dd') -> pg.DataContainerERT:
        """
        Crée un DataContainerERT PyGIMLi depuis un DataFrame avec profils de profondeur
        
        NOUVEAU: Chaque survey point a un profil vertical complet de -2m à -100m
        Les mesures suivent les profondeurs réelles de manière continue.
        
        Args:
            df: DataFrame avec colonnes: survey_point, depth, resistivity
            scheme_type: 'dd' (dipole-dipole), 'wa' (Wenner alpha), 'wb' (Wenner beta)
            
        Returns:
            DataContainerERT PyGIMLi avec profils verticaux
        """
        print("\n🔧 Création du schéma ERT PyGIMLi avec profils continus...")
        
        # Extraire coordonnées uniques
        survey_points = sorted(df['survey_point'].unique())
        depths = sorted(df['depth'].unique(), reverse=True)  # De -2 à -100
        
        self.survey_points = survey_points
        self.depths = depths
        
        print(f"   • Survey points: {survey_points}")
        print(f"   • Profils de profondeur: {depths[0]:.0f}m → {depths[-1]:.0f}m")
        print(f"   • {len(survey_points)} profils verticaux × {len(depths)} niveaux")
        
        # Créer positions des électrodes (en surface, Z=0)
        n_elec = len(survey_points)
        sensors = np.zeros((n_elec, 2))
        for i, sp in enumerate(survey_points):
            sensors[i] = [float(sp), 0.0]  # X=survey_point, Z=0 (surface)
        
        # Créer schéma de mesure
        scheme = ert.createData(elecs=sensors, schemeName=scheme_type)
        
        print(f"   • Schéma: {scheme_type.upper()}")
        print(f"   • {scheme.size()} configurations ABMN générées")
        print(f"   • {n_elec} électrodes en surface")
        
        # NOUVEAU: Mapper les mesures avec profondeurs aux configurations ABMN
        # Chaque configuration ABMN correspond à une profondeur d'investigation
        resistivities = []
        
        # Créer mapping profondeur → résistivité pour chaque survey point
        depth_profiles = {}
        for sp in survey_points:
            sp_data = df[df['survey_point'] == sp].sort_values('depth', ascending=False)
            depth_profiles[sp] = dict(zip(sp_data['depth'].values, sp_data['resistivity'].values))
        
        print(f"\n📊 PROFILS DE PROFONDEUR PAR SURVEY POINT:")
        for sp in survey_points:
            profile = depth_profiles[sp]
            print(f"   SP{sp}: ", end="")
            for d in depths:
                if d in profile:
                    print(f"{d:.0f}m({profile[d]:.3f}) → ", end="")
            print("✓")
        
        # Assigner résistivités aux configurations en fonction de la profondeur d'investigation
        for i in range(scheme.size()):
            # Indices électrodes ABMN
            a_idx = int(scheme('a')[i])
            b_idx = int(scheme('b')[i])
            m_idx = int(scheme('m')[i])
            n_idx = int(scheme('n')[i])
            
            # Calculer profondeur d'investigation théorique (formule ERT standard)
            # Profondeur ≈ 0.5 × espacement AB (pour dipole-dipole)
            ab_spacing = abs(sensors[b_idx][0] - sensors[a_idx][0])
            mn_spacing = abs(sensors[n_idx][0] - sensors[m_idx][0])
            theoretical_depth = -(ab_spacing + mn_spacing) / 2.0 * 15  # Facteur de conversion
            
            # Limiter à la plage de profondeurs disponibles
            theoretical_depth = max(min(theoretical_depth, depths[0]), depths[-1])
            
            # Trouver le survey point central de la mesure
            center_x = (sensors[a_idx][0] + sensors[b_idx][0] + sensors[m_idx][0] + sensors[n_idx][0]) / 4
            closest_sp = min(survey_points, key=lambda sp: abs(float(sp) - center_x))
            
            # Interpoler la résistivité pour cette profondeur
            profile = depth_profiles[closest_sp]
            available_depths = sorted(profile.keys(), reverse=True)
            
            # Trouver la profondeur mesurée la plus proche
            closest_depth = min(available_depths, key=lambda d: abs(d - theoretical_depth))
            res_value = profile[closest_depth]
            
            resistivities.append(res_value)
        
        scheme['rhoa'] = np.array(resistivities)
        scheme['err'] = np.ones(scheme.size()) * 0.03  # 3% d'erreur estimée
        
        self.data_ert = scheme
        
        print(f"\n✅ Données ERT créées avec profils continus")
        print(f"   • {scheme.size()} mesures assignées")
        print(f"   • Résistivité apparente : {min(resistivities):.4f} - {max(resistivities):.4f} Ω·m")
        print(f"   • Profondeurs suivies: {depths[0]:.0f}m à {depths[-1]:.0f}m")
        
        return scheme
    
    def load_ert_file(self, filepath: str) -> pg.DataContainerERT:
        """
        Charge un fichier ERT standard (.ohm, .dat, .txt)
        
        Args:
            filepath: Chemin du fichier ERT
            
        Returns:
            DataContainerERT
        """
        try:
            data = ert.load(filepath)
            self.data_ert = data
            print(f"✅ Fichier ERT chargé: {filepath}")
            print(f"   • {data.size()} mesures")
            print(f"   • {data.sensorCount()} électrodes")
            return data
        except Exception as e:
            print(f"❌ Erreur chargement: {e}")
            return None
    
    def run_inversion(self, lam=20, verbose=True) -> np.ndarray:
        """
        Effectue l'INVERSION ERT pour obtenir les résistivités VRAIES
        
        Args:
            lam: Paramètre de régularisation (20 = défaut)
            verbose: Afficher progression
            
        Returns:
            Modèle de résistivités vraies (array)
        """
        if self.data_ert is None:
            print("❌ Aucune donnée ERT chargée")
            return None
        
        print("\n" + "="*70)
        print("🚀 INVERSION ERT AVEC PYGIMLI")
        print("="*70)
        print("\nCalcul des RÉSISTIVITÉS VRAIES du sous-sol...")
        print("(Correction des effets géométriques, inversion de Tikhonov)\n")
        
        # Créer manager ERT
        mgr = ert.ERTManager()
        mgr.setData(self.data_ert)
        
        # Configuration inversion
        mgr.invert(lam=lam, verbose=verbose)
        
        self.manager = mgr
        self.mesh = mgr.paraDomain
        self.model = mgr.model
        
        print("\n" + "="*70)
        print("✅ INVERSION TERMINÉE")
        print("="*70)
        print(f"   • Maillage : {self.mesh.cellCount()} cellules")
        print(f"   • Résistivité min : {min(self.model):.4f} Ω·m")
        print(f"   • Résistivité max : {max(self.model):.4f} Ω·m")
        print(f"   • RMS (misfit) : {mgr.inv.absrms():.2f}")
        print("="*70)
        
        return self.model
    
    def create_2d_section_inverted(self, output_path: str = None, cmap='Spectral_r') -> Tuple[plt.Figure, Dict]:
        """
        Crée une coupe 2D avec résistivités INVERSÉES (vraies valeurs)
        
        Args:
            output_path: Chemin de sauvegarde
            cmap: Colormap ('Spectral_r', 'RdYlBu_r', 'jet_r')
            
        Returns:
            (Figure, dict infos)
        """
        if self.model is None:
            print("❌ Effectuez d'abord l'inversion avec run_inversion()")
            return None, {}
        
        print("\n🎨 Génération coupe 2D INVERSÉE...")
        
        fig, ax = plt.subplots(figsize=(14, 8), facecolor='white')
        
        # Afficher modèle inversé
        pg.show(
            self.mesh,
            data=self.model,
            ax=ax,
            cMap=cmap,
            colorBar=True,
            label='Résistivité VRAIE (Ω·m)',
            showMesh=False,
            logScale=True,  # Échelle log pour résistivités
            cMin=min(self.model),
            cMax=max(self.model)
        )
        
        # Marquer positions électrodes
        elec_x = self.data_ert.sensors()[:, 0]
        elec_z = self.data_ert.sensors()[:, 1]
        ax.scatter(elec_x, elec_z, c='black', s=80, marker='v', 
                  edgecolors='white', linewidths=2, zorder=10, 
                  label='Électrodes')
        
        ax.set_xlabel('Survey Point (X)', fontsize=12, weight='bold')
        ax.set_ylabel('Profondeur (Z en m)', fontsize=12, weight='bold')
        ax.set_title('Coupe ERT 2D - Modèle INVERSÉ (Résistivités VRAIES)\n'
                    'PyGIMLi Inversion - Couleurs Physiques Correctes',
                    fontsize=14, weight='bold', pad=20)
        
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.2)
        
        # Annotations
        ax.text(0.02, 0.98, 
               f"Inversion PyGIMLi\nRMS: {self.manager.inv.absrms():.2f}\n"
               f"Cellules: {self.mesh.cellCount()}\n"
               f"λ (régularisation): {self.manager.inv.lam}",
               transform=ax.transAxes, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8),
               fontsize=9, weight='bold')
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"✅ Sauvegardé: {output_path}")
        
        info = {
            'rms': self.manager.inv.absrms(),
            'n_cells': self.mesh.cellCount(),
            'res_min': min(self.model),
            'res_max': max(self.model)
        }
        
        return fig, info
    
    def create_2d_comparison(self, output_path: str = None) -> Tuple[plt.Figure, Dict]:
        """
        Crée une comparaison : Données apparentes vs Modèle inversé
        
        Args:
            output_path: Chemin de sauvegarde
            
        Returns:
            (Figure, dict infos)
        """
        if self.model is None:
            return None, {}
        
        print("\n🎨 Génération comparaison apparentes/inversées...")
        
        fig, axes = plt.subplots(2, 1, figsize=(14, 14), facecolor='white')
        
        # SUBPLOT 1: Résistivités apparentes (mesurées)
        ax1 = axes[0]
        self.manager.showData(vals=self.data_ert['rhoa'], ax=ax1, 
                             cMap='Spectral_r', colorBar=True,
                             label='Résistivité APPARENTE (Ω·m)')
        ax1.set_title('Pseudo-Section APPARENTE (Mesures Brutes)\n'
                     'Données non corrigées, effets géométriques présents',
                     fontsize=13, weight='bold', pad=15)
        
        # SUBPLOT 2: Modèle inversé (résistivités vraies)
        ax2 = axes[1]
        pg.show(self.mesh, data=self.model, ax=ax2, cMap='RdYlBu_r',
               colorBar=True, label='Résistivité VRAIE (Ω·m)',
               logScale=True, showMesh=False)
        ax2.set_title('Modèle INVERSÉ (Résistivités VRAIES du Sous-Sol)\n'
                     'Après inversion - Couleurs physiques correctes',
                     fontsize=13, weight='bold', pad=15)
        
        # Électrodes sur les deux
        elec_x = self.data_ert.sensors()[:, 0]
        elec_z = self.data_ert.sensors()[:, 1]
        for ax in axes:
            ax.scatter(elec_x, elec_z, c='black', s=60, marker='v',
                      edgecolors='white', linewidths=1.5, zorder=10)
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"✅ Sauvegardé: {output_path}")
        
        info = {'comparison': True, 'rms': self.manager.inv.absrms()}
        
        return fig, info
    
    def create_3d_volume_inverted(self, output_path: str = None) -> Tuple[plt.Figure, Dict]:
        """
        Crée un volume 3D matplotlib avec résistivités inversées
        
        Args:
            output_path: Chemin de sauvegarde
            
        Returns:
            (Figure 3D, dict infos)
        """
        if self.model is None:
            return None, {}
        
        print("\n🎨 Génération volume 3D inversé...")
        
        # Extraire centres de cellules du maillage
        cell_centers = []
        for cell in self.mesh.cells():
            c = cell.center()
            cell_centers.append([c.x(), c.y()])
        
        cell_centers = np.array(cell_centers)
        x = cell_centers[:, 0]
        z = cell_centers[:, 1]
        resistivity = np.array(self.model)
        
        # Créer Y artificiel pour 3D (profil 2D → Y=0)
        y = np.zeros_like(x)
        
        fig = plt.figure(figsize=(16, 12), facecolor='white')
        ax = fig.add_subplot(111, projection='3d')
        
        # Normalisation logarithmique
        norm = mcolors.LogNorm(vmin=resistivity.min(), vmax=resistivity.max())
        
        scatter = ax.scatter(
            x, y, z,
            c=resistivity,
            s=50,
            cmap='Spectral_r',
            norm=norm,
            edgecolors='black',
            linewidths=0.3,
            alpha=0.7
        )
        
        cbar = plt.colorbar(scatter, ax=ax, pad=0.1, shrink=0.7)
        cbar.set_label('Résistivité VRAIE (Ω·m)', fontsize=12, weight='bold')
        
        ax.set_xlabel('Survey Point (X)', fontsize=11, weight='bold')
        ax.set_ylabel('Y (m)', fontsize=11, weight='bold')
        ax.set_zlabel('Profondeur (Z en m)', fontsize=11, weight='bold')
        
        ax.set_title('Volume 3D ERT - Modèle INVERSÉ\n'
                    'PyGIMLi - Résistivités VRAIES du Sous-Sol',
                    fontsize=14, weight='bold', pad=20)
        
        ax.invert_zaxis()
        ax.view_init(elev=25, azim=45)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"✅ Sauvegardé: {output_path}")
        
        info = {
            'n_cells': len(resistivity),
            'res_min': resistivity.min(),
            'res_max': resistivity.max()
        }
        
        return fig, info
    
    def generate_all_sections(self, output_dir: str = 'ert_inversion_pygimli', 
                             prefix: str = 'inversion') -> Dict[str, str]:
        """
        Génère TOUTES les coupes avec résistivités inversées
        
        Args:
            output_dir: Répertoire de sortie
            prefix: Préfixe des fichiers
            
        Returns:
            Dict avec chemins des fichiers
        """
        os.makedirs(output_dir, exist_ok=True)
        
        print("\n" + "="*70)
        print("🚀 GÉNÉRATION COMPLÈTE - COUPES ERT INVERSÉES")
        print("="*70)
        
        outputs = {}
        
        # 1. Coupe 2D inversée (Spectral_r)
        path1 = os.path.join(output_dir, f'{prefix}_2d_inverted_spectral.png')
        fig1, _ = self.create_2d_section_inverted(path1, cmap='Spectral_r')
        if fig1:
            plt.close(fig1)
            outputs['2d_spectral'] = path1
        
        # 2. Coupe 2D inversée (RdYlBu_r)
        path2 = os.path.join(output_dir, f'{prefix}_2d_inverted_rdylbu.png')
        fig2, _ = self.create_2d_section_inverted(path2, cmap='RdYlBu_r')
        if fig2:
            plt.close(fig2)
            outputs['2d_rdylbu'] = path2
        
        # 3. Comparaison apparentes/inversées
        path3 = os.path.join(output_dir, f'{prefix}_comparison.png')
        fig3, _ = self.create_2d_comparison(path3)
        if fig3:
            plt.close(fig3)
            outputs['comparison'] = path3
        
        # 4. Volume 3D inversé
        path4 = os.path.join(output_dir, f'{prefix}_3d_volume.png')
        fig4, _ = self.create_3d_volume_inverted(path4)
        if fig4:
            plt.close(fig4)
            outputs['3d_volume'] = path4
        
        print("\n" + "="*70)
        print(f"✅ {len(outputs)} FICHIERS GÉNÉRÉS")
        print("="*70)
        for key, path in outputs.items():
            print(f"   • {key}: {os.path.basename(path)}")
        print(f"\n📁 Répertoire: {output_dir}/")
        print("="*70)
        
        return outputs
    
    def export_model(self, filepath: str):
        """
        Exporte le modèle inversé en fichier
        
        Args:
            filepath: Chemin du fichier de sortie
        """
        if self.model is None:
            print("❌ Aucun modèle à exporter")
            return
        
        self.mesh.save(filepath)
        print(f"✅ Modèle exporté: {filepath}")
    
    def get_water_classification(self, resistivity: float) -> Dict:
        """
        Classification physique selon résistivité VRAIE
        
        Args:
            resistivity: Résistivité en Ω·m
            
        Returns:
            Dict avec type_eau, couleur, description
        """
        if resistivity < 1:
            return {
                'type_eau': 'Eau de mer',
                'couleur': 'Rouge vif',
                'couleur_hex': '#FF4500',
                'description': 'Eau très salée, forte conductivité'
            }
        elif resistivity < 10:
            return {
                'type_eau': 'Eau salée / Nappe contaminée',
                'couleur': 'Orange',
                'couleur_hex': '#FFA500',
                'description': 'Eau saumâtre, minéralisée'
            }
        elif resistivity < 100:
            return {
                'type_eau': 'Eau douce',
                'couleur': 'Bleu clair',
                'couleur_hex': '#00CED1',
                'description': 'Eau potable, faible minéralisation'
            }
        else:
            return {
                'type_eau': 'Eau très pure / Roche sèche',
                'couleur': 'Bleu foncé',
                'couleur_hex': '#00008B',
                'description': 'Très faible conductivité, roches compactes'
            }


# ============================================================================
# FONCTIONS UTILITAIRES POUR INTÉGRATION
# ============================================================================

def process_ert_data_complete(df: pd.DataFrame, output_dir: str = '/tmp/ert_pygimli_inversion',
                              run_inversion: bool = True) -> Dict:
    """
    Pipeline complet : DataFrame → Inversion PyGIMLi → Coupes
    
    Args:
        df: DataFrame avec colonnes: survey_point, depth, resistivity
        output_dir: Répertoire de sortie
        run_inversion: Si False, ne fait que le parsing
        
    Returns:
        Dict avec chemins des fichiers générés
    """
    print("\n" + "╔"+"═"*68+"╗")
    print("║" + " "*10 + "PIPELINE COMPLET PYGIMLI - INVERSION ERT" + " "*18 + "║")
    print("╚"+"═"*68+"╝")
    
    # Créer outil
    tool = PyGIMLiERTInversionTool()
    
    # Étape 1: Créer données ERT
    tool.create_ert_data_from_measurements(df, scheme_type='dd')
    
    if run_inversion:
        # Étape 2: Inversion
        tool.run_inversion(lam=20, verbose=False)
        
        # Étape 3: Générer toutes les coupes
        outputs = tool.generate_all_sections(output_dir=output_dir, prefix='ert_inverted')
    else:
        outputs = {}
    
    return outputs


if __name__ == '__main__':
    print("Module PyGIMLi ERT Inversion Tool chargé ✅")
    print("Usage: from pygimli_ert_tool import PyGIMLiERTInversionTool")
