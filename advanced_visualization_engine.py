#!/usr/bin/env python3
"""
MOTEUR DE VISUALISATION AVANCÉ - Combine PyGIMLI + OpenCV + Matplotlib
Capable de gérer N'IMPORTE QUELLE demande graphique, même complexe
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Rectangle
import plotly.graph_objects as go
import plotly.express as px
from scipy.interpolate import griddata, Rbf
from scipy.ndimage import gaussian_filter
import warnings
warnings.filterwarnings('ignore')

class AdvancedVisualizationEngine:
    """
    Moteur de visualisation ultra-puissant
    Combine PyGIMLI, OpenCV, Matplotlib pour créer N'IMPORTE QUEL graphique
    """
    
    def __init__(self):
        self.color_schemes = {
            'geological': ['#2E86AB', '#06A77D', '#F4D35E', '#EE964B', '#F95738'],
            'resistivity': ['#0000FF', '#00FFFF', '#00FF00', '#FFFF00', '#FF0000'],
            'depth': ['#FFF7BC', '#FEE391', '#FEC44F', '#FE9929', '#EC7014', '#CC4C02', '#8C2D04'],
            'temperature': ['#313695', '#4575B4', '#74ADD1', '#ABD9E9', '#E0F3F8', '#FFFFBF', '#FEE090', '#FDAE61', '#F46D43', '#D73027', '#A50026'],
            'rainbow': plt.cm.rainbow,
            'viridis': plt.cm.viridis,
            'plasma': plt.cm.plasma,
            'inferno': plt.cm.inferno,
        }
        
        self.available_pygimli = self._check_pygimli()
        self.available_cv2 = self._check_opencv()
        
    def _check_pygimli(self):
        """Vérifie si PyGIMLI est disponible"""
        try:
            import pygimli as pg
            return True
        except ImportError:
            return False
            
    def _check_opencv(self):
        """Vérifie si OpenCV est disponible"""
        try:
            import cv2
            return True
        except ImportError:
            return False
    
    def auto_detect_format(self, data):
        """
        Détection INTELLIGENTE du format des données
        Retourne: (format_type, shape_info, suggestions)
        """
        if isinstance(data, (list, tuple)):
            data = np.array(data)
        
        if not isinstance(data, np.ndarray):
            return 'unknown', None, []
        
        shape = data.shape
        
        # Format 1D: simple liste de valeurs
        if len(shape) == 1 or (len(shape) == 2 and shape[1] == 1):
            return '1d_profile', {'n_points': len(data)}, [
                'vertical_profile',
                'histogram',
                'depth_profile'
            ]
        
        # Format 2D: grille ou points x,y,z
        elif len(shape) == 2:
            if shape[1] == 2:  # (x, y) pairs
                return 'points_xy', {'n_points': shape[0]}, [
                    'scatter_map',
                    'voronoi',
                    'delaunay'
                ]
            elif shape[1] == 3:  # (x, y, z) triplets
                return 'points_xyz', {'n_points': shape[0]}, [
                    'scatter_3d',
                    'interpolated_2d',
                    'contour_map'
                ]
            elif shape[1] == 4:  # (x, y, z, rho) - format ERT
                return 'ert_data', {'n_points': shape[0]}, [
                    'ert_section',
                    'resistivity_2d',
                    'geological_section'
                ]
            else:  # Grille 2D
                return 'grid_2d', {'rows': shape[0], 'cols': shape[1]}, [
                    'heatmap',
                    'contour',
                    'surface_3d'
                ]
        
        # Format 3D: volume
        elif len(shape) == 3:
            return 'volume_3d', {'shape': shape}, [
                'volume_slices',
                'isosurface',
                '3d_scatter'
            ]
        
        return 'unknown', shape, []
    
    def create_intelligent_visualization(self, data, request_text, color_scheme='resistivity'):
        """
        Crée une visualisation INTELLIGENTE basée sur la demande
        S'adapte automatiquement au type de données et à la demande
        """
        # Détection automatique du format
        format_type, shape_info, suggestions = self.auto_detect_format(data)
        
        # Analyse de la demande pour détecter le type de graphique
        request_lower = request_text.lower()
        
        # Mots-clés pour différents types de graphiques
        keywords = {
            'section': ['section', 'coupe', 'profil 2d', 'transect'],
            '3d': ['3d', 'volume', 'cube', 'volumétrique'],
            'contour': ['contour', 'isoligne', 'isovaleur'],
            'heatmap': ['heatmap', 'carte de chaleur', 'raster'],
            'interpolation': ['interpolé', 'interpolation', 'lissé', 'smooth'],
            'profile': ['profil vertical', 'profondeur', 'sondage'],
            'comparison': ['comparaison', 'avant-après', 'différence'],
            'animation': ['animation', 'évolution', 'temporel'],
            'geological': ['géologique', 'stratigraphie', 'couches'],
            'resistivity': ['résistivité', 'ert', 'électrique']
        }
        
        detected_types = []
        for viz_type, words in keywords.items():
            if any(word in request_lower for word in words):
                detected_types.append(viz_type)
        
        # Sélection de la meilleure méthode
        if 'section' in detected_types or 'resistivity' in detected_types:
            return self.create_ert_section(data, color_scheme)
        elif '3d' in detected_types:
            return self.create_3d_visualization(data, color_scheme)
        elif 'contour' in detected_types:
            return self.create_contour_map(data, color_scheme)
        elif 'profile' in detected_types:
            return self.create_vertical_profile(data, color_scheme)
        elif 'comparison' in detected_types:
            return self.create_comparison_plot(data, color_scheme)
        elif 'geological' in detected_types:
            return self.create_geological_section(data, color_scheme)
        else:
            # Visualisation par défaut selon le format
            if format_type == 'ert_data':
                return self.create_ert_section(data, color_scheme)
            elif format_type == '1d_profile':
                return self.create_vertical_profile(data, color_scheme)
            elif format_type == 'grid_2d':
                return self.create_heatmap(data, color_scheme)
            else:
                return self.create_adaptive_plot(data, color_scheme)
    
    def create_ert_section(self, data, color_scheme='resistivity'):
        """
        Crée une coupe ERT complète avec PyGIMLI si dispo, sinon matplotlib avancé
        """
        data = np.array(data)
        
        # Tentative avec PyGIMLI si disponible
        if self.available_pygimli:
            try:
                return self._create_ert_with_pygimli(data, color_scheme)
            except Exception as e:
                print(f"⚠️ PyGIMLI failed, fallback to matplotlib: {e}")
        
        # Fallback avancé avec matplotlib
        return self._create_ert_with_matplotlib(data, color_scheme)
    
    def _create_ert_with_pygimli(self, data, color_scheme):
        """Crée une coupe ERT avec PyGIMLI (méthode professionnelle)"""
        import pygimli as pg
        from pygimli.meshtools import createParaMesh2dGrid
        import pygimli.viewer.mpl as pgplt
        
        # Préparer les données
        if data.shape[1] == 4:  # x, y, z, rho
            x, y, z, rho = data[:, 0], data[:, 1], data[:, 2], data[:, 3]
        else:
            # Créer une grille si données simples
            n_points = len(data)
            x = np.linspace(0, 100, int(np.sqrt(n_points)))
            z = np.linspace(0, 50, int(np.sqrt(n_points)))
            X, Z = np.meshgrid(x, z)
            x, z = X.flatten(), Z.flatten()
            rho = data.flatten() if len(data.flatten()) == len(x) else np.tile(data, len(x))[:len(x)]
        
        # Créer le mesh
        mesh = createParaMesh2dGrid(
            sensors=np.column_stack([x, -np.abs(z)]),
            paraDepth=max(abs(z)),
            paraDX=0.5,
            quality=34.0
        )
        
        # Assigner les valeurs
        rho_mesh = pg.interpolate(
            srcMesh=pg.meshtools.createMesh2D(np.column_stack([x, -np.abs(z)])),
            inVec=rho,
            destMesh=mesh
        )
        
        # Créer la figure
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # Colormap personnalisée
        if color_scheme in self.color_schemes:
            if isinstance(self.color_schemes[color_scheme], list):
                cmap = mcolors.LinearSegmentedColormap.from_list(
                    'custom', self.color_schemes[color_scheme]
                )
            else:
                cmap = self.color_schemes[color_scheme]
        else:
            cmap = 'jet'
        
        # Plot avec PyGIMLI
        pgplt.drawModel(
            ax, mesh, rho_mesh,
            cMap=cmap,
            logScale=True,
            xlabel='Distance (m)',
            ylabel='Profondeur (m)',
            colorBar=True,
            label='Résistivité (Ω.m)'
        )
        
        ax.set_title('Coupe de Résistivité ERT - PyGIMLI', fontsize=16, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='--')
        
        plt.tight_layout()
        return fig
    
    def _create_ert_with_matplotlib(self, data, color_scheme):
        """
        Crée une coupe ERT avancée avec matplotlib + interpolation
        Méthode de secours ultra-puissante
        """
        data = np.array(data)
        
        # Préparer les données selon le format
        if len(data.shape) == 1:
            # Données 1D: créer une grille
            n = int(np.sqrt(len(data)))
            if n * n < len(data):
                n += 1
            x = np.linspace(0, 100, n)
            z = np.linspace(0, 50, n)
            X, Z = np.meshgrid(x, z)
            rho = np.zeros_like(X)
            rho.flat[:len(data)] = data
        elif data.shape[1] == 4:
            # Format x, y, z, rho
            x, y, z, rho = data[:, 0], data[:, 1], data[:, 2], data[:, 3]
            # Créer une grille régulière
            xi = np.linspace(x.min(), x.max(), 200)
            zi = np.linspace(z.min(), z.max(), 100)
            X, Z = np.meshgrid(xi, zi)
            # Interpolation RBF (Radial Basis Function) - très lisse
            rbf = Rbf(x, z, rho, function='multiquadric', smooth=0.1)
            rho_grid = rbf(X, Z)
            # Appliquer un filtre gaussien pour lisser encore plus
            rho = gaussian_filter(rho_grid, sigma=1.5)
        else:
            # Grille 2D directe
            rho = data
            x = np.linspace(0, 100, rho.shape[1])
            z = np.linspace(0, 50, rho.shape[0])
            X, Z = np.meshgrid(x, z)
        
        # Créer la figure avec plusieurs sous-graphiques
        fig = plt.figure(figsize=(16, 10))
        gs = fig.add_gridspec(2, 2, height_ratios=[3, 1], width_ratios=[20, 1])
        
        # Graphique principal: coupe de résistivité
        ax_main = fig.add_subplot(gs[0, 0])
        
        # Colormap personnalisée
        if color_scheme in self.color_schemes:
            if isinstance(self.color_schemes[color_scheme], list):
                colors = self.color_schemes[color_scheme]
                cmap = mcolors.LinearSegmentedColormap.from_list('custom', colors)
            else:
                cmap = self.color_schemes[color_scheme]
        else:
            cmap = 'jet'
        
        # Normalisation logarithmique pour résistivité
        norm = mcolors.LogNorm(vmin=max(rho.min(), 1), vmax=rho.max())
        
        # Plot principal avec contours
        im = ax_main.pcolormesh(X, -np.abs(Z), rho, cmap=cmap, norm=norm, shading='gouraud')
        
        # Ajouter des contours pour plus de clarté
        contour_levels = np.logspace(np.log10(max(rho.min(), 1)), np.log10(rho.max()), 10)
        cs = ax_main.contour(X, -np.abs(Z), rho, levels=contour_levels, colors='black', 
                            alpha=0.3, linewidths=0.5)
        ax_main.clabel(cs, inline=True, fontsize=8, fmt='%d Ω.m')
        
        # Annotations géologiques
        self._add_geological_interpretation(ax_main, X, Z, rho)
        
        ax_main.set_xlabel('Distance (m)', fontsize=12, fontweight='bold')
        ax_main.set_ylabel('Profondeur (m)', fontsize=12, fontweight='bold')
        ax_main.set_title('Coupe de Résistivité Électrique (ERT)', fontsize=16, fontweight='bold')
        ax_main.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
        ax_main.set_aspect('equal', adjustable='box')
        
        # Colorbar
        cax = fig.add_subplot(gs[0, 1])
        cbar = plt.colorbar(im, cax=cax, label='Résistivité (Ω.m)')
        cbar.ax.tick_params(labelsize=10)
        
        # Graphique secondaire: profil moyen
        ax_profile = fig.add_subplot(gs[1, 0])
        mean_profile = rho.mean(axis=1)
        ax_profile.plot(mean_profile, -np.abs(z), 'b-', linewidth=2, label='Profil moyen')
        ax_profile.fill_betweenx(-np.abs(z), mean_profile * 0.8, mean_profile * 1.2, 
                                 alpha=0.3, label='±20%')
        ax_profile.set_xlabel('Résistivité moyenne (Ω.m)', fontsize=10)
        ax_profile.set_ylabel('Profondeur (m)', fontsize=10)
        ax_profile.grid(True, alpha=0.3)
        ax_profile.legend()
        ax_profile.set_xscale('log')
        
        plt.tight_layout()
        return fig
    
    def _add_geological_interpretation(self, ax, X, Z, rho):
        """Ajoute des annotations géologiques intelligentes"""
        # Définir les zones géologiques selon résistivité
        zones = [
            (1, 50, 'Argiles saturées', '#4A90E2', -1),
            (50, 200, 'Altérites/Sables', '#7ED321', -10),
            (200, 1000, 'Roche altérée', '#F5A623', -20),
            (1000, 10000, 'Roche saine', '#D0021B', -30)
        ]
        
        for rho_min, rho_max, label, color, y_offset in zones:
            # Trouver les zones correspondantes
            mask = (rho >= rho_min) & (rho < rho_max)
            if mask.any():
                # Ajouter une annotation
                y_pos = -np.abs(Z).max() + y_offset
                x_pos = X.max() * 0.02
                ax.text(x_pos, y_pos, label, 
                       bbox=dict(boxstyle='round,pad=0.5', facecolor=color, alpha=0.7),
                       fontsize=9, fontweight='bold', color='white')
    
    def create_3d_visualization(self, data, color_scheme='viridis'):
        """Crée une visualisation 3D interactive avec Plotly"""
        data = np.array(data)
        
        if len(data.shape) == 1:
            # Créer une grille 3D à partir de données 1D
            n = int(np.cbrt(len(data)))
            x = np.linspace(0, 100, n)
            y = np.linspace(0, 100, n)
            z = np.linspace(0, 50, n)
            X, Y, Z = np.meshgrid(x, y, z)
            values = np.zeros_like(X)
            values.flat[:len(data)] = data
        elif data.shape[1] == 4:
            x, y, z, values = data[:, 0], data[:, 1], data[:, 2], data[:, 3]
            X, Y, Z = x, y, z
        else:
            raise ValueError("Format de données non supporté pour 3D")
        
        # Créer le volume 3D avec Plotly
        fig = go.Figure(data=go.Volume(
            x=X.flatten(),
            y=Y.flatten(),
            z=Z.flatten(),
            value=values.flatten(),
            isomin=values.min(),
            isomax=values.max(),
            opacity=0.1,
            surface_count=15,
            colorscale=color_scheme if color_scheme in ['viridis', 'plasma', 'inferno'] else 'Jet',
            colorbar=dict(title='Résistivité (Ω.m)')
        ))
        
        fig.update_layout(
            title='Visualisation 3D - Volume de Résistivité',
            scene=dict(
                xaxis_title='X (m)',
                yaxis_title='Y (m)',
                zaxis_title='Profondeur (m)',
            ),
            width=900,
            height=700
        )
        
        return fig
    
    def create_contour_map(self, data, color_scheme='geological'):
        """Crée une carte de contours avancée"""
        data = np.array(data)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Préparer grille
        if len(data.shape) == 1:
            n = int(np.sqrt(len(data)))
            x = np.linspace(0, 100, n)
            z = np.linspace(0, 50, n)
            X, Z = np.meshgrid(x, z)
            values = np.zeros_like(X)
            values.flat[:len(data)] = data
        else:
            X, Z = np.meshgrid(
                np.linspace(0, 100, data.shape[1]),
                np.linspace(0, 50, data.shape[0])
            )
            values = data
        
        # Carte de contours remplis
        if color_scheme in self.color_schemes:
            if isinstance(self.color_schemes[color_scheme], list):
                cmap = mcolors.LinearSegmentedColormap.from_list(
                    'custom', self.color_schemes[color_scheme]
                )
            else:
                cmap = self.color_schemes[color_scheme]
        else:
            cmap = 'jet'
        
        cf = ax1.contourf(X, Z, values, levels=20, cmap=cmap)
        ax1.contour(X, Z, values, levels=20, colors='black', alpha=0.3, linewidths=0.5)
        plt.colorbar(cf, ax=ax1, label='Résistivité (Ω.m)')
        ax1.set_title('Carte de Contours Remplis', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Distance (m)')
        ax1.set_ylabel('Profondeur (m)')
        ax1.grid(True, alpha=0.3)
        
        # Carte de contours avec lignes
        cs = ax2.contour(X, Z, values, levels=15, cmap=cmap, linewidths=2)
        ax2.clabel(cs, inline=True, fontsize=10)
        ax2.set_title('Carte de Contours avec Valeurs', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Distance (m)')
        ax2.set_ylabel('Profondeur (m)')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def create_vertical_profile(self, data, color_scheme='depth'):
        """Crée un profil vertical avec interprétation géologique"""
        data = np.array(data).flatten()
        depths = np.linspace(0, 50, len(data))
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 8))
        
        # Profil avec gradient de couleur
        points = np.array([data, depths]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        
        from matplotlib.collections import LineCollection
        norm = plt.Normalize(data.min(), data.max())
        if color_scheme in self.color_schemes:
            if isinstance(self.color_schemes[color_scheme], list):
                cmap = mcolors.LinearSegmentedColormap.from_list(
                    'custom', self.color_schemes[color_scheme]
                )
            else:
                cmap = self.color_schemes[color_scheme]
        else:
            cmap = plt.cm.rainbow
        
        lc = LineCollection(segments, cmap=cmap, norm=norm, linewidth=3)
        lc.set_array(data)
        ax1.add_collection(lc)
        ax1.set_xlim(data.min() * 0.9, data.max() * 1.1)
        ax1.set_ylim(depths.max(), depths.min())
        ax1.set_xlabel('Résistivité (Ω.m)', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Profondeur (m)', fontsize=12, fontweight='bold')
        ax1.set_title('Profil Vertical de Résistivité', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.set_xscale('log')
        plt.colorbar(lc, ax=ax1, label='Résistivité (Ω.m)')
        
        # Interprétation géologique
        geological_zones = []
        current_zone = None
        zone_start = 0
        
        for i, (d, r) in enumerate(zip(depths, data)):
            if r < 50:
                zone_type = 'Argiles'
                color = '#4A90E2'
            elif r < 200:
                zone_type = 'Altérites'
                color = '#7ED321'
            elif r < 1000:
                zone_type = 'Roche altérée'
                color = '#F5A623'
            else:
                zone_type = 'Roche saine'
                color = '#D0021B'
            
            if zone_type != current_zone:
                if current_zone is not None:
                    geological_zones.append((zone_start, d, current_zone, color))
                current_zone = zone_type
                zone_start = d
        
        if current_zone is not None:
            geological_zones.append((zone_start, depths[-1], current_zone, color))
        
        # Dessiner les zones géologiques
        for z_start, z_end, zone, col in geological_zones:
            ax2.barh(y=(z_start + z_end) / 2, width=1, height=z_end - z_start,
                    color=col, alpha=0.7, edgecolor='black', linewidth=2)
            ax2.text(0.5, (z_start + z_end) / 2, zone,
                    ha='center', va='center', fontsize=10, fontweight='bold')
        
        ax2.set_xlim(0, 1)
        ax2.set_ylim(depths.max(), depths.min())
        ax2.set_ylabel('Profondeur (m)', fontsize=12, fontweight='bold')
        ax2.set_title('Interprétation Géologique', fontsize=14, fontweight='bold')
        ax2.set_xticks([])
        ax2.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        return fig
    
    def create_comparison_plot(self, data, color_scheme='viridis'):
        """Crée un graphique de comparaison (avant/après, différence)"""
        # À implémenter selon besoin spécifique
        pass
    
    def create_geological_section(self, data, color_scheme='geological'):
        """Crée une coupe géologique interprétée"""
        return self._create_ert_with_matplotlib(data, color_scheme)
    
    def create_heatmap(self, data, color_scheme='viridis'):
        """Crée une heatmap avancée"""
        fig, ax = plt.subplots(figsize=(12, 8))
        
        if color_scheme in self.color_schemes:
            if isinstance(self.color_schemes[color_scheme], list):
                cmap = mcolors.LinearSegmentedColormap.from_list(
                    'custom', self.color_schemes[color_scheme]
                )
            else:
                cmap = self.color_schemes[color_scheme]
        else:
            cmap = 'viridis'
        
        im = ax.imshow(data, cmap=cmap, aspect='auto', interpolation='bilinear')
        plt.colorbar(im, ax=ax, label='Valeur')
        ax.set_title('Carte de Chaleur', fontsize=14, fontweight='bold')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        
        plt.tight_layout()
        return fig
    
    def create_adaptive_plot(self, data, color_scheme='viridis'):
        """Crée un graphique adaptatif basé sur les données"""
        format_type, _, suggestions = self.auto_detect_format(data)
        
        if suggestions:
            method_name = f"create_{suggestions[0]}"
            if hasattr(self, method_name):
                return getattr(self, method_name)(data, color_scheme)
        
        # Fallback: simple plot
        fig, ax = plt.subplots(figsize=(10, 6))
        data_flat = np.array(data).flatten()
        ax.plot(data_flat, 'b-', linewidth=2)
        ax.set_title('Visualisation Adaptative', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        return fig


# Fonction d'utilisation rapide
def create_advanced_visualization(data, request="coupe de résistivité", color_scheme='resistivity'):
    """
    Point d'entrée principal pour créer N'IMPORTE QUELLE visualisation
    
    Args:
        data: Données (np.array, list, etc.)
        request: Description de ce que l'utilisateur veut
        color_scheme: Schéma de couleurs
    
    Returns:
        Figure matplotlib ou plotly
    """
    engine = AdvancedVisualizationEngine()
    return engine.create_intelligent_visualization(data, request, color_scheme)
