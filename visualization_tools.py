"""
Outils de visualisation avancés pour Kibali
Génération automatique de graphiques, coupes, schémas et légendes
Utilisables pendant la conversation en temps réel
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import Rectangle, Circle, Polygon
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import io
import base64
from typing import Dict, List, Tuple, Optional, Any
import cv2
from PIL import Image, ImageDraw, ImageFont

class VisualizationEngine:
    """Moteur de visualisation pour générer des graphiques pendant la conversation"""
    
    def __init__(self):
        self.color_schemes = {
            'ert': ['#00008B', '#0000FF', '#00FFFF', '#00FF00', '#FFFF00', '#FF0000', '#8B0000'],
            'geological': ['#654321', '#8B4513', '#A0522D', '#CD853F', '#D2B48C', '#F5DEB3'],
            'depth': ['#000080', '#0000CD', '#4169E1', '#1E90FF', '#87CEEB', '#B0E0E6'],
            'scientific': ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
        }
    
    # ========================================
    # 1. GRAPHIQUES DE RÉSISTIVITÉ ERT
    # ========================================
    
    def create_resistivity_profile(self, values: List[float], depths: Optional[List[float]] = None, 
                                   title: str = "Profil de Résistivité", interactive: bool = True) -> str:
        """
        Crée un profil de résistivité vertical avec zones colorées
        Returns: HTML string (Plotly) ou path vers image (Matplotlib)
        """
        if depths is None:
            depths = list(range(len(values)))
        
        if interactive:
            # Version Plotly interactive
            fig = go.Figure()
            
            # Ligne principale
            fig.add_trace(go.Scatter(
                x=values,
                y=depths,
                mode='lines+markers',
                name='Résistivité',
                line=dict(color='royalblue', width=3),
                marker=dict(size=8, color=values, colorscale='Viridis', 
                          colorbar=dict(title="ρ (Ω·m)"), showscale=True)
            ))
            
            # Zones géologiques colorées
            zones = self._classify_resistivity_zones(values)
            for zone_name, zone_data in zones.items():
                if zone_data['indices']:
                    zone_values = [values[i] for i in zone_data['indices']]
                    zone_depths = [depths[i] for i in zone_data['indices']]
                    fig.add_trace(go.Scatter(
                        x=zone_values,
                        y=zone_depths,
                        mode='markers',
                        name=zone_name,
                        marker=dict(size=12, color=zone_data['color'], symbol='square')
                    ))
            
            fig.update_layout(
                title=dict(text=title, font=dict(size=20, family='Arial Black')),
                xaxis_title="Résistivité (Ω·m)",
                yaxis_title="Profondeur (m)",
                yaxis=dict(autorange='reversed'),
                hovermode='closest',
                template='plotly_white',
                height=600,
                showlegend=True
            )
            
            return fig.to_html(include_plotlyjs='cdn', div_id='resistivity_profile')
        
        else:
            # Version Matplotlib pour téléchargement
            fig, ax = plt.subplots(figsize=(10, 8))
            
            # Créer gradient de couleurs selon résistivité
            colors = [self._resistivity_to_color(v) for v in values]
            
            # Plot avec couleurs
            for i in range(len(values)-1):
                ax.plot([values[i], values[i+1]], [depths[i], depths[i+1]], 
                       color=colors[i], linewidth=3)
            
            ax.scatter(values, depths, c=colors, s=100, edgecolors='black', zorder=5)
            
            # Légende des zones
            legend_elements = self._create_geological_legend()
            ax.legend(handles=legend_elements, loc='best', fontsize=10)
            
            ax.set_xlabel('Résistivité (Ω·m)', fontsize=14, fontweight='bold')
            ax.set_ylabel('Profondeur (m)', fontsize=14, fontweight='bold')
            ax.set_title(title, fontsize=16, fontweight='bold')
            ax.invert_yaxis()
            ax.grid(True, alpha=0.3)
            
            # Sauvegarder
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=300, bbox_inches='tight')
            buf.seek(0)
            plt.close()
            
            return base64.b64encode(buf.read()).decode()
    
    def create_2d_resistivity_section(self, data_grid: np.ndarray, 
                                      x_coords: Optional[np.ndarray] = None,
                                      z_coords: Optional[np.ndarray] = None,
                                      title: str = "Coupe ERT 2D") -> str:
        """
        Crée une coupe 2D de résistivité avec colormap géologique
        """
        if x_coords is None:
            x_coords = np.arange(data_grid.shape[1])
        if z_coords is None:
            z_coords = np.arange(data_grid.shape[0])
        
        # Créer colormap personnalisée ERT
        cmap = LinearSegmentedColormap.from_list('ert', self.color_schemes['ert'])
        
        fig = go.Figure(data=go.Heatmap(
            z=data_grid,
            x=x_coords,
            y=z_coords,
            colorscale='RdYlBu_r',
            colorbar=dict(
                title="ρ (Ω·m)",
                titleside='right',
                tickmode='linear',
                tick0=data_grid.min(),
                dtick=(data_grid.max() - data_grid.min()) / 10
            )
        ))
        
        fig.update_layout(
            title=dict(text=title, font=dict(size=20)),
            xaxis_title="Distance (m)",
            yaxis_title="Profondeur (m)",
            yaxis=dict(autorange='reversed'),
            height=500,
            template='plotly_white'
        )
        
        return fig.to_html(include_plotlyjs='cdn', div_id='ert_section_2d')
    
    # ========================================
    # 2. SCHÉMAS GÉOLOGIQUES
    # ========================================
    
    def create_geological_column(self, layers: List[Dict[str, Any]], 
                                 width: int = 800, height: int = 1000) -> str:
        """
        Crée une colonne stratigraphique avec légendes
        layers = [
            {'name': 'Sol', 'depth_start': 0, 'depth_end': 2, 'resistivity': 50, 'color': '#8B4513'},
            ...
        ]
        """
        # Créer image avec PIL
        img = Image.new('RGB', (width, height), 'white')
        draw = ImageDraw.Draw(img)
        
        # Police
        try:
            font_title = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 24)
            font_label = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
        except:
            font_title = ImageFont.load_default()
            font_label = ImageFont.load_default()
        
        # Titre
        draw.text((width//2 - 150, 20), "COLONNE STRATIGRAPHIQUE", fill='black', font=font_title)
        
        # Calculer échelle
        max_depth = max([l['depth_end'] for l in layers])
        scale_factor = (height - 150) / max_depth
        
        margin_left = 50
        col_width = 200
        
        for layer in layers:
            y_start = 80 + layer['depth_start'] * scale_factor
            y_end = 80 + layer['depth_end'] * scale_factor
            
            # Dessiner couche
            draw.rectangle(
                [(margin_left, y_start), (margin_left + col_width, y_end)],
                fill=layer['color'],
                outline='black',
                width=2
            )
            
            # Texture selon type
            if 'pattern' in layer:
                self._add_pattern(draw, margin_left, y_start, col_width, y_end - y_start, layer['pattern'])
            
            # Annotations
            depth_text = f"{layer['depth_start']:.1f}m"
            draw.text((margin_left - 40, y_start), depth_text, fill='black', font=font_label)
            
            # Nom de la couche
            layer_name = layer['name']
            draw.text((margin_left + col_width + 20, (y_start + y_end)/2 - 10), 
                     layer_name, fill='black', font=font_label)
            
            # Résistivité
            if 'resistivity' in layer:
                res_text = f"ρ = {layer['resistivity']:.1f} Ω·m"
                draw.text((margin_left + col_width + 20, (y_start + y_end)/2 + 10), 
                         res_text, fill='blue', font=font_label)
        
        # Échelle profondeur finale
        draw.text((margin_left - 40, y_end), f"{max_depth:.1f}m", fill='black', font=font_label)
        
        # Convertir en base64
        buf = io.BytesIO()
        img.save(buf, format='PNG')
        buf.seek(0)
        
        return base64.b64encode(buf.read()).decode()
    
    def create_cross_section_diagram(self, measurements: List[Dict], 
                                     title: str = "Coupe Transversale") -> str:
        """
        Crée un schéma de coupe transversale annoté
        """
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # Trier par position x
        measurements = sorted(measurements, key=lambda x: x.get('x', 0))
        
        # Créer grille interpolée
        x_points = [m['x'] for m in measurements]
        depths = [m['depth'] for m in measurements]
        resistivities = [m['resistivity'] for m in measurements]
        
        # Interpolation pour affichage smooth
        from scipy.interpolate import griddata
        
        xi = np.linspace(min(x_points), max(x_points), 100)
        zi = np.linspace(min(depths), max(depths), 50)
        XI, ZI = np.meshgrid(xi, zi)
        
        RI = griddata((x_points, depths), resistivities, (XI, ZI), method='cubic')
        
        # Afficher avec contourf
        levels = np.logspace(np.log10(min(resistivities)), np.log10(max(resistivities)), 20)
        contour = ax.contourf(XI, ZI, RI, levels=levels, cmap='RdYlBu_r', alpha=0.8)
        
        # Contours noirs
        ax.contour(XI, ZI, RI, levels=levels[::3], colors='black', linewidths=0.5, alpha=0.4)
        
        # Points de mesure
        ax.scatter(x_points, depths, c='black', s=50, marker='o', edgecolors='white', 
                  linewidths=2, zorder=10, label='Points de mesure')
        
        # Annotations
        for m in measurements[::max(1, len(measurements)//10)]:  # Annoter 10 points max
            ax.annotate(f"{m['resistivity']:.1f}", 
                       xy=(m['x'], m['depth']), 
                       xytext=(5, 5), 
                       textcoords='offset points',
                       fontsize=8,
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
        
        # Colorbar
        cbar = plt.colorbar(contour, ax=ax, label='Résistivité (Ω·m)')
        
        ax.set_xlabel('Distance (m)', fontsize=14, fontweight='bold')
        ax.set_ylabel('Profondeur (m)', fontsize=14, fontweight='bold')
        ax.set_title(title, fontsize=16, fontweight='bold')
        ax.invert_yaxis()
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        
        # Sauvegarder
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=300, bbox_inches='tight')
        buf.seek(0)
        plt.close()
        
        return base64.b64encode(buf.read()).decode()
    
    # ========================================
    # 3. GRAPHIQUES STATISTIQUES
    # ========================================
    
    def create_histogram_with_zones(self, values: List[float], 
                                    title: str = "Distribution des Résistivités") -> str:
        """
        Histogramme avec zones géologiques colorées
        """
        fig = go.Figure()
        
        # Histogramme principal
        fig.add_trace(go.Histogram(
            x=values,
            nbinsx=50,
            name='Distribution',
            marker=dict(
                color=values,
                colorscale='Viridis',
                line=dict(color='white', width=1)
            )
        ))
        
        # Lignes verticales pour les seuils géologiques
        thresholds = [
            (10, 'Eau salée / Argile saturée', 'blue'),
            (50, 'Argile / Sable humide', 'green'),
            (200, 'Sable sec / Roche', 'orange'),
            (500, 'Roche compacte', 'red')
        ]
        
        for thresh, label, color in thresholds:
            if min(values) < thresh < max(values):
                fig.add_vline(x=thresh, line_dash="dash", line_color=color, 
                            annotation_text=label, annotation_position="top")
        
        fig.update_layout(
            title=title,
            xaxis_title="Résistivité (Ω·m)",
            yaxis_title="Fréquence",
            template='plotly_white',
            height=500,
            showlegend=False
        )
        
        return fig.to_html(include_plotlyjs='cdn', div_id='histogram_zones')
    
    def create_depth_statistics_chart(self, data: Dict[str, List[float]], 
                                     title: str = "Statistiques par Profondeur") -> str:
        """
        Graphique multi-courbes : min, max, moyenne, médiane par profondeur
        """
        depths = list(data.keys())
        
        fig = go.Figure()
        
        # Moyenne
        fig.add_trace(go.Scatter(
            x=[np.mean(data[d]) for d in depths],
            y=depths,
            mode='lines+markers',
            name='Moyenne',
            line=dict(color='blue', width=3),
            marker=dict(size=8)
        ))
        
        # Min et Max avec zone remplie
        fig.add_trace(go.Scatter(
            x=[np.min(data[d]) for d in depths],
            y=depths,
            mode='lines',
            name='Minimum',
            line=dict(color='lightblue', width=1),
            showlegend=False
        ))
        
        fig.add_trace(go.Scatter(
            x=[np.max(data[d]) for d in depths],
            y=depths,
            mode='lines',
            name='Plage',
            line=dict(color='lightblue', width=1),
            fill='tonextx',
            fillcolor='rgba(173, 216, 230, 0.3)'
        ))
        
        # Médiane
        fig.add_trace(go.Scatter(
            x=[np.median(data[d]) for d in depths],
            y=depths,
            mode='lines',
            name='Médiane',
            line=dict(color='red', width=2, dash='dot')
        ))
        
        fig.update_layout(
            title=title,
            xaxis_title="Résistivité (Ω·m)",
            yaxis_title="Profondeur (m)",
            yaxis=dict(autorange='reversed'),
            template='plotly_white',
            height=600,
            hovermode='y unified'
        )
        
        return fig.to_html(include_plotlyjs='cdn', div_id='depth_stats')
    
    # ========================================
    # 4. LÉGENDES ET ANNOTATIONS
    # ========================================
    
    def create_legend_table(self, legend_items: List[Dict[str, str]], 
                           title: str = "Légende Géologique") -> str:
        """
        Crée une table de légende HTML stylée
        legend_items = [
            {'range': '< 10 Ω·m', 'material': 'Eau salée', 'color': '#0000FF'},
            ...
        ]
        """
        html = f"""
        <div style="background: white; padding: 20px; border-radius: 10px; box-shadow: 0 4px 8px rgba(0,0,0,0.1);">
            <h3 style="color: #2196F3; text-align: center; margin-bottom: 15px;">{title}</h3>
            <table style="width: 100%; border-collapse: collapse;">
                <thead>
                    <tr style="background: #2196F3; color: white;">
                        <th style="padding: 10px; text-align: left; border: 1px solid #ddd;">Plage (Ω·m)</th>
                        <th style="padding: 10px; text-align: left; border: 1px solid #ddd;">Matériau</th>
                        <th style="padding: 10px; text-align: center; border: 1px solid #ddd; width: 80px;">Couleur</th>
                    </tr>
                </thead>
                <tbody>
        """
        
        for idx, item in enumerate(legend_items):
            bg_color = '#f9f9f9' if idx % 2 == 0 else 'white'
            html += f"""
                <tr style="background: {bg_color};">
                    <td style="padding: 10px; border: 1px solid #ddd; font-weight: bold;">{item['range']}</td>
                    <td style="padding: 10px; border: 1px solid #ddd;">{item['material']}</td>
                    <td style="padding: 10px; border: 1px solid #ddd; text-align: center;">
                        <div style="width: 50px; height: 25px; background: {item['color']}; 
                                    border: 2px solid #333; border-radius: 4px; margin: auto;"></div>
                    </td>
                </tr>
            """
        
        html += """
                </tbody>
            </table>
        </div>
        """
        
        return html
    
    def create_annotated_diagram(self, image_data: bytes, 
                                 annotations: List[Dict[str, Any]]) -> str:
        """
        Ajoute des annotations sur une image existante
        annotations = [
            {'x': 100, 'y': 200, 'text': 'Nappe phréatique', 'color': 'blue'},
            ...
        ]
        """
        # Convertir bytes en image OpenCV
        nparr = np.frombuffer(image_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        for ann in annotations:
            x, y = int(ann['x']), int(ann['y'])
            text = ann['text']
            color = self._hex_to_bgr(ann.get('color', '#FF0000'))
            
            # Dessiner point
            cv2.circle(img, (x, y), 8, color, -1)
            cv2.circle(img, (x, y), 10, (0, 0, 0), 2)
            
            # Dessiner ligne vers texte
            text_x, text_y = x + 30, y - 20
            cv2.line(img, (x, y), (text_x - 5, text_y + 5), color, 2)
            
            # Texte avec fond
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            thickness = 2
            (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, thickness)
            
            # Rectangle de fond
            cv2.rectangle(img, 
                         (text_x - 5, text_y - text_height - 5),
                         (text_x + text_width + 5, text_y + 5),
                         (255, 255, 255), -1)
            cv2.rectangle(img, 
                         (text_x - 5, text_y - text_height - 5),
                         (text_x + text_width + 5, text_y + 5),
                         color, 2)
            
            # Texte
            cv2.putText(img, text, (text_x, text_y), font, font_scale, (0, 0, 0), thickness)
        
        # Convertir en base64
        _, buffer = cv2.imencode('.png', img)
        return base64.b64encode(buffer).decode()
    
    # ========================================
    # 5. FONCTIONS UTILITAIRES
    # ========================================
    
    def _classify_resistivity_zones(self, values: List[float]) -> Dict:
        """Classifie les valeurs de résistivité par zones géologiques"""
        zones = {
            'Eau salée/Argile saturée': {'indices': [], 'color': '#0000FF', 'range': (0, 10)},
            'Argile/Sable humide': {'indices': [], 'color': '#00FF00', 'range': (10, 50)},
            'Sol mixte/Sable sec': {'indices': [], 'color': '#FFFF00', 'range': (50, 200)},
            'Roche/Gravier': {'indices': [], 'color': '#FF0000', 'range': (200, float('inf'))}
        }
        
        for idx, val in enumerate(values):
            for zone_name, zone_info in zones.items():
                if zone_info['range'][0] <= val < zone_info['range'][1]:
                    zone_info['indices'].append(idx)
                    break
        
        return zones
    
    def _resistivity_to_color(self, value: float) -> str:
        """Convertit une résistivité en couleur"""
        if value < 10:
            return '#0000FF'  # Bleu
        elif value < 50:
            return '#00FF00'  # Vert
        elif value < 200:
            return '#FFFF00'  # Jaune
        else:
            return '#FF0000'  # Rouge
    
    def _create_geological_legend(self) -> List:
        """Crée les éléments de légende matplotlib"""
        return [
            mpatches.Patch(color='#0000FF', label='Eau salée/Argile saturée (< 10 Ω·m)'),
            mpatches.Patch(color='#00FF00', label='Argile/Sable humide (10-50 Ω·m)'),
            mpatches.Patch(color='#FFFF00', label='Sol mixte/Sable sec (50-200 Ω·m)'),
            mpatches.Patch(color='#FF0000', label='Roche/Gravier (> 200 Ω·m)')
        ]
    
    def _add_pattern(self, draw, x, y, width, height, pattern: str):
        """Ajoute une texture à une couche géologique"""
        if pattern == 'dots':
            for i in range(int(x), int(x + width), 10):
                for j in range(int(y), int(y + height), 10):
                    draw.ellipse([i-2, j-2, i+2, j+2], fill='black')
        elif pattern == 'lines':
            for i in range(int(x), int(x + width), 5):
                draw.line([(i, y), (i, y + height)], fill='black', width=1)
        elif pattern == 'crosses':
            for i in range(int(x), int(x + width), 15):
                for j in range(int(y), int(y + height), 15):
                    draw.line([(i-5, j), (i+5, j)], fill='black', width=2)
                    draw.line([(i, j-5), (i, j+5)], fill='black', width=2)
    
    def _hex_to_bgr(self, hex_color: str) -> Tuple[int, int, int]:
        """Convertit couleur hex en BGR pour OpenCV"""
        hex_color = hex_color.lstrip('#')
        r, g, b = int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
        return (b, g, r)  # OpenCV utilise BGR


# ========================================
# FONCTIONS D'EXPORT POUR TÉLÉCHARGEMENT
# ========================================

def export_visualization_package(visualizations: List[Dict], 
                                 output_format: str = 'zip') -> bytes:
    """
    Exporte un package complet de visualisations
    visualizations = [
        {'type': 'image', 'name': 'profil.png', 'data': base64_string},
        {'type': 'html', 'name': 'section.html', 'data': html_string},
        ...
    ]
    """
    import zipfile
    
    buffer = io.BytesIO()
    
    if output_format == 'zip':
        with zipfile.ZipFile(buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
            for viz in visualizations:
                if viz['type'] == 'image':
                    # Décoder base64
                    img_data = base64.b64decode(viz['data'])
                    zf.writestr(viz['name'], img_data)
                elif viz['type'] == 'html':
                    zf.writestr(viz['name'], viz['data'])
                elif viz['type'] == 'json':
                    import json
                    zf.writestr(viz['name'], json.dumps(viz['data'], indent=2))
    
    buffer.seek(0)
    return buffer.getvalue()


def create_multi_page_pdf(visualizations: List[Dict], 
                         title: str = "Rapport ERT Complet") -> bytes:
    """
    Crée un PDF multi-pages avec toutes les visualisations
    """
    from matplotlib.backends.backend_pdf import PdfPages
    
    buffer = io.BytesIO()
    
    with PdfPages(buffer) as pdf:
        # Page de titre
        fig = plt.figure(figsize=(8.5, 11))
        fig.text(0.5, 0.5, title, ha='center', va='center', 
                fontsize=32, fontweight='bold', color='#2196F3')
        fig.text(0.5, 0.4, f"Généré le {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}", 
                ha='center', va='center', fontsize=14, color='gray')
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
        
        # Pages de visualisations
        for viz in visualizations:
            if viz['type'] == 'image':
                # Décoder et afficher image
                img_data = base64.b64decode(viz['data'])
                img = Image.open(io.BytesIO(img_data))
                
                fig, ax = plt.subplots(figsize=(8.5, 11))
                ax.imshow(img)
                ax.axis('off')
                ax.set_title(viz.get('title', viz['name']), fontsize=16, fontweight='bold')
                
                pdf.savefig(fig, bbox_inches='tight')
                plt.close()
    
    buffer.seek(0)
    return buffer.getvalue()
