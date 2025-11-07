"""
Système de génération automatique de visualisations pendant la conversation
Détecte automatiquement quand créer des graphiques et les génère en temps réel
"""

import streamlit as st
import re
from typing import Dict, List, Any, Optional
import numpy as np
from visualization_tools import VisualizationEngine

class AutoVisualizer:
    """Génère automatiquement des visualisations basées sur le contexte de la conversation"""
    
    def __init__(self):
        self.viz_engine = VisualizationEngine()
        self.generated_visualizations = []
    
    def detect_visualization_needs(self, query: str, extracted_data: Dict) -> List[str]:
        """
        Détecte automatiquement les besoins en visualisation
        Returns: Liste des types de visualisations à créer
        """
        query_lower = query.lower()
        viz_needs = []
        
        # Mots-clés pour graphiques
        graph_keywords = {
            'profile': ['profil', 'profile', 'vertical', 'colonne'],
            'section': ['coupe', 'section', 'transversal', '2d', 'horizontal'],
            'histogram': ['distribution', 'histogramme', 'répartition', 'fréquence'],
            'stats': ['statistique', 'moyenne', 'médiane', 'écart', 'variation'],
            'legend': ['légende', 'legend', 'explication', 'signification'],
            'comparison': ['comparaison', 'compare', 'différence', 'vs'],
            'evolution': ['évolution', 'evolution', 'temps', 'changement', 'tendance']
        }
        
        # Détection contextuelle
        has_resistivity_data = ('resistivity_values' in extracted_data and 
                               len(extracted_data.get('resistivity_values', [])) > 0)
        
        # Détection par mots-clés
        for viz_type, keywords in graph_keywords.items():
            if any(kw in query_lower for kw in keywords):
                viz_needs.append(viz_type)
        
        # Détection automatique si données disponibles
        if has_resistivity_data:
            # Toujours proposer un profil si pas déjà demandé
            if 'profile' not in viz_needs and len(viz_needs) == 0:
                viz_needs.append('profile')
            
            # Ajouter histogramme pour distribution
            if ('distribution' in query_lower or 'répartition' in query_lower):
                if 'histogram' not in viz_needs:
                    viz_needs.append('histogram')
            
            # Ajouter stats si question sur statistiques
            if any(word in query_lower for word in ['statistique', 'moyenne', 'max', 'min', 'médiane']):
                if 'stats' not in viz_needs:
                    viz_needs.append('stats')
        
        # Toujours ajouter légende si données ERT
        if has_resistivity_data and 'legend' not in viz_needs:
            viz_needs.append('legend')
        
        return viz_needs
    
    def generate_visualizations(self, viz_needs: List[str], data: Dict) -> Dict[str, Any]:
        """
        Génère toutes les visualisations demandées
        Returns: Dict avec HTML/base64 pour chaque visualisation
        """
        visualizations = {}
        
        # Extraire les données de résistivité
        resistivity_values = data.get('resistivity_values', [])
        
        if not resistivity_values:
            return visualizations
        
        # 1. PROFIL DE RÉSISTIVITÉ
        if 'profile' in viz_needs:
            try:
                html_profile = self.viz_engine.create_resistivity_profile(
                    resistivity_values,
                    title=f"Profil de Résistivité - {data.get('filename', 'Analyse')}"
                )
                visualizations['profile'] = {
                    'type': 'html',
                    'content': html_profile,
                    'title': '📊 Profil de Résistivité Vertical',
                    'download_name': 'profil_resistivite.html'
                }
            except Exception as e:
                print(f"Erreur génération profil: {e}")
        
        # 2. HISTOGRAMME
        if 'histogram' in viz_needs:
            try:
                html_hist = self.viz_engine.create_histogram_with_zones(
                    resistivity_values,
                    title=f"Distribution des Résistivités"
                )
                visualizations['histogram'] = {
                    'type': 'html',
                    'content': html_hist,
                    'title': '📊 Distribution des Valeurs',
                    'download_name': 'histogramme_resistivite.html'
                }
            except Exception as e:
                print(f"Erreur génération histogramme: {e}")
        
        # 3. LÉGENDE GÉOLOGIQUE
        if 'legend' in viz_needs:
            try:
                legend_items = [
                    {'range': '< 10 Ω·m', 'material': 'Eau salée / Argile saturée', 'color': '#0000FF'},
                    {'range': '10 - 50 Ω·m', 'material': 'Argile / Sable humide', 'color': '#00FF00'},
                    {'range': '50 - 200 Ω·m', 'material': 'Sol mixte / Sable sec', 'color': '#FFFF00'},
                    {'range': '> 200 Ω·m', 'material': 'Roche compacte / Gravier', 'color': '#FF0000'}
                ]
                html_legend = self.viz_engine.create_legend_table(legend_items)
                visualizations['legend'] = {
                    'type': 'html',
                    'content': html_legend,
                    'title': '🗺️ Légende Géologique',
                    'download_name': 'legende_geologique.html'
                }
            except Exception as e:
                print(f"Erreur génération légende: {e}")
        
        # 4. STATISTIQUES PAR PROFONDEUR (si données de profondeur disponibles)
        if 'stats' in viz_needs and 'depth_data' in data:
            try:
                html_stats = self.viz_engine.create_depth_statistics_chart(
                    data['depth_data'],
                    title="Statistiques par Profondeur"
                )
                visualizations['stats'] = {
                    'type': 'html',
                    'content': html_stats,
                    'title': '📈 Statistiques Multi-Niveaux',
                    'download_name': 'stats_profondeur.html'
                }
            except Exception as e:
                print(f"Erreur génération stats: {e}")
        
        # 5. COUPE 2D (si grille de données disponible)
        if 'section' in viz_needs and 'grid_data' in data:
            try:
                html_section = self.viz_engine.create_2d_resistivity_section(
                    data['grid_data'],
                    title="Coupe Transversale ERT"
                )
                visualizations['section'] = {
                    'type': 'html',
                    'content': html_section,
                    'title': '🗺️ Coupe 2D',
                    'download_name': 'coupe_2d.html'
                }
            except Exception as e:
                print(f"Erreur génération coupe 2D: {e}")
        
        return visualizations
    
    def display_visualizations(self, visualizations: Dict[str, Any]):
        """
        Affiche les visualisations dans Streamlit avec boutons de téléchargement
        """
        if not visualizations:
            return
        
        st.markdown("---")
        st.markdown("### 📊 Visualisations Générées Automatiquement")
        st.markdown("*Toutes les visualisations sont interactives et téléchargeables*")
        
        for viz_id, viz_data in visualizations.items():
            with st.expander(f"🎨 {viz_data['title']}", expanded=True):
                if viz_data['type'] == 'html':
                    # Afficher HTML interactif
                    st.components.v1.html(viz_data['content'], height=650, scrolling=True)
                    
                    # Bouton de téléchargement
                    st.download_button(
                        label=f"📥 Télécharger {viz_data['title']}",
                        data=viz_data['content'],
                        file_name=viz_data['download_name'],
                        mime='text/html',
                        key=f"download_{viz_id}"
                    )
                
                elif viz_data['type'] == 'image':
                    # Afficher image base64
                    import base64
                    img_html = f'<img src="data:image/png;base64,{viz_data["content"]}" style="width:100%; max-width:800px;">'
                    st.markdown(img_html, unsafe_allow_html=True)
                    
                    # Bouton de téléchargement
                    img_bytes = base64.b64decode(viz_data['content'])
                    st.download_button(
                        label=f"📥 Télécharger {viz_data['title']}",
                        data=img_bytes,
                        file_name=viz_data['download_name'],
                        mime='image/png',
                        key=f"download_{viz_id}"
                    )
        
        # Bouton pour télécharger tout en un package
        if len(visualizations) > 1:
            st.markdown("---")
            col1, col2 = st.columns(2)
            
            with col1:
                # Package ZIP
                from visualization_tools import export_visualization_package
                
                viz_list = [
                    {
                        'type': 'html' if v['type'] == 'html' else 'image',
                        'name': v['download_name'],
                        'data': v['content']
                    }
                    for v in visualizations.values()
                ]
                
                zip_data = export_visualization_package(viz_list, output_format='zip')
                
                st.download_button(
                    label="📦 Télécharger toutes les visualisations (ZIP)",
                    data=zip_data,
                    file_name="visualisations_ert_complete.zip",
                    mime="application/zip",
                    key="download_all_zip"
                )
            
            with col2:
                # PDF multi-pages
                try:
                    from visualization_tools import create_multi_page_pdf
                    
                    pdf_data = create_multi_page_pdf(viz_list, title="Rapport ERT Complet")
                    
                    st.download_button(
                        label="📄 Télécharger tout en PDF",
                        data=pdf_data,
                        file_name="rapport_ert_complet.pdf",
                        mime="application/pdf",
                        key="download_all_pdf"
                    )
                except Exception as e:
                    st.info(f"PDF non disponible: {e}")
    
    def auto_generate_and_display(self, query: str, data: Dict):
        """
        Fonction tout-en-un: détecte, génère et affiche automatiquement
        """
        # Détecter besoins
        viz_needs = self.detect_visualization_needs(query, data)
        
        if not viz_needs:
            return
        
        # Générer
        with st.spinner("🎨 Génération automatique des visualisations..."):
            visualizations = self.generate_visualizations(viz_needs, data)
        
        # Afficher
        if visualizations:
            self.display_visualizations(visualizations)
            
            # Stocker dans session pour réutilisation
            if 'generated_visualizations' not in st.session_state:
                st.session_state.generated_visualizations = []
            st.session_state.generated_visualizations.append({
                'query': query,
                'visualizations': visualizations
            })
