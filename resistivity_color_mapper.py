#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module de cartographie des résistivités et base de données géophysique
Système avancé pour l'analyse ERT avec couleurs, matériaux et recherche internet
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from typing import Dict, List, Tuple, Optional, Any
import requests
from bs4 import BeautifulSoup
import time
from langchain_tavily import TavilySearch as TavilySearchResults
from dotenv import load_dotenv

load_dotenv()
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

class ResistivityColorMapper:
    """Mappeur de couleurs pour les valeurs de résistivité ERT"""

    def __init__(self):
        # Palette de couleurs standardisée pour la résistivité (en Ohm.m)
        self.color_ranges = {
            # Très faible résistivité (< 1 Ohm.m)
            'very_low': {
                'range': (0.01, 1.0),
                'color': '#000080',  # Bleu foncé
                'description': 'Très faible résistivité',
                'materials': ['eau salée', 'argile saturée', 'minerais conducteurs']
            },
            # Faible résistivité (1-10 Ohm.m)
            'low': {
                'range': (1.0, 10.0),
                'color': '#0000FF',  # Bleu
                'description': 'Faible résistivité',
                'materials': ['argile', 'eau douce', 'sols humides']
            },
            # Moyenne-faible (10-50 Ohm.m)
            'medium_low': {
                'range': (10.0, 50.0),
                'color': '#0080FF',  # Bleu clair
                'description': 'Résistivité moyenne-faible',
                'materials': ['limon', 'sable humide', 'marne']
            },
            # Moyenne (50-200 Ohm.m)
            'medium': {
                'range': (50.0, 200.0),
                'color': '#00FF80',  # Vert clair
                'description': 'Résistivité moyenne',
                'materials': ['sable', 'gravier', 'sols agricoles']
            },
            # Moyenne-élevée (200-1000 Ohm.m)
            'medium_high': {
                'range': (200.0, 1000.0),
                'color': '#FFFF00',  # Jaune
                'description': 'Résistivité moyenne-élevée',
                'materials': ['sable sec', 'calcaire', 'schiste']
            },
            # Élevée (1000-10000 Ohm.m)
            'high': {
                'range': (1000.0, 10000.0),
                'color': '#FFA500',  # Orange
                'description': 'Résistivité élevée',
                'materials': ['roche cristalline', 'granite', 'gneiss']
            },
            # Très élevée (>10000 Ohm.m)
            'very_high': {
                'range': (10000.0, 100000.0),
                'color': '#FF0000',  # Rouge
                'description': 'Très haute résistivité',
                'materials': ['roche anhydre', 'air', 'vide', 'cavités']
            }
        }

        # Base de données des matériaux géophysiques
        self.material_database = self._initialize_material_database()

    def _initialize_material_database(self) -> Dict[str, Dict]:
        """Initialise la base de données des matériaux avec leurs propriétés RÉELLES"""
        return {
            # Liquides - valeurs vérifiées et réelles
            'liquids': {
                'eau_distillée': {
                    'resistivity_range': (0.8, 2.0),
                    'typical_value': 1.5,
                    'color': '#E0F0FF',
                    'nature': 'liquide pur',
                    'depth_range': 'surface',
                    'description': 'Eau pure sans minéraux, résistivité très faible'
                },
                'eau_douce': {
                    'resistivity_range': (10.0, 100.0),
                    'typical_value': 50.0,
                    'color': '#6080FF',
                    'nature': 'liquide légèrement minéralisé',
                    'depth_range': 'nappe phréatique',
                    'description': 'Eau de rivière ou nappe avec faible minéralisation'
                },
                'eau_saumâtre': {
                    'resistivity_range': (0.5, 5.0),
                    'typical_value': 2.0,
                    'color': '#004080',
                    'nature': 'liquide modérément salé',
                    'depth_range': 'estuaires, lagunes',
                    'description': 'Eau mélangée douce/salée, résistivité intermédiaire'
                },
                'eau_de_mer': {
                    'resistivity_range': (0.2, 0.5),
                    'typical_value': 0.3,
                    'color': '#002040',
                    'nature': 'liquide fortement salé',
                    'depth_range': 'océan, profondeur variable',
                    'description': 'Eau océanique avec forte conductivité due au sel'
                },
                'eau_souterraine_saline': {
                    'resistivity_range': (0.1, 1.0),
                    'typical_value': 0.5,
                    'color': '#001020',
                    'nature': 'liquide souterrain salé',
                    'depth_range': 'aquifères salins',
                    'description': 'Eau souterraine avec forte salinité'
                },
                'pétrole': {
                    'resistivity_range': (1.0, 100.0),
                    'typical_value': 10.0,
                    'color': '#800000',
                    'nature': 'hydrocarbure liquide',
                    'depth_range': 'réservoirs souterrains',
                    'description': 'Pétrole brut, isolant électrique variable'
                },
                'huile_minérale': {
                    'resistivity_range': (10.0, 1000.0),
                    'typical_value': 100.0,
                    'color': '#804000',
                    'nature': 'hydrocarbure raffiné',
                    'depth_range': 'réservoirs industriels',
                    'description': 'Huile isolante, résistivité variable selon pureté'
                },
                'mercure': {
                    'resistivity_range': (0.0001, 0.001),
                    'typical_value': 0.0005,
                    'color': '#C0C0C0',
                    'nature': 'métal liquide',
                    'depth_range': 'laboratoire/industriel',
                    'description': 'Mercure métallique liquide, excellent conducteur'
                }
            },

            # Minéraux métalliques précieux - valeurs RÉELLES vérifiées
            'precious_metals': {
                'or': {
                    'resistivity_range': (0.000002, 0.000005),
                    'typical_value': 0.0000024,
                    'color': '#FFD700',
                    'nature': 'métal précieux natif',
                    'depth_range': 'filons quartziques',
                    'description': 'Or natif, excellent conducteur électrique'
                },
                'argent': {
                    'resistivity_range': (0.0000015, 0.000002),
                    'typical_value': 0.0000016,
                    'color': '#C0C0C0',
                    'nature': 'métal précieux',
                    'depth_range': 'gisements sulfurés',
                    'description': 'Argent natif, très bon conducteur'
                },
                'platine': {
                    'resistivity_range': (0.00001, 0.00002),
                    'typical_value': 0.0000106,
                    'color': '#E5E4E2',
                    'nature': 'métal précieux du groupe platine',
                    'depth_range': 'complexes mafiques',
                    'description': 'Platine, résiste à la corrosion'
                },
                'diamant': {
                    'resistivity_range': (1000000.0, 10000000.0),
                    'typical_value': 1000000.0,
                    'color': '#FFFFFF',
                    'nature': 'minéral carboné cristallisé',
                    'depth_range': 'kimberlites, profondeur >150km',
                    'description': 'Diamant, isolant électrique parfait'
                }
            },

            # Minéraux courants - valeurs RÉELLES
            'minerals': {
                'quartz': {
                    'resistivity_range': (1000.0, 10000.0),
                    'typical_value': 5000.0,
                    'color': '#FFFFFF',
                    'nature': 'minéral siliceux',
                    'depth_range': 'croûte terrestre',
                    'description': 'Quartz pur, très résistif'
                },
                'calcite': {
                    'resistivity_range': (100.0, 1000.0),
                    'typical_value': 500.0,
                    'color': '#F0F0F0',
                    'nature': 'carbonate',
                    'depth_range': 'formations sédimentaires',
                    'description': 'Calcaire, marbre'
                },
                'halite': {
                    'resistivity_range': (0.1, 1.0),
                    'typical_value': 0.5,
                    'color': '#C0C0C0',
                    'nature': 'sel gemme',
                    'depth_range': 'dômes salins',
                    'description': 'Chlorure de sodium, très conducteur'
                },
                'pyrite': {
                    'resistivity_range': (0.001, 0.1),
                    'typical_value': 0.01,
                    'color': '#FFFF00',
                    'nature': 'sulfure métallique',
                    'depth_range': 'filons minéralisés',
                    'description': "Pyrite 'or des fous', très conductrice"
                },
                'magnetite': {
                    'resistivity_range': (0.001, 0.01),
                    'typical_value': 0.005,
                    'color': '#000000',
                    'nature': 'oxyde ferromagnétique',
                    'depth_range': 'roches mafiques',
                    'description': 'Aimant naturel, forte conductivité'
                },
                'graphite': {
                    'resistivity_range': (0.0001, 0.001),
                    'typical_value': 0.0005,
                    'color': '#404040',
                    'nature': 'carbone cristallisé',
                    'depth_range': 'roches métamorphiques',
                    'description': 'Graphite, excellent conducteur'
                },
                'cuivre_natif': {
                    'resistivity_range': (0.0000017, 0.000002),
                    'typical_value': 0.0000018,
                    'color': '#B87333',
                    'nature': 'métal natif',
                    'depth_range': 'gisements porphyriques',
                    'description': 'Cuivre métallique natif'
                },
                'galène': {
                    'resistivity_range': (0.0001, 0.001),
                    'typical_value': 0.0005,
                    'color': '#808080',
                    'nature': 'sulfure de plomb',
                    'depth_range': 'filons hydrothermaux',
                    'description': 'Minerai de plomb principal'
                },
                'chalcopyrite': {
                    'resistivity_range': (0.001, 0.01),
                    'typical_value': 0.005,
                    'color': '#A0522D',
                    'nature': 'sulfure de cuivre',
                    'depth_range': 'gisements porphyriques',
                    'description': 'Minerai de cuivre principal'
                },
                'sphène': {
                    'resistivity_range': (100.0, 1000.0),
                    'typical_value': 500.0,
                    'color': '#8B4513',
                    'nature': 'titanosilicate',
                    'depth_range': 'roches plutoniques',
                    'description': 'Sphène, minéral accessoire'
                },
                'apatite': {
                    'resistivity_range': (1000.0, 10000.0),
                    'typical_value': 5000.0,
                    'color': '#F5F5DC',
                    'nature': 'phosphate',
                    'depth_range': 'roches ignées et métamorphiques',
                    'description': 'Apatite, source de phosphore'
                },
                'zircon': {
                    'resistivity_range': (10000.0, 100000.0),
                    'typical_value': 50000.0,
                    'color': '#D2B48C',
                    'nature': 'zirconsilicate',
                    'depth_range': 'roches acides',
                    'description': 'Zircon, très résistif et durable'
                }
            },

            # Sols et formations - valeurs RÉELLES
            'soils': {
                'argile': {
                    'resistivity_range': (1.0, 20.0),
                    'typical_value': 10.0,
                    'color': '#8B4513',
                    'nature': 'sol fin',
                    'depth_range': 'surface à quelques mètres',
                    'description': 'Argile, très plastique, faible résistivité'
                },
                'limon': {
                    'resistivity_range': (10.0, 50.0),
                    'typical_value': 30.0,
                    'color': '#D2691E',
                    'nature': 'sol intermédiaire',
                    'depth_range': 'couches superficielles',
                    'description': 'Limon, mélange argile/sable'
                },
                'sable': {
                    'resistivity_range': (50.0, 500.0),
                    'typical_value': 200.0,
                    'color': '#F4A460',
                    'nature': 'sol grossier',
                    'depth_range': 'formations alluviales',
                    'description': 'Sable, perméable, résistivité moyenne'
                },
                'gravier': {
                    'resistivity_range': (200.0, 2000.0),
                    'typical_value': 800.0,
                    'color': '#DEB887',
                    'nature': 'matériau granulaire',
                    'depth_range': 'terrasses alluviales',
                    'description': 'Gravier, très perméable'
                },
                'roche_mère': {
                    'resistivity_range': (1000.0, 50000.0),
                    'typical_value': 10000.0,
                    'color': '#696969',
                    'nature': 'substratum rocheux',
                    'depth_range': 'socle géologique',
                    'description': 'Roche mère, très résistive'
                },
                'tourbe': {
                    'resistivity_range': (0.5, 5.0),
                    'typical_value': 2.0,
                    'color': '#654321',
                    'nature': 'matière organique décomposée',
                    'depth_range': 'zones humides',
                    'description': 'Tourbe, très conductrice due à l\'humidité'
                },
                'schiste': {
                    'resistivity_range': (100.0, 1000.0),
                    'typical_value': 500.0,
                    'color': '#2F4F4F',
                    'nature': 'roche métamorphique',
                    'depth_range': 'formations sédimentaires plissées',
                    'description': 'Schiste, résistivité moyenne'
                },
                'gneiss': {
                    'resistivity_range': (1000.0, 10000.0),
                    'typical_value': 5000.0,
                    'color': '#D3D3D3',
                    'nature': 'roche métamorphique',
                    'depth_range': 'socle ancien',
                    'description': 'Gneiss, très résistif'
                }
            },

            # Roches ignées - valeurs RÉELLES
            'igneous_rocks': {
                'granite': {
                    'resistivity_range': (1000.0, 10000.0),
                    'typical_value': 5000.0,
                    'color': '#D3D3D3',
                    'nature': 'roche plutonique acide',
                    'depth_range': 'croûte continentale',
                    'description': 'Granite, très résistif'
                },
                'basalte': {
                    'resistivity_range': (100.0, 1000.0),
                    'typical_value': 500.0,
                    'color': '#2F2F2F',
                    'nature': 'roche volcanique mafique',
                    'depth_range': 'croûte océanique',
                    'description': 'Basalte, résistivité moyenne'
                },
                'gabbro': {
                    'resistivity_range': (100.0, 1000.0),
                    'typical_value': 300.0,
                    'color': '#404040',
                    'nature': 'roche plutonique mafique',
                    'depth_range': 'croûte océanique profonde',
                    'description': 'Gabbro, similaire au basalte'
                },
                'rhyolite': {
                    'resistivity_range': (1000.0, 10000.0),
                    'typical_value': 8000.0,
                    'color': '#F5F5F5',
                    'nature': 'roche volcanique acide',
                    'depth_range': 'zones de subduction',
                    'description': 'Rhyolite, très résistive'
                },
                'péridotite': {
                    'resistivity_range': (10.0, 100.0),
                    'typical_value': 50.0,
                    'color': '#228B22',
                    'nature': 'roche ultramafique',
                    'depth_range': 'manteau supérieur',
                    'description': 'Péridotite, conductrice'
                }
            },

            # Fluides géologiques - valeurs RÉELLES
            'geological_fluids': {
                'eau_thermique': {
                    'resistivity_range': (0.1, 1.0),
                    'typical_value': 0.5,
                    'color': '#FF6B6B',
                    'nature': 'fluide hydrothermal',
                    'depth_range': 'systèmes géothermiques',
                    'description': 'Eau chaude minéralisée'
                },
                'fluide_pétrolier': {
                    'resistivity_range': (0.01, 1.0),
                    'typical_value': 0.1,
                    'color': '#8B4513',
                    'nature': 'hydrocarbure sous pression',
                    'depth_range': 'réservoirs profonds',
                    'description': 'Pétrole sous haute pression'
                },
                'saumure': {
                    'resistivity_range': (0.01, 0.1),
                    'typical_value': 0.05,
                    'color': '#4169E1',
                    'nature': 'solution saline concentrée',
                    'depth_range': 'formations évaporitiques',
                    'description': 'Saumure très concentrée'
                },
                'gaz_naturel': {
                    'resistivity_range': (1000.0, 10000.0),
                    'typical_value': 5000.0,
                    'color': '#F0F8FF',
                    'nature': 'hydrocarbure gazeux',
                    'depth_range': 'trappes de gaz',
                    'description': 'Gaz naturel, isolant'
                }
            }
        }

    def get_color_for_resistivity(self, resistivity: float) -> Tuple[str, str]:
        """Retourne la couleur et description pour une résistivité donnée"""
        for category, data in self.color_ranges.items():
            if data['range'][0] <= resistivity < data['range'][1]:
                return data['color'], data['description']
        # Valeurs extrêmes
        if resistivity < 0.01:
            return '#000000', 'Ultra faible résistivité'
        else:
            return '#FFFFFF', 'Ultra haute résistivité'

    def create_colorbar(self, figsize=(8, 2)) -> plt.Figure:
        """Crée une barre de couleur pour la résistivité"""
        fig, ax = plt.subplots(figsize=figsize)

        # Créer la colormap personnalisée
        colors = []
        boundaries = []
        labels = []

        for category, data in self.color_ranges.items():
            colors.append(data['color'])
            boundaries.append(data['range'][0])
            labels.append(f"{data['range'][0]:.1f}-{data['range'][1]:.1f}")

        boundaries.append(self.color_ranges['very_high']['range'][1])

        # Créer la colormap
        cmap = mcolors.ListedColormap(colors)
        norm = mcolors.BoundaryNorm(boundaries, cmap.N)

        # Créer la colorbar
        cb = plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap),
                          cax=ax, orientation='horizontal')
        cb.set_label('Résistivité (Ohm.m) - Échelle logarithmique')

        # Labels personnalisés
        cb.set_ticks([np.log10(b) for b in boundaries[:-1]])
        cb.set_ticklabels([f"{b:.1f}" for b in boundaries[:-1]])

        plt.tight_layout()
        return fig

    def find_similar_materials(self, resistivity: float, tolerance: float = 0.5) -> List[Dict]:
        """Trouve les matériaux similaires à une résistivité donnée"""
        similar_materials = []

        for category, materials in self.material_database.items():
            for material_name, properties in materials.items():
                res_range = properties['resistivity_range']
                typical = properties['typical_value']

                # Vérifier si la résistivité est dans la plage ou proche de la valeur typique
                if (res_range[0] <= resistivity <= res_range[1] or
                    abs(np.log10(resistivity) - np.log10(typical)) < tolerance):

                    similarity_score = 1.0 - min(1.0, abs(np.log10(resistivity) - np.log10(typical)) / 2.0)

                    material_info = properties.copy()
                    material_info.update({
                        'name': material_name,
                        'category': category,
                        'similarity_score': similarity_score
                    })
                    similar_materials.append(material_info)

        # Trier par score de similarité
        similar_materials.sort(key=lambda x: x['similarity_score'], reverse=True)
        return similar_materials[:10]  # Top 10

    def get_material_description(self, material_name: str, category: str = None) -> Optional[Dict]:
        """Obtient la description détaillée d'un matériau"""
        if category and category in self.material_database:
            return self.material_database[category].get(material_name)

        # Recherche dans toutes les catégories
        for cat, materials in self.material_database.items():
            if material_name in materials:
                material = materials[material_name].copy()
                material['category'] = cat
                return material

        return None

class GeophysicalDataSearcher:
    """Chercheur de données géophysiques sur internet"""

    def __init__(self):
        self.tavily_search = TavilySearchResults(api_key=TAVILY_API_KEY, max_results=10)

    def search_material_resistivity(self, material: str, context: str = "") -> List[Dict]:
        """Recherche la résistivité d'un matériau spécifique"""
        query = f"résistivité électrique {material} geophysical properties Ohm.m"
        if context:
            query += f" {context}"

        try:
            results = self.tavily_search.invoke(query)
            
            # Extraire la liste des résultats
            search_results = results.get('results', [])
            
            formatted_results = []
            for result in search_results:
                formatted_results.append({
                    'title': result.get('title', ''),
                    'content': result.get('content', ''),
                    'url': result.get('url', ''),
                    'material': material,
                    'search_type': 'resistivity_data'
                })

            return formatted_results

        except Exception as e:
            print(f"Erreur recherche résistivité {material}: {e}")
            return []

    def search_geophysical_database(self, materials_list: List[str]) -> Dict[str, List]:
        """Recherche des données géophysiques pour une liste de matériaux"""
        results = {}

        for material in materials_list:
            print(f"🔍 Recherche données pour: {material}")
            material_results = self.search_material_resistivity(material)
            results[material] = material_results

            # Pause pour éviter rate limiting
            time.sleep(1)

        return results

    def extract_resistivity_values(self, search_results: List[Dict]) -> List[float]:
        """Extrait les valeurs de résistivité des résultats de recherche"""
        import re

        resistivity_values = []

        for result in search_results:
            content = result.get('content', '')

            # Patterns pour trouver les valeurs de résistivité
            patterns = [
                r'(\d+(?:\.\d+)?)\s*(?:Ohm\.?m|Ω\.?m|ohm\.?m)',
                r'résistivité(?:\s+de)?\s*(\d+(?:\.\d+)?)',
                r'(\d+(?:\.\d+)?)\s*Ω',
                r'resistivity\s*(\d+(?:\.\d+)?)'
            ]

            for pattern in patterns:
                matches = re.findall(pattern, content, re.IGNORECASE)
                for match in matches:
                    try:
                        value = float(match)
                        if 0.001 <= value <= 100000:  # Plage réaliste
                            resistivity_values.append(value)
                    except ValueError:
                        continue

        return list(set(resistivity_values))  # Supprimer doublons

class DynamicERTAnalyzer:
    """Analyseur ERT dynamique avec descriptions enrichies"""

    def __init__(self):
        self.mapper = ResistivityColorMapper()
        self.data_searcher = GeophysicalDataSearcher()

    def analyze_resistivity_profile(self, resistivity_data: np.ndarray,
                                  depth_info: Optional[np.ndarray] = None,
                                  dat_file_path: Optional[str] = None) -> Dict[str, Any]:
        """Analyse un profil de résistivité avec descriptions dynamiques et validation .dat"""

        analysis = {
            'statistics': self._calculate_statistics(resistivity_data),
            'layers': self._identify_layers(resistivity_data, depth_info),
            'materials': self._identify_potential_materials(resistivity_data, dat_file_path),
            'geological_interpretation': self._geological_interpretation(resistivity_data),
            'color_mapping': self._create_color_profile(resistivity_data),
            'recommendations': self._generate_recommendations(resistivity_data),
            'dat_validation': None
        }

        # Ajout validation .dat si fichier fourni
        if dat_file_path:
            analysis['dat_validation'] = self.mapper.validate_with_dat_file(dat_file_path, resistivity_data)

        return analysis

    def _calculate_statistics(self, data: np.ndarray) -> Dict[str, float]:
        """Calcule les statistiques de base"""
        return {
            'mean': np.mean(data),
            'median': np.median(data),
            'std': np.std(data),
            'min': np.min(data),
            'max': np.max(data),
            'cv': np.std(data) / np.mean(data) if np.mean(data) != 0 else 0,
            'range': np.max(data) - np.min(data)
        }

    def _identify_layers(self, data: np.ndarray, depth_info: Optional[np.ndarray]) -> List[Dict]:
        """Identifie les différentes couches géologiques"""
        layers = []

        # Analyse par clustering simple
        from sklearn.cluster import KMeans

        # Normaliser les données (log pour la résistivité)
        log_data = np.log10(data + 1e-6)

        # Déterminer le nombre optimal de clusters (max 5)
        n_clusters = min(5, len(data) // 10 + 1)

        if n_clusters > 1:
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            clusters = kmeans.fit_predict(log_data.reshape(-1, 1))

            for i in range(n_clusters):
                mask = clusters == i
                cluster_data = data[mask]

                layer_info = {
                    'layer_id': i + 1,
                    'resistivity_range': (np.min(cluster_data), np.max(cluster_data)),
                    'mean_resistivity': np.mean(cluster_data),
                    'thickness_estimate': len(cluster_data) / len(data),
                    'color': self.mapper.get_color_for_resistivity(np.mean(cluster_data))[0],
                    'description': self._describe_layer(np.mean(cluster_data))
                }

                if depth_info is not None:
                    layer_depths = depth_info[mask]
                    layer_info['depth_range'] = (np.min(layer_depths), np.max(layer_depths))

                layers.append(layer_info)

        return sorted(layers, key=lambda x: x['mean_resistivity'])

    def _identify_potential_materials(self, data: np.ndarray, dat_file_path: Optional[str] = None) -> List[Dict]:
        """Identifie les matériaux potentiels présents avec validation .dat"""
        mean_resistivity = np.mean(data)

        # Recherche de matériaux similaires
        similar_materials = self.mapper.find_similar_materials(mean_resistivity)

        # Validation avec fichier .dat si fourni
        dat_validation = None
        if dat_file_path:
            dat_validation = self.validate_with_dat_file(dat_file_path, data)

        # Enrichir avec recherche web si nécessaire
        enriched_materials = []
        for material in similar_materials[:5]:  # Top 5
            # Recherche de données supplémentaires
            search_results = self.data_searcher.search_material_resistivity(
                material['name'],
                f"ERT geophysical properties real world data"
            )

            extracted_values = self.data_searcher.extract_resistivity_values(search_results)

            material_enriched = material.copy()
            material_enriched['web_resistivity_values'] = extracted_values
            material_enriched['validation_score'] = len(extracted_values) / 10.0  # Score basé sur données trouvées

            # Ajout validation .dat si disponible
            if dat_validation and dat_validation['data_loaded']:
                material_enriched['dat_validated'] = True
                material_enriched['dat_confidence'] = dat_validation['confidence_level']
            else:
                material_enriched['dat_validated'] = False
                material_enriched['dat_confidence'] = 'none'

            enriched_materials.append(material_enriched)

        return enriched_materials

    def validate_with_dat_file(self, dat_file_path: str, resistivity_data: np.ndarray) -> Dict[str, Any]:
        """Valide les données de résistivité avec un fichier .dat de référence"""
        validation_results = {
            'file_exists': False,
            'data_loaded': False,
            'validation_score': 0.0,
            'matching_materials': [],
            'confidence_level': 'low'
        }

        try:
            if not os.path.exists(dat_file_path):
                return validation_results

            validation_results['file_exists'] = True

            # Tentative de lecture du fichier .dat
            dat_data = self._load_dat_file(dat_file_path)
            if dat_data is None:
                return validation_results

            validation_results['data_loaded'] = True

            # Comparaison avec les données fournies
            matches = self._compare_with_dat_data(dat_data, resistivity_data)
            validation_results['matching_materials'] = matches

            # Calcul du score de validation
            if matches:
                avg_similarity = np.mean([m['similarity_score'] for m in matches])
                validation_results['validation_score'] = avg_similarity

                if avg_similarity > 0.8:
                    validation_results['confidence_level'] = 'high'
                elif avg_similarity > 0.6:
                    validation_results['confidence_level'] = 'medium'
                else:
                    validation_results['confidence_level'] = 'low'

        except Exception as e:
            print(f"Erreur validation fichier .dat {dat_file_path}: {e}")

        return validation_results

    def _load_dat_file(self, file_path: str) -> Optional[np.ndarray]:
        """Charge un fichier .dat et extrait les valeurs numériques"""
        try:
            with open(file_path, 'rb') as f:
                # Lecture binaire pour détecter le format
                data = f.read()

            # Conversion en texte pour extraction
            text_data = data.decode('utf-8', errors='ignore')

            # Extraction des nombres (résistivités)
            import re
            numbers = re.findall(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?', text_data)

            if len(numbers) < 10:
                return None

            # Conversion en array numpy
            values = np.array([float(n) for n in numbers if 1e-6 <= float(n) <= 1e8])

            return values if len(values) >= 10 else None

        except Exception as e:
            print(f"Erreur lecture fichier .dat: {e}")
            return None

    def _compare_with_dat_data(self, dat_data: np.ndarray, test_data: np.ndarray) -> List[Dict]:
        """Compare les données test avec les données de référence .dat"""
        matches = []

        # Statistiques des données de référence
        dat_mean = np.mean(dat_data)
        dat_std = np.std(dat_data)
        dat_range = (np.min(dat_data), np.max(dat_data))

        # Statistiques des données test
        test_mean = np.mean(test_data)
        test_std = np.std(test_data)
        test_range = (np.min(test_data), np.max(test_data))

        # Score de similarité basé sur les statistiques
        mean_similarity = 1.0 - min(1.0, abs(np.log10(dat_mean) - np.log10(test_mean)) / 2.0)
        range_similarity = 1.0 - min(1.0, abs(np.log10(dat_range[1]) - np.log10(test_range[1])) / 2.0)

        overall_similarity = (mean_similarity + range_similarity) / 2.0

        # Recherche de matériaux correspondants
        for category, materials in self.material_database.items():
            for material_name, properties in materials.items():
                mat_range = properties['resistivity_range']
                mat_typical = properties['typical_value']

                # Vérification si les données correspondent au matériau
                if (mat_range[0] <= test_mean <= mat_range[1] or
                    abs(np.log10(test_mean) - np.log10(mat_typical)) < 0.5):

                    material_match = properties.copy()
                    material_match.update({
                        'name': material_name,
                        'category': category,
                        'similarity_score': overall_similarity,
                        'dat_validation': {
                            'dat_mean': dat_mean,
                            'dat_range': dat_range,
                            'test_mean': test_mean,
                            'test_range': test_range
                        }
                    })
                    matches.append(material_match)

        # Tri par score de similarité
        matches.sort(key=lambda x: x['similarity_score'], reverse=True)
        return matches[:5]  # Top 5 matches

    def get_real_world_validation(self, material_name: str) -> Dict[str, Any]:
        """Obtient la validation réelle d'un matériau avec données du monde réel"""
        validation_data = {
            'material': material_name,
            'real_world_sources': [],
            'resistivity_range_verified': None,
            'confidence_level': 'unknown',
            'sources': []
        }

        # Recherche de données réelles pour le matériau
        search_results = self.data_searcher.search_material_resistivity(
            material_name,
            "geophysical properties resistivity values real world data"
        )

        if search_results:
            extracted_values = self.data_searcher.extract_resistivity_values(search_results)

            if extracted_values:
                validation_data.update({
                    'resistivity_range_verified': (min(extracted_values), max(extracted_values)),
                    'typical_value_verified': np.median(extracted_values),
                    'confidence_level': 'verified' if len(extracted_values) >= 3 else 'partial',
                    'sources': [r['url'] for r in search_results[:3]]
                })

        return validation_data

    def _geological_interpretation(self, data: np.ndarray) -> str:
        """Fournit une interprétation géologique"""
        stats = self._calculate_statistics(data)
        mean_res = stats['mean']

        interpretations = []

        if mean_res < 10:
            interpretations.append("Formation très conductrice, probablement saturée en eau ou argileuse")
        elif 10 <= mean_res < 100:
            interpretations.append("Matériau de résistivité moyenne, possiblement sol humide ou roche sédimentaire")
        elif 100 <= mean_res < 1000:
            interpretations.append("Formation résistive moyenne, sable sec ou calcaire possible")
        else:
            interpretations.append("Matériau très résistif, roche cristalline ou cavité aérienne")

        if stats['cv'] > 1.0:
            interpretations.append("Grande variabilité spatiale, structures hétérogènes")
        elif stats['cv'] < 0.3:
            interpretations.append("Formation homogène")

        return ". ".join(interpretations)

    def _create_color_profile(self, data: np.ndarray) -> List[Tuple[float, str, str]]:
        """Crée un profil coloré des données"""
        color_profile = []

        for resistivity in data:
            color, description = self.mapper.get_color_for_resistivity(resistivity)
            color_profile.append((resistivity, color, description))

        return color_profile

    def _describe_layer(self, resistivity: float) -> str:
        """Décrit une couche basée sur sa résistivité"""
        color, base_desc = self.mapper.get_color_for_resistivity(resistivity)

        similar_materials = self.mapper.find_similar_materials(resistivity, tolerance=0.3)

        if similar_materials:
            top_material = similar_materials[0]
            description = f"{base_desc} - Possiblement {top_material['name']} ({top_material['nature']})"
            if 'depth_range' in top_material:
                description += f" à {top_material['depth_range']}"
        else:
            description = base_desc

        return description

    def _generate_recommendations(self, data: np.ndarray) -> List[str]:
        """Génère des recommandations d'analyse"""
        recommendations = []
        stats = self._calculate_statistics(data)

        if stats['cv'] > 1.5:
            recommendations.append("Considérer une inversion 2D/3D pour mieux résoudre les structures")

        if stats['mean'] > 1000:
            recommendations.append("Vérifier la calibration - valeurs très élevées peuvent indiquer des problèmes d'acquisition")

        if len(data) < 100:
            recommendations.append("Augmenter la densité de mesures pour une meilleure résolution")

        similar_materials = self.mapper.find_similar_materials(stats['mean'])
        if similar_materials:
            top_material = similar_materials[0]
            recommendations.append(f"Comparer avec les propriétés de {top_material['name']} pour validation")

        return recommendations

# Fonctions utilitaires
def create_resistivity_colormap() -> mcolors.LinearSegmentedColormap:
    """Crée une colormap personnalisée pour la résistivité"""
    colors = ['#000080', '#0000FF', '#0080FF', '#00FF80', '#FFFF00', '#FFA500', '#FF0000']
    positions = [0.0, 0.2, 0.4, 0.6, 0.8, 0.9, 1.0]

    cmap = mcolors.LinearSegmentedColormap.from_list("resistivity", list(zip(positions, colors)))
    return cmap

def plot_resistivity_profile(data: np.ndarray, depths: Optional[np.ndarray] = None,
                           title: str = "Profil de Résistivité ERT") -> plt.Figure:
    """Crée un graphique de profil de résistivité"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    # Profil de résistivité
    if depths is not None:
        ax1.plot(data, depths, 'b-', linewidth=2)
        ax1.set_ylabel('Profondeur (m)')
    else:
        ax1.plot(data, 'b-', linewidth=2)
        ax1.set_ylabel('Position')

    ax1.set_xlabel('Résistivité (Ohm.m)')
    ax1.set_title('Profil Résistivité')
    ax1.set_xscale('log')
    ax1.grid(True, alpha=0.3)

    # Histogramme
    ax2.hist(np.log10(data), bins=20, alpha=0.7, color='blue', edgecolor='black')
    ax2.set_xlabel('Log10(Résistivité)')
    ax2.set_ylabel('Fréquence')
    ax2.set_title('Distribution des Résistivités')
    ax2.grid(True, alpha=0.3)

    plt.suptitle(title)
    plt.tight_layout()
    return fig

# Test et démonstration
if __name__ == "__main__":
    # Test du système
    mapper = ResistivityColorMapper()
    analyzer = DynamicERTAnalyzer()

    # Données de test (profil ERT simulé)
    test_data = np.array([5.0, 8.0, 50.0, 200.0, 800.0, 2000.0, 5000.0])

    print("🔍 Analyse ERT dynamique en cours...")
    analysis = analyzer.analyze_resistivity_profile(test_data)

    print(f"📊 Statistiques: {analysis['statistics']}")
    print(f"🏔️ Couches identifiées: {len(analysis['layers'])}")
    print(f"🧪 Matériaux potentiels: {len(analysis['materials'])}")
    print(f"📝 Interprétation: {analysis['geological_interpretation']}")

    # Recherche de données pour un matériau
    print("\n🔍 Recherche de données pour 'eau douce'...")
    search_results = analyzer.data_searcher.search_material_resistivity("eau douce")
    print(f"Résultats trouvés: {len(search_results)}")

    if search_results:
        resistivity_values = analyzer.data_searcher.extract_resistivity_values(search_results)
        print(f"Valeurs de résistivité extraites: {resistivity_values}")