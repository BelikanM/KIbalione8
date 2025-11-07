#!/usr/bin/env python3
"""
OUTIL D'ANALYSE DES VALEURS TYPIQUES D'EAU POUR LES COUPES ERT
Intègre les résistivités caractéristiques de l'eau souterraine
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt

class WaterResistivityAnalyzer:
    """
    Analyseur des valeurs de résistivité typiques pour l'eau dans les études ERT
    Utilisé pour interpréter les coupes géophysiques
    """
    
    def __init__(self):
        # Valeurs typiques de résistivité pour l'eau (en Ω·m)
        self.water_resistivity_ranges = {
            'eau_ultra_pure': {'min': 1.8e5, 'max': 1.8e6, 'description': 'Eau ultra-pure (laboratoire)'},
            'eau_distillee': {'min': 5e4, 'max': 1e5, 'description': 'Eau distillée'},
            'eau_pluie': {'min': 20, 'max': 100, 'description': 'Eau de pluie (contaminée)'},
            'eau_douce': {'min': 10, 'max': 100, 'description': 'Eau douce souterraine'},
            'eau_saumatre': {'min': 1, 'max': 10, 'description': 'Eau saumâtre'},
            'eau_salee': {'min': 0.1, 'max': 1, 'description': 'Eau salée'},
            'eau_brine': {'min': 0.01, 'max': 0.1, 'description': 'Eau très salée (brine)'},
            'eau_thermale': {'min': 0.5, 'max': 5, 'description': 'Eau thermale minéralisée'},
            'eau_polluee': {'min': 0.1, 'max': 5, 'description': 'Eau polluée/industrielle'}
        }
        
        # Facteurs d'influence sur la résistivité
        self.influence_factors = {
            'temperature': {
                'description': 'Température de l\'eau',
                'effect': 'Augmente la résistivité quand T↓',
                'typical_range': '5-25°C'
            },
            'salinite': {
                'description': 'Teneur en sels dissous',
                'effect': 'Diminue fortement la résistivité',
                'typical_range': '0-300 g/L'
            },
            'minerals': {
                'description': 'Minéraux dissous (Ca²⁺, Mg²⁺, Na⁺, etc.)',
                'effect': 'Diminue la résistivité',
                'typical_range': 'Variable'
            },
            'ph': {
                'description': 'pH de l\'eau',
                'effect': 'Influence la conductivité ionique',
                'typical_range': '4-9'
            },
            'pression': {
                'description': 'Pression hydrostatique',
                'effect': 'Légère influence sur la résistivité',
                'typical_range': '1-100 bar'
            }
        }
    
    def classify_water_type(self, resistivity: float) -> Dict:
        """
        Classifie le type d'eau basé sur sa résistivité
        
        Args:
            resistivity: Résistivité en Ω·m
            
        Returns:
            Dict avec classification et informations
        """
        for water_type, range_info in self.water_resistivity_ranges.items():
            if range_info['min'] <= resistivity <= range_info['max']:
                return {
                    'type': water_type,
                    'description': range_info['description'],
                    'resistivity_range': f"{range_info['min']:.1e} - {range_info['max']:.1e} Ω·m",
                    'confidence': 'high' if resistivity >= range_info['min'] * 0.5 and resistivity <= range_info['max'] * 2 else 'medium'
                }
        
        # Si hors des plages connues
        if resistivity > 1e3:
            return {
                'type': 'eau_tres_pure',
                'description': 'Eau très pure ou isolant',
                'resistivity_range': '> 1000 Ω·m',
                'confidence': 'low'
            }
        else:
            return {
                'type': 'conducteur_fort',
                'description': 'Conducteur fort (minerais métalliques?)',
                'resistivity_range': '< 0.01 Ω·m',
                'confidence': 'low'
            }
    
    def get_water_interpretation_guide(self) -> str:
        """
        Retourne un guide d'interprétation pour les valeurs d'eau dans les coupes ERT
        """
        guide = """
╔══════════════════════════════════════════════════════════════════════════╗
║              GUIDE D'INTERPRÉTATION - EAU DANS LES COUPES ERT                ║
╚══════════════════════════════════════════════════════════════════════════╝

🎯 VALEURS TYPIQUES DE RÉSISTIVITÉ POUR L'EAU SOUTERRAINE (Ω·m):

   🔵 EAU TRÈS PURE     : 180,000 - 1,800,000  (eau ultra-pure, laboratoire)
   🔵 EAU DISTILLÉE     :  50,000 - 100,000    (eau distillée)
   🔵 EAU DE PLUIE      :      20 - 100        (contaminée par CO₂)
   🔵 EAU DOUCE         :      10 - 100        (nappes phréatiques)
   🟢 EAU SAUMÂTRE      :       1 - 10         (estuaires, côtes)
   🟡 EAU SALÉE         :     0.1 - 1          (mers, océans)
   🔴 EAU BRINE         :   0.01 - 0.1         (très salée, minière)
   🟠 EAU THERmale      :     0.5 - 5          (sources thermales)
   ⚠️  EAU POLLUÉE      :     0.1 - 5          (industrielle, agricole)

📊 INTERPRÉTATION GÉOLOGIQUE DES COULEURS:

   • BLEU FONCÉ (ρ > 100 Ω·m)  → Zones sèches, aquifères pauvres en eau
   • BLEU CLAIR (50-100 Ω·m)   → Sols sableux, eaux douces diluées
   • VERT (10-50 Ω·m)         → Argiles, eaux saumâtres
   • JAUNE (1-10 Ω·m)         → Sols argileux humides, eaux salées
   • ORANGE (0.1-1 Ω·m)       → Zones très conductrices, eaux très salées
   • ROUGE (ρ < 0.1 Ω·m)      → Minerais conducteurs, fluides très minéralisés

⚠️  FACTEURS INFLUENÇANT LA RÉSISTIVITÉ:

   • TEMPÉRATURE: ↑T = ↓ρ (résistivité diminue avec température)
   • SALINITÉ: ↑Salinité = ↓ρ (plus de sels = plus conducteur)
   • MINÉRAUX: Ca²⁺, Mg²⁺, Na⁺, Cl⁻, SO₄²⁻ diminuent ρ
   • pH: Influence la dissociation ionique
   • PRESSION: Effet mineur sur la résistivité

🔍 APPLICATIONS PRATIQUES:

   • DÉTECTION DE NAPPES: Zones bleues = aquifères potentiels
   • POLLUTION: Chute brutale de ρ = contamination saline
   • SOURCES THERmales: Anomalies locales en zones volcaniques
   • KARSTS: Alternance rapide ρ = dissolution calcaire

╚══════════════════════════════════════════════════════════════════════════╝
"""
        return guide
    
    def analyze_resistivity_profile(self, resistivity_values: np.ndarray) -> Dict:
        """
        Analyse un profil de résistivité pour identifier les zones d'eau
        
        Args:
            resistivity_values: Array des valeurs de résistivité
            
        Returns:
            Dict avec analyse détaillée
        """
        analysis = {
            'statistics': {
                'mean': float(np.mean(resistivity_values)),
                'median': float(np.median(resistivity_values)),
                'std': float(np.std(resistivity_values)),
                'min': float(np.min(resistivity_values)),
                'max': float(np.max(resistivity_values)),
                'range': float(np.max(resistivity_values) - np.min(resistivity_values))
            },
            'water_zones': [],
            'interpretation': []
        }
        
        # Classifier chaque valeur
        for i, rho in enumerate(resistivity_values):
            classification = self.classify_water_type(rho)
            analysis['water_zones'].append({
                'index': i,
                'resistivity': rho,
                'classification': classification
            })
        
        # Analyse globale
        mean_rho = analysis['statistics']['mean']
        if mean_rho > 100:
            analysis['interpretation'].append("Profil majoritairement sec ou avec eaux très diluées")
        elif mean_rho > 10:
            analysis['interpretation'].append("Présence d'eaux douces à saumâtres")
        elif mean_rho > 1:
            analysis['interpretation'].append("Zonage avec eaux salées ou argiles conductrices")
        else:
            analysis['interpretation'].append("Fortes anomalies conductrices: eaux très salées ou minerais")
        
        # Détection d'anomalies
        std_rho = analysis['statistics']['std']
        if std_rho > mean_rho * 0.5:
            analysis['interpretation'].append("Forte variabilité: interfaces géologiques contrastées")
        
        return analysis
    
    def create_water_legend(self) -> str:
        """
        Crée une légende colorée pour les valeurs d'eau
        """
        legend = """
🌊 LÉGENDE DES VALEURS D'EAU POUR LES COUPES ERT

╔══════════════════════════════════════════════════════════════╗
║   COULEUR    │ RÉSISTIVITÉ │         TYPE D'EAU              ║
╠══════════════════════════════════════════════════════════════╣
║ 🔵 Bleu foncé │  ρ > 100 Ω·m │ Eau très pure/diluée          ║
║ 🔵 Bleu clair  │ 50-100 Ω·m  │ Eau douce souterraine         ║
║ 🟢 Vert        │ 10-50 Ω·m   │ Eau saumâtre/modérément salée  ║
║ 🟡 Jaune       │  1-10 Ω·m   │ Eau salée/argiles humides      ║
║ 🟠 Orange      │ 0.1-1 Ω·m   │ Eau très salée/brine           ║
║ 🔴 Rouge       │  ρ < 0.1 Ω·m│ Fluides très conducteurs       ║
╚══════════════════════════════════════════════════════════════╝

💡 CONSEILS D'INTERPRÉTATION:
• Les zones BLEUES indiquent des aquifères potentiels
• Les zones ROUGES peuvent signaler des pollutions ou minéralisations
• Les transitions brutales = interfaces géologiques
• La profondeur influence la température et donc la résistivité
"""
        return legend
    
    def get_typical_water_values_table(self) -> pd.DataFrame:
        """
        Retourne un tableau des valeurs typiques d'eau
        """
        data = []
        for water_type, info in self.water_resistivity_ranges.items():
            data.append({
                'Type_Eau': water_type.replace('_', ' ').title(),
                'Resistivite_Min_Ohm_m': info['min'],
                'Resistivite_Max_Ohm_m': info['max'],
                'Description': info['description']
            })
        
        return pd.DataFrame(data)

# Fonction d'intégration avec le parseur ERT existant
def integrate_water_analysis_with_ert(parser_instance, resistivity_threshold: float = 10.0) -> Dict:
    """
    Intègre l'analyse d'eau avec un parseur ERT existant
    
    Args:
        parser_instance: Instance du SurveyDepthDataParser
        resistivity_threshold: Seuil pour considérer comme zone d'eau (Ω·m)
    
    Returns:
        Dict avec analyse intégrée
    """
    if parser_instance.data is None:
        return {"error": "Aucune donnée chargée dans le parseur"}
    
    water_analyzer = WaterResistivityAnalyzer()
    
    # Analyser toutes les valeurs de résistivité
    all_resistivities = parser_instance.data['data'].values
    water_analysis = water_analyzer.analyze_resistivity_profile(all_resistivities)
    
    # Identifier les zones d'eau potentielles
    water_zones = parser_instance.data[parser_instance.data['data'] <= resistivity_threshold]
    
    integrated_analysis = {
        'water_analysis': water_analysis,
        'potential_water_zones': {
            'count': len(water_zones),
            'percentage': len(water_zones) / len(parser_instance.data) * 100,
            'locations': water_zones[['survey_point', 'depth', 'data']].to_dict('records')
        },
        'interpretation_guide': water_analyzer.get_water_interpretation_guide(),
        'water_legend': water_analyzer.create_water_legend()
    }
    
    return integrated_analysis

# Test de l'outil
if __name__ == "__main__":
    analyzer = WaterResistivityAnalyzer()
    
    print("🧪 TEST DE L'OUTIL D'ANALYSE D'EAU POUR LES COUPES ERT")
    print("=" * 60)
    
    # Test de classification
    test_values = [0.05, 2.5, 25, 150, 50000]
    print("\\n📊 CLASSIFICATION DES VALEURS DE RÉSISTIVITÉ:")
    for rho in test_values:
        result = analyzer.classify_water_type(rho)
        print(f"   {rho:8.1f} Ω·m → {result['type']:15} ({result['description']})")
    
    print("\\n📋 GUIDE D'INTERPRÉTATION:")
    print(analyzer.get_water_interpretation_guide()[:500] + "...")
    
    print("\\n🎨 LÉGENDE DES COULEURS:")
    print(analyzer.create_water_legend())
    
    print("\\n✅ OUTIL D'ANALYSE D'EAU INTÉGRÉ AVEC SUCCÈS !")