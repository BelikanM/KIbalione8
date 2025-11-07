"""
MODULE D'ANALYSE INTELLIGENTE ERT POUR KIBALI
==============================================

Kibali utilise ce module pour analyser les données ERT et les rendre cohérentes
grâce à son intelligence géophysique et géologique.

Architecture:
1. Lecture données brutes ERT (.dat)
2. Validation cohérence géologique
3. Détection et correction d'anomalies
4. Enrichissement contextuel (localisation, géologie régionale)
5. Génération rapport cohérent et intelligent
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class IntelligentERTAnalyzer:
    """
    Analyseur intelligent pour données ERT
    Utilisé par Kibali pour rendre les données cohérentes
    """
    
    # Références géologiques (résistivité en Ohm.m)
    GEOLOGICAL_REFERENCES = {
        "eau_douce": (1, 100),
        "eau_salee": (0.2, 1),
        "argile": (1, 100),
        "sable_humide": (50, 500),
        "sable_sec": (500, 5000),
        "calcaire": (100, 1000),
        "granite": (1000, 10000),
        "gres": (100, 5000),
        "basalte": (1000, 100000),
    }
    
    # Contextes géographiques
    CONTEXTS = {
        "gabon": {
            "climat": "tropical_humide",
            "nappe_moyenne": (2, 10),  # profondeur en mètres
            "sols_typiques": ["argile_lateritique", "sable_argileux"],
            "resistivite_surface": (20, 200),  # Ohm.m
        },
        "sahel": {
            "climat": "semi_aride",
            "nappe_moyenne": (20, 80),
            "sols_typiques": ["sable", "argile_seche"],
            "resistivite_surface": (100, 1000),
        }
    }
    
    def __init__(self, context: str = "gabon"):
        """
        Initialise l'analyseur avec un contexte géographique
        
        Args:
            context: Contexte géographique ("gabon", "sahel", etc.)
        """
        self.context = self.CONTEXTS.get(context.lower(), self.CONTEXTS["gabon"])
        self.context_name = context
        logger.info(f"🌍 Analyseur initialisé - Contexte: {context}")
    
    def validate_stratigraphy(self, depths: np.ndarray, resistivities: np.ndarray) -> Dict:
        """
        Valide la cohérence stratigraphique des données
        
        Vérifie:
        - Progression logique avec la profondeur
        - Absence de sauts aberrants
        - Cohérence avec modèles géologiques connus
        
        Returns:
            Dict avec statut validation et anomalies détectées
        """
        logger.info("🔍 Validation stratigraphique...")
        
        anomalies = []
        warnings = []
        
        # 1. Vérifier ordre des profondeurs
        if not np.all(depths[:-1] <= depths[1:]):
            anomalies.append({
                "type": "ordre_profondeur",
                "severity": "critique",
                "message": "Les profondeurs ne sont pas en ordre croissant"
            })
        
        # 2. Détecter sauts aberrants de résistivité
        resistivity_changes = np.diff(resistivities)
        max_change_ratio = 10  # Facteur 10 max entre couches adjacentes
        
        for i, change in enumerate(resistivity_changes):
            if abs(change) > 0:  # Éviter division par zéro
                ratio = abs(resistivities[i+1] / resistivities[i]) if resistivities[i] != 0 else float('inf')
                if ratio > max_change_ratio:
                    anomalies.append({
                        "type": "saut_aberrant",
                        "severity": "elevee",
                        "profondeur": depths[i],
                        "valeur_avant": resistivities[i],
                        "valeur_apres": resistivities[i+1],
                        "ratio": ratio,
                        "message": f"Saut de résistivité anormal à {depths[i]}m (ratio {ratio:.1f}x)"
                    })
        
        # 3. Vérifier cohérence avec surface attendue (contexte)
        if len(resistivities) > 0:
            surface_res = resistivities[0]
            expected_min, expected_max = self.context["resistivite_surface"]
            
            if not (expected_min <= surface_res <= expected_max):
                warnings.append({
                    "type": "surface_inhabituelle",
                    "severity": "moyenne",
                    "valeur": surface_res,
                    "attendu": (expected_min, expected_max),
                    "message": f"Résistivité de surface ({surface_res:.1f} Ω.m) inhabituelle pour {self.context_name}"
                })
        
        # 4. Détecter inversions stratigraphiques (plus résistant au-dessus)
        for i in range(len(resistivities) - 1):
            # Argile sous sable est logique, inverse est suspect
            if resistivities[i] > 1000 and resistivities[i+1] < 100:
                warnings.append({
                    "type": "inversion_possible",
                    "severity": "moyenne",
                    "profondeur": depths[i],
                    "message": f"Roche dure ({resistivities[i]:.0f} Ω.m) sur argile ({resistivities[i+1]:.0f} Ω.m) à {depths[i]}m - Vérifier"
                })
        
        is_valid = len(anomalies) == 0
        
        return {
            "valid": is_valid,
            "anomalies": anomalies,
            "warnings": warnings,
            "score_coherence": max(0, 100 - len(anomalies) * 30 - len(warnings) * 10)
        }
    
    def detect_and_correct_outliers(self, resistivities: np.ndarray, threshold: float = 3.0) -> Tuple[np.ndarray, List[Dict]]:
        """
        Détecte et corrige les valeurs aberrantes avec intelligence
        
        Utilise méthode statistique (Z-score) + connaissances géologiques
        
        Args:
            resistivities: Valeurs de résistivité
            threshold: Seuil Z-score (3.0 = 99.7% données normales)
        
        Returns:
            (données_corrigées, corrections_appliquées)
        """
        logger.info("🔧 Détection et correction d'anomalies...")
        
        corrected = resistivities.copy()
        corrections = []
        
        # Calcul Z-score
        mean = np.mean(resistivities)
        std = np.std(resistivities)
        
        if std == 0:
            return corrected, []
        
        z_scores = np.abs((resistivities - mean) / std)
        
        # Détecter outliers
        outliers = z_scores > threshold
        
        for i, is_outlier in enumerate(outliers):
            if is_outlier:
                old_value = resistivities[i]
                
                # Correction intelligente : moyenne des voisins ou médiane locale
                if i == 0:
                    # Premier point : utiliser suivant
                    new_value = resistivities[1] if len(resistivities) > 1 else mean
                elif i == len(resistivities) - 1:
                    # Dernier point : utiliser précédent
                    new_value = resistivities[i-1]
                else:
                    # Point intermédiaire : moyenne des voisins
                    new_value = (resistivities[i-1] + resistivities[i+1]) / 2
                
                corrected[i] = new_value
                
                corrections.append({
                    "index": i,
                    "valeur_originale": old_value,
                    "valeur_corrigee": new_value,
                    "z_score": z_scores[i],
                    "raison": f"Valeur aberrante détectée (Z-score={z_scores[i]:.2f})"
                })
        
        logger.info(f"✅ {len(corrections)} valeurs corrigées")
        return corrected, corrections
    
    def identify_layers(self, depths: np.ndarray, resistivities: np.ndarray) -> List[Dict]:
        """
        Identifie les couches géologiques à partir des données
        
        Utilise gradient de résistivité pour détecter changements de couche
        
        Returns:
            Liste de couches avec profondeur, type, résistivité
        """
        logger.info("🪨 Identification des couches géologiques...")
        
        layers = []
        
        # Détecter changements significatifs (gradient > 30%)
        threshold_change = 0.3
        
        current_layer_start = 0
        current_layer_res = resistivities[0]
        
        for i in range(1, len(resistivities)):
            change_ratio = abs(resistivities[i] - current_layer_res) / current_layer_res if current_layer_res != 0 else 1
            
            if change_ratio > threshold_change:
                # Fin de couche précédente
                layer_avg_res = np.mean(resistivities[current_layer_start:i])
                layer_type = self._classify_material(layer_avg_res)
                
                layers.append({
                    "profondeur_debut": depths[current_layer_start],
                    "profondeur_fin": depths[i-1],
                    "epaisseur": depths[i-1] - depths[current_layer_start],
                    "resistivite_moyenne": layer_avg_res,
                    "type_geologique": layer_type,
                    "description": self._get_layer_description(layer_type, layer_avg_res)
                })
                
                # Nouvelle couche
                current_layer_start = i
                current_layer_res = resistivities[i]
        
        # Dernière couche
        layer_avg_res = np.mean(resistivities[current_layer_start:])
        layer_type = self._classify_material(layer_avg_res)
        
        layers.append({
            "profondeur_debut": depths[current_layer_start],
            "profondeur_fin": depths[-1],
            "epaisseur": depths[-1] - depths[current_layer_start],
            "resistivite_moyenne": layer_avg_res,
            "type_geologique": layer_type,
            "description": self._get_layer_description(layer_type, layer_avg_res)
        })
        
        logger.info(f"✅ {len(layers)} couches identifiées")
        return layers
    
    def _classify_material(self, resistivity: float) -> str:
        """Classifie le matériau géologique selon résistivité"""
        for material, (min_r, max_r) in self.GEOLOGICAL_REFERENCES.items():
            if min_r <= resistivity <= max_r:
                return material
        
        if resistivity < 1:
            return "eau_salee_ou_argile_saturee"
        elif resistivity > 10000:
            return "roche_tres_resistante"
        else:
            return "materiau_intermediaire"
    
    def _get_layer_description(self, layer_type: str, resistivity: float) -> str:
        """Génère description intelligente de la couche"""
        descriptions = {
            "eau_douce": f"Couche aquifère ou formation très saturée ({resistivity:.1f} Ω.m)",
            "argile": f"Argile ou formation imperméable ({resistivity:.1f} Ω.m)",
            "sable_humide": f"Sable saturé ou formation perméable ({resistivity:.1f} Ω.m)",
            "sable_sec": f"Sable sec, faible teneur en eau ({resistivity:.1f} Ω.m)",
            "calcaire": f"Calcaire ou formation carbonatée ({resistivity:.1f} Ω.m)",
            "granite": f"Granite ou roche plutonique ({resistivity:.1f} Ω.m)",
            "gres": f"Grès ou formation détritique ({resistivity:.1f} Ω.m)",
        }
        return descriptions.get(layer_type, f"Formation de résistivité {resistivity:.1f} Ω.m")
    
    def analyze_hydrogeology(self, depths: np.ndarray, resistivities: np.ndarray, layers: List[Dict]) -> Dict:
        """
        Analyse hydrogéologique intelligente
        
        Détecte:
        - Zones aquifères probables
        - Profondeur nappe phréatique
        - Potentiel hydrique
        - Recommandations forages
        """
        logger.info("💧 Analyse hydrogéologique...")
        
        # Détecter zones à faible résistivité (eau)
        water_zones = []
        for i, res in enumerate(resistivities):
            if res < 100:  # Potentiellement saturé
                water_zones.append({
                    "profondeur": depths[i],
                    "resistivite": res,
                    "probabilite_eau": min(100, (100 - res) * 2)  # Plus faible = plus probable
                })
        
        # Estimer profondeur nappe
        nappe_depth = None
        if water_zones:
            # Première zone à très faible résistivité
            for zone in water_zones:
                if zone["resistivite"] < 50:
                    nappe_depth = zone["profondeur"]
                    break
        
        # Évaluer potentiel hydrique
        aquifer_layers = [l for l in layers if "sable" in l["type_geologique"].lower() or "eau" in l["type_geologique"].lower()]
        
        potential = "faible"
        if len(aquifer_layers) >= 2:
            potential = "excellent"
        elif len(aquifer_layers) == 1:
            total_thickness = sum(l["epaisseur"] for l in aquifer_layers)
            potential = "bon" if total_thickness > 5 else "moyen"
        
        # Recommandations forages
        recommendations = []
        if nappe_depth and nappe_depth < 15:
            recommendations.append(f"Forage peu profond recommandé à {nappe_depth:.1f}m")
        
        if aquifer_layers:
            best_layer = max(aquifer_layers, key=lambda l: l["epaisseur"])
            recommendations.append(
                f"Zone aquifère optimale: {best_layer['profondeur_debut']:.1f}-{best_layer['profondeur_fin']:.1f}m "
                f"(épaisseur {best_layer['epaisseur']:.1f}m)"
            )
        
        return {
            "zones_eau": water_zones,
            "profondeur_nappe_estimee": nappe_depth,
            "potentiel_hydrique": potential,
            "couches_aquiferes": aquifer_layers,
            "recommandations": recommendations
        }
    
    def generate_intelligent_report(self, depths: np.ndarray, resistivities: np.ndarray) -> Dict:
        """
        Génère un rapport complet et cohérent avec intelligence Kibali
        
        Pipeline complet:
        1. Validation stratigraphique
        2. Correction anomalies
        3. Identification couches
        4. Analyse hydrogéologique
        5. Synthèse intelligente
        """
        logger.info("📊 Génération rapport intelligent...")
        
        # 1. Validation
        validation = self.validate_stratigraphy(depths, resistivities)
        
        # 2. Correction
        corrected_resistivities, corrections = self.detect_and_correct_outliers(resistivities)
        
        # 3. Identification couches
        layers = self.identify_layers(depths, corrected_resistivities)
        
        # 4. Analyse hydro
        hydrogeology = self.analyze_hydrogeology(depths, corrected_resistivities, layers)
        
        # 5. Synthèse intelligente
        synthesis = self._generate_synthesis(validation, corrections, layers, hydrogeology)
        
        return {
            "contexte": self.context_name,
            "validation_stratigraphique": validation,
            "corrections_appliquees": corrections,
            "couches_geologiques": layers,
            "analyse_hydrogeologique": hydrogeology,
            "synthese_intelligente": synthesis,
            "donnees_corrigees": {
                "profondeurs": depths.tolist(),
                "resistivites_originales": resistivities.tolist(),
                "resistivites_corrigees": corrected_resistivities.tolist()
            }
        }
    
    def _generate_synthesis(self, validation: Dict, corrections: List, layers: List, hydro: Dict) -> str:
        """Génère synthèse textuelle intelligente"""
        
        parts = []
        
        # En-tête contexte
        parts.append(f"## 🌍 Analyse ERT - Contexte {self.context_name.upper()}\n")
        
        # Validation
        if validation["valid"]:
            parts.append(f"✅ **Données cohérentes** (score: {validation['score_coherence']}/100)")
        else:
            parts.append(f"⚠️ **Anomalies détectées** (score: {validation['score_coherence']}/100)")
            for anom in validation["anomalies"]:
                parts.append(f"  - {anom['message']}")
        
        # Corrections
        if corrections:
            parts.append(f"\n🔧 **{len(corrections)} corrections appliquées** par l'intelligence Kibali")
        
        # Stratigraphie
        parts.append(f"\n## 🪨 Stratigraphie identifiée ({len(layers)} couches)\n")
        for i, layer in enumerate(layers, 1):
            parts.append(
                f"{i}. **{layer['profondeur_debut']:.1f}-{layer['profondeur_fin']:.1f}m** "
                f"({layer['epaisseur']:.1f}m) - {layer['description']}"
            )
        
        # Hydrogéologie
        parts.append(f"\n## 💧 Potentiel hydrogéologique: **{hydro['potentiel_hydrique'].upper()}**\n")
        if hydro['profondeur_nappe_estimee']:
            parts.append(f"🎯 Nappe phréatique estimée: **{hydro['profondeur_nappe_estimee']:.1f}m**")
        
        if hydro['recommandations']:
            parts.append("\n### 📋 Recommandations:")
            for rec in hydro['recommandations']:
                parts.append(f"  - {rec}")
        
        return "\n".join(parts)


# Fonction utilitaire pour Kibali
def kibali_analyze_ert(depths: List[float], resistivities: List[float], context: str = "gabon") -> Dict:
    """
    Fonction d'analyse ERT pour Kibali
    
    Usage:
        results = kibali_analyze_ert([0, 5, 10, 15], [45, 78, 125, 245], "gabon")
        print(results["synthese_intelligente"])
    """
    analyzer = IntelligentERTAnalyzer(context=context)
    return analyzer.generate_intelligent_report(np.array(depths), np.array(resistivities))
