#!/usr/bin/env python3
"""Test du module d'analyse intelligente ERT"""

from intelligent_ert_analyzer import kibali_analyze_ert
import json

print("=" * 70)
print("TEST ANALYSE INTELLIGENTE ERT POUR KIBALI")
print("=" * 70)

# Données de test : Projet Archange Ondimba 2
depths = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
resistivities = [45.2, 78.3, 125.4, 245.6, 198.7, 89.3, 1250.5, 2340.8, 1980.3, 2105.6, 2450.9]

print(f"\n📊 Données d'entrée:")
print(f"   Profondeurs: {depths}")
print(f"   Résistivités: {resistivities}")

print(f"\n🚀 Lancement analyse intelligente Kibali...\n")

# Analyse avec contexte Gabon
results = kibali_analyze_ert(depths, resistivities, context="gabon")

# Afficher synthèse
print("\n" + "=" * 70)
print(results["synthese_intelligente"])
print("=" * 70)

# Détails validation
print("\n📋 VALIDATION STRATIGRAPHIQUE")
print("-" * 70)
validation = results["validation_stratigraphique"]
print(f"Statut: {'✅ VALIDE' if validation['valid'] else '⚠️ ANOMALIES'}")
print(f"Score cohérence: {validation['score_coherence']}/100")

if validation['anomalies']:
    print(f"\n❌ Anomalies critiques:")
    for anom in validation['anomalies']:
        print(f"   - {anom['message']}")

if validation['warnings']:
    print(f"\n⚠️  Avertissements:")
    for warn in validation['warnings']:
        print(f"   - {warn['message']}")

# Corrections
print("\n🔧 CORRECTIONS APPLIQUÉES")
print("-" * 70)
corrections = results["corrections_appliquees"]
if corrections:
    for corr in corrections:
        print(f"   Index {corr['index']}: {corr['valeur_originale']:.1f} → {corr['valeur_corrigee']:.1f} Ω.m")
        print(f"     Raison: {corr['raison']}")
else:
    print("   ✅ Aucune correction nécessaire")

# Couches géologiques
print("\n🪨 COUCHES GÉOLOGIQUES IDENTIFIÉES")
print("-" * 70)
for i, layer in enumerate(results["couches_geologiques"], 1):
    print(f"\n   Couche {i}:")
    print(f"   Profondeur: {layer['profondeur_debut']:.1f}m → {layer['profondeur_fin']:.1f}m")
    print(f"   Épaisseur: {layer['epaisseur']:.1f}m")
    print(f"   Type: {layer['type_geologique']}")
    print(f"   Résistivité moyenne: {layer['resistivite_moyenne']:.1f} Ω.m")
    print(f"   Description: {layer['description']}")

# Hydrogéologie
print("\n💧 ANALYSE HYDROGÉOLOGIQUE")
print("-" * 70)
hydro = results["analyse_hydrogeologique"]
print(f"Potentiel hydrique: {hydro['potentiel_hydrique'].upper()}")
if hydro['profondeur_nappe_estimee']:
    print(f"Nappe phréatique estimée: {hydro['profondeur_nappe_estimee']:.1f}m")
else:
    print("Nappe phréatique: Profonde ou non détectée")

print(f"\nZones aquifères potentielles: {len(hydro['couches_aquiferes'])}")
for aquifer in hydro['couches_aquiferes']:
    print(f"   - {aquifer['profondeur_debut']:.1f}-{aquifer['profondeur_fin']:.1f}m : {aquifer['type_geologique']}")

if hydro['recommandations']:
    print(f"\n📋 Recommandations:")
    for rec in hydro['recommandations']:
        print(f"   ✓ {rec}")

# Test avec données aberrantes
print("\n\n" + "=" * 70)
print("TEST 2: DONNÉES AVEC ANOMALIES")
print("=" * 70)

depths2 = [0, 5, 10, 15, 20, 25, 30]
resistivities2 = [45.2, 78.3, 9999.0, 245.6, 198.7, 15.2, 2340.8]  # Valeur aberrante à 10m

print(f"\n📊 Données avec anomalie:")
print(f"   Résistivités: {resistivities2}")
print(f"   ⚠️  Valeur 9999.0 Ω.m à 10m est aberrante pour contexte tropical\n")

results2 = kibali_analyze_ert(depths2, resistivities2, context="gabon")

print("\n" + results2["synthese_intelligente"])

print("\n✅ Tests terminés!")
print("=" * 70)
