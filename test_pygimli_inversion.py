#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test complet PyGIMLi ERT Inversion Tool
"""

from multi_freq_ert_parser import MultiFreqERTParser
from pygimli_ert_tool import PyGIMLiERTInversionTool, process_ert_data_complete

print("╔"+"═"*68+"╗")
print("║" + " "*15 + "TEST PYGIMLI ERT INVERSION TOOL" + " "*22 + "║")
print("╚"+"═"*68+"╝\n")

# Étape 1: Parser les données avec le parser existant
print("📄 ÉTAPE 1: Parsing des données multi-profondeurs...")
parser = MultiFreqERTParser()
df = parser.parse_file('frequ_multi_depth.dat')

print(f"✅ {len(df)} mesures parsées")
print(f"   • Survey points: {sorted(df['survey_point'].unique())}")
print(f"   • Profondeurs: {sorted(df['depth'].unique())}")
print(f"   • Résistivité: {df['resistivity'].min():.4f} - {df['resistivity'].max():.4f} Ω·m")

# Étape 2: Créer l'outil PyGIMLi
print("\n📊 ÉTAPE 2: Création schéma ERT PyGIMLi...")
tool = PyGIMLiERTInversionTool()
ert_data = tool.create_ert_data_from_measurements(df, scheme_type='dd')

# Étape 3: INVERSION pour obtenir résistivités VRAIES
print("\n🔬 ÉTAPE 3: INVERSION ERT...")
model = tool.run_inversion(lam=20, verbose=False)

# Étape 4: Générer toutes les coupes avec résistivités INVERSÉES
print("\n🎨 ÉTAPE 4: Génération des coupes...")
outputs = tool.generate_all_sections(
    output_dir='/tmp/pygimli_inversion_test',
    prefix='test_inversion'
)

print("\n" + "="*70)
print("✅ TEST TERMINÉ AVEC SUCCÈS")
print("="*70)
print(f"\n📁 Fichiers générés:")
for key, path in outputs.items():
    print(f"   • {key}: {path}")

print(f"\n📊 Qualité de l'inversion:")
print(f"   • RMS (misfit): {tool.manager.inv.absrms():.3f}")
print(f"   • Résistivité min (vraie): {min(model):.4f} Ω·m")
print(f"   • Résistivité max (vraie): {max(model):.4f} Ω·m")
print(f"   • Cellules maillage: {tool.mesh.cellCount()}")

print("\n🎉 Les coupes affichent maintenant les VRAIES résistivités")
print("   (après correction des effets géométriques)")
