#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Génération de coupes ERT PyGIMLi avec les VRAIES profondeurs
Analyse du fichier frequ_multi_depth.dat
"""

from multi_freq_ert_parser import MultiFreqERTParser
from pygimli_ert_tool import PyGIMLiERTInversionTool
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

print("╔" + "═"*68 + "╗")
print("║" + " "*10 + "ANALYSE ERT AVEC PROFONDEURS RÉELLES" + " "*21 + "║")
print("╚" + "═"*68 + "╝\n")

# Charger les données
print("📄 Lecture du fichier frequ_multi_depth.dat...")
df = pd.read_csv('frequ_multi_depth.dat', sep='\t')

print(f"\n✅ {len(df)} mesures chargées")
print(f"\n📊 STRUCTURE DES DONNÉES:")
print(f"   • Survey points: {sorted(df['survey-point'].unique())}")
print(f"   • Profondeurs (m): {sorted(df['depth'].unique())}")
print(f"   • Résistivité min: {df['data'].min():.4f} Ω·m")
print(f"   • Résistivité max: {df['data'].max():.4f} Ω·m")
print(f"   • Projet: {df['project'].iloc[0]}")

# Afficher échantillon des données
print(f"\n📋 APERÇU DES DONNÉES (10 premières lignes):")
print("┌─────────────┬───────────┬──────────────┬──────────┐")
print("│ Survey Pt   │ Depth (m) │ Résistivité  │ Project  │")
print("├─────────────┼───────────┼──────────────┼──────────┤")
for idx, row in df.head(10).iterrows():
    print(f"│ {row['survey-point']:>11} │ {row['depth']:>9.0f} │ {row['data']:>12.4f} │ {row['project']:>8} │")
print("└─────────────┴───────────┴──────────────┴──────────┘")

# Préparer DataFrame pour PyGIMLi avec profondeurs continues
print(f"\n🔧 Préparation des profils de profondeur continus...")
print(f"   Chaque survey point: -2m → -10m → -20m → -50m → -100m")

# Vérifier que chaque survey point a toutes les profondeurs
survey_points = sorted(df['survey-point'].unique())
depths = sorted(df['depth'].unique(), reverse=True)  # De -2 à -100

print(f"\n📏 PROFILS VERTICAUX PAR SURVEY POINT:")
for sp in survey_points:
    sp_data = df[df['survey-point'] == sp].sort_values('depth', ascending=False)
    print(f"\n   SP {sp}: ", end="")
    for d in depths:
        sp_d = sp_data[sp_data['depth'] == d]
        if len(sp_d) > 0:
            print(f"{d}m({sp_d['data'].values[0]:.3f}Ω·m) → ", end="")
        else:
            print(f"{d}m(--) → ", end="")
    print("FIN")

df_prepared = pd.DataFrame({
    'survey_point': df['survey-point'],
    'depth': df['depth'],
    'resistivity': df['data'],
    'project': df['project']
})

# Créer l'outil PyGIMLi
tool = PyGIMLiERTInversionTool()

print(f"\n🎯 Création du schéma ERT avec profondeurs continues...")
print(f"   Chaque survey point traverse {len(depths)} niveaux de profondeur")
ert_data = tool.create_ert_data_from_measurements(df_prepared, scheme_type='dd')

# Afficher matrice de couverture
print(f"\n📐 MATRICE DE COUVERTURE:")
print("   Survey Points (X) vs Profondeurs (Z):\n")
print("   Depth\\SP │ ", end="")
for sp in sorted(df['survey-point'].unique()):
    print(f"{sp:>4}", end=" │ ")
print()
print("   " + "─"*70)

for depth in sorted(df['depth'].unique()):
    print(f"   {depth:>7.0f}m │ ", end="")
    for sp in sorted(df['survey-point'].unique()):
        count = len(df[(df['survey-point'] == sp) & (df['depth'] == depth)])
        if count > 0:
            res = df[(df['survey-point'] == sp) & (df['depth'] == depth)]['data'].values[0]
            print(f"{res:.2f}", end=" │ ")
        else:
            print("  --  ", end=" │ ")
    print()

# INVERSION PyGIMLi
print(f"\n" + "="*70)
print("🚀 LANCEMENT INVERSION ERT PYGIMLI")
print("="*70)
print("Calcul des résistivités VRAIES du sous-sol...")
print("(Profondeurs: -2m, -10m, -20m, -50m, -100m)\n")

model = tool.run_inversion(lam=20, verbose=False)

# Générer les coupes
print(f"\n🎨 GÉNÉRATION DES COUPES ERT...")
outputs = tool.generate_all_sections(
    output_dir='/tmp/ert_vraies_profondeurs',
    prefix='frequ_multi_depth'
)

print(f"\n" + "="*70)
print("✅ ANALYSE TERMINÉE")
print("="*70)

print(f"\n📈 RÉSULTATS DE L'INVERSION:")
print(f"   • Résistivité min (inversée): {min(model):.4f} Ω·m")
print(f"   • Résistivité max (inversée): {max(model):.4f} Ω·m")
print(f"   • RMS (erreur): {tool.manager.inv.absrms():.4f}")
print(f"   • Cellules du maillage: {tool.mesh.cellCount()}")

print(f"\n📁 FICHIERS GÉNÉRÉS:")
for key, path in outputs.items():
    print(f"   • {key:15} → {path}")

print(f"\n💡 INTERPRÉTATION:")
print(f"   Les profondeurs analysées vont de -2m (surface) à -100m (profond)")
print(f"   PyGIMLi a calculé les VRAIES résistivités après correction géométrique")
print(f"   Les couleurs dans les coupes représentent les résistivités physiques réelles")

print(f"\n🌈 CLASSIFICATION EAU (basée sur résistivités inversées):")
for res_val in [min(model), np.median(model), max(model)]:
    classification = tool.get_water_classification(res_val)
    print(f"   • {res_val:.4f} Ω·m → {classification['type_eau']} ({classification['couleur']})")

print(f"\n🖼️ Visualisez les coupes:")
print(f"   cd /tmp/ert_vraies_profondeurs && ls -lh *.png")
