#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from multi_freq_ert_parser import MultiFreqERTParser
import pandas as pd

print('╔══════════════════════════════════════════════════════════════╗')
print('║   TEST CLASSIFICATION EAU + COULEURS RÉSISTIVITÉ            ║')
print('╚══════════════════════════════════════════════════════════════╝')
print()

# Parser le fichier
parser = MultiFreqERTParser()
df = parser.parse_file('freq.dat')

print(f'✅ {len(df)} mesures parsées')
print()

# Ajouter classification de l'eau
df_classified = parser.add_water_classification(df)

print('📊 TABLEAU AVEC CLASSIFICATION D\'EAU (20 premières lignes):')
print()
print('| Survey | Depth | Freq (MHz) | Résistivité (Ω·m) | Type d\'eau          | Couleur          |')
print('| ------ | ----- | ---------- | ----------------- | ------------------- | ---------------- |')

for idx, row in df_classified.head(20).iterrows():
    print(f'| {row["survey_point"]:<6} | {row["depth"]:<5.0f} | {row["frequency_MHz"]:<10.2f} | {row["resistivity"]:<17.3f} | {row["type_eau"]:<19} | {row["couleur"]:<16} |')

print()
print('📈 STATISTIQUES PAR TYPE D\'EAU:')
print()

# Compter par type d'eau
water_stats = df_classified.groupby('type_eau').agg({
    'resistivity': ['count', 'mean', 'min', 'max'],
    'couleur': 'first'
}).round(4)

print(water_stats)
print()

# Répartition
print('📊 RÉPARTITION DES TYPES D\'EAU:')
type_counts = df_classified['type_eau'].value_counts()
for water_type, count in type_counts.items():
    percentage = (count / len(df_classified)) * 100
    couleur = df_classified[df_classified['type_eau'] == water_type]['couleur'].iloc[0]
    print(f'   • {water_type:<20}: {count:>4} mesures ({percentage:>5.1f}%) - {couleur}')

print()
print('✅ Classification appliquée avec succès!')
print('💡 Les couleurs sont prêtes pour l\'affichage dans Streamlit')
