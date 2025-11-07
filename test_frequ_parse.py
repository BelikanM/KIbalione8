#!/usr/bin/env python3
from multi_freq_ert_parser import MultiFreqERTParser
import pandas as pd

print('╔══════════════════════════════════════════════════════════════╗')
print('║   TEST PARSING frequ.dat - DEBUG                             ║')
print('╚══════════════════════════════════════════════════════════════╝')
print()

# Lire le fichier manuellement
print('📄 CONTENU frequ.dat (5 premières lignes):')
with open('frequ.dat', 'r') as f:
    lines = f.readlines()[:5]
    for i, line in enumerate(lines, 1):
        print(f'   Ligne {i}: {repr(line[:80])}')
print()

# Parser avec MultiFreqERTParser
parser = MultiFreqERTParser()
df = parser.parse_file('frequ.dat')

print(f'✅ {len(df)} mesures parsées')

if len(df) > 0:
    print(f'✅ Colonnes: {list(df.columns)}')
    print()
    print('📊 Premières lignes:')
    print(df.head(10))
    print()
    
    # Coordonnées
    coords = parser.get_coordinates_corrected()
    print(f'📐 {len(coords)} coordonnées générées')
    print()
    
    if len(coords) > 0:
        print('📏 STRUCTURE X, Y, Z (10 premières lignes):')
        print(coords[['survey_point', 'x', 'y', 'z', 'resistivity']].head(10))
        print()
        
        print('PLAGES:')
        print(f'   X: {coords["x"].min():.0f} à {coords["x"].max():.0f}')
        print(f'   Y: {coords["y"].min():.0f} à {coords["y"].max():.0f}')
        print(f'   Z: {coords["z"].min():.1f} à {coords["z"].max():.1f}')
else:
    print('❌ Aucune donnée parsée')
