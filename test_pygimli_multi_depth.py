#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test PyGIMLi avec données multi-profondeurs (fusion freq.dat + frequ.dat)
"""

from multi_freq_ert_parser import MultiFreqERTParser
from pygimli_ert_sections import PyGIMLiERTSections

print("╔════════════════════════════════════════════════════════════╗")
print("║   TEST PYGIMLI - DONNÉES MULTI-PROFONDEURS (FUSION)       ║")
print("╚════════════════════════════════════════════════════════════╝\n")

# Créer données test frequ.dat avec profondeurs variées
frequ_content = '''survey-point\tdepth\tdata\tproject
1\t-2\t0.36289272\t20251030
1\t-10\t0.34289272\t20251030
1\t-20\t0.32289272\t20251030
1\t-50\t0.28289272\t20251030
1\t-100\t0.35072222\t20251030
2\t-2\t0.40952906\t20251030
2\t-10\t0.38952906\t20251030
2\t-20\t0.36952906\t20251030
2\t-50\t0.32952906\t20251030
2\t-100\t0.37070912\t20251030
3\t-2\t0.41214067\t20251030
3\t-10\t0.39214067\t20251030
3\t-20\t0.37214067\t20251030
3\t-50\t0.33214067\t20251030
3\t-100\t0.38214067\t20251030
4\t-2\t0.39500000\t20251030
4\t-10\t0.37500000\t20251030
4\t-20\t0.35500000\t20251030
4\t-50\t0.31500000\t20251030
4\t-100\t0.36500000\t20251030
5\t-2\t0.38800000\t20251030
5\t-10\t0.36800000\t20251030
5\t-20\t0.34800000\t20251030
5\t-50\t0.30800000\t20251030
5\t-100\t0.35800000\t20251030
6\t-2\t0.40100000\t20251030
6\t-10\t0.38100000\t20251030
6\t-20\t0.36100000\t20251030
6\t-50\t0.32100000\t20251030
6\t-100\t0.37100000\t20251030
7\t-2\t0.39300000\t20251030
7\t-10\t0.37300000\t20251030
7\t-20\t0.35300000\t20251030
7\t-50\t0.31300000\t20251030
7\t-100\t0.36300000\t20251030'''

with open('frequ_multi_depth.dat', 'w') as f:
    f.write(frequ_content)

print("✅ Fichier frequ_multi_depth.dat créé (7 survey points × 5 profondeurs = 35 mesures)\n")

# Parser avec multi-profondeurs
parser = MultiFreqERTParser()
df = parser.parse_file('frequ_multi_depth.dat')

print(f"📊 Données parsées: {len(df)} mesures")
print(f"   • Profondeurs: {sorted(df['depth'].unique())}")
print(f"   • Survey points: {sorted(df['survey_point'].unique())}")
print()

# Générer coupes PyGIMLi
gimli_gen = PyGIMLiERTSections()
gimli_gen.load_data_from_parser(df)

# Générer les 3 formats
outputs = gimli_gen.generate_all_formats(
    output_dir='/tmp/ert_pygimli_multi_depth',
    prefix='fusion_multi_profondeurs'
)

print(f"\n✅ {len(outputs)} coupes générées!")
print("\n📂 Ouvrir les fichiers:")
for output in outputs:
    print(f"   xdg-open {output}")
