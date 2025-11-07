#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from multi_freq_ert_parser import MultiFreqERTParser
import pandas as pd

# Parser
parser = MultiFreqERTParser()
df = parser.parse_file('freq.dat')

# Ajouter classification
df_classified = parser.add_water_classification(df)

# Créer HTML avec couleurs
html = """
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Classification Résistivité - Types d'Eau</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }
        h1 { color: #333; text-align: center; }
        .stats { background: white; padding: 20px; border-radius: 8px; margin: 20px 0; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        table { width: 100%; border-collapse: collapse; background: white; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        th { background: #667eea; color: white; padding: 12px; text-align: left; }
        td { padding: 10px; border-bottom: 1px solid #ddd; }
        tr:hover { background: #f0f0f0; }
        .color-box { display: inline-block; width: 20px; height: 20px; border: 1px solid #333; margin-right: 10px; vertical-align: middle; }
    </style>
</head>
<body>
    <h1>📊 Classification ERT - Types d'Eau par Résistivité</h1>
    
    <div class="stats">
        <h2>📈 Légende des Types d'Eau</h2>
        <table>
            <tr>
                <th>Type d'Eau</th>
                <th>Résistivité (Ω·m)</th>
                <th>Couleur Associée</th>
                <th>Nombre de mesures</th>
            </tr>
"""

# Ajouter légende
water_types = [
    ('Eau de mer', '0.1 - 1', '#FF4500', 'Rouge vif / Orange'),
    ('Eau salée (nappe)', '1 - 10', '#FFA500', 'Jaune / Orange'),
    ('Eau douce', '10 - 100', '#00CED1', 'Vert / Bleu clair'),
    ('Eau très pure', '> 100', '#00008B', 'Bleu foncé')
]

for water_type, range_res, color, desc in water_types:
    count = len(df_classified[df_classified['type_eau'] == water_type])
    html += f"""
            <tr>
                <td><strong>{water_type}</strong></td>
                <td>{range_res}</td>
                <td><span class="color-box" style="background-color: {color};"></span>{desc}</td>
                <td>{count} mesures ({count/len(df_classified)*100:.1f}%)</td>
            </tr>
"""

html += """
        </table>
    </div>
    
    <div class="stats">
        <h2>📊 Données ERT Classifiées (50 premières lignes)</h2>
        <table>
            <tr>
                <th>Survey Point</th>
                <th>Depth (m)</th>
                <th>Fréquence (MHz)</th>
                <th>Résistivité (Ω·m)</th>
                <th>Type d'Eau</th>
                <th>Couleur</th>
            </tr>
"""

# Ajouter données
for idx, row in df_classified.head(50).iterrows():
    html += f"""
            <tr style="background-color: {row['couleur_hex']}22;">
                <td>{row['survey_point']}</td>
                <td>{row['depth']:.1f}</td>
                <td>{row['frequency_MHz']:.2f}</td>
                <td>{row['resistivity']:.4f}</td>
                <td><strong>{row['type_eau']}</strong></td>
                <td><span class="color-box" style="background-color: {row['couleur_hex']};"></span>{row['couleur']}</td>
            </tr>
"""

html += """
        </table>
    </div>
</body>
</html>
"""

# Sauvegarder
with open('/tmp/water_classification.html', 'w', encoding='utf-8') as f:
    f.write(html)

print('✅ Tableau HTML généré: /tmp/water_classification.html')
print(f'✅ {len(df_classified)} mesures classifiées')
print()
print('📊 Répartition:')
for water_type, count in df_classified['type_eau'].value_counts().items():
    print(f'   • {water_type}: {count} ({count/len(df_classified)*100:.1f}%)')
