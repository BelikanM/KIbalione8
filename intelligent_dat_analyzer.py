#!/usr/bin/env python3
"""
ANALYSEUR INTELLIGENT DE STRUCTURE DE FICHIERS .DAT
Comprend automatiquement la structure des fichiers ERT/géophysiques
"""

import numpy as np
import pandas as pd
import re
from typing import Dict, List, Tuple, Any, Optional
import chardet
from dataclasses import dataclass
from pathlib import Path

@dataclass
class DATFileStructure:
    """Structure détectée d'un fichier .DAT"""
    file_type: str  # 'ert_frequency', 'ert_standard', 'seismic', 'unknown'
    encoding: str  # 'utf-8', 'utf-16', 'latin1', etc.
    delimiter: str  # ',', '\t', ' ', etc.
    header_lines: int
    data_start_line: int
    columns: List[str]
    column_types: Dict[str, str]  # {col_name: 'frequency'|'resistivity'|'depth'|'coordinate'|'identifier'}
    frequencies_mhz: Optional[List[float]]  # Si fichier fréquence EM
    num_stations: int
    num_measurements: int
    metadata: Dict[str, Any]
    parsing_instructions: str  # Instructions pour parser
    confidence_score: float  # 0-100 % de confiance dans l'analyse

class IntelligentDATAnalyzer:
    """
    Analyseur intelligent qui comprend automatiquement la structure des fichiers .DAT
    """
    
    def __init__(self):
        self.known_patterns = {
            'frequency_em': {
                'indicators': ['MHz', 'frequency', 'fréquence', 'kHz'],
                'format': 'electromagnetic_frequency_domain'
            },
            'resistivity_standard': {
                'indicators': ['resistivity', 'résistivité', 'Ohm.m', 'Ω.m'],
                'format': 'electrical_resistivity_tomography'
            },
            'coordinates': {
                'indicators': ['X', 'Y', 'Z', 'Depth', 'Distance', 'Position'],
                'format': 'spatial_coordinates'
            },
            'time_series': {
                'indicators': ['Time', 'Temps', 'Date', 'Timestamp'],
                'format': 'time_series_data'
            }
        }
    
    def detect_encoding(self, file_path: str) -> str:
        """
        Détecte automatiquement l'encodage du fichier
        Teste plusieurs encodages courants pour fichiers géophysiques
        """
        # Lire les premiers bytes
        with open(file_path, 'rb') as f:
            raw_data = f.read(10000)  # Premiers 10KB
        
        # Utiliser chardet pour détection automatique
        detected = chardet.detect(raw_data)
        confidence = detected['confidence']
        encoding = detected['encoding']
        
        # Si confiance faible, tester encodages courants
        if confidence < 0.7:
            test_encodings = ['utf-8', 'utf-16', 'utf-16-le', 'latin1', 'cp1252', 'gbk', 'iso-8859-1']
            
            for enc in test_encodings:
                try:
                    raw_data.decode(enc)
                    return enc
                except:
                    continue
        
        return encoding if encoding else 'utf-8'
    
    def detect_delimiter(self, sample_lines: List[str]) -> str:
        """Détecte le délimiteur (virgule, tab, espace, point-virgule)"""
        delimiters = {
            ',': 0,
            '\t': 0,
            ' ': 0,
            ';': 0,
            '|': 0
        }
        
        for line in sample_lines[:10]:
            for delim in delimiters:
                delimiters[delim] += line.count(delim)
        
        # Exclure les espaces si trop nombreux (probablement pas un délimiteur)
        if delimiters[' '] > 100:
            delimiters[' '] = 0
        
        return max(delimiters, key=delimiters.get)
    
    def detect_header_lines(self, lines: List[str], delimiter: str) -> int:
        """Détecte combien de lignes constituent l'en-tête"""
        for i, line in enumerate(lines[:20]):
            parts = line.split(delimiter)
            
            # Si ligne contient majoritairement des nombres, c'est probablement des données
            numeric_count = sum(1 for p in parts if self._is_numeric(p.strip()))
            
            if numeric_count / len(parts) > 0.5 and i > 0:
                return i
        
        return 1  # Par défaut, 1 ligne d'en-tête
    
    def _is_numeric(self, s: str) -> bool:
        """Vérifie si une chaîne est numérique"""
        try:
            s_clean = s.replace('MHz', '').replace('kHz', '').replace('Ω', '').replace(',', '.')
            float(s_clean)
            return True
        except:
            return False
    
    def detect_frequency_columns(self, header: List[str]) -> List[float]:
        """Détecte et extrait les fréquences en MHz"""
        frequencies = []
        
        for col in header:
            col_clean = col.strip()
            
            # Chercher pattern fréquence
            if 'MHz' in col_clean or 'mhz' in col_clean.lower():
                try:
                    freq = float(re.sub(r'[^\d.]', '', col_clean))
                    frequencies.append(freq)
                except:
                    pass
            elif 'kHz' in col_clean or 'khz' in col_clean.lower():
                try:
                    freq = float(re.sub(r'[^\d.]', '', col_clean)) / 1000
                    frequencies.append(freq)
                except:
                    pass
        
        return sorted(frequencies, reverse=True)  # Trier par ordre décroissant
    
    def classify_column_type(self, col_name: str, sample_values: List) -> str:
        """Classifie le type d'une colonne"""
        col_lower = col_name.lower()
        
        # Identifiants
        if any(kw in col_lower for kw in ['projet', 'project', 'ligne', 'line', 'station', 'id']):
            return 'identifier'
        
        # Coordonnées
        if any(kw in col_lower for kw in ['x', 'y', 'z', 'lat', 'lon', 'distance', 'position']):
            return 'coordinate'
        
        # Profondeur
        if any(kw in col_lower for kw in ['depth', 'profondeur', 'z', 'elevation']):
            return 'depth'
        
        # Fréquence
        if 'mhz' in col_lower or 'khz' in col_lower or 'frequency' in col_lower:
            return 'frequency_measurement'
        
        # Résistivité
        if any(kw in col_lower for kw in ['resistivity', 'résistivité', 'rho', 'ρ', 'ohm']):
            return 'resistivity'
        
        # Si échantillon numérique, c'est une mesure
        numeric_ratio = sum(1 for v in sample_values if self._is_numeric(str(v))) / len(sample_values)
        if numeric_ratio > 0.8:
            return 'measurement'
        
        return 'text'
    
    def analyze_file_structure(self, file_path: str) -> DATFileStructure:
        """
        ANALYSE COMPLÈTE de la structure du fichier .DAT
        Retourne une structure détaillée avec instructions de parsing
        """
        print(f"🔍 Analyse intelligente de: {Path(file_path).name}")
        
        # 1. Détection encodage
        encoding = self.detect_encoding(file_path)
        print(f"   📝 Encodage détecté: {encoding}")
        
        # 2. Lecture du fichier
        try:
            with open(file_path, 'r', encoding=encoding) as f:
                lines = [line.strip() for line in f.readlines() if line.strip()]
        except:
            # Fallback
            with open(file_path, 'r', encoding='latin1', errors='ignore') as f:
                lines = [line.strip() for line in f.readlines() if line.strip()]
            encoding = 'latin1'
        
        if not lines:
            return self._create_unknown_structure()
        
        # 3. Détection délimiteur
        delimiter = self.detect_delimiter(lines)
        print(f"   📊 Délimiteur: '{delimiter}' (code: {ord(delimiter)})")
        
        # 4. Détection nombre lignes en-tête
        header_lines = self.detect_header_lines(lines, delimiter)
        print(f"   📋 Lignes d'en-tête: {header_lines}")
        
        # 5. Extraction colonnes
        header_parts = lines[header_lines - 1].split(delimiter)
        columns = [col.strip() for col in header_parts]
        print(f"   🔢 Nombre de colonnes: {len(columns)}")
        
        # 6. Détection fréquences
        frequencies = self.detect_frequency_columns(columns)
        is_frequency_file = len(frequencies) > 5
        
        if is_frequency_file:
            print(f"   📡 Fichier fréquence EM détecté: {len(frequencies)} canaux")
            print(f"      Plage: {frequencies[-1]:.2f} - {frequencies[0]:.2f} MHz")
        
        # 7. Classification colonnes
        data_start = header_lines
        sample_data = []
        for i in range(data_start, min(data_start + 10, len(lines))):
            sample_data.append(lines[i].split(delimiter))
        
        column_types = {}
        for i, col in enumerate(columns):
            sample_vals = [row[i] if i < len(row) else '' for row in sample_data]
            column_types[col] = self.classify_column_type(col, sample_vals)
        
        # 8. Déterminer type de fichier
        if is_frequency_file:
            file_type = 'ert_frequency_domain'
            confidence = 95.0
        elif any('resistivity' in ct.lower() for ct in column_types.values()):
            file_type = 'ert_standard'
            confidence = 90.0
        elif len(columns) >= 3 and column_types.get(columns[0]) in ['identifier', 'text']:
            file_type = 'ert_custom'
            confidence = 75.0
        else:
            file_type = 'unknown'
            confidence = 50.0
        
        # 9. Compter stations et mesures
        num_data_lines = len(lines) - data_start
        num_measurements = len([ct for ct in column_types.values() if 'measurement' in ct or 'frequency' in ct])
        
        # 10. Métadonnées
        metadata = {
            'file_size_bytes': Path(file_path).stat().st_size,
            'total_lines': len(lines),
            'data_lines': num_data_lines,
            'first_data_sample': lines[data_start][:100] + '...' if data_start < len(lines) else '',
            'column_preview': columns[:5] if len(columns) > 5 else columns
        }
        
        # 11. Instructions de parsing
        parsing_instructions = self._generate_parsing_instructions(
            file_type, encoding, delimiter, header_lines, columns, column_types, frequencies
        )
        
        print(f"   ✅ Type: {file_type} (confiance: {confidence:.1f}%)")
        
        return DATFileStructure(
            file_type=file_type,
            encoding=encoding,
            delimiter=delimiter,
            header_lines=header_lines,
            data_start_line=data_start,
            columns=columns,
            column_types=column_types,
            frequencies_mhz=frequencies if is_frequency_file else None,
            num_stations=num_data_lines,
            num_measurements=num_measurements,
            metadata=metadata,
            parsing_instructions=parsing_instructions,
            confidence_score=confidence
        )
    
    def _generate_parsing_instructions(
        self, file_type: str, encoding: str, delimiter: str, 
        header_lines: int, columns: List[str], column_types: Dict,
        frequencies: Optional[List[float]]
    ) -> str:
        """Génère instructions Python pour parser ce fichier"""
        
        if file_type == 'ert_frequency_domain':
            return f"""
# INSTRUCTIONS DE PARSING - Fichier ERT Fréquence EM

import pandas as pd
import numpy as np

# 1. Charger le fichier
df = pd.read_csv(
    "votre_fichier.dat",
    encoding="{encoding}",
    delimiter="{delimiter}",
    skiprows={header_lines - 1}
)

# 2. Colonnes détectées:
# - Colonne 0: {columns[0]} ({column_types.get(columns[0], 'unknown')})
# - Colonne 1: {columns[1]} ({column_types.get(columns[1], 'unknown')})
# - Colonnes suivantes: Mesures aux fréquences (MHz)

# 3. Fréquences disponibles:
frequencies = {frequencies[:10]}  # (+ {len(frequencies) - 10} autres)

# 4. Extraire les données
project_names = df.iloc[:, 0]  # Noms de projet
station_ids = df.iloc[:, 1]     # IDs de station
measurements = df.iloc[:, 2:].values  # Matrice de mesures

# 5. Structure:
# - Lignes: stations/points de mesure
# - Colonnes: fréquences (de {frequencies[0]:.2f} à {frequencies[-1]:.2f} MHz)
# - Valeurs: résistivités apparentes ou amplitudes normalisées

# 6. Visualisation simple
import matplotlib.pyplot as plt

# Profil d'une station
station_idx = 0
plt.semilogx(frequencies, measurements[station_idx, :])
plt.xlabel('Fréquence (MHz)')
plt.ylabel('Mesure')
plt.title(f'Profil station {{station_ids.iloc[station_idx]}}')
plt.grid(True)
plt.show()
"""
        
        elif file_type == 'ert_standard':
            return f"""
# INSTRUCTIONS DE PARSING - Fichier ERT Standard

import pandas as pd

# 1. Charger
df = pd.read_csv(
    "votre_fichier.dat",
    encoding="{encoding}",
    delimiter="{delimiter}",
    skiprows={header_lines - 1}
)

# 2. Colonnes: {', '.join(columns[:5])}...

# 3. Extraire données géométriques et résistivité
# (adapter selon vos colonnes spécifiques)
"""
        
        else:
            return f"""
# INSTRUCTIONS DE PARSING - Fichier personnalisé

import pandas as pd

df = pd.read_csv(
    "votre_fichier.dat",
    encoding="{encoding}",
    delimiter="{delimiter}",
    skiprows={header_lines - 1}
)

# Colonnes: {columns[:10]}
# Types: {column_types}
"""
    
    def _create_unknown_structure(self) -> DATFileStructure:
        """Structure par défaut si analyse échoue"""
        return DATFileStructure(
            file_type='unknown',
            encoding='utf-8',
            delimiter=',',
            header_lines=0,
            data_start_line=0,
            columns=[],
            column_types={},
            frequencies_mhz=None,
            num_stations=0,
            num_measurements=0,
            metadata={},
            parsing_instructions="# Analyse automatique échouée",
            confidence_score=0.0
        )
    
    def load_data_with_structure(self, file_path: str, structure: DATFileStructure) -> pd.DataFrame:
        """
        Charge les données selon la structure détectée
        Retourne un DataFrame propre et formaté
        """
        print(f"📥 Chargement données avec structure détectée...")
        
        try:
            df = pd.read_csv(
                file_path,
                encoding=structure.encoding,
                delimiter=structure.delimiter,
                skiprows=structure.header_lines - 1,
                names=structure.columns if structure.columns else None
            )
            
            # Nettoyer les colonnes
            if structure.file_type == 'ert_frequency_domain':
                # Séparer identifiants et mesures
                id_cols = [col for col, ctype in structure.column_types.items() 
                          if ctype in ['identifier', 'text', 'coordinate']]
                measure_cols = [col for col, ctype in structure.column_types.items()
                               if ctype == 'frequency_measurement']
                
                # Convertir mesures en float
                for col in measure_cols:
                    if col in df.columns:
                        df[col] = pd.to_numeric(df[col], errors='coerce')
            
            print(f"   ✅ {len(df)} lignes chargées avec succès")
            return df
            
        except Exception as e:
            print(f"   ❌ Erreur chargement: {e}")
            return pd.DataFrame()
    
    def generate_structure_report(self, structure: DATFileStructure) -> str:
        """Génère un rapport textuel de la structure détectée"""
        
        report = f"""
╔══════════════════════════════════════════════════════════════════════════╗
║              RAPPORT D'ANALYSE DE STRUCTURE - Fichier .DAT               ║
╚══════════════════════════════════════════════════════════════════════════╝

📋 IDENTIFICATION
   Type de fichier      : {structure.file_type.upper()}
   Confiance            : {structure.confidence_score:.1f}%
   Encodage             : {structure.encoding}
   Délimiteur           : '{structure.delimiter}' (ASCII {ord(structure.delimiter)})

📊 STRUCTURE
   Lignes d'en-tête     : {structure.header_lines}
   Ligne de début données : {structure.data_start_line + 1}
   Nombre de colonnes   : {len(structure.columns)}
   Nombre de stations   : {structure.num_stations}
   Nombre de mesures    : {structure.num_measurements}

🔢 COLONNES DÉTECTÉES
"""
        
        for i, (col, ctype) in enumerate(structure.column_types.items(), 1):
            report += f"   [{i:2d}] {col[:40]:40s} → {ctype}\n"
        
        if structure.frequencies_mhz:
            report += f"""
📡 FRÉQUENCES (Domaine EM)
   Nombre de canaux     : {len(structure.frequencies_mhz)}
   Fréquence max        : {structure.frequencies_mhz[0]:.2f} MHz
   Fréquence min        : {structure.frequencies_mhz[-1]:.2f} MHz
   Ratio max/min        : {structure.frequencies_mhz[0] / structure.frequencies_mhz[-1]:.1f}x
"""
        
        report += f"""
📁 MÉTADONNÉES
   Taille fichier       : {structure.metadata.get('file_size_bytes', 0) / 1024:.1f} KB
   Lignes totales       : {structure.metadata.get('total_lines', 0)}
   Lignes de données    : {structure.metadata.get('data_lines', 0)}

💻 INSTRUCTIONS DE PARSING
{structure.parsing_instructions}

╚══════════════════════════════════════════════════════════════════════════╝
"""
        return report


# Fonction d'utilisation rapide
def analyze_dat_file(file_path: str, verbose: bool = True) -> DATFileStructure:
    """
    Point d'entrée principal pour analyser un fichier .DAT
    
    Args:
        file_path: Chemin vers le fichier .dat
        verbose: Afficher rapport détaillé
    
    Returns:
        Structure détectée
    """
    analyzer = IntelligentDATAnalyzer()
    structure = analyzer.analyze_file_structure(file_path)
    
    if verbose:
        print("\n" + analyzer.generate_structure_report(structure))
    
    return structure


# Test
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
        structure = analyze_dat_file(file_path, verbose=True)
        
        # Charger données
        analyzer = IntelligentDATAnalyzer()
        df = analyzer.load_data_with_structure(file_path, structure)
        print(f"\n📊 Aperçu données:\n{df.head()}")
    else:
        print("Usage: python intelligent_dat_analyzer.py <fichier.dat>")
