# 📦 INSTALLATION DES DÉPENDANCES
!apt-get -q install -y tesseract-ocr
!pip install -q construct python-magic bitstring h5py numpy matplotlib scikit-learn ipywidgets plyfile plotly scipy scikit-image pytesseract
# 📚 IMPORTS
import os
import io
import base64
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from construct import Struct, Int16ul, Array, GreedyRange
import magic
import h5py
from ipywidgets import FileUpload, Button, VBox
from IPython.display import display, clear_output, Javascript
from plyfile import PlyData, PlyElement
import plotly.graph_objects as go
import scipy.ndimage as ndi
from scipy.signal import hilbert
from scipy.ndimage import uniform_filter, sobel
from scipy.optimize import least_squares
from skimage.transform import hough_line, hough_line_peaks, probabilistic_hough_line
from itertools import groupby
import pytesseract
from PIL import Image
import struct
import xml.etree.ElementTree as ET
import pandas as pd
# 📄 STRUCTURE POUR DT1
Trace = Struct("samples" / Array(512, Int16ul))
GPRData = Struct("traces" / GreedyRange(Trace))
# 🔍 PARSE .HD
def parse_hd_text(content: str) -> dict:
    lines = content.strip().splitlines()
    metadata = {}
    for i, line in enumerate(lines):
        upper = line.strip().upper()
        try:
            if 'SAMPLES PER TRACE' in upper:
                metadata['samples_per_trace'] = int(lines[i + 1].strip())
            elif 'TOTAL TRACES' in upper:
                metadata['total_traces'] = int(lines[i + 1].strip())
            elif 'FREQUENCY' in upper:
                metadata['antenna_frequency'] = lines[i + 1].strip()
            elif 'UNITS' in upper:
                metadata['units'] = lines[i + 1].strip()
            elif 'RANGE' in upper or 'TIME WINDOW' in upper:
                metadata['time_window_ns'] = float(lines[i + 1].strip())
            elif 'START POSITION' in upper or 'START POS' in upper:
                metadata['start_position'] = float(lines[i + 1].strip())
            elif 'FINAL POSITION' in upper or 'STOP POSITION' in upper:
                metadata['final_position'] = float(lines[i + 1].strip())
            elif 'ANTENNA' in upper:
                metadata['antenna'] = lines[i + 1].strip()
        except:
            continue
    return metadata
# 📥 LECTURE DT1 AVEC OU SANS HD
def read_dt1_auto(path, hd_path=None):
    meta = {}
    num_samples, num_traces = None, None
    if hd_path and os.path.exists(hd_path):
        try:
            with open(hd_path, 'r', errors='ignore') as f:
                content = f.read()
            meta = parse_hd_text(content)
            num_samples = meta.get("samples_per_trace", None)
            num_traces = meta.get("total_traces", None)
        except:
            print("❌ Erreur lecture .HD")
    with open(path, "rb") as f:
        content = f.read()
    if num_samples and num_traces:
        try:
            data = np.frombuffer(content, dtype=np.int16).reshape((num_samples, num_traces))
            return data, meta
        except:
            print("⚠️ .HD incohérent, fallback Construct")
    try:
        data = GPRData.parse(content)
        traces = [t.samples for t in data.traces]
        arr = np.array(traces).T
        print(f"✅ Auto-chargé : {arr.shape}")
        return arr, meta
    except Exception as e:
        raise ValueError(f"Échec lecture Construct : {e}")
# 📥 LECTURE DZT AVEC OU SANS DZX
def read_dzt_auto(path, dzx_path=None):
    meta = {}
    antenna_freq_map = {
        '3101A': 900,
        '3200MLF': 2000,
        '5103': 400,
        '50400': 400,
        '51600': 1600,
        # Ajoutez plus si nécessaire
    }
    if dzx_path and os.path.exists(dzx_path):
        try:
            tree = ET.parse(dzx_path)
            root = tree.getroot()
            antenna_elem = root.find('.//ANTENNA')
            if antenna_elem is not None:
                meta['antenna'] = antenna_elem.text
                ant_key = meta['antenna'].upper().replace('.ANT', '')
                if ant_key in antenna_freq_map:
                    meta['antenna_frequency'] = f"{antenna_freq_map[ant_key]} MHz"
        except Exception as e:
            print(f"❌ Erreur lecture .DZX: {e}")
    with open(path, "rb") as f:
        f.seek(0)
        tag = struct.unpack('<H', f.read(2))[0]
        data_offset_sector = struct.unpack('<H', f.read(2))[0]
        data_offset = data_offset_sector * 512
        nsamp = struct.unpack('<H', f.read(2))[0]
        bits = struct.unpack('<H', f.read(2))[0]
        zero = struct.unpack('<H', f.read(2))[0]
        sps = struct.unpack('<f', f.read(4))[0]
        spm = struct.unpack('<f', f.read(4))[0]
        mpm = struct.unpack('<f', f.read(4))[0]
        position = struct.unpack('<f', f.read(4))[0]
        range_ns = struct.unpack('<f', f.read(4))[0]
        meta['samples_per_trace'] = nsamp
        meta['time_window_ns'] = range_ns
        meta['start_position'] = position
        meta['units'] = 'm'  # Assumé
        f.seek(0, 2)
        file_size = f.tell()
        data_size = file_size - data_offset
        if bits == 8:
            sample_size = 1
            dtype = np.uint8
        elif bits == 16:
            sample_size = 2
            dtype = np.uint16
        elif bits == 32:
            sample_size = 4
            dtype = np.uint32
        else:
            raise ValueError(f"Bits non supportés: {bits}")
        trace_size = nsamp * sample_size
        num_traces = data_size // trace_size
        meta['total_traces'] = num_traces
        if spm > 0:
            dx = 1.0 / spm
        else:
            dx = 0.05  # Défaut
        meta['final_position'] = position + (num_traces - 1) * dx if num_traces > 1 else position
        f.seek(data_offset)
        raw_data = np.frombuffer(f.read(), dtype=dtype)
        data = raw_data.reshape((num_traces, nsamp)).T
        data = (data.astype(np.int32) - zero).astype(np.int16)
    # Détermination dynamique du sens (orientation) - Assumé samples (profondeur/time) x traces (position)
    # Si besoin de transposer basé sur meta, ajoutez logique ici (ex: si 'orientation' in meta)
    # Pour l'instant, assume standard GPR: pas de transposition nécessaire
    # Pour sens x/y/z: assume ligne le long de x par défaut; si GPS dans DZX, pourrait analyser direction
    # Mais pas implémenté pour simplicité (ajoutez parsing GPS si besoin)
    print(f"✅ Auto-chargé DZT: {data.shape}")
    return data, meta
# 🖼️ TRAITEMENT D'IMAGE AVEC OCR
def process_image(path):
    image = Image.open(path)
    w, h = image.size
    # Extraction du texte global
    text = pytesseract.image_to_string(image)
    # Crop top pour titre
    top_crop = image.crop((0, 0, w, h * 0.1))
    title_text = pytesseract.image_to_string(top_crop).strip()
    title = title_text if title_text else "Untitled"
    # Crop left pour y_label (rotaté)
    left_crop = image.crop((0, 0, w * 0.1, h))
    left_rot = left_crop.rotate(90, expand=True)
    y_text = pytesseract.image_to_string(left_rot).strip()
    # Crop bottom pour x_label
    bottom_crop = image.crop((0, h * 0.9, w, h))
    x_text = pytesseract.image_to_string(bottom_crop).strip()
    print(f"Titre extrait : {title}")
    print(f"Étiquette X : {x_text}, Étiquette Y : {y_text}")
    # Crop zone plot (marges 10%)
    plot_crop = image.crop((w * 0.1, h * 0.1, w * 0.9, h * 0.9))
    gray = plot_crop.convert('L')
    data = np.array(gray)
    # Inverser si fond clair
    if np.mean(data) > 127:
        data = 255 - data
    # Déterminer si transposition nécessaire (si y est signal)
    transpose = False
    if 'signal' in y_text.lower() or 'amplitude' in y_text.lower():
        transpose = True
    elif 'trace' in y_text.lower() or 'position' in y_text.lower():
        transpose = True
    if transpose:
        data = data.T
    # Meta par défaut
    meta = {}
    return data, meta, title
# 📊 AFFICHAGE 2D
def display_data(data, title="Radar"):
    plt.figure(figsize=(12, 5))
    plt.imshow(data, cmap='gray', aspect='auto')
    plt.title(title)
    plt.xlabel("Trace")
    plt.ylabel("Sample")
    plt.colorbar()
    plt.show()
# 🧪 DÉTECTION D'ANOMALIES GÉNÉRIQUES (autres anomalies)
def detect_anomalies(data, threshold=2.5, min_size=10):
    """
    Détecte des zones d'anomalies génériques dans les données GPR.
    Args:
        data (np.array): matrice 2D (samples x traces)
        threshold (float): seuil en nombre d'écarts-types pour considérer une anomalie
        min_size (int): taille minimale d'une zone détectée pour la conserver
    Returns:
        anomalies_mask (np.array bool): masque 2D des anomalies détectées
        zones (list of tuples): liste des bounding boxes (x_min, x_max, y_min, y_max)
    """
    # Normaliser chaque trace (colonne)
    norm_data = (data - np.mean(data, axis=0)) / (np.std(data, axis=0) + 1e-6)
    # Points où le signal dépasse le seuil
    anomalies_mask = np.abs(norm_data) > threshold
    # Labelisation des zones connectées
    labeled, num_features = ndi.label(anomalies_mask)
    sizes = ndi.sum(anomalies_mask, labeled, range(1, num_features + 1))
    filtered_mask = np.zeros_like(anomalies_mask, dtype=bool)
    zones = []
    for i, size in enumerate(sizes):
        if size >= min_size:
            filtered_mask[labeled == i + 1] = True
            coords = np.argwhere(labeled == i + 1)
            y_min, x_min = coords.min(axis=0)
            y_max, x_max = coords.max(axis=0)
            zones.append((x_min, x_max, y_min, y_max))
    return filtered_mask, zones
# 🔎 DÉTECTION D'HYPERBOLES (objets métalliques)
def detect_hyperbolas(data, rho_b=0.1, min_segment_size=5, min_cluster_width=5, fit_res_threshold=10):
    """
    Détecte les hyperboles dans les données GPR pour identifier les objets métalliques.
    Basé sur une approche de reconnaissance en temps réel avec prétraitement, seuillage adaptatif,
    clustering par connexion de colonnes et ajustement d'hyperbole.
    Retourne les zones et les paramètres fitted pour classification ultérieure.
    """
    # Prétraitement
    filtered = uniform_filter(data.astype(float), size=3) # Filtre moyenne mobile
    row_mean = np.mean(filtered, axis=1)[:, np.newaxis]
    preprocessed = filtered - row_mean
    # Détection des bords (Sobel)
    edges = np.hypot(sobel(preprocessed, axis=0), sobel(preprocessed, axis=1))
    edges = np.abs(edges)
    max_edge = np.max(edges)
    if max_edge > 0:
        edges /= max_edge
    # Seuil adaptatif
    high_edges = edges[edges > rho_b * np.max(edges)]
    thresh_b = np.mean(high_edges) if len(high_edges) > 0 else 0.5
    binary = edges > thresh_b
    # Clustering par connexion de colonnes (C3 simplifié)
    height, width = binary.shape
    clusters = []
    for col in range(width):
        col_data = binary[:, col]
        labeled_col, num_labels = ndi.label(col_data)
        segments = []
        for lbl in range(1, num_labels + 1):
            pos = np.where(labeled_col == lbl)[0]
            if len(pos) >= min_segment_size:
                segments.append((col, min(pos), max(pos)))
        if col == 0:
            for seg in segments:
                clusters.append([seg])
        else:
            new_clusters = []
            for seg in segments:
                added = False
                for cl in clusters:
                    last_seg = cl[-1]
                    if last_seg[0] == col - 1:
                        overlap = min(last_seg[2], seg[2]) - max(last_seg[1], seg[1]) + 1
                        if overlap > 0:
                            cl.append(seg)
                            added = True
                if not added:
                    new_clusters.append([seg])
            clusters.extend(new_clusters)
    # Filtrage et ajustement d'hyperbole
    hyperbola_zones = []
    fitted_params = []
    for cluster in clusters:
        if len(cluster) > min_cluster_width:
            points = []
            for seg in cluster:
                for y in range(seg[1], seg[2] + 1):
                    points.append((seg[0], y))
            points = np.array(points)
            # Fonction d'hyperbole (forme simplifiée: y = sqrt(y0^2 + a*(x - x0)^2))
            def residuals(params, x, y):
                x0, y0, a = params
                return y - np.sqrt(y0**2 + a**2 * (x - x0)**2)
            # Estimation initiale
            x_min, x_max = np.min(points[:, 0]), np.max(points[:, 0])
            y_min = np.min(points[:, 1])
            x0_guess = (x_min + x_max) / 2
            y0_guess = y_min
            a_guess = 1.0 / (x_max - x_min + 1e-6)
            try:
                result = least_squares(residuals, [x0_guess, y0_guess, a_guess], args=(points[:, 0], points[:, 1]))
                if result.success and (result.cost / len(points)) < fit_res_threshold:
                    hyperbola_zones.append((x_min, x_max, np.min(points[:, 1]), np.max(points[:, 1])))
                    fitted_params.append(result.x) # x0, y0, a
            except:
                pass
    return hyperbola_zones, fitted_params
# 🕳️ DÉTECTION DE TROUS/VIDES
def detect_voids(data, energy_threshold=0.5, min_size=20):
    """
    Détecte les zones de trous ou vides en se basant sur les régions de faible énergie du signal.
    Utilise la transformée de Hilbert pour l'enveloppe et identifie les zones connectées de faible énergie.
    """
    # Enveloppe via Hilbert
    envelope = np.abs(hilbert(data.astype(float), axis=0))
    envelope /= np.max(envelope + 1e-6)
    # Énergie
    energy = envelope ** 2
    mean_energy = np.mean(energy)
    low_energy = energy < energy_threshold * mean_energy
    # Labelisation
    labeled, num = ndi.label(low_energy)
    sizes = ndi.sum(low_energy, labeled, range(1, num + 1))
    void_zones = []
    for i, size in enumerate(sizes):
        if size >= min_size:
            coords = np.argwhere(labeled == i + 1)
            y_min, x_min = coords.min(0)
            y_max, x_max = coords.max(0)
            void_zones.append((x_min, x_max, y_min, y_max))
    return void_zones
# 🚰 DÉTECTION DE L'EAU (zones à forte atténuation)
def detect_water(data, anomaly_zones, att_threshold=0.3):
    water_zones = []
    for (x_min, x_max, y_min, y_max) in anomaly_zones:
        if y_max + 10 < data.shape[0]:
            below_energy = np.mean(np.abs(data[y_max:y_max+50, x_min:x_max]))
            zone_energy = np.mean(np.abs(data[y_min:y_max, x_min:x_max]))
            if below_energy < att_threshold * zone_energy and (y_max - y_min) < 20: # Forte réflexion horizontale avec atténuation dessous
                water_zones.append((x_min, x_max, y_min, y_max))
    return water_zones
# 🛢️ DÉTECTION DE PIPES/TUYAUX (fonctionnalités linéaires)
def detect_pipes(data, min_length=10):
    # Prétraitement similaire
    filtered = uniform_filter(data.astype(float), size=3)
    row_mean = np.mean(filtered, axis=1)[:, np.newaxis]
    preprocessed = filtered - row_mean
    edges = np.hypot(sobel(preprocessed, axis=0), sobel(preprocessed, axis=1))
    edges = np.abs(edges)
    edges = (edges > 0.5 * np.max(edges)).astype(np.uint8)
    pipe_zones = []
    lines = probabilistic_hough_line(edges, threshold=10, line_length=min_length, line_gap=3)
    for line in lines:
        p0, p1 = line
        x_min, x_max = min(p0[0], p1[0]), max(p0[0], p1[0])
        y_min, y_max = min(p0[1], p1[1]), max(p0[1], p1[1])
        if x_max - x_min > min_length:
            pipe_zones.append((x_min, x_max, y_min, y_max))
    return pipe_zones
# 🟥 AFFICHAGE DES DÉTECTIONS
def highlight_detections(data, anomaly_zones, metal_zones, void_zones, water_zones, pipe_zones):
    plt.figure(figsize=(12, 5))
    plt.imshow(data, cmap='gray', aspect='auto')
    plt.title("Détections GPR: Jaune=Anomalies/Rocher, Rouge=Métal/Fer/Or/Diamant, Bleu=Vides, Cyan=Eau, Vert=Pipes")
    for (x_min, x_max, y_min, y_max) in anomaly_zones:
        plt.gca().add_patch(plt.Rectangle(
            (x_min, y_min), x_max - x_min, y_max - y_min,
            edgecolor='yellow', facecolor='none', linewidth=2))
    for (x_min, x_max, y_min, y_max) in metal_zones:
        plt.gca().add_patch(plt.Rectangle(
            (x_min, y_min), x_max - x_min, y_max - y_min,
            edgecolor='red', facecolor='none', linewidth=2))
    for (x_min, x_max, y_min, y_max) in void_zones:
        plt.gca().add_patch(plt.Rectangle(
            (x_min, y_min), x_max - x_min, y_max - y_min,
            edgecolor='blue', facecolor='none', linewidth=2))
    for (x_min, x_max, y_min, y_max) in water_zones:
        plt.gca().add_patch(plt.Rectangle(
            (x_min, y_min), x_max - x_min, y_max - y_min,
            edgecolor='cyan', facecolor='none', linewidth=2))
    for (x_min, x_max, y_min, y_max) in pipe_zones:
        plt.gca().add_patch(plt.Rectangle(
            (x_min, y_min), x_max - x_min, y_max - y_min,
            edgecolor='green', facecolor='none', linewidth=2))
    plt.xlabel("Trace")
    plt.ylabel("Sample")
    plt.colorbar()
    plt.show()
# 🔬 DÉTECTION ET DÉTERMINATION DES MATÉRIAUX
def detect_materials(data, anomaly_zones, metal_zones, fitted_params, void_zones, water_zones, pipe_zones):
    """
    Détermine les types de matériaux avec précision en se basant sur les zones détectées.
    Assigne des labels à chaque point des données GPR :
    - 0: Sol (background)
    - 1: Rocher (anomalie générique)
    - 2: Fer (métal fort)
    - 3: Or/Diamant (métal/précieux petit ou spécifique)
    - 4: Eau
    - 5: Vide/Trou
    - 6: Pipe/Tuyau (pétrole ou autre)
    Priorité : Pipe > Métal > Eau > Vide > Anomalie > Sol
    Utilise params fitted pour classifier métaux basés sur v et amplitude.
    """
    label_mask = np.zeros_like(data, dtype=int) # 0: Sol
    # Assigner anomalies génériques (1: Rocher)
    for (x_min, x_max, y_min, y_max) in anomaly_zones:
        label_mask[y_min:y_max+1, x_min:x_max+1] = 1
    # Assigner vides (5)
    for (x_min, x_max, y_min, y_max) in void_zones:
        label_mask[y_min:y_max+1, x_min:x_max+1] = 5
    # Assigner eau (4)
    for (x_min, x_max, y_min, y_max) in water_zones:
        label_mask[y_min:y_max+1, x_min:x_max+1] = 4
    # Assigner métaux avec subclass (2: Fer, 3: Or/Diamant)
    for idx, (x_min, x_max, y_min, y_max) in enumerate(metal_zones):
        if idx < len(fitted_params):
            _, _, a = fitted_params[idx]
            amplitude = np.mean(np.abs(data[y_min:y_max+1, x_min:x_max+1]))
            size = (x_max - x_min) * (y_max - y_min)
            # Assume dx=0.05 m, dt=0.2 ns
            dx = 0.05
            dt = 0.2
            v = 2 * dx / (a * dt) if a != 0 else 0.1
            epsilon = (0.3 / v) ** 2 if v != 0 else 1
            if size < 50: # Petit -> Diamant/Or
                label = 3
            elif amplitude > np.mean(np.abs(data)) * 3: # Fort -> Fer
                label = 2
            else:
                label = 2
            label_mask[y_min:y_max+1, x_min:x_max+1] = label
        else:
            label_mask[y_min:y_max+1, x_min:x_max+1] = 2
    # Assigner pipes (6)
    for (x_min, x_max, y_min, y_max) in pipe_zones:
        label_mask[y_min:y_max+1, x_min:x_max+1] = 6
    return label_mask
# 🌍 CLASSIFICATION DES TYPES DE TERRES (SOLS)
def classify_soil_types(data, label_mask, n_soil_types=3):
    """
    Utilise un modèle IA (PCA + KMeans) pour classifier automatiquement les types de sols dans les zones background.
    Applique le clustering sur les lignes (profondeurs) pour identifier les couches horizontales.
    Retourne un dictionnaire de mapping label -> nom de type de sol, basé sur l'énergie moyenne (réflexion).
    Met à jour label_mask avec des labels 10 + cluster_id pour les sols.
    """
    # Réduction dimensionnelle pour clustering efficace
    pca = PCA(n_components=min(10, data.shape[1]))
    reduced_rows = pca.fit_transform(data)
    # Clustering des lignes (couches)
    row_kmeans = KMeans(n_clusters=n_soil_types, random_state=42).fit(reduced_rows)
    soil_labels = row_kmeans.labels_ # Labels par profondeur (sample)
    # Calcul des énergies moyennes pour chaque cluster (sur données originales)
    cluster_energies = [np.mean(np.abs(data[soil_labels == i])) for i in range(n_soil_types)]
    # Trier les clusters par énergie croissante (faible: sable sec, moyen: loam, haut: argile humide)
    sorted_idx = np.argsort(cluster_energies)
    soil_type_names = ['Sable Sec', 'Loam', 'Argile Humide']
    type_map = {sorted_idx[j]: soil_type_names[j] for j in range(min(n_soil_types, 3))}
    color_map = {sorted_idx[0]: (255, 255, 0), # Jaune: Sable
                 sorted_idx[1]: (139, 69, 19), # Marron: Loam
                 sorted_idx[2]: (100, 50, 10)} # Marron foncé: Argile
    soil_v = {'Sable Sec': 0.2, 'Loam': 0.1, 'Argile Humide': 0.06}
    if n_soil_types > 3:
        for j in range(3, n_soil_types):
            type_map[sorted_idx[j]] = f'Autre Sol {j-2}'
            color_map[sorted_idx[j]] = (150, 150, 150)
            soil_v[type_map[sorted_idx[j]]] = 0.1
    # Trouver les segments consécutifs de labels (couches)
    soil_segments = []
    for label, group in groupby(enumerate(soil_labels), key=lambda x: x[1]):
        group_list = list(group)
        y_min = group_list[0][0]
        y_max = group_list[-1][0]
        soil_segments.append((label, y_min, y_max))
    # Assigner les labels de sol dans label_mask (seulement où background)
    for label, y_min, y_max in soil_segments:
        sub_mask = label_mask[y_min:y_max+1, :] == 0
        label_mask[y_min:y_max+1, :][sub_mask] = 10 + label
    soil_names = {10 + k: v for k, v in type_map.items()}
    soil_colors = {10 + k: v for k, v in color_map.items()}
    soil_v_map = {10 + k: soil_v[v] for k, v in type_map.items()}
    return type_map, color_map, soil_names, soil_colors, soil_segments, soil_v
# 📏 CALCUL DES PROFONDEURS (MIDAS-like approach simplifié)
def calculate_depths(all_zones, depth_array):
    """
    Calcule les profondeurs estimées pour les zones détectées en utilisant le tableau de profondeurs cumulées.
    """
    depths = []
    for (x_min, x_max, y_min, y_max) in all_zones:
        depth_min = depth_array[y_min]
        depth_max = depth_array[y_max]
        depths.append((depth_min, depth_max))
    return depths
# 🧠 CLUSTERING & ANALYSE PCA + EXPORT + TÉLÉCHARGEMENT AUTOMATIQUE
def advanced_process(data, meta, anomaly_zones, metal_zones, fitted_params, void_zones, water_zones, pipe_zones):
    flattened = data.reshape(data.shape[0], -1).T
    pca = PCA(n_components=2)
    reduced = pca.fit_transform(flattened)
    kmeans = KMeans(n_clusters=3, random_state=42).fit(reduced)
    plt.scatter(reduced[:, 0], reduced[:, 1], c=kmeans.labels_, cmap='viridis')
    plt.title("PCA + KMeans")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.show()
    # Détection des matériaux
    label_mask = detect_materials(data, anomaly_zones, metal_zones, fitted_params, void_zones, water_zones, pipe_zones)
    # Classification des types de sols
    type_map, color_map, soil_names, soil_colors, soil_segments, soil_v = classify_soil_types(data, label_mask)
    # Calcul dt, dx
    dt = 0.2
    if 'time_window_ns' in meta:
        dt = meta['time_window_ns'] / data.shape[0]
    dx = 0.05
    start_x = 0.0
    if 'start_position' in meta:
        start_x = meta['start_position']
    if 'final_position' in meta:
        num_tr = data.shape[1]
        if num_tr > 1:
            dx = (meta['final_position'] - start_x) / (num_tr - 1)
    # Calcul du tableau de profondeurs cumulées
    depth_array = np.zeros(data.shape[0])
    current_depth = 0.0
    for lbl, y_min, y_max in soil_segments:
        v = soil_v[type_map[lbl]]
        for y in range(y_min, y_max + 1):
            depth_array[y] = current_depth + (y - y_min) * (dt * v / 2)
        current_depth = depth_array[y_max]
    # Calcul des profondeurs pour zones
    all_zones = void_zones + anomaly_zones + metal_zones + water_zones + pipe_zones
    depths = calculate_depths(all_zones, depth_array)
    zone_types = ['Vide'] * len(void_zones) + ['Anomalie/Rocher'] * len(anomaly_zones) + ['Métal'] * len(metal_zones) + ['Eau'] * len(water_zones) + ['Pipe'] * len(pipe_zones)
    df = pd.DataFrame({
        'Zone': [i+1 for i in range(len(depths))],
        'Type': zone_types,
        'Profondeur Min (m)': [f"{d[0]:.2f}" for d in depths],
        'Profondeur Max (m)': [f"{d[1]:.2f}" for d in depths]
    })
    print("Profondeurs estimées (min-max en m) pour zones détectées:")
    display(df)
    # EXPORT HDF5
    with h5py.File("gpr_data.h5", "w") as hf:
        hf.create_dataset("data", data=data)
    # EXPORT PLY avec couleurs, axes dynamiques: x=position, y=signal (scalé), z=profondeur
    height, width = data.shape
    vertices = []
    color_rgb = {1: (139, 69, 19), # Marron: Rocher
                 2: (255, 0, 0), # Rouge: Fer
                 3: (255, 215, 0), # Or: Or/Diamant
                 4: (0, 0, 255), # Bleu: Eau
                 5: (173, 216, 230), # Bleu clair: Vide
                 6: (0, 255, 0)} # Vert: Pipe
    color_rgb.update(soil_colors) # Ajouter couleurs des sols
    material_names = {1: 'Rocher',
                      2: 'Fer (Métal)',
                      3: 'Or ou Diamant (Précieux)',
                      4: 'Eau',
                      5: 'Vide/Trou',
                      6: 'Pipe/Tuyau (Pétrole ou autre)'}
    material_names.update(soil_names) # Ajouter noms des sols
    max_signal = np.max(np.abs(data))
    scan_length = dx * (width - 1)
    y_scale = (scan_length / 10) / max_signal if max_signal > 0 and scan_length > 0 else 1.0
    for trace in range(width):
        for sample in range(height):
            x_val = start_x + trace * dx
            y_val = data[sample, trace] * y_scale
            z_val = -depth_array[sample] # Profondeur négative pour bas
            label = label_mask[sample, trace]
            r, g, b = color_rgb.get(label, (128, 128, 128))
            vertices.append((x_val, y_val, z_val, r, g, b))
    verts_np = np.array(vertices, dtype=[("x", "f4"), ("y", "f4"), ("z", "f4"),
                                         ("red", "u1"), ("green", "u1"), ("blue", "u1")])
    el = PlyElement.describe(verts_np, "vertex")
    PlyData([el], text=True).write("gpr_cloud.ply")
    print("✅ Fichiers exportés : gpr_data.h5, gpr_cloud.ply")
    # Lecture et encodage base64 du fichier PLY pour téléchargement auto
    with open("gpr_cloud.ply", "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    filename = "gpr_cloud.ply"
    js_code = f"""
    var data = "data:application/octet-stream;base64,{b64}";
    var link = document.createElement('a');
    link.href = data;
    link.download = "{filename}";
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    """
    display(Javascript(js_code))
    # Visualisation 2D des matériaux
    rgb_data = np.zeros((data.shape[0], data.shape[1], 3), dtype=np.uint8)
    for label in np.unique(label_mask):
        if label != 0:
            rgb_data[label_mask == label] = color_rgb.get(label, (128, 128, 128))
    plt.figure(figsize=(12, 5))
    plt.imshow(rgb_data, aspect='auto')
    plt.title("Carte 2D des Matériaux")
    plt.xlabel("Trace")
    plt.ylabel("Sample")
    plt.show()
    # Visualisation 3D avec couleurs par matériau et noms
    fig = go.Figure()
    labels_flat = [label_mask[sample, trace] for trace in range(width) for sample in range(height)]
    unique_labels = np.unique(labels_flat)
    for label in unique_labels:
        mask = np.array(labels_flat) == label
        fig.add_trace(go.Scatter3d(
            x=verts_np["x"][mask], y=verts_np["y"][mask], z=verts_np["z"][mask],
            mode='markers', marker=dict(size=1, color=f'rgb{color_rgb.get(label, (128,128,128))}'),
            name=material_names.get(label, 'Inconnu'),
            hovertext=[material_names.get(label, 'Inconnu') for _ in range(sum(mask))]
        ))
    fig.update_layout(title="Nuage de points GPR avec Matériaux", scene=dict(xaxis_title='X (Position)', yaxis_title='Y (Signal)', zaxis_title='Z (Profondeur)'))
    fig.show()
# 📁 DÉTECTION MIME
def detect_file_type(path):
    mime = magic.Magic(mime=True).from_file(path)
    print(f"📁 Type MIME : {mime}")
# 🔁 TRAITEMENT GLOBAL DE TOUS LES FICHIERS
def process_uploaded_files(file_dict):
    for name, content in file_dict.items():
        temp_path = f"/tmp/{name}"
        with open(temp_path, "wb") as f:
            f.write(content)
    for name in file_dict.keys():
        upper_name = name.upper()
        path = f"/tmp/{name}"
        try:
            detect_file_type(path)
            if upper_name.endswith(".DT1"):
                base = upper_name[:-4]
                hd_path = f"/tmp/{base}.HD" if os.path.exists(f"/tmp/{base}.HD") else None
                data, meta = read_dt1_auto(path, hd_path)
                display_title = meta.get('antenna', meta.get('antenna_frequency', upper_name))
                display_data(data, title=display_title)
                # Détection anomalies génériques
                _, anomaly_zones = detect_anomalies(data, threshold=2.5, min_size=10)
                # Détection objets métalliques (hyperboles)
                metal_zones, fitted_params = detect_hyperbolas(data)
                # Détection trous/vides
                void_zones = detect_voids(data)
                # Détection eau
                water_zones = detect_water(data, anomaly_zones)
                # Détection pipes
                pipe_zones = detect_pipes(data)
                # Affichage unifié
                highlight_detections(data, anomaly_zones, metal_zones, void_zones, water_zones, pipe_zones)
                advanced_process(data, meta, anomaly_zones, metal_zones, fitted_params, void_zones, water_zones, pipe_zones)
            elif upper_name.endswith(".DZT"):
                base = upper_name[:-4]
                dzx_path = f"/tmp/{base}.DZX" if os.path.exists(f"/tmp/{base}.DZX") else None
                data, meta = read_dzt_auto(path, dzx_path)
                display_title = meta.get('antenna', meta.get('antenna_frequency', upper_name))
                display_data(data, title=display_title)
                # Détection anomalies génériques
                _, anomaly_zones = detect_anomalies(data, threshold=2.5, min_size=10)
                # Détection objets métalliques (hyperboles)
                metal_zones, fitted_params = detect_hyperbolas(data)
                # Détection trous/vides
                void_zones = detect_voids(data)
                # Détection eau
                water_zones = detect_water(data, anomaly_zones)
                # Détection pipes
                pipe_zones = detect_pipes(data)
                # Affichage unifié
                highlight_detections(data, anomaly_zones, metal_zones, void_zones, water_zones, pipe_zones)
                advanced_process(data, meta, anomaly_zones, metal_zones, fitted_params, void_zones, water_zones, pipe_zones)
            elif upper_name.endswith((".PNG", ".JPG", ".JPEG", ".BMP")):
                data, meta, extracted_title = process_image(path)
                display_data(data, title=extracted_title)
                # Détection anomalies génériques
                _, anomaly_zones = detect_anomalies(data, threshold=2.5, min_size=10)
                # Détection objets métalliques (hyperboles)
                metal_zones, fitted_params = detect_hyperbolas(data)
                # Détection trous/vides
                void_zones = detect_voids(data)
                # Détection eau
                water_zones = detect_water(data, anomaly_zones)
                # Détection pipes
                pipe_zones = detect_pipes(data)
                # Affichage unifié
                highlight_detections(data, anomaly_zones, metal_zones, void_zones, water_zones, pipe_zones)
                advanced_process(data, meta, anomaly_zones, metal_zones, fitted_params, void_zones, water_zones, pipe_zones)
            elif upper_name.endswith(".HD"):
                print(f"ℹ️ Fichier .HD seul détecté ({upper_name}), en attente de .DT1 correspondant.")
            elif upper_name.endswith(".DZX"):
                print(f"ℹ️ Fichier .DZX seul détecté ({upper_name}), en attente de .DZT correspondant.")
            else:
                print(f"⛔ Format non supporté : {upper_name}")
        except Exception as e:
            print(f"❌ Erreur sur {upper_name} : {e}")
# 🎛️ INTERFACE WIDGETS
uploader = FileUpload(accept='', multiple=True)
btn = Button(description="Analyser les fichiers", button_style='success')
def on_click(b):
    clear_output(wait=True)
    print("Fichiers uploadés :")
    files = uploader.value
    file_dict = {}
    if isinstance(files, dict):
        for filename, fileinfo in files.items():
            file_dict[filename] = fileinfo['content']
            print(f"- {filename}")
    elif isinstance(files, (tuple, list)):
        for file in files:
            try:
                filename = getattr(file, 'name', 'unknown')
                content = getattr(file, 'content', b'')
                file_dict[filename] = content
                print(f"- {filename}")
            except Exception as e:
                print(f"❓ Format inattendu dans tuple : {file} / Erreur : {e}")
    else:
        print("❓ Format de fichiers inattendu :", type(files))
    process_uploaded_files(file_dict)
btn.on_click(on_click)
display(VBox([uploader, btn]))