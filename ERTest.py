# app_sonic_ravensgate.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.cluster import KMeans
import chardet
import os
import tempfile
import io
import plotly.graph_objects as go
from datetime import datetime

# --- Table de réglage température (Ts) ---
temperature_control_table = {
    36: {0:31, 5:31, 10:32, 15:33, 20:34, 25:34, 30:35, 35:36, 40:37, 45:37, 50:38, 55:39, 60:40, 65:40, 70:41, 75:42, 80:43, 85:43, 90:44, 95:45},
    38: {0:32, 5:33, 10:34, 15:35, 20:35, 25:36, 30:37, 35:38, 40:39, 45:39, 50:40, 55:41, 60:41, 65:42, 70:43, 75:44, 80:44, 85:45, 90:46, 95:47},
    40: {0:34, 5:35, 10:36, 15:36, 20:37, 25:38, 30:39, 35:39, 40:40, 45:41, 50:42, 55:42, 60:43, 65:44, 70:45, 75:45, 80:46, 85:47, 90:48, 95:48},
    42: {0:36, 5:36, 10:37, 15:38, 20:39, 25:39, 30:40, 35:41, 40:42, 45:42, 50:43, 55:44, 60:45, 65:45, 70:46, 75:47, 80:48, 85:48, 90:49, 95:50},
    44: {0:37, 5:38, 10:39, 15:40, 20:40, 25:41, 30:42, 35:43, 40:43, 45:44, 50:45, 55:46, 60:46, 65:47, 70:48, 75:49, 80:49, 85:50, 90:51, 95:52},
    46: {0:39, 5:40, 10:41, 15:41, 20:42, 25:43, 30:44, 35:44, 40:45, 45:46, 50:47, 55:47, 60:48, 65:49, 70:50, 75:50, 80:51, 85:52, 90:53, 95:53},
    48: {0:41, 5:42, 10:42, 15:43, 20:44, 25:45, 30:45, 35:46, 40:47, 45:48, 50:48, 55:49, 60:50, 65:51, 70:51, 75:52, 80:53, 85:54, 90:54, 95:55},
    50: {0:43, 5:43, 10:44, 15:45, 20:45, 25:46, 30:47, 35:48, 40:49, 45:49, 50:50, 55:51, 60:52, 65:52, 70:53, 75:54, 80:55, 85:55, 90:56, 95:57},
    52: {0:44, 5:45, 10:46, 15:46, 20:47, 25:48, 30:49, 35:49, 40:50, 45:51, 50:52, 55:52, 60:53, 65:54, 70:55, 75:55, 80:56, 85:57, 90:58, 95:58},
    54: {0:46, 5:47, 10:47, 15:48, 20:49, 25:50, 30:50, 35:51, 40:52, 45:53, 50:53, 55:54, 60:55, 65:56, 70:55, 75:57, 80:58, 85:59, 90:59, 95:60},
    56: {0:48, 5:48, 10:49, 15:50, 20:51, 25:51, 30:52, 35:53, 40:54, 45:54, 50:55, 55:56, 60:57, 65:57, 70:58, 75:59, 80:60, 85:60, 90:61, 95:62},
    58: {0:49, 5:50, 10:51, 15:52, 20:52, 25:53, 30:54, 35:55, 40:55, 45:56, 50:57, 55:58, 60:58, 65:59, 70:60, 75:61, 80:61, 85:62, 90:63, 95:64},
    60: {0:51, 5:52, 10:53, 15:53, 20:54, 25:55, 30:56, 35:56, 40:57, 45:58, 50:59, 55:59, 60:60, 65:61, 70:62, 75:62, 80:63, 85:64, 90:65, 95:65},
    62: {0:53, 5:53, 10:54, 15:55, 20:56, 25:56, 30:57, 35:58, 40:59, 45:59, 50:60, 55:61, 60:62, 65:62, 70:63, 75:64, 80:65, 85:65, 90:66, 95:67},
    64: {0:54, 5:55, 10:56, 15:57, 20:57, 25:58, 30:59, 35:60, 40:60, 45:61, 50:62, 55:63, 60:63, 65:64, 70:65, 75:66, 80:66, 85:67, 90:68, 95:69},
    66: {0:56, 5:57, 10:58, 15:58, 20:59, 25:60, 30:61, 35:61, 40:62, 45:63, 50:64, 55:64, 60:65, 65:66, 70:67, 75:67, 80:68, 85:69, 90:70, 95:70},
    68: {0:58, 5:59, 10:59, 15:60, 20:61, 25:62, 30:62, 35:63, 40:64, 45:65, 50:65, 55:66, 60:67, 65:68, 70:68, 75:69, 80:70, 85:71, 90:71, 95:72},
    70: {0:60, 5:60, 10:61, 15:62, 20:63, 25:63, 30:64, 35:65, 40:66, 45:66, 50:67, 55:68, 60:69, 65:69, 70:70, 75:71, 80:72, 85:72, 90:73, 95:74},
    72: {0:61, 5:62, 10:63, 15:63, 20:64, 25:65, 30:66, 35:66, 40:67, 45:68, 50:69, 55:70, 60:71, 65:72, 70:72, 75:73, 80:74, 85:75, 90:75, 95:75},
    74: {0:63, 5:64, 10:64, 15:65, 20:66, 25:67, 30:67, 35:68, 40:69, 45:70, 50:70, 55:71, 60:72, 65:73, 70:73, 75:74, 80:75, 85:76, 90:76, 95:77},
    76: {0:65, 5:65, 10:66, 15:67, 20:68, 25:68, 30:69, 35:70, 40:71, 45:71, 50:72, 55:73, 60:74, 65:74, 70:75, 75:76, 80:77, 85:77, 90:78, 95:79},
    78: {0:66, 5:67, 10:68, 15:69, 20:69, 25:70, 30:71, 35:72, 40:72, 45:73, 50:74, 55:75, 60:75, 65:76, 70:77, 75:78, 80:78, 85:79, 90:80, 95:81},
    80: {0:68, 5:69, 10:70, 15:70, 20:71, 25:72, 30:73, 35:73, 40:74, 45:75, 50:76, 55:76, 60:77, 65:78, 70:79, 75:79, 80:80, 85:81, 90:82, 95:82},
    82: {0:70, 5:70, 10:71, 15:72, 20:73, 25:73, 30:74, 35:75, 40:76, 45:76, 50:77, 55:78, 60:79, 65:79, 70:80, 75:81, 80:82, 85:82, 90:83, 95:84},
    84: {0:71, 5:72, 10:73, 15:74, 20:74, 25:75, 30:76, 35:77, 40:77, 45:78, 50:79, 55:80, 60:80, 65:81, 70:82, 75:83, 80:83, 85:84, 90:85, 95:86},
    86: {0:73, 5:74, 10:75, 15:75, 20:76, 25:77, 30:78, 35:78, 40:79, 45:80, 50:81, 55:81, 60:82, 65:83, 70:84, 75:84, 80:85, 85:86, 90:87, 95:87},
    88: {0:75, 5:76, 10:76, 15:77, 20:78, 25:79, 30:79, 35:80, 40:81, 45:82, 50:82, 55:83, 60:84, 65:85, 70:85, 75:86, 80:87, 85:88, 90:88, 95:89},
    90: {0:77, 5:77, 10:78, 15:79, 20:80, 25:80, 30:81, 35:82, 40:83, 45:83, 50:84, 55:85, 60:86, 65:86, 70:87, 75:88, 80:89, 85:89, 90:90, 95:91}
}

def get_ts(tw_f: float, tg_f: float) -> int:
    tw = int(tw_f / 2 + 0.5) * 2
    tg = int(tg_f / 5 + 0.5) * 5
    tw = max(36, min(90, tw))
    tg = max(0, min(95, tg))
    return temperature_control_table[tw][tg]

# --- Fonction pour créer un rapport PDF complet ---
def create_pdf_report(df, unit, figures_dict):
    """
    Crée un rapport PDF complet avec tous les tableaux et graphiques
    
    Args:
        df: DataFrame avec les données
        unit: Unité de mesure
        figures_dict: Dictionnaire contenant toutes les figures matplotlib
        
    Returns:
        Bytes du fichier PDF
    """
    buffer = io.BytesIO()
    
    with PdfPages(buffer) as pdf:
        # Page 1: Page de titre
        fig_title = plt.figure(figsize=(8.5, 11))
        fig_title.text(0.5, 0.7, 'Rapport d\'Analyse ERT', 
                      ha='center', va='center', fontsize=24, fontweight='bold')
        fig_title.text(0.5, 0.6, 'Ravensgate Sonic Water Level Meter', 
                      ha='center', va='center', fontsize=16)
        fig_title.text(0.5, 0.5, f'Date: {datetime.now().strftime("%d/%m/%Y %H:%M")}', 
                      ha='center', va='center', fontsize=12)
        fig_title.text(0.5, 0.4, f'Total mesures: {len(df)}', 
                      ha='center', va='center', fontsize=12)
        fig_title.text(0.5, 0.35, f'Points de sondage: {df["survey_point"].nunique()}', 
                      ha='center', va='center', fontsize=12)
        fig_title.text(0.5, 0.3, f'Unité: {unit}', 
                      ha='center', va='center', fontsize=12)
        plt.axis('off')
        pdf.savefig(fig_title, bbox_inches='tight')
        plt.close(fig_title)
        
        # Page 2: Statistiques descriptives
        fig_stats = plt.figure(figsize=(8.5, 11))
        ax_stats = fig_stats.add_subplot(111)
        
        stats_data = [
            ['Total mesures', len(df)],
            ['Points de sondage', df['survey_point'].nunique()],
            ['Profondeurs uniques', df['depth'].nunique()],
            [f'DTW moyen ({unit})', f"{df['data'].mean():.2f}"],
            [f'DTW min ({unit})', f"{df['data'].min():.2f}"],
            [f'DTW max ({unit})', f"{df['data'].max():.2f}"],
            [f'Écart-type ({unit})', f"{df['data'].std():.2f}"],
        ]
        
        table_stats = ax_stats.table(cellText=stats_data, 
                                     colLabels=['Statistique', 'Valeur'],
                                     cellLoc='left', loc='center',
                                     colWidths=[0.6, 0.4])
        table_stats.auto_set_font_size(False)
        table_stats.set_fontsize(10)
        table_stats.scale(1, 2)
        ax_stats.axis('off')
        ax_stats.set_title('Statistiques descriptives', fontsize=16, fontweight='bold', pad=20)
        pdf.savefig(fig_stats, bbox_inches='tight')
        plt.close(fig_stats)
        
        # Page 3+: Statistiques par profondeur
        depth_stats = df.groupby('depth')['data'].agg(['mean', 'min', 'max', 'std']).round(2)
        
        fig_depth = plt.figure(figsize=(8.5, 11))
        ax_depth = fig_depth.add_subplot(111)
        
        depth_data = [[f"{idx:.1f}", f"{row['mean']:.2f}", f"{row['min']:.2f}", 
                      f"{row['max']:.2f}", f"{row['std']:.2f}"] 
                     for idx, row in depth_stats.iterrows()]
        
        table_depth = ax_depth.table(cellText=depth_data,
                                    colLabels=['Profondeur', 'Moyenne DTW', 'Min DTW', 'Max DTW', 'Écart-type'],
                                    cellLoc='center', loc='center',
                                    colWidths=[0.2, 0.2, 0.2, 0.2, 0.2])
        table_depth.auto_set_font_size(False)
        table_depth.set_fontsize(9)
        table_depth.scale(1, 1.5)
        ax_depth.axis('off')
        ax_depth.set_title(f'Statistiques par profondeur ({unit})', fontsize=16, fontweight='bold', pad=20)
        pdf.savefig(fig_depth, bbox_inches='tight')
        plt.close(fig_depth)
        
        # Ajouter toutes les figures fournies
        for fig_name, fig in figures_dict.items():
            if fig is not None:
                pdf.savefig(fig, bbox_inches='tight')
        
        # Métadonnées du PDF
        d = pdf.infodict()
        d['Title'] = 'Rapport Analyse ERT - Ravensgate Sonic'
        d['Author'] = 'ERTest Application'
        d['Subject'] = 'Analyse des niveaux d\'eau souterraine'
        d['Keywords'] = 'ERT, Ravensgate, Water Level, DTW'
        d['CreationDate'] = datetime.now()
    
    buffer.seek(0)
    return buffer.getvalue()

# --- Parsing .dat robuste avec cache ---
@st.cache_data
def detect_encoding(file_bytes):
    """Détecte l'encodage depuis les bytes du fichier"""
    result = chardet.detect(file_bytes[:100000])
    return result['encoding'] or 'utf-8'

@st.cache_data
def parse_dat(file_content, encoding):
    """Parse le contenu du fichier .dat avec mise en cache"""
    try:
        from io import StringIO
        df = pd.read_csv(
            StringIO(file_content.decode(encoding)), 
            delim_whitespace=True, header=None, comment='#',
            names=['survey_point', 'depth', 'data', 'project'],
            on_bad_lines='skip', engine='python'
        )
        df['depth'] = pd.to_numeric(df['depth'], errors='coerce')
        df['data'] = pd.to_numeric(df['data'], errors='coerce')
        df = df.dropna(subset=['depth', 'data'])
        return df
    except Exception as e:
        st.error(f"Erreur parsing : {e}")
        return pd.DataFrame()

# --- Tableau des types d'eau ---
water_html = """
<style>
.water-table th { background-color: #333; color: white; padding: 12px; text-align: center; }
.water-table td { padding: 12px; text-align: center; border-bottom: 1px solid #ddd; }
</style>
<table class="water-table" style="width:100%; border-collapse: collapse; margin: 20px 0;">
  <tr>
    <th>Type d'eau</th>
    <th>Résistivité (Ω.m)</th>
    <th>Couleur associée</th>
    <th>Description</th>
  </tr>
  <tr style="background-color: #FF4500; color: white;">
    <td><strong>Eau de mer</strong></td>
    <td>0.1 – 1</td>
    <td>Rouge vif / Orange</td>
    <td>Eau océanique hautement salée (∼35 g/L de sel). Très forte conductivité électrique due aux ions Na⁺ et Cl⁻. Typique des mers et océans.</td>
  </tr>
  <tr style="background-color: #FFD700; color: black;">
    <td><strong>Eau salée (nappe)</strong></td>
    <td>1 – 10</td>
    <td>Jaune / Orange</td>
    <td>Eau saumâtre dans les nappes phréatiques côtières (intrusion saline). Salinité intermédiaire, souvent non potable sans traitement.</td>
  </tr>
  <tr style="background-color: #90EE90; color: black;">
    <td><strong>Eau douce</strong></td>
    <td>10 – 100</td>
    <td>Vert / Bleu clair</td>
    <td>Eau potable standard (rivières, lacs, nappes intérieures). Faiblement minéralisée, conductivité modérée.</td>
  </tr>
  <tr style="background-color: #00008B; color: white;">
    <td><strong>Eau très pure</strong></td>
    <td>> 100</td>
    <td>Bleu foncé</td>
    <td>Eau ultra-pure (distillée, déminéralisée, pluie). Presque pas d'ions → très faible conductivité. Utilisée en laboratoire/industrie.</td>
  </tr>
</table>
"""

# --- Seed pour reproductibilité des exemples ---
np.random.seed(42)

# --- Interface Streamlit ---
st.set_page_config(page_title="Ravensgate Sonic Tool", layout="wide", initial_sidebar_state="expanded")
st.title("🪣 Ravensgate Sonic Water Level Meter – Outil Complet (07 Novembre 2025)")

tab1, tab2, tab3 = st.tabs(["🌡️ Calculateur Réglage Température", "📊 Analyse Fichiers .dat", "🌍 ERT Pseudo-sections 2D/3D"])

# ===================== TAB 1 : TEMPÉRATURE =====================
with tab1:
    st.header("Calculateur de réglage Ts (Table officielle Ravensgate)")
    st.markdown("""
    Entrez la température de l'eau du puits (**Tw**) et la température moyenne quotidienne de surface (**Tg**).  
    L'app arrondit **conventionnellement (half-up)** aux pas du tableau et clamp automatiquement.
    
    **Exemple du manuel** : Tw = 58 °F (14 °C), Tg = 85 °F (29 °C) → **Ts = 62 °F** (17 °C).
    """)

    unit = st.radio("Unité", options=["°F", "°C"], horizontal=True)

    if unit == "°C":
        col1, col2 = st.columns(2)
        with col1:
            tw_c = st.number_input("Tw – Température eau puits (°C)", value=10.0, min_value=-10.0, max_value=50.0, step=0.1)
        with col2:
            tg_c = st.number_input("Tg – Température surface moyenne (°C)", value=20.0, min_value=-30.0, max_value=50.0, step=0.1)
        tw_f = tw_c * 9/5 + 32
        tg_f = tg_c * 9/5 + 32
    else:
        col1, col2 = st.columns(2)
        with col1:
            tw_f = st.number_input("Tw – Température eau puits (°F)", value=60.0, min_value=20.0, max_value=120.0, step=0.5)
        with col2:
            tg_f = st.number_input("Tg – Température surface moyenne (°F)", value=70.0, min_value=-20.0, max_value=120.0, step=0.5)

    if st.button("🔥 Calculer Ts", type="primary", use_container_width=True):
        ts = get_ts(tw_f, tg_f)
        tw_used = max(36, min(90, int(tw_f / 2 + 0.5) * 2))
        tg_used = max(0, min(95, int(tg_f / 5 + 0.5) * 5))

        st.success(f"**Réglage recommandé sur l'appareil → Ts = {ts} °F**")

        if unit == "°C":
            st.info(f"Tw utilisée → {tw_used} °F ({(tw_used - 32)*5/9:.1f} °C) | Tg utilisée → {tg_used} °F ({(tg_used - 32)*5/9:.1f} °C)")
        else:
            st.info(f"Tw utilisée → {tw_used} °F | Tg utilisée → {tg_used} °F")

    with st.expander("📋 Tableau complet Ravensgate (cliquer pour déplier)"):
        tg_cols = list(range(0, 96, 5))
        df_table = pd.DataFrame.from_dict(temperature_control_table, orient='index', columns=tg_cols)
        df_table.index.name = "Tw \\ Tg"
        df_table = df_table.sort_index()
        df_table.insert(0, "Tw (°F)", df_table.index)
        st.dataframe(df_table.style.background_gradient(cmap='coolwarm', axis=None), use_container_width=True)

    with st.expander("💧 Valeurs typiques pour l'eau – Résistivité & Couleurs associées"):
        st.markdown("### **2. Valeurs typiques pour l'eau**")
        st.markdown(water_html, unsafe_allow_html=True)
        st.caption("Ces valeurs sont indicatives. Les couleurs sont couramment utilisées dans les cartes de résistivité électrique (ERT) pour visualiser la salinité/qualité de l'eau souterraine.")

# ===================== TAB 2 : ANALYSE .DAT =====================
with tab2:
    st.header("2 Analyse de fichiers .dat de Ravensgate Sonic Water Level Meter")
    
    st.markdown("""
    ### Format attendu dans le .dat :
    - **Date** : Format YYYY/MM/DD HH:MM:SS
    - **Survey Point** (Point de forage)
    - **Depth From** et **Depth To** (Profondeur de mesure)
    - **Data** : Niveau d'eau (DTW - Depth To Water)
    """)
    
    # Initialiser l'état de session
    if 'uploaded_data' not in st.session_state:
        st.session_state['uploaded_data'] = None
    
    uploaded_file = st.file_uploader("📂 Uploader un fichier .dat", type=["dat"])
    
    if uploaded_file is not None:
        # Lire le contenu du fichier en bytes (avec cache)
        file_bytes = uploaded_file.read()
        encoding = detect_encoding(file_bytes)
        
        # Parser le fichier (avec cache)
        df = parse_dat(file_bytes, encoding)
        
        # Déterminer l'unité
        unit = 'm'  # Par défaut
        
        if not df.empty:
            st.success(f"✅ {len(df)} lignes chargées avec succès")
            
            # Sauvegarder dans l'état de session pour l'onglet 3
            st.session_state['uploaded_data'] = df.copy()
            st.session_state['unit'] = unit
            
            # Affichage du DataFrame
            st.dataframe(df.head(50), use_container_width=True)
            
            # Statistiques de base
            st.subheader("📊 Statistiques descriptives")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total mesures", len(df))
            with col2:
                st.metric("Points de sondage", df['survey_point'].nunique())
            with col3:
                st.metric(f"DTW moyen ({unit})", f"{df['data'].mean():.2f}")
            with col4:
                st.metric(f"DTW max ({unit})", f"{df['data'].max():.2f}")
            
            # Graphique temporel
            st.subheader("📈 Évolution temporelle du niveau d'eau")
            
            # Dictionnaire pour stocker toutes les figures
            figures_dict = {}
            
            # Vérifier si colonne 'date' existe
            if 'date' in df.columns:
                fig_time, ax = plt.subplots(figsize=(12, 5), dpi=150)
                for sp in sorted(df['survey_point'].unique()):
                    subset = df[df['survey_point'] == sp]
                    ax.plot(subset['date'], subset['data'], marker='o', label=f'SP {int(sp)}', markersize=4)
                ax.set_xlabel('Date', fontsize=11)
                ax.set_ylabel(f'DTW ({unit})', fontsize=11)
                ax.set_title('Niveau d\'eau par point de sondage', fontsize=13, fontweight='bold')
                ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
                ax.grid(True, alpha=0.3)
                plt.xticks(rotation=45)
                plt.tight_layout()
                st.pyplot(fig_time)
                
                # Sauvegarder pour PDF
                figures_dict['temporal_evolution'] = fig_time
            else:
                st.info("⚠️ Pas de colonne 'date' dans le fichier - graphique temporel indisponible")
                fig_time = None
            
            # Détection d'anomalies
            st.subheader("🔍 Détection d'anomalies (K-Means)")
            n_clusters = st.slider("Nombre de clusters", 2, 5, 3, key='kmeans_slider')
            
            # Cache du calcul KMeans basé sur les données + nombre de clusters
            @st.cache_data
            def compute_kmeans(data_hash, n_clust):
                """Calcul KMeans avec cache"""
                X = df[['survey_point', 'depth', 'data']].values
                kmeans = KMeans(n_clusters=n_clust, random_state=42, n_init=10)
                return kmeans.fit_predict(X)
            
            # Hash unique des données pour invalidation du cache
            data_hash = hash(tuple(df[['survey_point', 'depth', 'data']].values.flatten()))
            clusters = compute_kmeans(data_hash, n_clusters)
            df_viz = df.copy()
            df_viz['cluster'] = clusters
            
            fig_cluster, ax = plt.subplots(figsize=(12, 6), dpi=150)
            scatter = ax.scatter(df_viz['survey_point'], df_viz['depth'], c=df_viz['cluster'], 
                                cmap='viridis', s=50, alpha=0.6, edgecolors='black', linewidths=0.5)
            plt.colorbar(scatter, ax=ax, label='Cluster')
            ax.set_xlabel('Point de sondage', fontsize=11)
            ax.set_ylabel(f'Profondeur ({unit})', fontsize=11)
            ax.set_title(f'Classification en {n_clusters} groupes', fontsize=13, fontweight='bold')
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig_cluster)
            
            # Sauvegarder pour PDF
            figures_dict['kmeans_clustering'] = fig_cluster
            
            # Export
            st.subheader("💾 Exporter les résultats")
            col1, col2, col3 = st.columns(3)
            with col1:
                csv = df.to_csv(index=False).encode('utf-8')
                st.download_button("📥 CSV", csv, "analysis.csv", "text/csv", key='download_csv')
            with col2:
                # Créer Excel uniquement à la demande (lazy loading)
                if st.button("� Préparer Excel", key='prepare_excel'):
                    buffer = io.BytesIO()
                    with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                        df.to_excel(writer, index=False, sheet_name='Data')
                    st.session_state['excel_buffer'] = buffer.getvalue()
                    st.success("✅ Excel prêt !")
                
                if 'excel_buffer' in st.session_state:
                    st.download_button("📥 Excel", st.session_state['excel_buffer'], 
                                      "analysis.xlsx", 
                                      "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                      key='download_excel')
            with col3:
                # Générer PDF avec tous les graphiques et tableaux
                if st.button("📄 Générer Rapport PDF", key='generate_pdf'):
                    with st.spinner('Génération du PDF en cours...'):
                        pdf_bytes = create_pdf_report(df, unit, figures_dict)
                        st.session_state['pdf_buffer'] = pdf_bytes
                        st.success("✅ PDF prêt !")
                
                if 'pdf_buffer' in st.session_state:
                    st.download_button(
                        "📥 PDF Complet",
                        st.session_state['pdf_buffer'],
                        f"rapport_ert_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                        "application/pdf",
                        key='download_pdf'
                    )
# ===================== TAB 3 : ERT PSEUDO-SECTIONS 2D/3D =====================
with tab3:
    st.header("4 Interprétation des pseudo-sections et modèles de résistivité (FicheERT.pdf)")

    st.subheader("4.1 Définition d'une pseudo-section")
    st.markdown("""
La première étape dans l'interprétation des données en tomographie électrique consiste à construire une **pseudo-section**. Une pseudo-section est une carte de résultat qui présente les valeurs des résistivités apparentes calculées à partir de la différence de potentiel mesurée aux bornes de deux électrodes de mesure ainsi que de la valeur du courant injecté entre les deux électrodes d'injection.

La couleur d'un point sur la pseudo-section représente donc la valeur de la résistivité apparente en ce point.
    """)

    # Vérifier si des données ont été chargées dans l'onglet 2
    if st.session_state.get('uploaded_data') is not None:
        df = st.session_state['uploaded_data']
        unit = st.session_state.get('unit', 'm')
        
        st.success(f"✅ Utilisation des données du fichier uploadé : {len(df)} mesures")
        
        st.markdown("**Pseudo-sections générées à partir de vos données réelles**")
        
        # Cache de la préparation des données 2D
        @st.cache_data
        def prepare_2d_data(data_hash):
            """Prépare les données pour visualisation 2D avec cache"""
            survey_points = sorted(df['survey_point'].unique())
            depths = sorted(df['depth'].unique())
            
            X_real = []
            Z_real = []
            Rho_real = []
            
            for sp in survey_points:
                for depth in depths:
                    subset = df[(df['survey_point'] == sp) & (df['depth'] == depth)]
                    if len(subset) > 0:
                        X_real.append(float(sp))
                        Z_real.append(abs(float(depth)))
                        Rho_real.append(float(subset['data'].values[0]))
            
            return np.array(X_real), np.array(Z_real), np.array(Rho_real)
        
        # Cache de l'interpolation (très coûteuse)
        @st.cache_data
        def interpolate_grid(X, Z, Rho, data_hash):
            """Interpolation cubique avec cache"""
            from scipy.interpolate import griddata
            xi = np.linspace(X.min(), X.max(), 100)
            zi = np.linspace(Z.min(), Z.max(), 50)
            Xi, Zi = np.meshgrid(xi, zi)
            Rhoi = griddata((X, Z), Rho, (Xi, Zi), method='cubic')
            return Xi, Zi, Rhoi, xi, zi
        
        # Hash unique des données
        data_hash = hash(tuple(df[['survey_point', 'depth', 'data']].values.flatten()))
        
        st.subheader("📊 Pseudo-section 2D - Données réelles du fichier .dat")
        
        # Dictionnaire pour stocker les figures du Tab 3
        figures_tab3 = {}
        
        # Préparer les données (avec cache)
        X_real, Z_real, Rho_real = prepare_2d_data(data_hash)
        
        # Interpoler (avec cache)
        Xi, Zi, Rhoi, xi, zi = interpolate_grid(X_real, Z_real, Rho_real, data_hash)
        
        # Pseudo-section 2D avec données réelles (haute résolution pour PDF)
        fig_real, ax = plt.subplots(figsize=(14, 7), dpi=150)
        
        # Utiliser une échelle de couleur adaptée aux valeurs d'eau
        vmin, vmax = Rho_real.min(), Rho_real.max()
        
        pcm = ax.pcolormesh(Xi, Zi, Rhoi, cmap='jet_r', shading='auto', 
                           vmin=vmin, vmax=vmax)
        
        # Ajouter les points de mesure réels
        scatter = ax.scatter(X_real, Z_real, c=Rho_real, cmap='jet_r', 
                            s=50, edgecolors='black', linewidths=0.5,
                            vmin=vmin, vmax=vmax, zorder=10)
        
        fig_real.colorbar(pcm, ax=ax, label=f'Niveau d\'eau DTW ({unit})', extend='both')
        ax.invert_yaxis()
        ax.set_xlabel('Point de sondage (Survey Point)', fontsize=11)
        ax.set_ylabel(f'Profondeur totale ({unit})', fontsize=11)
        ax.set_title(f'Pseudo-section 2D - Données réelles ({len(df)} mesures)', 
                    fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='--')
        plt.tight_layout()
        
        st.pyplot(fig_real)
        
        # Sauvegarder pour PDF
        figures_tab3['pseudo_section_2d'] = fig_real
        
        # Légende des couleurs basée sur les valeurs réelles
        st.markdown(f"""
**Interprétation des couleurs (basée sur vos données) :**
- Valeur minimale : **{vmin:.2f} {unit}** (niveau d'eau le plus bas) → couleur bleue
- Valeur moyenne : **{Rho_real.mean():.2f} {unit}** → couleur intermédiaire
- Valeur maximale : **{vmax:.2f} {unit}** (niveau d'eau le plus haut) → couleur rouge

Les zones rouges indiquent des niveaux d'eau plus élevés (DTW plus grand).
Les zones bleues indiquent des niveaux d'eau plus bas (nappe plus proche de la surface).
        """)
        
        # Vue 3D des données réelles
        survey_points = sorted(df['survey_point'].unique())
        depths = sorted(df['depth'].unique())
        
        if len(survey_points) > 2 and len(depths) > 2:
            st.subheader("🌐 Modèle 3D - Volume d'eau (données réelles)")
            
            fig3d_real = go.Figure(data=go.Scatter3d(
                x=X_real,
                y=np.zeros_like(X_real),  # Y=0 pour profil 2D
                z=-Z_real,  # Négatif pour afficher en profondeur
                mode='markers',
                marker=dict(
                    size=8,
                    color=Rho_real,
                    colorscale='Jet',
                    showscale=True,
                    colorbar=dict(title=f'DTW ({unit})'),
                    line=dict(width=0.5, color='black')
                ),
                text=[f'SP: {int(X_real[i])}<br>Depth: {Z_real[i]:.1f}{unit}<br>DTW: {Rho_real[i]:.2f}{unit}' 
                      for i in range(len(X_real))],
                hoverinfo='text'
            ))
            
            fig3d_real.update_layout(
                scene=dict(
                    xaxis_title='Point de sondage',
                    yaxis_title='Transect (m)',
                    zaxis_title=f'Profondeur ({unit})',
                    aspectmode='data'
                ),
                title='Visualisation 3D des mesures de niveau d\'eau',
                height=600
            )
            
            st.plotly_chart(fig3d_real, use_container_width=True)
        
        # Statistiques par profondeur
        st.subheader("📈 Analyse par profondeur")
        
        # Cache du calcul statistique
        @st.cache_data
        def compute_depth_stats(data_hash):
            """Calcul des statistiques par profondeur avec cache"""
            depth_stats = df.groupby('depth')['data'].agg(['mean', 'min', 'max', 'std']).round(2)
            depth_stats.columns = ['Moyenne DTW', 'Min DTW', 'Max DTW', 'Écart-type']
            return depth_stats
        
        depth_stats = compute_depth_stats(data_hash)
        st.dataframe(depth_stats.style.background_gradient(cmap='RdYlBu_r', axis=0), use_container_width=True)
        
        # Export PDF des pseudo-sections
        st.subheader("📄 Export PDF des Pseudo-sections")
        col_pdf1, col_pdf2 = st.columns([1, 2])
        with col_pdf1:
            if st.button("📄 Générer PDF Pseudo-sections", key='generate_pdf_tab3'):
                with st.spinner('Génération du PDF des pseudo-sections...'):
                    pdf_bytes = create_pdf_report(df, unit, figures_tab3)
                    st.session_state['pdf_tab3_buffer'] = pdf_bytes
                    st.success("✅ PDF pseudo-sections prêt !")
        
        with col_pdf2:
            if 'pdf_tab3_buffer' in st.session_state:
                st.download_button(
                    "📥 Télécharger PDF Pseudo-sections",
                    st.session_state['pdf_tab3_buffer'],
                    f"pseudo_sections_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                    "application/pdf",
                    key='download_pdf_tab3'
                )
        
    else:
        st.warning("⚠️ Aucune donnée chargée. Veuillez d'abord uploader un fichier .dat dans l'onglet 'Analyse Fichiers .dat'")
        st.info("💡 Uploadez un fichier .dat dans l'onglet 'Analyse Fichiers .dat' pour visualiser vos données avec interprétation des couleurs de résistivité.")

# --- Sidebar ---
st.sidebar.image("logo_belikan.png", use_container_width=True)
st.sidebar.markdown("""
**Belikan M. - Analyse ERT**  
Outil d'analyse géophysique  
Expert en hydrogéologie et ERT

**Outil optimisé – 07 Novembre 2025**  
✅ Calculateur Ts intelligent (Ravensgate Sonic)
✅ Analyse .dat + détection anomalies (K-Means avec cache)  
✅ Tableau résistivité eau (descriptions détaillées)  
✅ Pseudo-sections 2D/3D basées sur vos données réelles  
✅ Interprétation couleurs : résistivité → type d'eau → minéraux  
✅ Performance optimisée avec @st.cache_data  
✅ Interpolation cubique cachée pour fluidité  
✅ Zéro exemples synthétiques - Données réelles uniquement  
✅ **Export PDF** : Rapports complets avec tous les graphiques

**Exports disponibles** :  
📥 CSV - Données brutes  
📊 Excel - Tableaux formatés  
📄 PDF - Rapports graphiques haute qualité (150 DPI)

**Légende couleurs ERT** :  
- 🔴 Rouge/orange (<10 Ω·m) : Eau salée/mer  
- 🟡 Jaune (10-100 Ω·m) : Eau saumâtre/douce  
- 🔵 Bleu (>100 Ω·m) : Eau pure/roche sèche  
""")

