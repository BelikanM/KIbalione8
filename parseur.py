import streamlit as st
import pandas as pd
import os

st.set_page_config(page_title="ERT Parser", layout="wide")

st.title("Logiciel de Parsing ERT / Multi-fichier")

uploaded_files = st.file_uploader("Chargez vos fichiers .dat ou .txt", type=['dat','txt'], accept_multiple_files=True)

if uploaded_files:
    all_data = []

    for uploaded_file in uploaded_files:
        # Lecture du fichier
        content = uploaded_file.read().decode('utf-8', errors='ignore')
        lines = content.split('\n')

        # Détection des fréquences (en-tête contenant MHz)
        freq_line = None
        for line in lines[:5]:
            if 'MHz' in line:
                freq_line = line
                break

        if freq_line:
            freqs = [float(f.strip().replace('MHz','')) for f in freq_line.split(',') if 'MHz' in f]
        else:
            freqs = [0]  # si fichier mono-fréquence

        # Lecture des données (on suppose que la première ligne après l'en-tête contient le projet)
        data_lines = [line for line in lines if line and not any(c in line for c in ['MHz', '����'])]

        for line in data_lines:
            parts = line.strip().split(',')
            if len(parts) < 4:
                continue
            project = parts[0]
            survey_point = int(parts[1])
            depth = float(parts[2])
            resistivities = [float(r) for r in parts[3:] if r]

            # On fait le jumelage avec les fréquences si elles correspondent
            for i, r in enumerate(resistivities):
                freq = freqs[i] if i < len(freqs) else 0
                all_data.append([project, survey_point, depth, freq, r])

    # Création du DataFrame final
    df = pd.DataFrame(all_data, columns=['project','survey_point','depth','frequency_MHz','resistivity'])

    st.subheader("Tableau ERT fusionné")
    st.dataframe(df)

    # Export CSV
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Télécharger CSV",
        data=csv,
        file_name='ERT_fusion.csv',
        mime='text/csv'
    )

    # Export Excel
    excel_path = 'ERT_fusion.xlsx'
    df.to_excel(excel_path, index=False)
    with open(excel_path, 'rb') as f:
        st.download_button('Télécharger Excel', f.read(), file_name='ERT_fusion.xlsx')
