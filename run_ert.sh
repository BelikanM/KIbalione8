#!/bin/bash
# Script de lancement de ERT.py avec l'environnement gestmodo

cd /home/belikan/KIbalione8
~/miniconda3/envs/gestmodo/bin/python -m streamlit run ERT.py --server.port 8503
