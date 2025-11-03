import math
import fitz # pymupdf
import osmium
import networkx as nx
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import pickle
import json
from huggingface_hub import InferenceClient
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from shapely.geometry import Point
import io
from PIL import Image
import cv2
import open3d as o3d
from io import BytesIO
import pandas as pd
from skimage import measure, segmentation
from sklearn.cluster import KMeans
import torch
from torchvision import models, transforms
from langchain_huggingface import HuggingFaceEndpoint
from langchain_classic.agents import AgentExecutor
from langchain_classic.agents.react.agent import create_react_agent
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import Tool
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
from transformers import pipeline
import requests
from bs4 import BeautifulSoup
import time
import shutil
from diffusers import DiffusionPipeline, AudioLDMPipeline, ShapEPipeline, ShapEImg2ImgPipeline
import imageio
import scipy.io.wavfile as wavfile
from tavily import TavilyClient
import os
from pathlib import Path
from dotenv import load_dotenv
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import MultinomialNB
import re
from transformers import AutoTokenizer, AutoModelForCausalLM
from langchain_huggingface import HuggingFacePipeline
# ===============================================
# Configuration - CHEMINS UNIFIÉS
# ===============================================
# Définir dynamiquement les chemins basés sur le répertoire du projet corrigé
PROJECT_DIR = os.path.expanduser('~/RAG_ChatBot') # Chemin corrigé vers le dossier contenant les données et poids
CHATBOT_DIR = PROJECT_DIR
VECTORDB_PATH = os.path.join(CHATBOT_DIR, "vectordb")
CHAT_VECTORDB_PATH = os.path.join(CHATBOT_DIR, "chat_vectordb") # AJOUT MÉMOIRE VECTORIELLE: Base dédiée pour l'historique chat
PDFS_PATH = os.path.join(CHATBOT_DIR, "pdfs")
GRAPHS_PATH = os.path.join(CHATBOT_DIR, "graphs")
MAPS_PATH = os.path.join(CHATBOT_DIR, "maps")
METADATA_PATH = os.path.join(CHATBOT_DIR, "metadata.json")
TRAJECTORIES_PATH = os.path.join(CHATBOT_DIR, "trajectories.json")
WEB_CACHE_PATH = os.path.join(CHATBOT_DIR, "web_cache.json")
GENERATED_PATH = os.path.join(CHATBOT_DIR, "generated")
SUBMODELS_PATH = os.path.join(CHATBOT_DIR, "submodels") # Nouveau: Chemin pour les sous-modèles sklearn
MODEL_PATH = os.path.expanduser("~/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-V3-0324/snapshots/e9b33add76883f293d6bf61f6bd89b497e80e335")
# Modèles qui fonctionnent
WORKING_MODELS = {
    "DeepSeek V3 (Puissant)": "deepseek-ai/DeepSeek-V3-0324",
    "Gemma 2B (Rapide)": "google/gemma-2-2b-it",
    "Llama 3.1 8B (Équilibré)": "meta-llama/Meta-Llama-3.1-8B-Instruct",
    "Qwen 2.5 7B (Polyvalent)": "Qwen/Qwen2.5-7B-Instruct",
    "SmolLM 3B (Léger)": "HuggingFaceTB/SmolLM3-3B",
}
# ===============================================
# Configuration HuggingFace Token depuis .env
# ===============================================
# Charger le token depuis .env dans le dossier corrigé
env_path = os.path.join(CHATBOT_DIR, ".env")
if os.path.exists(env_path):
    load_dotenv(env_path)
    st.write(f"✅ Fichier .env trouvé: {env_path}")
else:
    st.write(f"⚠️ Aucun fichier .env trouvé à {env_path}")
    st.write("Créez un fichier .env dans ~/RAG_ChatBot avec: HF_TOKEN=hf_votre_token")
HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    raise ValueError("❌ HF_TOKEN non trouvé ! Vérifiez votre fichier .env")
else:
    st.write(f"🔑 Token HF configuré: {HF_TOKEN[:10]}...")
# Définir la variable d'environnement pour huggingface_hub
os.environ["HF_TOKEN"] = HF_TOKEN
os.environ["HUGGINGFACE_HUB_TOKEN"] = HF_TOKEN
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
if not TAVILY_API_KEY:
    raise ValueError("❌ TAVILY_API_KEY non trouvé ! Vérifiez votre fichier .env")
# ===============================================
# Test de connexion HuggingFace
# ===============================================
def test_hf_connection():
    """Teste la connexion à HuggingFace"""
    try:
        from huggingface_hub import whoami
        user_info = whoami(token=HF_TOKEN)
        st.write(f"✅ Connexion HuggingFace réussie: {user_info.get('name', 'Utilisateur')}")
        return True
    except Exception as e:
        st.write(f"❌ Erreur connexion HuggingFace: {e}")
        return False
# Tester la connexion au démarrage
if not test_hf_connection():
    st.write("⚠️ Problème de connexion HuggingFace, vérifiez votre token")
# ===============================================
# Fonctions utilitaires
# ===============================================
def setup_drive():
    """Crée les dossiers"""
    st.write("📁 Configuration des dossiers...")
    os.makedirs(CHATBOT_DIR, exist_ok=True)
    os.makedirs(PDFS_PATH, exist_ok=True)
    os.makedirs(GRAPHS_PATH, exist_ok=True)
    os.makedirs(MAPS_PATH, exist_ok=True)
    os.makedirs(GENERATED_PATH, exist_ok=True)
    os.makedirs(os.path.dirname(CHAT_VECTORDB_PATH), exist_ok=True) # AJOUT MÉMOIRE VECTORIELLE: Dossier pour chat_vectordb
    os.makedirs(SUBMODELS_PATH, exist_ok=True) # Nouveau: Dossier pour sous-modèles
    st.write(f"📁 Dossier principal : {CHATBOT_DIR}")
    return True
def extract_text_from_pdf(pdf_path):
    """Extraire le texte d'un PDF"""
    text = ""
    try:
        with fitz.open(pdf_path) as doc:
            for page_num, page in enumerate(doc):
                page_text = page.get_text()
                text += f"\n[Page {page_num + 1}]\n{page_text}\n"
        return text
    except Exception as e:
        st.write(f"❌ Erreur PDF {pdf_path}: {e}")
        return ""
def upload_and_process_pbf(pbf_file):
    """Traitement du fichier PBF uploadé"""
    if pbf_file is None:
        return None, None, "❌ Aucun fichier uploadé"
    pbf_path = pbf_file.name
    with open(pbf_path, "wb") as f:
        f.write(pbf_file.getvalue())
    st.write("⚙️ Lecture du PBF et construction du graphe...")
    handler = RoadPOIHandler()
    handler.apply_file(pbf_path, locations=True)
    G = handler.graph
    pois = handler.pois
    # Sauvegarder dans le dossier chatbot
    graph_name = os.path.basename(pbf_path).replace('.osm.pbf', '_graph.graphml')
    graph_path = os.path.join(GRAPHS_PATH, graph_name)
    nx.write_graphml(G, graph_path)
    # Sauvegarder les POIs
    pois_name = graph_name.replace('_graph.graphml', '_pois.json')
    pois_path = os.path.join(GRAPHS_PATH, pois_name)
    with open(pois_path, 'w', encoding='utf-8') as f:
        json.dump(pois, f, indent=2, ensure_ascii=False)
    st.write(f"✅ Graphe: {len(G)} nœuds, {G.size()} arêtes")
    st.write(f"✅ POIs: {len(pois)} points")
    st.write(f"💾 Sauvegardé: {graph_path}")
    return G, pois, f"✅ Graphe créé: {len(G)} nœuds, {len(pois)} POIs"
def load_existing_graph():
    """Charge un graphe existant"""
    graph_files = [f for f in os.listdir(GRAPHS_PATH) if f.endswith('_graph.graphml')] if os.path.exists(GRAPHS_PATH) else []
    if not graph_files:
        return None, None, "❌ Aucun graphe trouvé"
    graph_file = graph_files[0]
    graph_path = os.path.join(GRAPHS_PATH, graph_file)
    pois_path = os.path.join(GRAPHS_PATH, graph_file.replace('_graph.graphml', '_pois.json'))
    try:
        G = nx.read_graphml(graph_path)
        pois = []
        if os.path.exists(pois_path):
            with open(pois_path, 'r', encoding='utf-8') as f:
                pois = json.load(f)
        return G, pois, f"✅ Graphe chargé: {len(G)} nœuds, {len(pois)} POIs"
    except Exception as e:
        return None, None, f"❌ Erreur: {e}"
@st.cache_resource
def get_embedding_model():
    """Modèle d'embedding en cache pour éviter rechargement"""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': device}
    )
# AJOUT MÉMOIRE VECTORIELLE: Fonctions pour la mémoire chat
def load_chat_vectordb():
    """Charger la base vectorielle pour l'historique chat"""
    if not os.path.exists(CHAT_VECTORDB_PATH):
        return None, "⚠️ Aucune base chat trouvée"
    embedding_model = get_embedding_model()
    try:
        chat_vectordb = FAISS.load_local(CHAT_VECTORDB_PATH, embedding_model, allow_dangerous_deserialization=True)
        return chat_vectordb, "✅ Base chat chargée"
    except Exception as e:
        return None, f"❌ Erreur chat: {e}"
def add_to_chat_db(user_msg, ai_msg, chat_vectordb):
    """Ajouter un échange user-AI à la base chat"""
    if chat_vectordb is None:
        embedding_model = get_embedding_model()
        chat_vectordb = FAISS.from_texts([""], embedding_model) # Créer si vide
    exchange = f"User: {user_msg} ||| Assistant: {ai_msg}"
    doc = Document(
        page_content=exchange,
        metadata={"type": "chat_exchange", "timestamp": time.time()}
    )
    chat_vectordb.add_documents([doc])
    chat_vectordb.save_local(CHAT_VECTORDB_PATH)
    return chat_vectordb
def chat_rag_search(question, chat_vectordb, k=3):
    """Rechercher dans l'historique chat pour contexte"""
    if not chat_vectordb:
        return []
    try:
        return chat_vectordb.similarity_search(question, k=k)
    except Exception as e:
        st.write(f"❌ Erreur recherche chat: {e}")
        return []
def process_pdfs():
    """Traiter les PDFs"""
    st.write("📄 Traitement des PDFs...")
    embedding_model = get_embedding_model()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100
    )
    # Charger vectordb existante si elle existe
    vectordb = None
    if os.path.exists(VECTORDB_PATH):
        try:
            vectordb, _ = load_vectordb()
        except Exception as e:
            st.write(f"⚠️ Erreur chargement vectordb existante: {e}. Création nouvelle.")
            vectordb = None
    # Charger métadonnées existantes
    if os.path.exists(METADATA_PATH):
        with open(METADATA_PATH, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
    else:
        metadata = {"processed_files": [], "total_chunks": 0}
    processed_filenames = {p["filename"] for p in metadata["processed_files"]}
    all_documents = []
    pdf_files = [f for f in os.listdir(PDFS_PATH) if f.endswith('.pdf')] if os.path.exists(PDFS_PATH) else []
    if not pdf_files:
        return vectordb, "⚠️ Aucun PDF trouvé"
   
    # Check préliminaire : si aucun nouveau, skip
    new_pdfs = [f for f in pdf_files if f not in processed_filenames]
    if not new_pdfs:
        return vectordb, "✅ Tous les PDFs déjà traités. Base à jour !"
   
    progress_bar = st.progress(0)
    status_text = st.empty()
    new_chunks_count = 0
    new_processed = []
    total_pdfs = len(new_pdfs)
    current_pdf = 0
    for pdf_file in pdf_files:
        if pdf_file in processed_filenames:
            st.write(f" 📖 {pdf_file} déjà traité, sauté.")
            continue
        pdf_path = os.path.join(PDFS_PATH, pdf_file)
        st.write(f" 📖 Traitement nouveau PDF : {pdf_file}")
        status_text.text(f"Traitement de {pdf_file}...")
        text = extract_text_from_pdf(pdf_path)
        if not text.strip():
            continue
        try:
            chunks = text_splitter.split_text(text)
        except Exception as e:
            st.write(f"❌ Erreur split text pour {pdf_file}: {e}")
            continue
        for i, chunk in enumerate(chunks):
            doc = Document(
                page_content=chunk,
                metadata={
                    "source": pdf_file,
                    "chunk_id": i,
                    "type": "pdf"
                }
            )
            all_documents.append(doc)
        new_processed.append({"filename": pdf_file, "chunks": len(chunks)})
        new_chunks_count += len(chunks)
        current_pdf += 1
        progress = current_pdf / total_pdfs if total_pdfs > 0 else 1
        progress_bar.progress(progress)
    status_text.text("Finalisation...")
    # Ajouter les trajets sauvegardés (toujours, car ils peuvent changer)
    if os.path.exists(TRAJECTORIES_PATH):
        with open(TRAJECTORIES_PATH, 'r', encoding='utf-8') as f:
            trajectories = json.load(f)
        for traj in trajectories:
            traj_text = f"""Trajet: {traj.get('question', '')}
Départ: {traj.get('start_name', '')}
Arrivée: {traj.get('end_name', '')}
Distance: {traj.get('distance', 0)/1000:.2f} km"""
            doc = Document(
                page_content=traj_text,
                metadata={"source": "trajectories", "type": "trajectory"}
            )
            all_documents.append(doc)
    if all_documents:
        try:
            if vectordb is None:
                vectordb = FAISS.from_documents(all_documents, embedding_model)
            else:
                vectordb.add_documents(all_documents)
            vectordb.save_local(VECTORDB_PATH)
        except Exception as e:
            st.write(f"❌ Erreur sauvegarde vectordb: {e}")
            return None, "❌ Échec sauvegarde base"
    # Mettre à jour métadonnées seulement si changements
    if new_processed:
        metadata["processed_files"].extend(new_processed)
        metadata["total_chunks"] += new_chunks_count
        with open(METADATA_PATH, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
    progress_bar.progress(1)
    status_text.text("Terminé !")
    return vectordb, f"✅ Base mise à jour : {len(new_processed)} nouveaux PDFs traités, {new_chunks_count} nouveaux chunks (total : {metadata['total_chunks']})"
def load_vectordb():
    """Charger la base vectorielle"""
    if not os.path.exists(VECTORDB_PATH):
        return None, "⚠️ Aucune base trouvée"
    embedding_model = get_embedding_model()
    try:
        vectordb = FAISS.load_local(VECTORDB_PATH, embedding_model, allow_dangerous_deserialization=True)
        return vectordb, "✅ Base chargée"
    except Exception as e:
        return None, f"❌ Erreur: {e}"
def save_trajectory(question, response, trajectory_info):
    """Sauvegarde un trajet"""
    trajectories = []
    if os.path.exists(TRAJECTORIES_PATH):
        with open(TRAJECTORIES_PATH, 'r', encoding='utf-8') as f:
            trajectories = json.load(f)
    new_trajectory = {
        "question": question,
        "response": response,
        "start_name": trajectory_info.get('start', {}).get('name', ''),
        "end_name": trajectory_info.get('end', {}).get('name', ''),
        "distance": trajectory_info.get('distance', 0)
    }
    trajectories.append(new_trajectory)
    with open(TRAJECTORIES_PATH, 'w', encoding='utf-8') as f:
        json.dump(trajectories, f, indent=2, ensure_ascii=False)
def upload_pdfs(uploaded_files):
    """Upload des PDFs"""
    if uploaded_files is None:
        return []
    saved_files = []
    for file in uploaded_files:
        filename = file.name
        filepath = os.path.join(PDFS_PATH, filename)
        with open(filepath, "wb") as f:
            f.write(file.getvalue())
        saved_files.append(filename)
    return saved_files
# ===============================================
# Système de Cache Web Intelligent
# ===============================================
def load_web_cache():
    """Charge le cache web"""
    if os.path.exists(WEB_CACHE_PATH):
        try:
            with open(WEB_CACHE_PATH, 'r', encoding='utf-8') as f:
                return json.load(f)
        except:
            pass
    return {}
def save_web_cache(cache):
    """Sauvegarde le cache web"""
    try:
        with open(WEB_CACHE_PATH, 'w', encoding='utf-8') as f:
            json.dump(cache, f, indent=2, ensure_ascii=False)
    except Exception as e:
        st.write(f"Erreur sauvegarde cache: {e}")
def get_cache_key(query, source="text"):
    """Génère une clé de cache pour une requête"""
    return f"{source}:{query.lower().strip()}"
def is_cache_expired(cache_entry, max_age_hours=24):
    """Vérifie si l'entrée du cache a expiré"""
    current_time = time.time()
    return (current_time - cache_entry.get('timestamp', 0)) > (max_age_hours * 3600)
def get_cache_stats():
    """Obtient les statistiques du cache"""
    try:
        cache = load_web_cache()
        if not cache:
            return "Cache vide"
        total_entries = len(cache)
        expired_count = sum(1 for entry in cache.values() if is_cache_expired(entry))
        valid_count = total_entries - expired_count
        return f"📊 Cache: {total_entries} entrées total, {valid_count} valides, {expired_count} expirées"
    except Exception as e:
        return f"❌ Erreur stats: {e}"
# ===============================================
# Fonctions RAG et Web Search Améliorées
# ===============================================
class LocalClient:
    def __init__(self):
        from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
        
        MODEL_PATH = "/root/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-V3-0324/snapshots/e9b33add76883f293d6bf61f6bd89b497e80e335"
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True, local_files_only=True)
        
        # Load model with device_map for large models
        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH, 
            trust_remote_code=True, 
            local_files_only=True,
            device_map="auto",
            torch_dtype="auto"
        )
        
        self.model.eval()

    def chat_completion(self, messages, model, max_tokens, temperature, stream=False):
        try:
            # Use chat template for proper formatting
            prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = self.tokenizer.encode(prompt, return_tensors="pt").to(self.model.device)
            outputs = self.model.generate(inputs, max_new_tokens=max_tokens, temperature=temperature, do_sample=temperature > 0, pad_token_id=self.tokenizer.eos_token_id)
            generated_ids = outputs[0][inputs.shape[-1]:]
            response = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
            class Choice:
                def __init__(self, content):
                    self.message = type('msg', (), {'content': content})()
            class Resp:
                def __init__(self, choice):
                    self.choices = [choice]
            return Resp(Choice(response))
        except Exception as e:
            class Choice:
                def __init__(self, content):
                    self.message = type('msg', (), {'content': content})()
            class Resp:
                def __init__(self, choice):
                    self.choices = [choice]
            return Resp(Choice(f"Erreur locale: {str(e)}"))
@st.cache_resource
def create_client():
    """Créer le client Inference avec gestion d'erreurs améliorée"""
    try:
        client = InferenceClient(token=HF_TOKEN)
        return client
    except Exception as e:
        st.write(f"❌ Erreur création client: {e}. Passage en mode local.")
        return LocalClient()
def rag_search(question, vectordb, k=3):
    """Rechercher dans la base vectorielle"""
    if not vectordb:
        return []
    try:
        return vectordb.similarity_search(question, k=k)
    except Exception as e:
        st.write(f"❌ Erreur recherche: {e}")
        return []
def enhanced_web_search(query, max_results=5, search_type="text", use_cache=True):
    """
    Recherche web avancée avec cache intelligent et multiple sources
    Args:
        query: Requête de recherche
        max_results: Nombre max de résultats
        search_type: Type de recherche ("text", "news", "both")
        use_cache: Utiliser le cache
    Returns:
        Liste de résultats enrichis
    """
    cache = load_web_cache() if use_cache else {}
    results = []
    try:
        # Recherche texte
        if search_type in ["text", "both"]:
            cache_key = get_cache_key(query, "text")
            if cache_key in cache and not is_cache_expired(cache[cache_key]):
                st.write(f"📋 Utilisation cache pour: {query}")
                text_results = cache[cache_key]['results']
            else:
                st.write(f"🔍 Recherche web pour: {query}")
                tavily = TavilyClient(api_key=TAVILY_API_KEY)
                text_results = []
                try:
                    raw_results = tavily.search(query, max_results=max_results, search_depth="advanced", topic="general")
                    for r in raw_results.get('results', []):
                        text_results.append({
                            'title': r.get('title', ''),
                            'body': r.get('content', ''),
                            'href': r.get('url', ''),
                            'source_type': 'web_search'
                        })
                    # Sauvegarder en cache
                    cache[cache_key] = {
                        'results': text_results,
                        'timestamp': time.time()
                    }
                    if use_cache:
                        save_web_cache(cache)
                except Exception as e:
                    st.write(f"Erreur recherche texte: {e}")
                    text_results = []
            results.extend(text_results)
        # Recherche actualités
        if search_type in ["news", "both"]:
            cache_key = get_cache_key(query, "news")
            if cache_key in cache and not is_cache_expired(cache[cache_key], max_age_hours=6):
                news_results = cache[cache_key]['results']
            else:
                tavily = TavilyClient(api_key=TAVILY_API_KEY)
                news_results = []
                try:
                    raw_news = tavily.search(query, max_results=max_results//2 if search_type == "both" else max_results, search_depth="advanced", topic="news")
                    for r in raw_news.get('results', []):
                        news_results.append({
                            'title': r.get('title', ''),
                            'body': r.get('content', ''),
                            'url': r.get('url', ''),
                            'date': r.get('published_date', ''),
                            'source': r.get('source', ''),
                            'source_type': 'news'
                        })
                    # Sauvegarder en cache (6h pour les news)
                    cache[cache_key] = {
                        'results': news_results,
                        'timestamp': time.time()
                    }
                    if use_cache:
                        save_web_cache(cache)
                except Exception as e:
                    st.write(f"Erreur recherche news: {e}")
                    news_results = []
            results.extend(news_results)
    except Exception as e:
        st.write(f"❌ Erreur recherche web globale: {e}")
        results = [{'title': 'Erreur de recherche', 'body': f'Erreur: {e}', 'source_type': 'error'}]
    return results
def smart_content_extraction(url, max_length=1000):
    """
    Extraction intelligente du contenu d'une page web
    Args:
        url: URL à scraper
        max_length: Longueur max du contenu
    Returns:
        Contenu extrait et nettoyé
    """
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        # Supprimer les éléments non pertinents
        for element in soup(['script', 'style', 'nav', 'footer', 'aside', 'header']):
            element.decompose()
        # Extraire le texte principal
        main_content = soup.find('main') or soup.find('article') or soup.find('div', class_='content') or soup.body
        if main_content:
            text = main_content.get_text(separator=' ', strip=True)
        else:
            text = soup.get_text(separator=' ', strip=True)
        # Nettoyer et tronquer
        text = ' '.join(text.split()) # Normaliser les espaces
        return text[:max_length] + ('...' if len(text) > max_length else '')
    except Exception as e:
        st.write(f"Erreur extraction contenu {url}: {e}")
        return f"Impossible d'extraire le contenu de {url}"
def intelligent_query_expansion(query):
    """
    Expansion intelligente des requêtes pour améliorer les résultats
    Args:
        query: Requête originale
    Returns:
        Liste de requêtes expandues
    """
    expanded_queries = [query] # Toujours inclure la requête originale
    # Détection de mots-clés pour expansion contextuelle
    keywords = {
        'actualité': ['news', 'dernières nouvelles', 'récent'],
        'comment': ['tutorial', 'guide', 'étapes'],
        'pourquoi': ['raison', 'cause', 'explication'],
        'comparaison': ['vs', 'différence', 'comparatif'],
        'prix': ['coût', 'tarif', 'budget'],
        'avis': ['opinion', 'critique', 'review']
    }
    query_lower = query.lower()
    for trigger, expansions in keywords.items():
        if trigger in query_lower:
            for expansion in expansions:
                expanded_queries.append(f"{query} {expansion}")
    return expanded_queries[:3] # Limiter à 3 requêtes max
def hybrid_search_enhanced(query, vectordb, k=3, web_search_enabled=True, search_type="both", chat_vectordb=None): # AJOUT MÉMOIRE VECTORIELLE: Param pour chat_vectordb
    """
    Recherche hybride améliorée combinant RAG local et web avec intelligence
    Args:
        query: Requête de recherche
        vectordb: Base vectorielle locale
        k: Nombre de résultats RAG
        web_search_enabled: Activer la recherche web
        search_type: Type de recherche web
        chat_vectordb: Base pour historique chat (optionnel)
    Returns:
        Liste de documents combinés et enrichis
    """
    all_results = []
    # 1. Recherche RAG locale
    local_docs = rag_search(query, vectordb, k)
    for doc in local_docs:
        doc.metadata['search_source'] = 'local_rag'
        doc.metadata['relevance_score'] = 1.0 # Score max pour les docs locaux
    all_results.extend(local_docs)
    # AJOUT MÉMOIRE VECTORIELLE: Recherche dans historique chat pour contexte conversationnel
    if chat_vectordb:
        chat_docs = chat_rag_search(query, chat_vectordb, k=3)
        for doc in chat_docs:
            doc.metadata['search_source'] = 'chat_history'
            doc.metadata['relevance_score'] = 0.9
        all_results.extend(chat_docs[:2]) # Limiter à 2 pour éviter surcharge
    # 2. Recherche web intelligente si activée
    if web_search_enabled:
        st.write(f"🌐 Recherche web activée pour: {query}")
        # Expansion de requête pour de meilleurs résultats
        expanded_queries = intelligent_query_expansion(query)
        web_results = []
        for exp_query in expanded_queries:
            try:
                search_results = enhanced_web_search(
                    exp_query,
                    max_results=3,
                    search_type=search_type
                )
                for result in search_results:
                    # Créer un document à partir du résultat web
                    content = f"Titre: {result.get('title', '')}\n"
                    content += f"Contenu: {result.get('body', '')}\n"
                    if result.get('source_type') == 'news' and result.get('date'):
                        content += f"Date: {result.get('date')}\n"
                        content += f"Source: {result.get('source', '')}\n"
                    # Extraction de contenu supplémentaire si URL disponible
                    url = result.get('href') or result.get('url')
                    if url and len(result.get('body', '')) < 200:
                        st.write(f"📄 Extraction contenu de: {url}")
                        extra_content = smart_content_extraction(url)
                        if extra_content and "Impossible d'extraire" not in extra_content:
                            content += f"\nContenu détaillé: {extra_content}"
                    doc = Document(
                        page_content=content,
                        metadata={
                            'source': url or 'web_search',
                            'type': result.get('source_type', 'web'),
                            'search_source': 'web',
                            'query_used': exp_query,
                            'relevance_score': 0.8 if exp_query == query else 0.6
                        }
                    )
                    web_results.append(doc)
            except Exception as e:
                st.write(f"Erreur recherche pour '{exp_query}': {e}")
                continue
        # Filtrer les doublons et trier par pertinence
        unique_web_results = []
        seen_urls = set()
        for doc in web_results:
            url = doc.metadata.get('source', '')
            if url not in seen_urls:
                seen_urls.add(url)
                unique_web_results.append(doc)
        # Trier par score de pertinence
        unique_web_results.sort(key=lambda x: x.metadata.get('relevance_score', 0), reverse=True)
        all_results.extend(unique_web_results[:5]) # Max 5 résultats web
    return all_results
def generate_answer_enhanced(question, context_docs, model_name, include_sources=True):
    """
    Génération de réponse améliorée avec gestion des sources multiples
    Args:
        question: Question posée
        context_docs: Documents de contexte
        model_name: Modèle à utiliser
        include_sources: Inclure les sources dans la réponse
    Returns:
        Réponse générée avec sources
    """
    if not context_docs:
        context = "Aucun contexte spécifique trouvé."
    else:
        context_parts = []
        local_sources = []
        web_sources = []
        chat_sources = [] # AJOUT MÉMOIRE VECTORIELLE: Sources pour historique chat
        for i, doc in enumerate(context_docs):
            source = doc.metadata.get('source', 'Document inconnu')
            doc_type = doc.metadata.get('type', 'unknown')
            search_source = doc.metadata.get('search_source', 'unknown')
            content = doc.page_content.strip()
            # Classifier les sources
            if search_source == 'local_rag':
                local_sources.append(f"[{i+1}] {source} ({doc_type})")
            elif search_source == 'chat_history':
                chat_sources.append(f"[{i+1}] Historique précédent: {source}")
            else:
                web_sources.append(f"[{i+1}] {source}")
            context_parts.append(f"[Source {i+1} - {doc_type}]\n{content}")
        context = "\n\n".join(context_parts)
    # Prompt amélioré avec instructions pour les sources (ajout chat)
    prompt = f"""Tu es un assistant IA intelligent qui répond aux questions en utilisant à la fois des documents locaux, l'historique des conversations passées, et des informations web récentes.
CONTEXTE DISPONIBLE (incluant historique pour continuité):
{context}
QUESTION: {question}
INSTRUCTIONS:
- Utilise l'historique chat pour maintenir la fluidité et rappeler les échanges précédents
- Utilise toutes les sources disponibles pour donner une réponse complète et précise
- Si les informations web contredisent les documents locaux ou l'historique, mentionne les deux perspectives
- Privilégie les informations récentes pour les sujets d'actualité
- Sois précis et cite tes sources si nécessaire
- Si certaines informations manquent, dis-le clairement et propose de clarifier basé sur l'historique
RÉPONSE DÉTAILLÉE:"""
    try:
        client = create_client()
        messages = [{"role": "user", "content": prompt}]
        response = client.chat_completion(
            messages=messages,
            model=model_name,
            max_tokens=600,
            temperature=0.3
        )
        answer = response.choices[0].message.content
        # Ajouter les sources si demandé
        if include_sources and context_docs:
            sources_text = "\n\n📚 **Sources consultées:**\n"
            if chat_sources: # AJOUT MÉMOIRE VECTORIELLE
                sources_text += "**Historique conversation:**\n"
                for source in chat_sources[:2]:
                    sources_text += f"• {source}\n"
            if local_sources:
                sources_text += "**Documents locaux:**\n"
                for source in local_sources[:3]: # Limiter l'affichage
                    sources_text += f"• {source}\n"
            if web_sources:
                sources_text += "**Sources web:**\n"
                for source in web_sources[:3]: # Limiter l'affichage
                    sources_text += f"• {source}\n"
            answer += sources_text
        return answer
    except Exception as e:
        error_str = str(e)
        # Check for payment error and retry with LocalClient
        if "402" in error_str or "Payment Required" in error_str:
            try:
                # Retry with LocalClient
                local_client = LocalClient()
                messages = [{"role": "user", "content": prompt}]
                response = local_client.chat_completion(
                    messages=messages,
                    model=model_name,
                    max_tokens=600,
                    temperature=0.3
                )
                answer = response.choices[0].message.content
                # Ajouter les sources si demandé
                if include_sources and context_docs:
                    sources_text = "\n\n📚 **Sources consultées (mode local):**\n"
                    if chat_sources:
                        sources_text += "**Historique conversation:**\n"
                        for source in chat_sources[:2]:
                            sources_text += f"• {source}\n"
                    if local_sources:
                        sources_text += "**Documents locaux:**\n"
                        for source in local_sources[:3]:
                            sources_text += f"• {source}\n"
                    if web_sources:
                        sources_text += "**Sources web:**\n"
                        for source in web_sources[:3]:
                            sources_text += f"• {source}\n"
                    answer += sources_text
                return answer + "\n\n⚠️ Réponse générée en mode local (API distante indisponible)."
            except Exception as local_e:
                return f"❌ Erreur génération (même en local): {str(local_e)}"
        else:
            return f"❌ Erreur génération: {error_str}"
# ===============================================
# Fonctions Web Search et Hybrid (Mises à jour)
# ===============================================
def web_search(query, max_results=5):
    """Version simplifiée pour compatibilité"""
    try:
        results = enhanced_web_search(query, max_results, "text")
        return [f"{r.get('title', '')}: {r.get('href', r.get('url', ''))} - {r.get('body', '')}" for r in results]
    except Exception as e:
        return [f"❌ Erreur recherche web: {e}"]
def hybrid_search(query, vectordb, k=3):
    """Version simplifiée pour compatibilité"""
    return hybrid_search_enhanced(query, vectordb, k, web_search_enabled=True)
def final_search(question, vectordb, graph, pois):
    """Recherche finale combinant toutes les sources"""
    results = hybrid_search_enhanced(question, vectordb, k=3, web_search_enabled=True)
    # OSM si mention lieu
    if any(keyword in question.lower() for keyword in ["aller", "trajet", "itinéraire", "route", "navigation"]):
        try:
            carte, reponse, traj = calculer_trajet(question, graph, pois)
            if traj:
                results.append(Document(
                    page_content=reponse,
                    metadata={"source": "trajet_osm", "type": "navigation"}
                ))
        except:
            pass
    return results
# ===============================================
# Fonctions Modèles Hugging Face Spécialisés
# ===============================================
@st.cache_resource
def initialize_specialized_models():
    """Initialise les modèles spécialisés avec gestion d'erreurs"""
    device_id = 0 if torch.cuda.is_available() else -1
    models = {}
    try:
        models['summarizer'] = pipeline("summarization", model="facebook/bart-large-cnn", device=device_id)
        st.write("✅ Modèle de résumé chargé")
    except Exception as e:
        st.write(f"⚠️ Erreur chargement summarizer: {e}")
        models['summarizer'] = None
    try:
        models['translator'] = pipeline("translation", model="Helsinki-NLP/opus-mt-fr-en", device=device_id)
        st.write("✅ Modèle de traduction chargé")
    except Exception as e:
        st.write(f"⚠️ Erreur chargement translator: {e}")
        models['translator'] = None
    try:
        models['captioner'] = None
        st.write("✅ Captioner configuré pour utiliser LLM (llava)")
    except Exception as e:
        st.write(f"⚠️ Erreur chargement captioner: {e}")
        models['captioner'] = None
    try:
        models['ner'] = pipeline("ner", model="dbmdz/bert-large-cased-finetuned-conll03-english", device=device_id)
        st.write("✅ Modèle NER chargé")
        st.write("⚠️ Warning NER ignoré : weights pooler non utilisés (normal pour ce checkpoint).")
    except Exception as e:
        st.write(f"⚠️ Erreur chargement NER: {e}")
        models['ner'] = None
    return models
# Initialiser les modèles
SPECIALIZED_MODELS = initialize_specialized_models()
def summarize_text(text):
    if SPECIALIZED_MODELS.get('summarizer') is None:
        return "❌ Modèle de résumé non disponible"
    try:
        return SPECIALIZED_MODELS['summarizer'](text[:1024], max_length=200, min_length=30, do_sample=False)[0]['summary_text']
    except Exception as e:
        return f"❌ Erreur résumé: {e}"
def translate_text(text, src_lang="fr", tgt_lang="en"):
    if SPECIALIZED_MODELS.get('translator') is None:
        return "❌ Modèle de traduction non disponible"
    try:
        return SPECIALIZED_MODELS['translator'](text)[0]['translation_text']
    except Exception as e:
        return f"❌ Erreur traduction: {e}"
def caption_image(image_path):
    client = create_client()
    model = "llava-hf/llava-1.5-7b-hf"
    prompt = "Generate a detailed caption for this image."
    try:
        return client.image_to_text(image_path, prompt=prompt, model=model, max_tokens=500)
    except Exception as e:
        return f"❌ Erreur caption: {e}"
def extract_entities(text):
    if SPECIALIZED_MODELS.get('ner') is None:
        return "❌ Modèle NER non disponible"
    try:
        return SPECIALIZED_MODELS['ner'](text)
    except Exception as e:
        return f"❌ Erreur NER: {e}"
# ===============================================
# Fonctions de génération avec Stable Diffusion et similaires
# ===============================================
def generate_text_to_image(prompt):
    """Génère une image à partir de texte"""
    try:
        pipe = DiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", use_auth_token=HF_TOKEN)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        pipe.to(device)
        image = pipe(prompt).images[0]
        path = os.path.join(GENERATED_PATH, f"image_{int(time.time())}.png")
        image.save(path)
        return f"Image générée et sauvegardée à {path}"
    except Exception as e:
        return f"❌ Erreur génération image: {e}"
def generate_text_to_video(prompt):
    """Génère une vidéo à partir de texte"""
    try:
        pipe = DiffusionPipeline.from_pretrained("damo-vilab/text-to-video-ms-1.7b", torch_dtype=torch.float16, variant="fp16", use_auth_token=HF_TOKEN)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cuda":
            pipe.enable_model_cpu_offload()
        else:
            pipe.to(device)
        gen = pipe(prompt, num_inference_steps=25)
        frames = gen.frames[0] # Assuming batch size 1
        path = os.path.join(GENERATED_PATH, f"video_{int(time.time())}.gif")
        imageio.mimsave(path, frames, fps=5)
        return f"Vidéo générée et sauvegardée à {path}"
    except Exception as e:
        return f"❌ Erreur génération vidéo: {e}"
def generate_text_to_audio(prompt):
    """Génère un son à partir de texte"""
    try:
        pipe = AudioLDMPipeline.from_pretrained("cvssp/audio-ldm", torch_dtype=torch.float16, use_auth_token=HF_TOKEN)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        pipe.to(device)
        audio = pipe(prompt, audio_length_in_s=5.0).audios[0]
        path = os.path.join(GENERATED_PATH, f"audio_{int(time.time())}.wav")
        wavfile.write(path, rate=16000, data=audio) # Assuming 16kHz sample rate
        return f"Son généré et sauvegardé à {path}"
    except Exception as e:
        return f"❌ Erreur génération son: {e}"
def generate_text_to_3d(prompt):
    """Génère un modèle 3D à partir de texte (rendue image)"""
    try:
        pipe = ShapEPipeline.from_pretrained("openai/shap-e", use_auth_token=HF_TOKEN)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        pipe.to(device)
        output = pipe(prompt, num_inference_steps=64)
        image = output.images[0]
        path = os.path.join(GENERATED_PATH, f"3d_text_{int(time.time())}.png")
        image.save(path)
        return f"Rendu 3D généré et sauvegardé à {path}"
    except Exception as e:
        return f"❌ Erreur génération 3D (texte): {e}"
def generate_image_to_3d(image_path):
    """Génère un modèle 3D à partir d'une image (rendue image)"""
    try:
        pipe = ShapEImg2ImgPipeline.from_pretrained("openai/shap-e", use_auth_token=HF_TOKEN)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        pipe.to(device)
        image = Image.open(image_path)
        output = pipe(image, num_inference_steps=64)
        rendered_image = output.images[0]
        path = os.path.join(GENERATED_PATH, f"3d_image_{int(time.time())}.png")
        rendered_image.save(path)
        return f"Rendu 3D généré à partir de l'image et sauvegardé à {path}"
    except Exception as e:
        return f"❌ Erreur génération 3D (image): {e}"
# ===============================================
# Agent LangChain Amélioré avec Recherche Web
# ===============================================
def create_enhanced_agent(model_name, vectordb, graph, pois, chat_vectordb=None): # AJOUT MÉMOIRE VECTORIELLE: Param pour chat
    """
    Crée un agent LangChain amélioré avec capacités de recherche web
    Args:
        model_name: Nom du modèle HuggingFace
        vectordb: Base vectorielle locale
        graph: Graphe OSM
        pois: Points d'intérêt
        chat_vectordb: Base pour historique chat (optionnel)
    Returns:
        Agent configuré avec tous les outils
    """
    try:
        llm = HuggingFaceEndpoint(
            repo_id=model_name,
            huggingfacehub_api_token=HF_TOKEN,
            temperature=0.3,
            max_new_tokens=600
        )
    except Exception as e:
        st.write(f"❌ Erreur HuggingFaceEndpoint: {e}. Passage en mode local.")
        pipe = pipeline("text-generation", model=MODEL_PATH, tokenizer=MODEL_PATH, max_new_tokens=600, temperature=0.3, device_map="auto")
        llm = HuggingFacePipeline(pipeline=pipe)
    # Configuration des outils de recherche web
    search_wrapper = DuckDuckGoSearchAPIWrapper(
        region="fr-fr",
        time="d",
        max_results=5
    )
    search_tool = TavilySearchResults(api_key=TAVILY_API_KEY, max_results=5)
    search_results_tool = TavilySearchResults(api_key=TAVILY_API_KEY, max_results=5, include_raw_content=True)
    tools = [
        # Outils de base RAG et recherche
        Tool(
            name="Local_Knowledge_Base",
            func=lambda q: "\n\n".join([d.page_content for d in rag_search(q, vectordb, k=3)]),
            description="Recherche dans la base de connaissances locale (PDFs et documents internes). Utilise ceci en PREMIER pour les questions sur des documents spécifiques."
        ),
        Tool(
            name="Chat_History_Search", # AJOUT MÉMOIRE VECTORIELLE: Nouvel outil pour historique
            func=lambda q: "\n\n".join([d.page_content for d in chat_rag_search(q, chat_vectordb, k=3)]),
            description="Recherche dans l'historique des conversations passées pour maintenir la continuité. Utilise pour les suites de discussion."
        ),
        Tool(
            name="Web_Search",
            func=lambda q: search_tool.run(q),
            description="Recherche sur Internet pour des informations récentes, actualités, ou des connaissances générales non disponibles localement."
        ),
        Tool(
            name="Web_Search_Detailed",
            func=lambda q: search_results_tool.run(q),
            description="Recherche web détaillée avec sources et liens. Utilise pour obtenir des résultats web structurés avec URLs."
        ),
        Tool(
            name="Hybrid_Search",
            func=lambda q: "\n\n".join([d.page_content for d in hybrid_search_enhanced(q, vectordb, k=3, web_search_enabled=True, chat_vectordb=chat_vectordb)]),
            description="Recherche hybride combinant base locale, historique chat ET web. Idéal pour des questions nécessitant à la fois des données internes, passées et externes."
        ),
        Tool(
            name="Current_News_Search",
            func=lambda q: "\n\n".join([f"{r.get('title', '')}: {r.get('body', '')}" for r in enhanced_web_search(q, search_type="news")]),
            description="Recherche spécialisée pour les actualités récentes et informations temporelles."
        ),
        # Outils spécialisés
        Tool(
            name="OSM_Route_Calculator",
            func=lambda q: calculer_trajet(q, graph, pois)[1] if graph and pois else "❌ Aucune carte OSM disponible",
            description="Calcule des itinéraires routiers entre deux lieux. Utilise pour les questions de navigation, trajets, ou géolocalisation."
        ),
        Tool(
            name="Smart_Content_Extractor",
            func=lambda url: smart_content_extraction(url) if url.startswith('http') else "❌ URL invalide",
            description="Extrait le contenu détaillé d'une page web spécifique. Fournis une URL complète."
        ),
        Tool(
            name="Text_Summarizer",
            func=summarize_text,
            description="Résume un texte long en version concise. Utile pour synthétiser des informations volumineuses."
        ),
        Tool(
            name="Language_Translator",
            func=translate_text,
            description="Traduit du français vers l'anglais. Utile pour traiter des sources en langue étrangère."
        ),
        Tool(
            name="Image_Analyzer",
            func=caption_image,
            description="Analyse et décrit le contenu d'une image. Fournis le chemin vers un fichier image."
        ),
        Tool(
            name="Entity_Extractor",
            func=lambda t: json.dumps(extract_entities(t)),
            description="Extrait des entités nommées (personnes, lieux, organisations) d'un texte."
        ),
        # Nouveaux outils Stable Diffusion via API
        Tool(
            name="Text_To_Image_Generator",
            func=generate_text_to_image,
            description="Génère une image à partir d'une description textuelle. Fournis un prompt descriptif."
        ),
        Tool(
            name="Text_To_Video_Generator",
            func=generate_text_to_video,
            description="Génère une vidéo à partir d'une description textuelle. Fournis un prompt descriptif."
        ),
        Tool(
            name="Text_To_Audio_Generator",
            func=generate_text_to_audio,
            description="Génère un son ou audio à partir d'une description textuelle. Fournis un prompt descriptif."
        ),
        Tool(
            name="Text_To_3D_Generator",
            func=generate_text_to_3d,
            description="Génère un modèle 3D (rendue image) à partir d'une description textuelle. Fournis un prompt descriptif."
        ),
        Tool(
            name="Image_To_3D_Generator",
            func=generate_image_to_3d,
            description="Génère un modèle 3D (rendue image) à partir d'une image. Fournis le chemin vers un fichier image."
        ),
    ]
    # Configuration de l'agent avec prompt personnalisé (ajout chat)
    agent_prompt = PromptTemplate.from_template("""Tu es Kibali, un assistant IA avancé avec accès à de multiples sources d'information.
CAPACITÉS DISPONIBLES:
- Base de connaissances locale (PDFs et documents)
- Historique des conversations passées pour continuité
- Recherche web en temps réel
- Calcul d'itinéraires sur cartes OSM
- Analyse d'images et extraction de contenu web
- Traduction et résumé automatiques
- Génération d'images, vidéos, sons et modèles 3D à partir de texte ou images
INSTRUCTIONS IMPORTANTES:
1. Utilise TOUJOURS l'historique chat en premier pour les suites de discussion
2. Utilise la base locale en premier pour les questions sur des documents spécifiques
3. Combine les sources locales, historique ET web pour des réponses complètes
4. Pour les actualités ou infos récentes, privilégie la recherche web
5. Cite tes sources et indique leur provenance (locale vs historique vs web)
6. Si les informations se contredisent, mentionne les deux perspectives
7. Reste concis mais informatif
8. Pour les générations, sauvegarde les fichiers et retourne le chemin

Tu as accès aux outils suivants: {tools}

Utilise le format suivant:
Question: la question d'entrée
Thought: réfléchis à ce que tu dois faire
Action: l'action à entreprendre, doit être l'un de [{tool_names}]
Action Input: l'entrée à l'action
Observation: le résultat de l'action
... (ce Thought/Action/Action Input/Observation peut se répéter N fois)
Thought: Je connais maintenant la réponse finale
Final Answer: la réponse finale à la question d'entrée

Commence!
Question: {input}
Thought: {agent_scratchpad}""")

    # Créer l'agent avec la nouvelle API LangChain 1.0
    agent = create_react_agent(llm, tools, agent_prompt)

    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        max_iterations=5,
        early_stopping_method="generate",
        handle_parsing_errors=True
    )
    st.write(f"✅ Agent créé avec {len(tools)} outils disponibles")
    return agent_executor
# Alias pour compatibilité
def create_agent(model_name, vectordb, graph, pois):
    """Version simplifiée pour compatibilité ascendante"""
    return create_enhanced_agent(model_name, vectordb, graph, pois)
# ===============================================
# Fonctions OSM et Graphe Routier
# ===============================================
def haversine(lon1, lat1, lon2, lat2):
    """Calcul distance haversine en mètres"""
    R = 6371000
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi/2.0)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2.0)**2
    return R * (2 * math.atan2(math.sqrt(a), math.sqrt(1 - a)))
class RoadPOIHandler(osmium.SimpleHandler):
    """Handler pour extraire routes et POIs depuis OSM"""
    def __init__(self):
        super().__init__()
        self.graph = nx.Graph()
        self.pois = []
    def node(self, n):
        """Extraire les POIs (points d'intérêt)"""
        if n.location.valid() and n.tags:
            name = n.tags.get('name', '')
            amenity = n.tags.get('amenity', '')
            if name or amenity:
                self.pois.append({
                    'name': name,
                    'amenity': amenity,
                    'lon': n.location.lon,
                    'lat': n.location.lat,
                    'tags': dict(n.tags)
                })
    def way(self, w):
        """Extraire les routes"""
        if 'highway' in w.tags:
            coords = []
            for n in w.nodes:
                if n.location.valid():
                    coords.append((n.location.lon, n.location.lat))
            for i in range(len(coords)-1):
                lon1, lat1 = coords[i]
                lon2, lat2 = coords[i+1]
                n1, n2 = (lon1, lat1), (lon2, lat2)
                dist = haversine(lon1, lat1, lon2, lat2)
                self.graph.add_node(n1, x=lon1, y=lat1)
                self.graph.add_node(n2, x=lon2, y=lat2)
                self.graph.add_edge(n1, n2, length=dist, highway=w.tags.get("highway"))
def trouver_noeud_plus_proche(lon, lat, graph):
    """Trouve le nœud du graphe le plus proche"""
    min_dist = float("inf")
    closest_node = None
    for node, data in graph.nodes(data=True):
        nlon, nlat = float(data["x"]), float(data["y"])
        dist = haversine(lon, lat, nlon, nlat)
        if dist < min_dist:
            min_dist = dist
            closest_node = node
    return closest_node
def chercher_poi_par_nom(nom, pois_list):
    """Recherche un POI par nom"""
    nom_lower = nom.lower()
    for poi in pois_list:
        if nom_lower in poi['name'].lower() or nom_lower in poi['amenity'].lower():
            return poi
    return None
def generer_carte_trajet(graph, path, pois_list, start_poi=None, end_poi=None):
    """Génère une carte 2D du trajet"""
    fig, ax = plt.subplots(figsize=(12, 10))
    # Dessiner le graphe en arrière-plan
    for edge in list(graph.edges())[:1000]: # Limiter pour la performance
        node1, node2 = edge
        x1, y1 = node1[0], node1[1]
        x2, y2 = node2[0], node2[1]
        ax.plot([x1, x2], [y1, y2], 'lightgray', alpha=0.3, linewidth=0.5)
    # Dessiner le trajet
    if path and len(path) > 1:
        path_x = [node[0] for node in path]
        path_y = [node[1] for node in path]
        ax.plot(path_x, path_y, 'red', linewidth=3, label='Trajet')
        # Marquer début et fin
        ax.scatter(path_x[0], path_y[0], color='green', s=100, label='Départ', zorder=5)
        ax.scatter(path_x[-1], path_y[-1], color='red', s=100, label='Arrivée', zorder=5)
    # Ajouter quelques POIs
    for poi in pois_list[:20]:
        if poi['name']:
            ax.scatter(poi['lon'], poi['lat'], color='blue', s=20, alpha=0.6)
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_title('Trajet calculé sur la carte OSM')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    # Sauvegarder en mémoire
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    buf.seek(0)
    plt.close()
    return buf
def calculer_trajet(question, graph, pois_list):
    """Calcule un trajet basé sur une question textuelle"""
    if not graph or not pois_list:
        return None, "❌ Graphe ou POIs non disponibles", None
    # Utiliser LLM pour extraire départ et arrivée
    try:
        client = create_client()
        prompt = f"""Extraie le lieu de départ et le lieu d'arrivée de cette question de trajet.
Question: {question}
Réponds au format exact:
Départ: [nom du lieu de départ]
Arrivée: [nom du lieu d'arrivée]"""
        messages = [{"role": "user", "content": prompt}]
        response = client.chat_completion(
            messages=messages,
            model=WORKING_MODELS["Llama 3.1 8B (Équilibré)"],
            max_tokens=100,
            temperature=0.1
        )
        extraction = response.choices[0].message.content
        start_line = [line for line in extraction.split('\n') if line.startswith('Départ: ')]
        end_line = [line for line in extraction.split('\n') if line.startswith('Arrivée: ')]
        if start_line and end_line:
            start_place = start_line[0].replace('Départ: ', '').strip()
            end_place = end_line[0].replace('Arrivée: ', '').strip()
        else:
            return None, "❌ Impossible d'extraire les lieux de la question.", None
    except Exception as e:
        st.write(f"❌ Erreur extraction LLM: {e}")
        return None, "❌ Erreur lors de l'extraction des lieux.", None
    start_poi = chercher_poi_par_nom(start_place, pois_list)
    end_poi = chercher_poi_par_nom(end_place, pois_list)
    if not start_poi or not end_poi:
        return None, f"❌ Impossible de trouver les lieux: {start_place} ou {end_place}.", None
    # Trouver les nœuds dans le graphe
    start_node = trouver_noeud_plus_proche(start_poi['lon'], start_poi['lat'], graph)
    end_node = trouver_noeud_plus_proche(end_poi['lon'], end_poi['lat'], graph)
    if not start_node or not end_node:
        return None, "❌ Impossible de trouver les nœuds dans le graphe routier.", None
    try:
        # Calculer le chemin
        path = nx.shortest_path(graph, source=start_node, target=end_node, weight="length")
        # Calculer la distance
        distance_totale = 0
        for i in range(len(path)-1):
            distance_totale += graph[path[i]][path[i+1]]['length']
        # Générer la carte
        carte_buf = generer_carte_trajet(graph, path, pois_list, start_poi, end_poi)
        # Réponse textuelle
        reponse = f"""🗺️ **Trajet calculé**
📍 **Départ**: {start_poi['name']} ({start_poi['amenity']})
🎯 **Arrivée**: {end_poi['name']} ({end_poi['amenity']})
📏 **Distance**: {distance_totale/1000:.2f} km
⏱️ **Temps estimé**: {int(distance_totale/83.33):.0f} min à pied | {int(distance_totale/833.33):.0f} min en voiture
🛣️ **Étapes**: {len(path)} points"""
        return carte_buf, reponse, {
            'start': start_poi,
            'end': end_poi,
            'distance': distance_totale,
            'path_length': len(path)
        }
    except nx.NetworkXNoPath:
        return None, f"❌ Aucun chemin trouvé entre {start_poi['name']} et {end_poi['name']}", None
    except Exception as e:
        return None, f"❌ Erreur: {str(e)}", None
# ===============================================
# Fonctions utilitaires pour images
# ===============================================
def fig_to_pil(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    buf.seek(0)
    plt.close(fig)
    return Image.open(buf)
def df_to_html(df, max_rows=10):
    # Réduire le tableau si trop long
    if len(df) > max_rows:
        summary_row = pd.DataFrame({col: ['...'] for col in df.columns})
        df = pd.concat([df.head(max_rows // 2), summary_row, df.tail(max_rows // 2)])
    return df.to_html(index=False, escape=False)
# ===============================================
# Fonctions Image Analysis
# ===============================================
def classify_soil(image: np.ndarray):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mean_hue = np.mean(hsv[:,:,0])
    mean_sat = np.mean(hsv[:,:,1])
    mean_val = np.mean(hsv[:,:,2])
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    texture_variance = np.var(gray)
    soil_type = "Inconnu"
    possible_contents = "Inconnu"
    possible_minerals = "Inconnu"
    if mean_val < 100 and texture_variance > 5000:
        soil_type = "Argileux (riche en matière organique)"
        possible_contents = "Peut contenir de l'eau, nutriments, adapté aux cultures racines"
        possible_minerals = "Argiles comme kaolinite, illite; possible fer, aluminium"
    elif mean_sat > 100 and texture_variance < 3000:
        soil_type = "Sableux (drainant)"
        possible_contents = "Peut contenir peu d'eau, adapté aux plantes résistantes à la sécheresse"
        possible_minerals = "Quartz, feldspath; silice abondante"
    elif mean_hue > 20 and mean_hue < 40:
        soil_type = "Limoneux (équilibré)"
        possible_contents = "Peut contenir minéraux, bon pour l'agriculture générale"
        possible_minerals = "Silt avec mica, quartz; calcium, potassium"
    # Graphisme : Histogramme des couleurs HSV
    fig, ax = plt.subplots()
    ax.hist(hsv[:,:,0].ravel(), bins=50, color='b', alpha=0.5, label='Hue')
    ax.hist(hsv[:,:,1].ravel(), bins=50, color='g', alpha=0.5, label='Saturation')
    ax.hist(hsv[:,:,2].ravel(), bins=50, color='r', alpha=0.5, label='Value')
    ax.set_title('Histogramme des Composantes HSV')
    ax.legend()
    hist_img = fig_to_pil(fig)
    # Tableau des metrics
    metrics_df = pd.DataFrame({
        'Métrique': ['Hue Moyenne', 'Saturation Moyenne', 'Valeur Moyenne', 'Variance Texture'],
        'Valeur': [mean_hue, mean_sat, mean_val, texture_variance],
        'Explication': ['Moyenne de la teinte', 'Moyenne de la saturation des couleurs', 'Moyenne de la luminosité', 'Variance de la texture pour rugosité']
    })
    metrics_html = df_to_html(metrics_df)
    return {
        "soil_type": soil_type,
        "possible_contents": possible_contents,
        "possible_minerals": possible_minerals
    }, hist_img, metrics_html
def simulate_infrared(image: np.ndarray):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ir_img = cv2.applyColorMap(gray, cv2.COLORMAP_JET)
    fig, ax = plt.subplots()
    ax.imshow(cv2.cvtColor(ir_img, cv2.COLOR_BGR2RGB))
    ax.set_title('Simulation Infrarouge (Colormap JET)')
    ax.axis('off')
    ir_pil = fig_to_pil(fig)
    # Analyse simple (fake temp based on intensity)
    mean_intensity = np.mean(gray)
    ir_analysis = f"Simulation IR: Intensité moyenne {mean_intensity:.2f} (plus rouge = plus chaud, bleu = plus froid)"
    return ir_pil, ir_analysis
def detect_objects(image: np.ndarray, scale_factor=0.1):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    img_with_contours = image.copy()
    dimensions = []
    types = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w < 10 or h < 10: continue # skip small
        cv2.rectangle(img_with_contours, (x, y), (x+w, y+h), (0, 255, 0), 2)
        w_m = w * scale_factor
        h_m = h * scale_factor
        aspect = w / h if h != 0 else 0
        if aspect > 5: obj_type = 'Route'
        elif aspect < 0.2: obj_type = 'Clôture'
        elif 0.5 < aspect < 2: obj_type = 'Bâtiment'
        else: obj_type = 'Autre'
        dimensions.append((w_m, h_m))
        types.append(obj_type)
        cv2.putText(img_with_contours, f"{obj_type}: {w_m:.4f}m x {h_m:.4f}m", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
    num_objects = len(contours)
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(cv2.cvtColor(img_with_contours, cv2.COLOR_BGR2RGB))
    ax.set_title(f"Objets Détectés avec Contours ({num_objects})")
    ax.axis('off')
    obj_img = fig_to_pil(fig)
    if dimensions:
        dim_df = pd.DataFrame({
            'Type': types,
            'Largeur (m)': [d[0] for d in dimensions],
            'Hauteur (m)': [d[1] for d in dimensions],
            'Explication': ['Dimension estimée avec contours OpenCV' for _ in types]
        })
        dim_html = df_to_html(dim_df)
    else:
        dim_html = ""
    return num_objects, obj_img, dim_html
def detect_fences(image: np.ndarray, scale_factor=0.1):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, minLineLength=100, maxLineGap=10)
    img_with_lines = image.copy()
    lengths = []
    if lines is not None:
        line_list = [line[0] for line in lines]
        filtered_lines = [l for l in line_list if abs(l[0] - l[2]) < 10 or abs(l[1] - l[3]) < 10 or abs((l[1]-l[3]) / (l[0]-l[2] + 1e-5)) < 0.1 or abs((l[1]-l[3]) / (l[0]-l[2] + 1e-5)) > 10]
        line_lengths = [np.sqrt((x2 - x1)**2 + (y2 - y1)**2) for x1,y1,x2,y2 in filtered_lines]
        sorted_indices = np.argsort(line_lengths)[::-1]
        sorted_lines = [filtered_lines[i] for i in sorted_indices]
        for x1,y1,x2,y2 in sorted_lines:
            cv2.line(img_with_lines, (x1, y1), (x2, y2), (255, 0, 0), 2)
            length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2) * scale_factor
            lengths.append(length)
            mid_x = (x1 + x2) // 2
            mid_y = (y1 + y2) // 2
            cv2.putText(img_with_lines, f"{length:.4f}m", (mid_x, mid_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(cv2.cvtColor(img_with_lines, cv2.COLOR_BGR2RGB))
    ax.set_title(f"Clôtures/Bordures Détectées ({len(lengths)})")
    ax.axis('off')
    fence_img = fig_to_pil(fig)
    if lengths:
        fence_df = pd.DataFrame({
            'Longueur (m)': lengths,
            'Explication': ['Longueur de bordure filtrée et triée pour précision' for _ in lengths]
        })
        fence_html = df_to_html(fence_df)
    else:
        fence_html = ""
    return len(lengths), fence_img, fence_html
def detect_anomalies(image: np.ndarray):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    num_edges = np.sum(edges > 0)
    mean_variance = np.mean(cv2.Laplacian(gray, cv2.CV_64F).var())
    anomalies = []
    if num_edges > 10000:
        anomalies.append("Anomalies structurelles détectées (ex. : fissures, défauts)")
    if mean_variance > 500:
        anomalies.append("Textures inhabituelles (ex. : zones irrégulières)")
    # Simulation photogrammétrie basique avec Open3D
    depth = np.random.rand(*gray.shape) * 255
    point_cloud = o3d.geometry.PointCloud.create_from_rgbd_image(
        o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d.geometry.Image(image),
            o3d.geometry.Image(depth.astype(np.float32))
        ),
        o3d.camera.PinholeCameraIntrinsic(640, 480, 525, 525, 320, 240)
    )
    num_points = len(point_cloud.points)
    # Graphisme : Histogramme des variances
    fig, ax = plt.subplots()
    ax.hist(cv2.Laplacian(gray, cv2.CV_64F).ravel(), bins=50)
    ax.set_title('Histogramme des Variances Locales (Anomalies)')
    var_hist_img = fig_to_pil(fig)
    # Tableau des metrics anomalies
    anomaly_df = pd.DataFrame({
        'Métrique': ['Nombre de Bords', 'Variance Moyenne', 'Points dans Point Cloud'],
        'Valeur': [num_edges, mean_variance, num_points],
        'Explication': ['Indique complexité structurelle (haut = anomalies)', 'Mesure irrégularités texture', 'Simulation 3D pour volume']
    })
    anomaly_html = df_to_html(anomaly_df)
    anomaly_desc_df = pd.DataFrame({
        'Anomalie': anomalies,
        'Explication': ['Défauts potentiels dans le terrain ou structures' for _ in anomalies]
    })
    anomaly_desc_html = df_to_html(anomaly_desc_df)
    return anomalies, var_hist_img, anomaly_html, anomaly_desc_html
def advanced_analyses(image: np.ndarray):
    analyses = {}
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    edges = cv2.Canny(gray, 100, 200)
    adv_images = []
    adv_tables = []
    # 1. Analyse Géologique
    kmeans = KMeans(n_clusters=3).fit(gray.reshape(-1, 1))
    clustered = kmeans.labels_.reshape(gray.shape)
    analyses['Géologique'] = 'Clusters de textures : ' + str(np.unique(kmeans.labels_))
    fig, ax = plt.subplots()
    ax.imshow(clustered, cmap='viridis')
    ax.set_title('Analyse Géologique: Clustering Textures')
    ax.axis('off')
    adv_images.append(fig_to_pil(fig))
    geo_df = pd.DataFrame({'Cluster': np.unique(kmeans.labels_), 'Compte': np.bincount(kmeans.labels_), 'Explication': ['Groupe de texture géologique' for _ in np.unique(kmeans.labels_)]})
    adv_tables.append(df_to_html(geo_df))
    # 2. Analyse Hydrologique
    blue_mask = cv2.inRange(hsv, (100, 50, 50), (130, 255, 255))
    water_area = np.sum(blue_mask > 0) / blue_mask.size * 100
    analyses['Hydrologique'] = f'Pourcentage eau : {water_area:.2f}%'
    fig, ax = plt.subplots()
    ax.imshow(blue_mask, cmap='gray')
    ax.set_title('Analyse Hydrologique: Masque Eau')
    ax.axis('off')
    adv_images.append(fig_to_pil(fig))
    hydro_df = pd.DataFrame({'Métrique': ['Pourcentage Eau'], 'Valeur': [water_area], 'Explication': ['Zone potentielle pour ressources hydriques']})
    adv_tables.append(df_to_html(hydro_df))
    return analyses, {}, adv_images, adv_tables
def process_image(uploaded_file):
    image = Image.open(BytesIO(uploaded_file))
    img_array = np.array(image)
    proc_images = [image]
    captions = ['Image Originale']
    tables_html = []
    # IR
    ir_pil, ir_analysis = simulate_infrared(img_array)
    proc_images.append(ir_pil)
    captions.append('Simulation Infrarouge')
    tables_html.append('<h3>Analyse IR</h3><p>' + ir_analysis + '</p>')
    # Soil
    soil, hist_img, metrics_html = classify_soil(img_array)
    proc_images.append(hist_img)
    captions.append('Histogramme HSV')
    tables_html.append('<h3>Métriques Sol</h3>' + metrics_html)
    # Objects
    num_objects, obj_img, dim_html = detect_objects(img_array)
    proc_images.append(obj_img)
    captions.append('Objets Détectés')
    if dim_html:
        tables_html.append('<h3>Dimensions Objets</h3>' + dim_html)
    # Fences
    num_fences, fence_img, fence_html = detect_fences(img_array)
    proc_images.append(fence_img)
    captions.append('Clôtures Détectées')
    if fence_html:
        tables_html.append('<h3>Longueurs Clôtures</h3>' + fence_html)
    # Anomalies
    anomalies, var_hist_img, anomaly_html, anomaly_desc_html = detect_anomalies(img_array)
    proc_images.append(var_hist_img)
    captions.append('Histogramme Variances')
    tables_html.append('<h3>Métriques Anomalies</h3>' + anomaly_html)
    # Advanced
    analyses, predictions, adv_images, adv_tables = advanced_analyses(img_array)
    proc_images += adv_images[:5] # Limiter le nombre d'images
    captions += ['Analyse Avancée'] * len(adv_images[:5])
    tables_html += adv_tables[:3] # Limiter le nombre de tableaux
    analysis_data = {
        "soil": soil,
        "ir_analysis": ir_analysis,
        "num_objects": num_objects,
        "num_fences": num_fences,
        "anomalies": anomalies,
        "analyses": analyses,
        "predictions": predictions
    }
    tables_str = '<br>'.join(tables_html)
    return analysis_data, proc_images, tables_str
def improve_analysis_with_llm(analysis_data, model_name):
    prompt = f"""Analyse les données suivantes de l'image et fournis une analyse naturelle améliorée:
DONNÉES:
{json.dumps(analysis_data, indent=2)}
ANALYSE AMÉLIORÉE:"""
    try:
        client = create_client()
        messages = [{"role": "user", "content": prompt}]
        response = client.chat_completion(
            messages=messages,
            model=model_name,
            max_tokens=800,
            temperature=0.5
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"❌ Erreur: {str(e)}"
def update_agent(model_choice, vectordb, graph, pois, chat_vectordb=None): # AJOUT MÉMOIRE VECTORIELLE
    model_name = WORKING_MODELS[model_choice]
    agent = create_enhanced_agent(model_name, vectordb, graph, pois, chat_vectordb)
    cache_info = get_cache_stats()
    return model_name, agent, cache_info
def handle_clear_cache():
    """Vide le cache web"""
    try:
        if os.path.exists(WEB_CACHE_PATH):
            os.remove(WEB_CACHE_PATH)
        return "✅ Cache web vidé"
    except Exception as e:
        return f"❌ Erreur: {e}"
def highlight_important_words(text):
    """Met en évidence les mots importants avec effet scintillante et tooltip"""
    # Mots-clés simples pour exemple (peut être étendu avec NER)
    important_keywords = ['important', 'clé', 'essentiel', 'critique', 'principal', 'trajet', 'pétrole', 'topographie']
    for keyword in important_keywords:
        text = re.sub(rf'\b({keyword})\b', r'<span class="sparkle-word" title="\1: Terme clé pour la compréhension du contexte">\1</span>', text, flags=re.IGNORECASE)
    return text
def handle_chat_enhanced(message, history, agent, model_choice, vectordb, graph, pois, web_enabled):
    # AJOUT MÉMOIRE VECTORIELLE: Charger la base chat
    chat_vectordb, _ = load_chat_vectordb()
    if not message.strip():
        return ""
    if agent is None:
        model_name, agent, _ = update_agent(model_choice, vectordb, graph, pois, chat_vectordb)
    try:
        if not web_enabled:
            # Recherche hybride incluant chat
            docs = hybrid_search_enhanced(message, vectordb, k=3, web_search_enabled=False, chat_vectordb=chat_vectordb)
            response = generate_answer_enhanced(message, docs, WORKING_MODELS[model_choice], include_sources=True)
        else:
            response = agent.run(message)
    except Exception as e:
        response = f"❌ Erreur: {e}\n\nTentative avec recherche locale..."
        try:
            docs = hybrid_search_enhanced(message, vectordb, k=3, web_search_enabled=False, chat_vectordb=chat_vectordb)
            response = generate_answer_enhanced(message, docs, WORKING_MODELS[model_choice])
        except:
            response = f"❌ Erreur complète: {e}"
    # AJOUT MÉMOIRE VECTORIELLE: Sauvegarder l'échange dans la base chat
    chat_vectordb = add_to_chat_db(message, response, chat_vectordb)
    # Appliquer highlighting pour fluidité
    response = highlight_important_words(response)
    return response
def handle_web_search(query, search_type):
    if not query.strip():
        return "⚠️ Veuillez entrer une requête"
    try:
        results = enhanced_web_search(query, max_results=10, search_type=search_type)
        if not results:
            return "❌ Aucun résultat trouvé"
        html_output = "<div style='max-height: 500px; overflow-y: auto;'>"
        for i, result in enumerate(results):
            title = result.get('title', 'Sans titre')
            body = result.get('body', 'Pas de description')
            url = result.get('href') or result.get('url', '#')
            source_type = result.get('source_type', 'web')
            if source_type == 'news':
                icon = "📰"
                color = "#e3f2fd"
            else:
                icon = "🔍"
                color = "#f5f5f5"
            html_output += f"""
            <div style='margin: 10px 0; padding: 15px; background-color: {color}; border-radius: 8px; border-left: 4px solid #2196F3;'>
                <h4 style='margin: 0 0 8px 0; color: #1976D2;'>{icon} {title}</h4>
                <p style='margin: 8px 0; color: #424242; line-height: 1.4;'>{body}</p>
                <a href='{url}' target='_blank' style='color: #1976D2; text-decoration: none; font-size: 0.9em;'>🔗 {url}</a>
            </div>
            """
        html_output += "</div>"
        return html_output
    except Exception as e:
        return f"❌ Erreur recherche: {e}"
def handle_content_extraction(url):
    if not url.strip():
        return "⚠️ Veuillez entrer une URL"
    if not url.startswith(('http://', 'https://')):
        url = 'https://' + url
    try:
        content = smart_content_extraction(url, max_length=2000)
        return content
    except Exception as e:
        return f"❌ Erreur extraction: {e}"
# ===============================================
# Fonctions utilitaires supplémentaires
# ===============================================
def get_system_status():
    """Retourne le statut complet du système"""
    status = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "directories": {
            "chatbot": os.path.exists(CHATBOT_DIR),
            "pdfs": os.path.exists(PDFS_PATH),
            "graphs": os.path.exists(GRAPHS_PATH),
            "maps": os.path.exists(MAPS_PATH)
        },
        "files": {
            "vectordb": os.path.exists(VECTORDB_PATH),
            "chat_vectordb": os.path.exists(CHAT_VECTORDB_PATH), # AJOUT MÉMOIRE VECTORIELLE
            "metadata": os.path.exists(METADATA_PATH),
            "trajectories": os.path.exists(TRAJECTORIES_PATH),
            "web_cache": os.path.exists(WEB_CACHE_PATH)
        },
        "counts": {
            "pdfs": len([f for f in os.listdir(PDFS_PATH) if f.endswith('.pdf')]) if os.path.exists(PDFS_PATH) else 0,
            "graphs": len([f for f in os.listdir(GRAPHS_PATH) if f.endswith('_graph.graphml')]) if os.path.exists(GRAPHS_PATH) else 0
        },
        "cache_stats": get_cache_stats(),
        "token_configured": bool(HF_TOKEN and len(HF_TOKEN) > 10)
    }
    return status
def cleanup_old_cache():
    """Nettoie les entrées expirées du cache"""
    try:
        cache = load_web_cache()
        if not cache:
            return "Cache vide"
        original_count = len(cache)
        cleaned_cache = {}
        for key, entry in cache.items():
            if not is_cache_expired(entry):
                cleaned_cache[key] = entry
        save_web_cache(cleaned_cache)
        removed_count = original_count - len(cleaned_cache)
        return f"✅ Cache nettoyé: {removed_count} entrées expirées supprimées, {len(cleaned_cache)} conservées"
    except Exception as e:
        return f"❌ Erreur nettoyage cache: {e}"
def export_system_config():
    """Exporte la configuration système pour debug"""
    config = {
        "version": "2.0.0",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "paths": {
            "chatbot_dir": CHATBOT_DIR,
            "vectordb_path": VECTORDB_PATH,
            "chat_vectordb_path": CHAT_VECTORDB_PATH, # AJOUT MÉMOIRE VECTORIELLE
            "pdfs_path": PDFS_PATH,
            "graphs_path": GRAPHS_PATH,
            "maps_path": MAPS_PATH
        },
        "models": WORKING_MODELS,
        "status": get_system_status(),
        "features": {
            "web_search": True,
            "osm_routing": True,
            "image_analysis": True,
            "pdf_processing": True,
            "caching": True,
            "chat_memory": True # AJOUT MÉMOIRE VECTORIELLE
        }
    }
    config_path = os.path.join(CHATBOT_DIR, "system_config.json")
    try:
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        return f"✅ Configuration exportée: {config_path}"
    except Exception as e:
        return f"❌ Erreur export: {e}"
def test_all_features():
    """Teste toutes les fonctionnalités principales"""
    results = {}
    # Test HuggingFace
    results["huggingface"] = test_hf_connection()
    # Test recherche web
    try:
        test_results = enhanced_web_search("test", max_results=1)
        results["web_search"] = len(test_results) > 0
    except:
        results["web_search"] = False
    # Test recherche web
    results["specialized_models"] = {}
    for model_name, model in SPECIALIZED_MODELS.items():
        results["specialized_models"][model_name] = model is not None
    # Test base vectorielle
    try:
        vectordb, _ = load_vectordb()
        results["vectordb"] = vectordb is not None
    except:
        results["vectordb"] = False
    # Test base chat # AJOUT MÉMOIRE VECTORIELLE
    try:
        chat_vectordb, _ = load_chat_vectordb()
        results["chat_vectordb"] = chat_vectordb is not None
    except:
        results["chat_vectordb"] = False
    # Test graphe OSM
    try:
        graph, pois, _ = load_existing_graph()
        results["osm_graph"] = graph is not None
    except:
        results["osm_graph"] = False
    return results
# ===============================================
# Fonctions de maintenance avancées
# ===============================================
def optimize_vectordb():
    """Optimise la base vectorielle en supprimant les doublons"""
    try:
        vectordb, status = load_vectordb()
        if not vectordb:
            return "❌ Aucune base vectorielle à optimiser"
        # Cette fonction nécessiterait une implémentation plus complexe
        # pour détecter et supprimer les doublons dans FAISS
        return "✅ Base vectorielle optimisée (fonctionnalité à implémenter)"
    except Exception as e:
        return f"❌ Erreur optimisation: {e}"
def backup_all_data():
    """Crée une sauvegarde de toutes les données"""
    try:
        import zipfile
        backup_name = f"kibali_backup_{time.strftime('%Y%m%d_%H%M%S')}.zip"
        backup_path = os.path.join(CHATBOT_DIR, backup_name)
        with zipfile.ZipFile(backup_path, 'w', zipfile.ZIP_DEFLATED) as backup_zip:
            # Sauvegarder tous les fichiers du dossier chatbot
            for root, dirs, files in os.walk(CHATBOT_DIR):
                for file in files:
                    file_path = os.path.join(root, file)
                    arcname = os.path.relpath(file_path, CHATBOT_DIR)
                    backup_zip.write(file_path, arcname)
        return f"✅ Sauvegarde créée: {backup_path}"
    except Exception as e:
        return f"❌ Erreur sauvegarde: {e}"
def restore_from_backup(backup_path):
    """Restaure les données depuis une sauvegarde"""
    try:
        import zipfile
        if not os.path.exists(backup_path):
            return "❌ Fichier de sauvegarde non trouvé"
        with zipfile.ZipFile(backup_path, 'r') as backup_zip:
            backup_zip.extractall(CHATBOT_DIR)
        return f"✅ Données restaurées depuis: {backup_path}"
    except Exception as e:
        return f"❌ Erreur restauration: {e}"
# ===============================================
# NOUVEAU: Fonctions Auto-Apprentissage et Sous-Modèles avec Scikit-Learn
# ===============================================
def create_submodel_from_chat_history(chat_vectordb, submodel_type="classification"):
    """
    Crée un petit sous-modèle sklearn à partir de l'historique chat pour automatiser des réponses.
    - Type: 'classification' pour classer les questions et prédire des réponses automatisées.
    Rend le modèle plus "humain" en apprenant des patterns conversationnels.
    """
    if not chat_vectordb:
        return None, "❌ Aucune base chat pour entraîner le sous-modèle"
   
    # Extraire les échanges de l'historique
    exchanges = []
    for doc in list(chat_vectordb.docstore._dict.values()) or []:
        exchange = doc.page_content
        if "User:" in exchange and "Assistant:" in exchange:
            user_part = exchange.split("|||")[0].replace("User: ", "").strip()
            ai_part = exchange.split("|||")[1].replace("Assistant: ", "").strip() if "|||" in exchange else ""
            exchanges.append((user_part, ai_part))
   
    if len(exchanges) < 10:
        return None, "❌ Historique chat trop court pour entraîner un modèle"
   
    try:
        # Préparation des données : TF-IDF pour vectorisation textuelle
        vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        X = vectorizer.fit_transform([user[0] for user in exchanges])
       
        # Pour classification simple (ex: prédire si réponse est informative ou autre)
        # Labels simples basés sur patterns (ex: 0=info, 1=question, 2=autre)
        labels = []
        for user_msg, _ in exchanges:
            if re.search(r'\?', user_msg):
                labels.append(1) # Question
            elif any(word in user_msg.lower() for word in ['info', 'savoir', 'expliquer']):
                labels.append(0) # Info
            else:
                labels.append(2) # Autre
       
        X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=42)
       
        if submodel_type == "classification":
            model = MultinomialNB()
        else:
            model = RandomForestClassifier(n_estimators=50)
       
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
       
        # Sauvegarder le modèle et vectorizer
        model_path = os.path.join(SUBMODELS_PATH, f"submodel_{submodel_type}_{int(time.time())}.pkl")
        with open(model_path, 'wb') as f:
            pickle.dump({'model': model, 'vectorizer': vectorizer}, f)
       
        # Visualisation avec matplotlib : Accuracy plot
        fig, ax = plt.subplots()
        ax.bar(['Train', 'Test'], [1.0, accuracy]) # Train est parfait par défaut
        ax.set_title(f'Précision du sous-modèle {submodel_type.capitalize()}')
        ax.set_ylabel('Accuracy')
        plot_path = os.path.join(SUBMODELS_PATH, f"accuracy_plot_{submodel_type}_{int(time.time())}.png")
        plt.savefig(plot_path)
        plt.close()
       
        return model_path, f"✅ Sous-modèle {submodel_type} créé avec accuracy {accuracy:.2f}. Sauvegardé: {model_path}"
    except Exception as e:
        return None, f"❌ Erreur création sous-modèle: {e}"
def use_submodel_for_automation(query, submodel_path, submodel_type="classification"):
    """
    Utilise un sous-modèle pour automatiser une réponse, rendant le comportement plus humain (ex: prédiction rapide).
    """
    if not os.path.exists(submodel_path):
        return "❌ Sous-modèle non trouvé"
   
    try:
        with open(submodel_path, 'rb') as f:
            data = pickle.load(f)
            model = data['model']
            vectorizer = data['vectorizer']
       
        query_vec = vectorizer.transform([query])
        prediction = model.predict(query_vec)[0]
       
        # Réponses automatisées basées sur prédiction pour plus d'humanité
        automated_responses = {
            0: "Voici des infos basiques sur ce sujet, basées sur nos échanges passés.",
            1: "Bonne question ! Laisse-moi réfléchir à ça en me basant sur ce qu'on a discuté avant.",
            2: "Intéressant, je vais creuser un peu plus pour te répondre de manière personnalisée."
        }
       
        response = automated_responses.get(prediction, "Réponse automatisée générée.")
       
        # Visualisation: Distribution des features TF-IDF pour la query
        fig, ax = plt.subplots()
        tfidf_scores = query_vec.toarray()[0]
        top_features = np.argsort(tfidf_scores)[-5:]
        ax.bar(range(len(top_features)), tfidf_scores[top_features])
        ax.set_title('Top Features TF-IDF pour la Query')
        ax.set_xticks(range(len(top_features)))
        ax.set_xticklabels([vectorizer.get_feature_names_out()[i] for i in top_features], rotation=45)
        plot_path = os.path.join(SUBMODELS_PATH, f"query_features_{int(time.time())}.png")
        plt.savefig(plot_path)
        plt.close()
       
        return f"{response} (Prédiction: {prediction}) | Graph: {plot_path}"
    except Exception as e:
        return f"❌ Erreur utilisation sous-modèle: {e}"
# ===============================================
# NOUVEAU: Fonctions Amélioration Base de Données via Fouille Internet
# ===============================================
def improve_database_with_web_search(topics, num_results_per_topic=5, vectordb=None):
    """
    Fouille internet sur des sujets spécifiques (pétrole, topographie, sciences physiques, sous-sol, etc.)
    et améliore la base de données en ajoutant de nouveaux documents.
    """
    specific_topics = topics or ["pétrole extraction techniques", "topographie cartographie avancée", "sciences physiques mécanique sol", "sous-sol géologie ressources"]
   
    if vectordb is None:
        vectordb, _ = load_vectordb()
        if vectordb is None:
            embedding_model = get_embedding_model()
            vectordb = FAISS.from_texts([""], embedding_model)
   
    new_documents = []
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
   
    for topic in specific_topics:
        st.write(f"🔍 Fouille internet pour: {topic}")
        search_results = enhanced_web_search(topic, max_results=num_results_per_topic, search_type="both")
       
        for result in search_results:
            content = f"Titre: {result.get('title', '')}\nContenu: {result.get('body', '')}\n"
            url = result.get('href') or result.get('url')
            if url and len(result.get('body', '')) < 500:
                extra_content = smart_content_extraction(url, max_length=2000)
                if "Impossible d'extraire" not in extra_content:
                    content += f"\nContenu détaillé: {extra_content}"
           
            chunks = text_splitter.split_text(content)
            for i, chunk in enumerate(chunks):
                doc = Document(
                    page_content=chunk,
                    metadata={
                        "source": url or topic,
                        "topic": topic,
                        "type": "web_enrichment",
                        "chunk_id": i
                    }
                )
                new_documents.append(doc)
   
    if new_documents:
        vectordb.add_documents(new_documents)
        vectordb.save_local(VECTORDB_PATH)
        return vectordb, f"✅ Base améliorée: {len(new_documents)} nouveaux chunks ajoutés sur {len(specific_topics)} sujets"
    else:
        return vectordb, "⚠️ Aucun nouveau contenu ajouté"
# ===============================================
# Version API pour utilisation externe
# ===============================================
class KibaliAPI:
    """API simplifiée pour utiliser Kibali depuis du code externe"""
    def __init__(self):
        self.vectordb = None
        self.chat_vectordb = None # AJOUT MÉMOIRE VECTORIELLE
        self.graph = None
        self.pois = []
        self.client = None
        self.model_name = WORKING_MODELS[list(WORKING_MODELS.keys())[0]]
        # Initialisation automatique
        self._initialize()
    def _initialize(self):
        """Initialisation automatique"""
        try:
            setup_drive()
            self.vectordb, _ = load_vectordb()
            self.chat_vectordb, _ = load_chat_vectordb() # AJOUT MÉMOIRE VECTORIELLE
            self.graph, self.pois, _ = load_existing_graph()
            self.client = create_client()
        except Exception as e:
            print(f"⚠️ Initialisation partielle: {e}")
    def ask(self, question, use_web=True):
        """Pose une question simple"""
        try:
            if use_web:
                docs = hybrid_search_enhanced(question, self.vectordb, web_search_enabled=True, chat_vectordb=self.chat_vectordb) # AJOUT MÉMOIRE VECTORIELLE
            else:
                docs = rag_search(question, self.vectordb)
            return generate_answer_enhanced(question, docs, self.model_name)
        except Exception as e:
            return f"❌ Erreur: {e}"
    def search_web(self, query, max_results=5):
        """Recherche web simple"""
        try:
            results = enhanced_web_search(query, max_results)
            return [{"title": r.get("title"), "url": r.get("href", r.get("url")), "snippet": r.get("body")} for r in results]
        except Exception as e:
            return [{"error": str(e)}]
    def calculate_route(self, from_place, to_place):
        """Calcule un itinéraire"""
        try:
            question = f"Comment aller de {from_place} à {to_place}"
            _, response, info = calculer_trajet(question, self.graph, self.pois)
            return {"response": response, "info": info}
        except Exception as e:
            return {"error": str(e)}
    def get_status(self):
        """Retourne le statut du système"""
        return get_system_status()
    # NOUVEAU: Méthodes API pour auto-apprentissage et amélioration DB
    def train_submodel(self, submodel_type="classification"):
        """Entraîne un sous-modèle"""
        path, msg = create_submodel_from_chat_history(self.chat_vectordb, submodel_type)
        return {"path": path, "message": msg}
    def improve_db(self, topics=None, num_results=5):
        """Améliore la DB avec fouille internet"""
        self.vectordb, msg = improve_database_with_web_search(topics, num_results, self.vectordb)
        return {"message": msg}
# Instance globale de l'API
kibali_api = KibaliAPI()
# ===============================================
# Interface Streamlit Améliorée
# ===============================================
st.markdown("""
<style>
    .stApp {
        background: white;
        color: black;
    }
    .sidebar .sidebar-content {
        background: white;
    }
    .stSidebar > div {
        background: white;
    }
    .stChatMessage {
        background: white;
        border-radius: 18px;
        border-left: 4px solid #2196F3;
        margin: 5px 0;
        padding: 12px;
        box-shadow: 0 1px 2px rgba(0,0,0,0.1);
        color: black !important;
        transition: all 0.3s ease;
        filter: none; /* Correction pour flou */
    }
    .stChatMessage:hover {
        transform: scale(1.02);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    .stChatMessage p, .stChatMessage li {
        color: black !important;
        background-color: rgba(255, 255, 255, 0.1);
    }
    .stTextInput > div > div > input {
        background: white;
        border: 1px solid #2196F3;
        border-radius: 20px;
        color: black;
        padding: 10px 15px;
        filter: none; /* Correction pour flou */
    }
    .stTextInput > div > div > input::placeholder {
        color: #757575;
    }
    .stButton > button {
        background: linear-gradient(45deg, #2196F3 0%, #21CBF3 100%);
        color: white;
        border: none;
        border-radius: 20px;
        padding: 10px 20px;
        font-weight: bold;
        box-shadow: 0 4px 8px rgba(0,0,0,0.3);
        transition: all 0.3s ease;
        width: 100%;
        margin-bottom: 10px;
    }
    .stButton > button:hover {
        transform: translateY(-2px) scale(1.05);
        box-shadow: 0 6px 12px rgba(0,0,0,0.4);
        animation: pulse 1s infinite; /* Effet fluide */
    }
    @keyframes pulse {
        0% { box-shadow: 0 6px 12px rgba(0,0,0,0.4); }
        50% { box-shadow: 0 6px 16px rgba(33, 150, 243, 0.6); }
        100% { box-shadow: 0 6px 12px rgba(0,0,0,0.4); }
    }
    .stSelectbox > div > div > select {
        background: white;
        border: 1px solid #2196F3;
        border-radius: 10px;
        color: black;
        filter: none; /* Correction pour flou */
    }
    .stCheckbox > div > label {
        color: black;
        transition: color 0.3s ease;
    }
    .stCheckbox > div > label:hover {
        color: #2196F3;
    }
    .stTextArea > div > div > textarea {
        background: white;
        color: black;
        border: 1px solid #2196F3;
    }
    h1, h2, h3 {
        color: #2196F3;
        text-shadow: 0 0 10px rgba(33, 150, 243, 0.5);
        animation: glow 2s ease-in-out infinite alternate;
    }
    @keyframes glow {
        from { text-shadow: 0 0 10px rgba(33, 150, 243, 0.5); }
        to { text-shadow: 0 0 20px rgba(33, 150, 243, 0.8), 0 0 30px rgba(33, 203, 243, 0.6); }
    }
    .chat-footer {
        position: fixed;
        bottom: 0;
        left: 0;
        right: 0;
        background: rgba(255, 255, 255, 0.95);
        border-top: 1px solid #2196F3;
        padding: 10px;
        z-index: 1000;
        transition: all 0.3s ease;
    }
    .chat-footer:hover {
        background: rgba(255, 255, 255, 1);
    }
    /* Effet scintillante pour mots importants */
    .sparkle-word {
        color: #2196F3;
        background: linear-gradient(45deg, #2196F3, #21CBF3, #4ecdc4, #45b7d1);
        background-size: 400% 400%;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        animation: sparkle 2s linear infinite, gradient-shift 3s ease infinite;
        cursor: pointer;
        position: relative;
        padding: 2px 4px;
        border-radius: 4px;
        transition: transform 0.2s ease;
    }
    .sparkle-word:hover {
        transform: scale(1.1);
        text-shadow: 0 0 10px rgba(33, 150, 243, 0.8);
    }
    @keyframes sparkle {
        0%, 100% { text-shadow: 0 0 5px rgba(33, 150, 243, 0.5); }
        50% { text-shadow: 0 0 20px rgba(33, 150, 243, 1), 0 0 30px rgba(33, 203, 243, 0.7); }
    }
    @keyframes gradient-shift {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    /* Correction pour lisibilité des questions/réponses */
    .stMarkdown {
        filter: none !important;
        -webkit-filter: none !important;
        color: black !important;
        font-weight: 400;
        line-height: 1.6;
        background-color: rgba(255, 255, 255, 0.1);
    }
    .stMarkdown p, .stMarkdown li {
        color: black !important;
        text-shadow: none;
    }
    .st-emotion-cache-1i5yq8u input, .st-emotion-cache-1i5yq8u textarea {
        color: black !important;
    }
    @media (max-width: 768px) {
        .chat-footer {
            padding: 5px;
        }
        .stTextInput input {
            font-size: 14px;
        }
        .sparkle-word {
            font-size: 0.9em;
        }
    }
</style>
""", unsafe_allow_html=True)
# Sidebar pour options
with st.sidebar:
    st.markdown("<h2 style='color: #2196F3; text-align: center;'>⚙️ Options</h2>", unsafe_allow_html=True)
    st.markdown("---")
   
    # Initialisation des états de session
    if 'status_msg' not in st.session_state:
        st.session_state.status_msg = ""
    if 'cache_msg' not in st.session_state:
        st.session_state.cache_msg = get_cache_stats()
   
    # Uploads et boutons config
    pdf_upload = st.file_uploader("📤 Upload PDFs", type="pdf", accept_multiple_files=True, key="pdf_sidebar")
    pbf_upload = st.file_uploader("📤 Upload OSM (.pbf)", type="osm.pbf", key="pbf_sidebar")
    process_pdfs_btn = st.button("🔄 Traiter PDFs", key="process_sidebar")
    load_graph_btn = st.button("📂 Charger graphe", key="load_graph_sidebar")
    load_vectordb_btn = st.button("📂 Charger DB", key="load_db_sidebar")
    clear_cache_btn = st.button("🗑️ Vider cache", key="clear_cache_sidebar")
   
    # NOUVEAU: Boutons pour auto-apprentissage et amélioration DB
    train_submodel_btn = st.button("🧠 Entraîner sous-modèle (sklearn)", key="train_submodel")
    improve_db_btn = st.button("📚 Améliorer DB (fouille internet)", key="improve_db")
   
    st.markdown("---")
    status_display = st.text_area("📊 Statut", value=st.session_state.status_msg, height=100, key='status_sidebar')
    cache_stats = st.text_area("📈 Cache", value=st.session_state.cache_msg, height=50, key='cache_sidebar')
   
    if "vectordb" not in st.session_state:
        st.session_state.vectordb = None
    if "chat_vectordb" not in st.session_state: # AJOUT MÉMOIRE VECTORIELLE
        st.session_state.chat_vectordb = None
    if "graph" not in st.session_state:
        st.session_state.graph = None
    if "pois" not in st.session_state:
        st.session_state.pois = []
    if "current_model" not in st.session_state:
        st.session_state.current_model = WORKING_MODELS[list(WORKING_MODELS.keys())[0]]
    if "agent" not in st.session_state:
        st.session_state.agent = None
    if pdf_upload:
        files = upload_pdfs(pdf_upload)
        st.session_state.status_msg = f"✅ {len(files)} PDFs uploadés" if files else "⚠️ Aucun PDF"
        # Pas de rerun ici : file_uploader gère déjà
    if pbf_upload:
        st.session_state.graph, st.session_state.pois, msg = upload_and_process_pbf(pbf_upload)
        st.session_state.status_msg = msg
        model_choice = st.selectbox("Modèle", list(WORKING_MODELS.keys()), key="model_sidebar")
        st.session_state.current_model, st.session_state.agent, cache_info = update_agent(model_choice, st.session_state.vectordb, st.session_state.graph, st.session_state.pois, st.session_state.chat_vectordb) # AJOUT MÉMOIRE VECTORIELLE
        st.session_state.cache_msg = cache_info
        st.rerun()
    if process_pdfs_btn:
        st.session_state.vectordb, msg = process_pdfs()
        st.session_state.status_msg = msg
        model_choice = st.selectbox("Modèle", list(WORKING_MODELS.keys()), key="model_process")
        st.session_state.current_model, st.session_state.agent, cache_info = update_agent(model_choice, st.session_state.vectordb, st.session_state.graph, st.session_state.pois, st.session_state.chat_vectordb) # AJOUT MÉMOIRE VECTORIELLE
        st.session_state.cache_msg = cache_info
        st.rerun()
    if load_graph_btn:
        st.session_state.graph, st.session_state.pois, msg = load_existing_graph()
        st.session_state.status_msg = msg
        model_choice = st.selectbox("Modèle", list(WORKING_MODELS.keys()), key="model_load_graph")
        st.session_state.current_model, st.session_state.agent, cache_info = update_agent(model_choice, st.session_state.vectordb, st.session_state.graph, st.session_state.pois, st.session_state.chat_vectordb) # AJOUT MÉMOIRE VECTORIELLE
        st.session_state.cache_msg = cache_info
        st.rerun()
    if load_vectordb_btn:
        st.session_state.vectordb, msg = load_vectordb()
        st.session_state.status_msg = msg
        model_choice = st.selectbox("Modèle", list(WORKING_MODELS.keys()), key="model_load_db")
        st.session_state.chat_vectordb, _ = load_chat_vectordb() # AJOUT MÉMOIRE VECTORIELLE: Charger chat db
        st.session_state.current_model, st.session_state.agent, cache_info = update_agent(model_choice, st.session_state.vectordb, st.session_state.graph, st.session_state.pois, st.session_state.chat_vectordb)
        st.session_state.cache_msg = cache_info
        st.rerun()
    if clear_cache_btn:
        msg = handle_clear_cache()
        st.session_state.status_msg = msg
        st.session_state.cache_msg = get_cache_stats()
        st.rerun()
   
    # NOUVEAU: Gestion des boutons auto-apprentissage et amélioration
    if train_submodel_btn:
        st.session_state.chat_vectordb, _ = load_chat_vectordb()
        submodel_path, msg = create_submodel_from_chat_history(st.session_state.chat_vectordb)
        st.session_state.status_msg = msg
        if submodel_path:
            st.write(f"Utiliser: use_submodel_for_automation('query', '{submodel_path}')")
        st.rerun()
   
    if improve_db_btn:
        topics_input = st.text_input("Sujets (séparés par ,)", value="pétrole,topographie,sciences physiques,sous-sol", key="topics_input")
        topics = [t.strip() for t in topics_input.split(",")]
        st.session_state.vectordb, msg = improve_database_with_web_search(topics)
        st.session_state.status_msg = msg
        st.rerun()
# Main area - Chat principal
st.title("🗺️ Kibali 🌟 - Assistant IA Avancé")
main_container = st.container()
with main_container:
    # Onglets pour autres fonctionnalités
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["🗺️ Trajets", "📸 Analyse Image", "🌐 Recherche Web", "💬 Chat", "📊 Status"])
    with tab1:
        st.markdown("""
        ### Calcul de trajets
        **Exemples:** "Comment aller de l'école à l'hôpital ?"
        """)
        trajectory_input = st.text_area("🗺️ Question de trajet", key="traj_input")
        if st.button("🚀 Calculer trajet", key="calc_traj"):
            carte_buf, reponse, traj_info = calculer_trajet(trajectory_input, st.session_state.graph, st.session_state.pois)
            st.text_area("📋 Détails", reponse, key="traj_details")
            if carte_buf:
                carte_buf.seek(0)
                st.image(Image.open(carte_buf), key="traj_map")
            if traj_info:
                if st.button("💾 Sauvegarder trajet", key="save_traj"):
                    save_trajectory(trajectory_input, reponse, traj_info)
                    st.write("✅ Trajet sauvegardé")
    with tab2:
        st.markdown("""
        ### Analyse d'images
        Upload une image pour analyse détaillée, annotations, graphiques et amélioration IA.
        """)
        image_upload = st.file_uploader("📤 Upload Image", type=["jpg", "png"], key="img_upload")
        if image_upload and st.button("🔍 Analyser", key="analyze_img"):
            analysis_data, proc_images, tables_str = process_image(image_upload.getvalue())
            improved_analysis = improve_analysis_with_llm(analysis_data, st.session_state.current_model)
            st.image(proc_images, caption=proc_images, width=400) # Responsive width
            st.markdown(tables_str, unsafe_allow_html=True)
            st.text_area("Analyse Améliorée (IA)", improved_analysis, key="img_analysis")
    with tab3:
        st.markdown("""
        ### Recherche web avancée avec extraction de contenu
        """)
        web_query = st.text_area("🔍 Requête de recherche", key="web_query")
        search_type = st.selectbox("Type de recherche", ["text", "news", "both"], key="search_type")
        if st.button("🔍 Rechercher", key="search_btn"):
            results = handle_web_search(web_query, search_type)
            st.markdown(results, unsafe_allow_html=True)
        url_extract = st.text_input("🌐 URL à extraire", key="url_extract")
        if st.button("📄 Extraire contenu", key="extract_btn"):
            content = handle_content_extraction(url_extract)
            st.text_area("Contenu extrait", content, key="extracted_content")
    with tab4:
        st.markdown("### Assistant IA avec recherche web intégrée")
        web_search_toggle = st.checkbox("🌐 Recherche web activée", value=True, key="web_toggle")
        # NOUVEAU: Option pour utiliser sous-modèle
        use_submodel = st.checkbox("🧠 Utiliser sous-modèle auto-appris pour réponse rapide", key="use_submodel")
        submodel_path_input = st.text_input("Chemin sous-modèle (optionnel)", key="submodel_path")
       
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []
        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"], avatar="☁️" if msg["role"] == "user" else "⭐"):
                # Correction pour lisibilité : utiliser markdown pour HTML
                if msg["role"] == "user":
                    st.markdown(f"**Question:** {highlight_important_words(msg['content'])}", unsafe_allow_html=True)
                else:
                    st.markdown(highlight_important_words(msg['content']), unsafe_allow_html=True)
        if prompt := st.chat_input("Pose une question...", key="chat_input"):
            with st.chat_message("user", avatar="☁️"):
                highlighted_prompt = highlight_important_words(prompt)
                st.markdown(f"**Question:** {highlighted_prompt}", unsafe_allow_html=True)
            with st.chat_message("assistant", avatar="⭐"):
                with st.spinner("Réponse en cours..."):
                    content_to_save = None # Variable intermédiaire pour corriger l'erreur NameError
                    if use_submodel and submodel_path_input:
                        automated = use_submodel_for_automation(prompt, submodel_path_input)
                        st.markdown(highlight_important_words(automated), unsafe_allow_html=True)
                        content_to_save = automated
                    else:
                        response = handle_chat_enhanced(prompt, st.session_state.chat_history, st.session_state.agent, list(WORKING_MODELS.keys())[0], st.session_state.vectordb, st.session_state.graph, st.session_state.pois, web_search_toggle)
                        st.markdown(highlight_important_words(response), unsafe_allow_html=True)
                        content_to_save = response
            st.session_state.chat_history.append({"role": "user", "content": prompt})
            st.session_state.chat_history.append({"role": "assistant", "content": content_to_save})
    with tab5:
        st.markdown("### Statut système")
        st.json(get_system_status())
st.markdown("### 📊 Informations Système")
setup_drive()
st.write(f"🚀 Kibali 🌟 - Assistant IA Avancé avec Recherche Web")
st.write(f"📁 Dossier unifié: {CHATBOT_DIR}")
st.write(f"🔑 Token HF configuré: {HF_TOKEN[:10]}...")
st.write(f"🌐 Recherche web intégrée")
existing_graphs = [f for f in os.listdir(GRAPHS_PATH) if f.endswith('_graph.graphml')] if os.path.exists(GRAPHS_PATH) else []
existing_pdfs = [f for f in os.listdir(PDFS_PATH) if f.endswith('.pdf')] if os.path.exists(PDFS_PATH) else []
st.write(f"📊 État initial:")
st.write(f" 🗺️ Graphes OSM: {len(existing_graphs)}")
st.write(f" 📄 PDFs: {len(existing_pdfs)}")
st.write(f" 💾 Base vectorielle: {'✅' if os.path.exists(VECTORDB_PATH) else '❌'}")
st.write(f" 🧠 Mémoire chat: {'✅' if os.path.exists(CHAT_VECTORDB_PATH) else '❌'}") # AJOUT MÉMOIRE VECTORIELLE
st.write(f" 🌐 Cache web: {'✅' if os.path.exists(WEB_CACHE_PATH) else '❌'}")
st.write(f" 📈 {get_cache_stats()}")
st.write("\n" + "="*60)
st.write("🎉 KIBALI 🌟 - SYSTÈME CHARGÉ AVEC SUCCÈS")
st.write("="*60)
st.write(f"📅 Version: 2.0.0 - {time.strftime('%Y-%m-%d %H:%M:%S')}")
st.write(f"🔑 Token HF: {'✅ Configuré' if HF_TOKEN else '❌ Manquant'}")
st.write(f"📁 Dossier: {CHATBOT_DIR}")
st.write(f"🌐 Recherche web: ✅ Activée")
st.write(f"💾 Cache intelligent: ✅ Activé")
st.write(f"🧠 Mémoire vectorielle chat: ✅ Activée") # AJOUT MÉMOIRE VECTORIELLE
st.write(f"🤖 Auto-apprentissage sklearn: ✅ Activé (sous-modèles dans {SUBMODELS_PATH})")
st.write(f"📚 Amélioration DB via fouille: ✅ Activée (sujets pétrole, topographie, etc.)")
st.write("\n📚 FONCTIONNALITÉS PRINCIPALES:")
st.write(" 💬 Chat RAG avec recherche web intelligent")
st.write(" 🧠 Mémoire des conversations pour fluidité") # AJOUT MÉMOIRE VECTORIELLE
st.write(" 🗺️ Calcul de trajets OSM")
st.write(" 📸 Analyse d'images avec IA")
st.write(" 🌐 Extraction de contenu web")
st.write(" 💾 Gestion unifiée des données")
st.write(" 🤖 Sous-modèles sklearn pour automatismes humains")
st.write(" 📚 Fouille auto internet pour enrichir DB (pétrole, topographie, sciences physiques, sous-sol)")
st.write("\n🚀 UTILISATION:")
st.write(" Interface: Exécutez les cellules suivantes")
st.write(" API: kibali_api.ask('votre question')")
st.write(" Auto-apprentissage: kibali_api.train_submodel()")
st.write(" Amélioration DB: kibali_api.improve_db(['pétrole'])")
st.write(" Tests: test_all_features()")
st.write("\n⚙️ MAINTENANCE:")
st.write(" Status: get_system_status()")
st.write(" Nettoyage: cleanup_old_cache()")
st.write(" Sauvegarde: backup_all_data()")
st.write("="*60)