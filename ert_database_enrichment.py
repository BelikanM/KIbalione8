#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module d'enrichissement de base de données ERT
Inspiré des techniques de Kibali pour améliorer les connaissances ERT via fouille internet
"""

import os
import time
import json
import numpy as np
from langchain_tavily import TavilySearch as TavilySearchResults
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from dotenv import load_dotenv

# Configuration
load_dotenv()
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

class ERTDatabaseEnricher:
    """Enrichisseur de base de données ERT utilisant la fouille internet"""
    
    def __init__(self, vectorstore=None):
        self.vectorstore = vectorstore
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, 
            chunk_overlap=100
        )
        
        # Sujets spécialisés ERT pour fouille
        self.ert_topics = [
            "ERT electrical resistivity tomography interpretation",
            "résistivité électrique géophysique subsurface",
            "tomographie résistivité électrique environnement",
            "ERT data processing inversion algorithms",
            "géophysique électrique aquifère détection",
            "electrical resistivity soil characterization",
            "ERT archaeological prospection methods",
            "résistivité électrique pollution contamination",
            "electrical tomography geological structures",
            "ERT hydrogeology groundwater exploration",
            "géotechnique résistivité électrique stabilité",
            "electrical resistivity mining exploration",
            "ERT environmental monitoring techniques",
            "tomographie électrique fractures detection",
            "résistivité électrique argile sable gravier"
        ]
    
    def enhanced_web_search(self, query, max_results=5):
        """Recherche web améliorée pour ERT"""
        try:
            tool = TavilySearchResults(api_key=TAVILY_API_KEY, max_results=max_results)
            
            # Enrichir la requête pour ERT
            enhanced_query = f"{query} ERT electrical resistivity tomography geophysics"
            results = tool.invoke(enhanced_query)
            
            if not results:
                return []
                
            formatted_results = []
            for result in results:
                formatted_results.append({
                    'title': result.get('title', ''),
                    'content': result.get('content', ''),
                    'url': result.get('url', ''),
                    'source_type': 'ert_web_search'
                })
            
            return formatted_results
            
        except Exception as e:
            print(f"❌ Erreur recherche web ERT: {e}")
            return []
    
    def improve_ert_database(self, specific_topics=None, num_results_per_topic=3):
        """Améliore la base de données avec des connaissances ERT spécialisées"""
        topics = specific_topics or self.ert_topics
        new_documents = []
        
        print(f"🔍 Fouille internet ERT sur {len(topics)} sujets...")
        
        for i, topic in enumerate(topics):
            print(f"📊 Progression: {i+1}/{len(topics)} - {topic[:50]}...")
            
            search_results = self.enhanced_web_search(topic, max_results=num_results_per_topic)
            
            for result in search_results:
                content = f"Titre: {result.get('title', '')}\n"
                content += f"Contenu: {result.get('content', '')}\n"
                
                # Diviser en chunks
                chunks = self.text_splitter.split_text(content)
                
                for j, chunk in enumerate(chunks):
                    doc = Document(
                        page_content=chunk,
                        metadata={
                            "source": result.get('url', topic),
                            "topic": topic,
                            "type": "ert_enrichment",
                            "chunk_id": j,
                            "timestamp": time.time()
                        }
                    )
                    new_documents.append(doc)
            
            # Pause pour éviter rate limiting
            time.sleep(1)
        
        if new_documents:
            print(f"✅ {len(new_documents)} nouveaux documents ERT collectés")
            return new_documents
        else:
            print("⚠️ Aucun nouveau document collecté")
            return []
    
    def analyze_ert_data_context(self, numbers):
        """Analyse les données ERT pour enrichir la recherche"""
        if numbers is None or len(numbers) < 10:
            return None
            
        arr = np.array(numbers)
        
        analysis = {
            "mean": np.mean(arr),
            "std": np.std(arr),
            "min": np.min(arr),
            "max": np.max(arr),
            "cv": np.std(arr) / np.mean(arr) if np.mean(arr) != 0 else 0,
            "is_ert_like": 0.1 <= np.min(arr) <= 10000 and len(numbers) >= 20
        }
        
        return analysis
    
    def get_contextual_ert_topics(self, ert_analysis):
        """Génère des sujets de recherche contextuels basés sur l'analyse ERT"""
        if not ert_analysis or not ert_analysis.get('is_ert_like'):
            return self.ert_topics[:5]  # Topics génériques
        
        mean_res = ert_analysis['mean']
        contextual_topics = []
        
        # Sujets basés sur la valeur de résistivité
        if mean_res < 10:
            contextual_topics.extend([
                "ERT low resistivity clay detection interpretation",
                "résistivité faible argile eau saline conducteur",
                "electrical resistivity contamination pollution"
            ])
        elif 10 <= mean_res <= 100:
            contextual_topics.extend([
                "ERT medium resistivity soil sediment analysis",
                "résistivité moyenne sable limon géotechnique",
                "electrical resistivity groundwater aquifer"
            ])
        else:
            contextual_topics.extend([
                "ERT high resistivity rock bedrock detection",
                "résistivité élevée roche cristalline vide",
                "electrical resistivity fracture cavity detection"
            ])
        
        # Sujets généraux ERT
        contextual_topics.extend([
            f"ERT {mean_res:.1f} ohm.m interpretation geophysical",
            "electrical resistivity tomography data processing",
            "ERT inversion algorithms geological structures"
        ])
        
        return contextual_topics

def create_ert_knowledge_base(vectorstore_path, ert_data=None):
    """Crée une base de connaissances ERT enrichie"""
    enricher = ERTDatabaseEnricher()
    
    # Analyser les données ERT si fournies
    ert_analysis = None
    if ert_data:
        ert_analysis = enricher.analyze_ert_data_context(ert_data)
        print(f"📊 Analyse ERT: {ert_analysis}")
    
    # Obtenir des sujets contextuels
    if ert_analysis:
        topics = enricher.get_contextual_ert_topics(ert_analysis)
    else:
        topics = enricher.ert_topics[:10]  # Limiter pour test
    
    # Collecter les documents
    documents = enricher.improve_ert_database(topics, num_results_per_topic=2)
    
    if documents:
        # Créer ou charger la base vectorielle
        try:
            from sentence_transformers import SentenceTransformer
            
            class SentenceTransformerEmbeddings:
                def __init__(self, model_name, device='cpu'):
                    self.model = SentenceTransformer(model_name, device=device)
                
                def embed_documents(self, texts):
                    return self.model.encode(texts, convert_to_numpy=True).tolist()
                
                def embed_query(self, text):
                    return self.model.encode([text], convert_to_numpy=True)[0].tolist()
            
            embeddings = SentenceTransformerEmbeddings('sentence-transformers/all-MiniLM-L6-v2')
            
            if os.path.exists(vectorstore_path):
                # Charger et enrichir
                vectorstore = FAISS.load_local(vectorstore_path, embeddings, allow_dangerous_deserialization=True)
                vectorstore.add_documents(documents)
            else:
                # Créer nouveau
                vectorstore = FAISS.from_documents(documents, embeddings)
            
            # Sauvegarder
            vectorstore.save_local(vectorstore_path)
            
            print(f"✅ Base de connaissances ERT enrichie: {len(documents)} nouveaux documents")
            print(f"💾 Sauvegardée dans: {vectorstore_path}")
            
            return vectorstore, f"Base ERT enrichie avec {len(documents)} documents sur {len(topics)} sujets"
            
        except Exception as e:
            print(f"❌ Erreur création base vectorielle: {e}")
            return None, f"Erreur: {e}"
    
    else:
        return None, "Aucun document collecté"

# Exemple d'utilisation
if __name__ == "__main__":
    # Test avec données ERT simulées
    test_ert_data = np.random.lognormal(2, 1, 100) * 10  # Simulation données ERT
    
    vectorstore, message = create_ert_knowledge_base(
        "/tmp/test_ert_vectordb", 
        ert_data=test_ert_data
    )
    
    print(f"Résultat: {message}")