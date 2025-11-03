#!/usr/bin/env python3
"""Script pour ajouter informations_kibali.pdf à la base vectorielle de Kibali"""

import os
import sys
import json
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

# Chemins
CHATBOT_DIR = os.path.expanduser('~/RAG_ChatBot')
VECTORDB_PATH = os.path.join(CHATBOT_DIR, "vectordb")
PDFS_PATH = os.path.join(CHATBOT_DIR, "pdfs")
METADATA_PATH = os.path.join(CHATBOT_DIR, "metadata.json")

def main():
    print("🚀 Mise à jour de la base vectorielle de Kibali...")
    
    # 1. Charger le modèle d'embeddings
    print("📦 Chargement du modèle d'embeddings...")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'}
    )
    
    # 2. Charger la vectordb existante
    print(f"📂 Chargement de la base vectorielle depuis {VECTORDB_PATH}...")
    try:
        vectordb = FAISS.load_local(
            VECTORDB_PATH, 
            embeddings, 
            allow_dangerous_deserialization=True
        )
        print(f"✅ Base vectorielle chargée ({vectordb.index.ntotal} vecteurs)")
    except Exception as e:
        print(f"❌ Erreur chargement vectordb: {e}")
        return 1
    
    # 3. Charger les métadonnées
    with open(METADATA_PATH, 'r', encoding='utf-8') as f:
        metadata = json.load(f)
    
    processed_filenames = {p["filename"] for p in metadata["processed_files"]}
    print(f"📋 {len(processed_filenames)} PDFs déjà indexés")
    
    # 4. Traiter informations_kibali.pdf
    pdf_file = "informations_kibali.pdf"
    
    if pdf_file in processed_filenames:
        print(f"ℹ️  {pdf_file} déjà indexé, réindexation...")
        # Retirer de la liste pour le rajouter
        metadata["processed_files"] = [
            p for p in metadata["processed_files"] 
            if p["filename"] != pdf_file
        ]
    
    pdf_path = os.path.join(PDFS_PATH, pdf_file)
    
    if not os.path.exists(pdf_path):
        print(f"❌ {pdf_path} n'existe pas")
        return 1
    
    print(f"📄 Extraction du texte de {pdf_file}...")
    
    # Extraire avec PyPDFLoader
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()
    
    # Combiner le texte
    full_text = "\n\n".join([page.page_content for page in pages])
    
    print(f"📏 Texte extrait : {len(full_text)} caractères")
    
    # 5. Découper en chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100
    )
    
    chunks = text_splitter.split_text(full_text)
    print(f"✂️  {len(chunks)} chunks créés")
    
    # Créer des documents
    documents = []
    for i, chunk in enumerate(chunks):
        doc = Document(
            page_content=chunk,
            metadata={
                "source": pdf_file,
                "chunk_id": i,
                "type": "pdf"
            }
        )
        documents.append(doc)
    
    # 6. Ajouter à la vectordb
    print(f"➕ Ajout de {len(documents)} documents à la vectordb...")
    vectordb.add_documents(documents)
    
    # 7. Sauvegarder
    print(f"💾 Sauvegarde dans {VECTORDB_PATH}...")
    vectordb.save_local(VECTORDB_PATH)
    
    # 8. Mettre à jour les métadonnées
    metadata["processed_files"].append({
        "filename": pdf_file,
        "chunks": len(chunks)
    })
    metadata["total_chunks"] = metadata.get("total_chunks", 0) + len(chunks)
    
    with open(METADATA_PATH, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Métadonnées mises à jour")
    print(f"\n🎉 Base vectorielle mise à jour avec succès !")
    print(f"📊 Total : {vectordb.index.ntotal} vecteurs")
    print(f"📚 {len(metadata['processed_files'])} PDFs indexés")
    
    # 9. Test de recherche
    print(f"\n🔍 Test de recherche : 'Nyundu Francis Arnaud'...")
    retriever = vectordb.as_retriever(search_kwargs={"k": 3})
    docs = retriever.invoke("Nyundu Francis Arnaud créateur Kibali")
    
    if docs:
        print(f"✅ {len(docs)} documents trouvés :")
        for i, doc in enumerate(docs, 1):
            print(f"\n📄 Document {i} (Source: {doc.metadata.get('source', 'Unknown')}):")
            print(f"   {doc.page_content[:200]}...")
    else:
        print("❌ Aucun document trouvé")
    
    return 0

if __name__ == '__main__':
    sys.exit(main())
