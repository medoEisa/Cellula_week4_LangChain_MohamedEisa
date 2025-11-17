import os
import json
import warnings
import sys
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from config import CHUNKS_FILE, CHROMA_DIR
import streamlit as st


# Silence TF imports in case they exist
sys.modules["tensorflow"] = None
sys.modules["tf_keras"] = None
warnings.filterwarnings("ignore", message="MessageFactory")

def load_chunks():
    with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
    texts = [d["content"] for d in data]
    metadatas = [d.get("metadata", {}) for d in data]
    return texts, metadatas
@st.cache_resource
def create_or_load_chroma_db(embedding_model_name="BAAI/bge-base-en-v1.5"):
    print(" Initializing embedding model...", flush=True)
    embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)
    if os.path.exists(CHROMA_DIR):
        print(" Loading existing Chroma database...", flush=True)
        db = Chroma(persist_directory=CHROMA_DIR, embedding_function=embeddings)
    else:
        print(" Creating new Chroma database (this may take a while)...", flush=True)
        texts, metadatas = load_chunks()
        db = Chroma.from_texts(
            texts=texts,
            embedding=embeddings,
            metadatas=metadatas,
            persist_directory=CHROMA_DIR
        )
        db.persist()
        print(f" Chroma DB created and saved to {CHROMA_DIR}", flush=True)

    return db

def query_chroma(db, query_text, k=3):
    # returns list of Document objects (page_content, metadata)
    results = db.similarity_search(query_text, k=k)
    return results

