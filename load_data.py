from langchain_community.document_loaders import WikipediaLoader
from config import TOPICS, MAX_DOCS_PER_TOPIC, MAX_CHARS_PER_DOC, CHUNK_SIZE, CHUNK_OVERLAP , DATA_DIR, RAW_FILE, CHUNKS_FILE
import json
import os
from langchain_text_splitters import NLTKTextSplitter
import nltk
import sys
nltk.download("punkt")

sys.modules["tensorflow"] = None
sys.modules["tf_keras"] = None

def load_wikipedia_articles():
    docs = []
    print(".... Loading Wikipedia articles...")
    for topic in TOPICS:
        print(f"   ↳ {topic}")
        loader = WikipediaLoader(
            query=topic,
            load_max_docs=MAX_DOCS_PER_TOPIC,
            doc_content_chars_max=MAX_CHARS_PER_DOC
        )
        docs.extend(loader.load())
    print(f"Loaded {len(docs)} documents total.")
    return docs

def split_documents(docs):
    print("...Splitting documents into chunks...")
    text_splitter = NLTKTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    split_docs = text_splitter.split_documents(docs)
    print(f"....Split into {len(split_docs)} chunks.")
    return split_docs

def save_raw_docs(docs):
    os.makedirs(DATA_DIR, exist_ok=True)
    documents = [doc.page_content for doc in docs]
    metadata = [{"title": doc.metadata["title"], "source": doc.metadata["source"]} for doc in docs]

    data = [{"content": c, "metadata": m} for c, m in zip(documents, metadata)]
    with open(RAW_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
    print(f" ! Saved raw documents to {RAW_FILE}")

def save_chunks(split_docs):
    data = [{"content": doc.page_content, "metadata": doc.metadata} for doc in split_docs]
    with open(CHUNKS_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
    print(f" ! Saved chunks to {CHUNKS_FILE}")

if __name__ == "__main__":  
    documents = load_wikipedia_articles()
    save_raw_docs(documents)
    split_docs = split_documents(documents)
    save_chunks(split_docs)
    print("\n Wikipedia data loading and chunking completed successfully!")
