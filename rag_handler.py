from chroma_handler import create_or_load_chroma_db, query_chroma
from model import ask_with_retrieval
from config import CHROMA_DIR

def get_rag_response(query):
    db = create_or_load_chroma_db()
    docs = query_chroma(db, query, k=4)
    texts = [d.page_content for d in docs]
    return ask_with_retrieval(query, texts)
