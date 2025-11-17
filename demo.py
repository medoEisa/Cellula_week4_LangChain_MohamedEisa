
# # #deepseek-chat-api=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
import os
from model import ask_with_retrieval, memory
from chroma_handler import create_or_load_chroma_db, query_chroma
from config import CHROMA_DIR


def run_demo():
    """Simple CLI demo for EduBuddy RAG system."""
    db = create_or_load_chroma_db()
    print("\n=== EduBuddy RAG Demo ===")
    print("Type 'exit' to quit.\n")

    while True:
        q = input("You: ").strip()
        if not q:
            continue
        if q.lower() in ("exit", "quit"):
            print("Goodbye !!!!")
            break

        # Retrieve top-4 most relevant chunks
        docs = query_chroma(db, q, k=4)
        texts = [d.page_content for d in docs] if docs else []

        # Ask EduBuddy with retrieved context
        answer = ask_with_retrieval(q, texts)
        print("\nEduBuddy:", answer, "\n")


if __name__ == "__main__":
    run_demo()
