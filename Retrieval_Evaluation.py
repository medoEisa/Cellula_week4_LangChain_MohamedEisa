from chroma_handler import create_or_load_chroma_db, query_chroma
from model import ask_with_retrieval
import os

# --------------------------
# Load Chroma DB
# --------------------------
db = create_or_load_chroma_db()

# --------------------------
# Example queries
# --------------------------
queries = [
    "Explain the difference between supervised and unsupervised learning.",
    "What is the purpose of a convolutional layer in CNNs?",
    "How does gradient descent work in deep learning?"
]

# --------------------------
# Function to evaluate retrieval metrics
# --------------------------
def evaluate_retrieval(query, db, k=3):
    retrieved_docs = query_chroma(db, query_text=query, k=k)

    # Send to LLM for answer generation
    retrieved_texts = [d.page_content for d in retrieved_docs]
    answer = ask_with_retrieval(query, retrieved_docs_texts=retrieved_texts)

    # Simple evaluation metrics (for demo purposes)
    # Normally you need ground truth relevance labels
    precision_at_k = min(1.0, len(retrieved_docs)/k)  # placeholder
    recall_at_k = 1.0  # placeholder
    mrr = 1.0  # placeholder
    ndcg = 1.0  # placeholder

    return {
        "Query": query,
        "Retrieved Docs": [d.page_content[:80]+"..." for d in retrieved_docs],
        "Answer": answer[:200]+"...",
        "Precision@K": precision_at_k,
        "Recall@K": recall_at_k,
        "MRR": mrr,
        "nDCG": ndcg
    }

# --------------------------
# Run evaluation
# --------------------------
for q in queries:
    results = evaluate_retrieval(q, db)
    print("\nQuery:", results["Query"])
    print("Retrieved Docs:", results["Retrieved Docs"])
    print("Generated Answer:", results["Answer"])
    print("Metrics:", {k: v for k, v in results.items() if k not in ["Query", "Retrieved Docs", "Answer"]})
