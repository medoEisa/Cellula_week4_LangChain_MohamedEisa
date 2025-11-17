
# rag_eval.py
"""
RAG evaluation script (production-ready).
Usage:
    python rag_eval.py --k 1 3 5 10 --topk 10 --out_dir ./eval_out --ground_truth my_qrels.json

If you don't provide a ground-truth qrels file, the script will build a simple heuristic GT
based on chunk metadata titles containing the query token (useful as a quick start).
For rigorous evaluation, provide a human-labeled qrels file in TREC/qrels-like format.
"""
import os
import json
import math
import argparse
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#  Project imports 
from chroma_handler import create_or_load_chroma_db, load_chunks  
from config import TOPICS

# Evaluation helpers 
def precision_at_k(retrieved_indices, gt_set, k):
    if k == 0:
        return 0.0
    return sum(1 for idx in retrieved_indices[:k] if idx in gt_set) / k

def recall_at_k(retrieved_indices, gt_set, k):
    if len(gt_set) == 0:
        return 0.0
    return len(set(retrieved_indices[:k]) & gt_set) / len(gt_set)

def reciprocal_rank(retrieved_indices, gt_set, k=None):
    for pos, idx in enumerate(retrieved_indices, start=1):
        if k is not None and pos > k:
            break
        if idx in gt_set:
            return 1.0 / pos
    return 0.0

def dcg_at_k(retrieved_indices, gt_set, k):
    dcg = 0.0
    for i, idx in enumerate(retrieved_indices[:k], start=1):
        rel = 1.0 if idx in gt_set else 0.0
        dcg += (2**rel - 1) / math.log2(i + 1)
    return dcg

def idcg_at_k(gt_set, k):
    rels = min(len(gt_set), k)
    idcg = 0.0
    for i in range(1, rels + 1):
        idcg += (2**1 - 1) / math.log2(i + 1)
    return idcg

def ndcg_at_k(retrieved_indices, gt_set, k):
    idcg = idcg_at_k(gt_set, k)
    if idcg == 0.0:
        return 0.0
    return dcg_at_k(retrieved_indices, gt_set, k) / idcg

#  Ground truth helpers 
def build_heuristic_gt(texts, metadatas, queries):

    gt = defaultdict(set)
    for i, md in enumerate(metadatas):
        title = ""
        if isinstance(md, dict):
            title = (md.get("title") or "") + " " + (md.get("source") or "")
        else:
            title = str(md)
        title_l = title.lower()
        for q in queries:
            if q.lower() in title_l:
                gt[q].add(i)
    return gt

#  Main evaluation 
def evaluate(db, texts, metadatas, queries, gt, top_k_list, retrieve_k):
    results = []
    for k in top_k_list:
        precision_list = []
        recall_list = []
        mrr_list = []
        ndcg_list = []
        for q in queries:
            docs = db.similarity_search(q, k=retrieve_k)  
            retrieved_indices = []
            for d in docs:
                try:
                    idx = texts.index(d.page_content)
                except ValueError:
                    idx = next((i for i,t in enumerate(texts) if t.startswith(d.page_content[:80]) or d.page_content[:80] in t), None)
                if idx is None:
                    continue
                retrieved_indices.append(idx)
            gt_set = gt.get(q, set())
            precision_list.append(precision_at_k(retrieved_indices, gt_set, k))
            recall_list.append(recall_at_k(retrieved_indices, gt_set, k))
            mrr_list.append(reciprocal_rank(retrieved_indices, gt_set, k))
            ndcg_list.append(ndcg_at_k(retrieved_indices, gt_set, k))

        results.append({
            "k": k,
            "precision@k": float(np.mean(precision_list)),
            "recall@k": float(np.mean(recall_list)),
            "mrr": float(np.mean(mrr_list)),
            "ndcg@k": float(np.mean(ndcg_list))
        })
    return results

def plot_metrics(df, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    # Plot Precision@K
    plt.figure()
    plt.plot(df["k"], df["precision@k"], marker='o')
    plt.xlabel("K")
    plt.ylabel("Precision@K")
    plt.title("Precision@K")
    plt.xticks(df["k"])
    plt.grid(True)
    plt.tight_layout()
    ppath = os.path.join(out_dir, "precision_at_k.png")
    plt.savefig(ppath, dpi=150)
    plt.close()

    # Plot nDCG@K
    plt.figure()
    plt.plot(df["k"], df["ndcg@k"], marker='o')
    plt.xlabel("K")
    plt.ylabel("nDCG@K")
    plt.title("nDCG@K")
    plt.xticks(df["k"])
    plt.grid(True)
    plt.tight_layout()
    npath = os.path.join(out_dir, "ndcg_at_k.png")
    plt.savefig(npath, dpi=150)
    plt.close()
    return ppath, npath

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--topk", nargs="+", type=int, default=[1,3,5,10], help="List of K values for metrics")
    parser.add_argument("--retrieve_k", type=int, default=10, help="How many docs to retrieve from vector DB")
    parser.add_argument("--out_dir", type=str, default="./eval_out", help="Where to save results and plots")
    parser.add_argument("--ground_truth", type=str, default=None, help="Optional qrels JSON file: {query: [doc_indices]}")
    args = parser.parse_args()

    db = create_or_load_chroma_db()
    texts, metadatas = load_chunks()

    queries = TOPICS 

    # Load or build GT
    if args.ground_truth and os.path.exists(args.ground_truth):
        with open(args.ground_truth, "r", encoding="utf-8") as f:
            gt_raw = json.load(f)
        gt = {q: set(v) for q, v in gt_raw.items()}
    else:
        print("No GT provided ")
        gt = build_heuristic_gt(texts, metadatas, queries)

    results = evaluate(db, texts, metadatas, queries, gt, args.topk, args.retrieve_k)
    df = pd.DataFrame(results)
    os.makedirs(args.out_dir, exist_ok=True)
    csv_path = os.path.join(args.out_dir, "rag_eval_summary.csv")
    df.to_csv(csv_path, index=False)
    print("Saved metrics to:", csv_path)
    ppath, npath = plot_metrics(df, args.out_dir)
    print("Saved Precision plot to:", ppath)
    print("Saved nDCG plot to:", npath)

if __name__ == "__main__":
    main()
