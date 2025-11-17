
# eval_rag_deepeval.py
import os
import sys
import json
from deepeval import assert_test
from deepeval.metrics.ragas import (
    RAGASContextualPrecisionMetric,
    RAGASContextualRecallMetric,
    RAGASAnswerRelevancyMetric,
    RAGASFaithfulnessMetric,
)
from deepeval.test_case import LLMTestCase

from model import ask_with_retrieval
from chroma_handler import create_or_load_chroma_db, query_chroma
# Make sure your project modules are in sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(__file__)))


with open("eval_test/test_cases.json", "r", encoding="utf-8") as f:
    TEST_CASES_DATA = json.load(f)

db = create_or_load_chroma_db()

metrics = [
    RAGASContextualPrecisionMetric(model="meta-llama/llama-3.3-70b-instruct:free"),
    RAGASContextualRecallMetric(model="meta-llama/llama-3.3-70b-instruct:free"),
    RAGASAnswerRelevancyMetric(model="meta-llama/llama-3.3-70b-instruct:free"),
    RAGASFaithfulnessMetric(model="meta-llama/llama-3.3-70b-instruct:free"),
]

TEST_CASES = []
for case in TEST_CASES_DATA:
    query = case["query"]
    expected_output = case["expected_output"]

    retrieved_docs = query_chroma(db, query_text=query, k=3)
    retrieved_texts = [d.page_content for d in retrieved_docs]

    actual_output = ask_with_retrieval(query, retrieved_docs_texts=retrieved_texts)

    TEST_CASES.append(
        LLMTestCase(
            input=query,
            retrieval_context=retrieved_texts,
            expected_output=expected_output,
            actual_output=actual_output,
        )
    )

print("\n===== RAG DeepEval Benchmark =====\n")
for tc in TEST_CASES:
    print(f"Query: {tc.input}")
    print(f"Expected: {tc.expected_output}")
    print(f"Actual: {tc.actual_output}\n")

    for metric in metrics:
        score = metric.measure(tc)
        print(f"{metric.__class__.__name__}: {score.score:.4f}")
    print("-" * 60)
