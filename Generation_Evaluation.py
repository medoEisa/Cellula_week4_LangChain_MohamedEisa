from deepeval.metrics import AnswerRelevancyMetric, FaithfulnessMetric
from deepeval import evaluate
import json

from model import ask_with_retrieval

def test_generation():
    relevancy = AnswerRelevancyMetric()
    faithfulness = FaithfulnessMetric()

    with open("ground_truth.json") as f:
        gt = json.load(f)

    test_cases = []

    for query, data in gt.items():
        contexts = data["contexts"]
        llm_output = ask_with_retrieval(query, retrieved_docs_texts=contexts)

        test_cases.append({
            "input": query,
            "actual_output": llm_output,
            "expected_output": data["answer"],
            "retrieval_contexts": contexts
        })

    evaluate(
        test_cases,
        metrics=[relevancy, faithfulness]
    )

if __name__ == "__main__":  
    test_generation()
    
