"""
2_rag_bot/tests/run_offline_eval.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Offline evaluation runner for the Career Bot using RAGAS framework.
Runs the gold-standard dataset through the agent and scores results using Ragas.
"""

import os
import sys
import json
import time
from typing import List, Dict

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import CareerAgent
from src import EvaluationSystem

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)

def run_eval():
    print("=" * 60)
    print("Career Bot Offline Evaluation with RAGAS")
    print("=" * 60)

    tests_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_path = os.path.join(tests_dir, "eval_dataset.json")
    if not os.path.exists(dataset_path):
        print(f"Error: Dataset not found at {dataset_path}")
        return

    with open(dataset_path, "r") as f:
        dataset = json.load(f)

    agent = CareerAgent()
    eval_sys = EvaluationSystem()
    
    data = {
        "question": [],
        "answer": [],
        "contexts": [],
        "ground_truth": []
    }
    latencies = []
    
    for i, item in enumerate(dataset):
        query = item["query"]
        expected = item.get("expected_answer", item.get("expected", ""))
        
        print(f"\n[{i+1}/{len(dataset)}] Processing Query: {query}")
        
        start_time = time.time()
        response = ""
        # Consume the generator
        for chunk in agent.chat(query, []):
            response = chunk
            
        latency = (time.time() - start_time) * 1000
        latencies.append(latency)
        
        # Retrieve context used (approximate since agent uses internal retriever)
        try:
            context = agent.retriever.retrieve(query)
            contexts = [context] if context else [""]
        except Exception as e:
            print(f"Error retrieving context: {e}")
            contexts = [""]
        
        data["question"].append(query)
        data["answer"].append(response)
        data["contexts"].append(contexts)
        data["ground_truth"].append(expected)

    # Convert to HuggingFace Dataset
    print("\n" + "=" * 60)
    print("Running RAGAS Evaluation...")
    print("This may take some time depending on the dataset size.")
    print("=" * 60)
    
    hf_dataset = Dataset.from_dict(data)
    
    try:
        result = evaluate(
            hf_dataset,
            metrics=[
                faithfulness,
                answer_relevancy,
                context_precision,
                context_recall,
            ],
            raise_exceptions=False
        )
    except Exception as e:
        print(f"RAGAS evaluation failed: {e}")
        return

    print("\n" + "=" * 60)
    metrics = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
    scores = {}
    for m in metrics:
        try:
            val = result[m]
            if isinstance(val, list):
                scores[m] = float(sum(val) / len(val)) if val else 0.0
            else:
                scores[m] = float(val)
        except Exception:
            scores[m] = 0.0

    print(f"Faithfulness:       {scores['faithfulness']:.4f}")
    print(f"Answer Relevancy:   {scores['answer_relevancy']:.4f}")
    print(f"Context Precision:  {scores['context_precision']:.4f}")
    print(f"Context Recall:     {scores['context_recall']:.4f}")
    
    avg_latency = sum(latencies) / len(latencies) if latencies else 0
    print(f"Avg Latency:        {avg_latency:.0f}ms")
    print("=" * 60)

    # Save to database
    from src.core.config import EMBEDDING_VERSION
    report = {
        "avg_groundedness": scores["faithfulness"],
        "avg_relevance": scores["answer_relevancy"],
        "avg_hallucination": 1.0 - scores["faithfulness"],
        "avg_correctness": (scores["context_precision"] + scores["context_recall"]) / 2.0,
        "context_precision": scores["context_precision"],
        "context_recall": scores["context_recall"],
        "avg_latency": avg_latency,
        "total_queries": len(dataset)
    }
    eval_sys.log_benchmark(version=EMBEDDING_VERSION, results=report)
    print("[Database] Saved benchmark results to evaluations.db")

if __name__ == "__main__":
    run_eval()
