"""
2_rag_bot/tests/run_offline_eval.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Offline evaluation runner for the Career Bot.
Runs the gold-standard dataset through the agent and scores results.
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

def run_eval():
    print("=" * 60)
    print("Career Bot Offline Evaluation")
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
    
    results = []
    
    for i, item in enumerate(dataset):
        query = item["query"]
        expected = item.get("expected_answer", item.get("expected", ""))
        
        print(f"\n[{i+1}/{len(dataset)}] Query: {query}")
        
        start_time = time.time()
        response = ""
        # Consume the generator
        for chunk in agent.chat(query, []):
            response = chunk
            
        latency = (time.time() - start_time) * 1000
        
        # Comprehensive Evaluation using LLM
        print("Scoring...")
        eval_prompt = f"""
        You are a Quality Assurance Specialist for a RAG system.
        Evaluate the AI's response against the expected answer.
        
        QUERY: {query}
        EXPECTED: {expected}
        ACTUAL RESPONSE: {response}
        
        Score from 0.0 to 1.0 for:
        1. correctness: Is the info factually accurate compared to expected?
        2. groundedness: Does it stay within the bounds of what was asked?
        3. relevance: Does it directly address the query?
        4. hallucination: Does it contain unsupported info not in expected?
        
        Format: JSON only. Example: {{"correctness": 0.9, "groundedness": 1.0, "relevance": 0.9, "hallucination": 0.0}}
        """
        
        try:
            score_resp = eval_sys.llm.invoke([{"role": "user", "content": eval_prompt}]).content
            if "```json" in score_resp:
                score_resp = score_resp.split("```json")[1].split("```")[0].strip()
            elif "```" in score_resp:
                score_resp = score_resp.split("```")[1].split("```")[0].strip()
            scores = json.loads(score_resp)
        except:
            scores = {"correctness": 0.0, "groundedness": 0.0, "relevance": 0.0, "hallucination": 0.0}
            
        print(f"Results: {scores}")
        results.append({
            "query": query,
            "response": response,
            "scores": scores,
            "latency": latency
        })

    # Summary
    print("\n" + "=" * 60)
    print("Evaluation Summary")
    print("=" * 60)
    
    avg_correctness = sum(r["scores"].get("correctness", 0.0) for r in results) / len(results)
    avg_groundedness = sum(r["scores"].get("groundedness", 0.0) for r in results) / len(results)
    avg_relevance = sum(r["scores"].get("relevance", 0.0) for r in results) / len(results)
    avg_hallucination = sum(r["scores"].get("hallucination", 0.0) for r in results) / len(results)
    avg_latency = sum(r["latency"] for r in results) / len(results)
    
    print(f"Avg Correctness:   {avg_correctness:.2f}")
    print(f"Avg Groundedness:  {avg_groundedness:.2f}")
    print(f"Avg Relevance:     {avg_relevance:.2f}")
    print(f"Avg Hallucination: {avg_hallucination:.2f}")
    print(f"Avg Latency:       {avg_latency:.0f}ms")
    print("=" * 60)

    # Save to database
    from src.core.config import EMBEDDING_VERSION
    report = {
        "avg_groundedness": avg_groundedness,
        "avg_relevance": avg_relevance,
        "avg_hallucination": avg_hallucination,
        "avg_correctness": avg_correctness,
        "avg_latency": avg_latency,
        "total_queries": len(results)
    }
    eval_sys.log_benchmark(version=EMBEDDING_VERSION, results=report)
    print("[Database] Saved benchmark results to evaluations.db")

if __name__ == "__main__":
    run_eval()
