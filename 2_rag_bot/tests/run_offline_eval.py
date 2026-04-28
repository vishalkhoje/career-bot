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

from src.agent import CareerAgent
from src.evaluation import EvaluationSystem

def run_eval():
    print("=" * 60)
    print("Career Bot Offline Evaluation")
    print("=" * 60)

    dataset_path = "2_rag_bot/tests/eval_dataset.json"
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
        expected = item["expected"]
        
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
        1. Correctness: Is the info factually accurate compared to expected?
        2. Groundedness: Does it stay within the bounds of what was asked?
        3. Completeness: Does it answer all parts of the query?
        
        Format: JSON only. Example: {{"correctness": 0.9, "groundedness": 1.0, "completeness": 0.8}}
        """
        
        try:
            score_resp = eval_sys.llm.invoke([{"role": "user", "content": eval_prompt}]).content
            scores = json.loads(score_resp)
        except:
            scores = {"correctness": 0.0, "groundedness": 0.0, "completeness": 0.0}
            
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
    
    avg_correctness = sum(r["scores"]["correctness"] for r in results) / len(results)
    avg_groundedness = sum(r["scores"]["groundedness"] for r in results) / len(results)
    avg_latency = sum(r["latency"] for r in results) / len(results)
    
    print(f"Avg Correctness:  {avg_correctness:.2f}")
    print(f"Avg Groundedness: {avg_groundedness:.2f}")
    print(f"Avg Latency:     {avg_latency:.0f}ms")
    print("=" * 60)

if __name__ == "__main__":
    run_eval()
