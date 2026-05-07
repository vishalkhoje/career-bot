import sys
import os
from datetime import datetime

# Add 2_rag_bot to path
sys.path.append(os.path.abspath("2_rag_bot"))

from src import Observability
from src import EvaluationSystem

def test_monitoring_and_eval():
    print("--- Starting Verification Test ---")
    
    query = "Test Query for Monitoring Upgrade"
    response = "This is a simulated response for verification."
    context = "Verification context."
    latency = 1200.5
    tokens = 500
    steps = ["Cache Miss", "Intent: FACTUAL", "Retrieval", "LLM Generation"]
    
    # 1. Test Monitoring (Cost, Steps, Alerts)
    print("\n[1/2] Testing Observability.log_request...")
    Observability.log_request(
        query=query,
        latency_ms=latency,
        tokens=tokens,
        status="success",
        steps=steps,
        retrieved_docs=[context],
        hallucination_score=0.1
    )
    
    # 2. Test Evaluation (Correctness Score Migration)
    print("\n[2/2] Testing EvaluationSystem.log_evaluation...")
    eval_sys = EvaluationSystem(db_path="evaluations.db")
    eval_id = eval_sys.log_evaluation(
        query=query,
        response=response,
        latency_ms=latency,
        tokens=tokens,
        cache_hit=False,
        hallucination_score=0.1,
        relevance_score=0.9,
        correctness_score=0.95,
        metadata={"test": True}
    )
    print(f"Successfully saved evaluation entry ID: {eval_id}")
    
    # Verify the file exists
    if os.path.exists("career_bot_metrics.jsonl"):
        print("\n✅ Verification successful: career_bot_metrics.jsonl updated.")
    else:
        print("\n❌ Verification failed: career_bot_metrics.jsonl not found.")

if __name__ == "__main__":
    test_monitoring_and_eval()
