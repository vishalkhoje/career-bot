import sys
import os
import sqlite3
import json
import time

# Add 2_rag_bot to path
sys.path.append(os.path.abspath("2_rag_bot"))

from src import EvaluationSystem
from src import CareerAgent

def test_feedback_loop():
    print("--- 🔄 Feedback Loop Verification Test ---")
    
    eval_sys = EvaluationSystem(db_path="evaluations.db")
    agent = CareerAgent()
    
    query = "What is Vishal's favorite color?"
    bad_response = "Vishal's favorite color is Red." # Hallucination (not in context)
    context = "Vishal is a Senior AI Engineer. He likes coding in Python."
    
    # 1. Log a bad evaluation
    print("\n[1/4] Logging a simulated bad response...")
    eval_id = eval_sys.log_evaluation(
        query=query,
        response=bad_response,
        latency_ms=100.0,
        tokens=50,
        cache_hit=False,
        metadata={"context_preview": context}
    )
    
    # 2. Simulate 👎 Feedback
    print(f"\n[2/4] Simulating 👎 feedback for ID {eval_id}...")
    eval_sys.update_feedback(eval_id, "down")
    
    # 3. Run Analysis
    print("\n[3/4] Running analyze_and_improve...")
    eval_sys.analyze_and_improve(eval_id)
    
    # Check if gold_response was created
    with sqlite3.connect("evaluations.db") as conn:
        row = conn.execute("SELECT gold_response, failure_reason FROM evaluations WHERE id = ?", (eval_id,)).fetchone()
        if row and row[0]:
            print(f"✅ Analysis Successful! Reason: {row[1]}")
            print(f"   Gold Response: {row[0][:50]}...")
        else:
            print("❌ Analysis Failed: No gold response found.")
            return

    # 4. Verify Learning in Agent
    print("\n[4/4] Verifying if Agent learns from this correction...")
    # We'll just check if the learner pulls it
    learned = agent.learner.get_relevant_corrections(query)
    if learned and query in learned:
        print("✅ Learning Verified: Corrected example found in Learner!")
        print(f"--- Learned Section Preview ---\n{learned}\n-------------------------------")
    else:
        print("❌ Learning Failed: Corrected example not found or not relevant.")

if __name__ == "__main__":
    test_feedback_loop()
