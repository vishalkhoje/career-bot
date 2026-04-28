import sys
import os
from datetime import datetime

# Add 2_rag_bot to path
sys.path.append(os.path.abspath("2_rag_bot"))

from src.monitoring import Observability

def trigger_test_alerts():
    print("--- 🚨 Alert System Verification Test ---")
    
    # Example 1: Triggering a SLOW RESPONSE alert
    print("\n[Case 1] Simulating a Slow Response (>15s)...")
    Observability.log_request(
        query="Why is this so slow?",
        latency_ms=18500.0,  # Above 15000ms threshold
        tokens=1000,
        status="success",
        steps=["Cache Miss", "Complex Reasoning", "Tool Call", "LLM Generation"],
        hallucination_score=0.05
    )
    
    # Example 2: Triggering a HIGH HALLUCINATION alert
    print("\n[Case 2] Simulating a High Hallucination Risk (>0.7)...")
    Observability.log_request(
        query="Tell me something you don't know.",
        latency_ms=1200.0,
        tokens=200,
        status="success",
        steps=["Cache Miss", "LLM Generation"],
        hallucination_score=0.85  # Above 0.7 threshold
    )
    
    # Example 3: Triggering BOTH alerts
    print("\n[Case 3] Simulating both Slow Response and High Hallucination...")
    Observability.log_request(
        query="Super complex hallucinated query.",
        latency_ms=25000.0,
        tokens=500,
        status="success",
        steps=["Chain of Thought"],
        hallucination_score=0.95
    )

if __name__ == "__main__":
    trigger_test_alerts()
