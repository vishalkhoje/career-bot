"""
2_rag_bot/tests/verify_observability.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Verification script for Production Observability.
"""

import os
import sys
import json
import time

# Add the project directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import CareerAgent

def test_observability():
    print("=" * 60)
    print("Production Observability Verification")
    print("=" * 60)

    # Clean up old metrics file if it exists
    metrics_file = "career_bot_metrics.jsonl"
    if os.path.exists(metrics_file):
        os.remove(metrics_file)

    agent = CareerAgent()
    
    # 1. Test a simple query (should hit LLM and generate metrics)
    query = f"Unique query at {time.time()}: Who are you?"
    print(f"\nRunning query: '{query}'")
    
    # Consume the generator
    response = ""
    for chunk in agent.chat(query, []):
        response += chunk
        
    # 2. Check if the metrics file was created
    if os.path.exists(metrics_file):
        print(f"✅ PASS: Metrics file '{metrics_file}' created.")
        
        # 3. Verify the content of the file
        with open(metrics_file, "r") as f:
            lines = f.readlines()
            if lines:
                last_event = json.loads(lines[-1])
                print(f"\nLast Event in Logs:\n{json.dumps(last_event, indent=2)}")
                
                if "latency_ms" in last_event and "token_usage" in last_event:
                    print("\n✅ PASS: Metrics contains latency and token usage.")
                else:
                    print("\n❌ FAIL: Metrics missing expected fields.")
            else:
                print("\n❌ FAIL: Metrics file is empty.")
    else:
        print(f"\n❌ FAIL: Metrics file '{metrics_file}' NOT created.")

    print("=" * 60)

if __name__ == "__main__":
    test_observability()
