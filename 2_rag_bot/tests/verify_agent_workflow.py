"""
2_rag_bot/tests/verify_agent_workflow.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Verification script for Multi-Agent Workflow (Intent -> Planner -> Critic).
"""

import os
import sys
import time

# Add the project directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.agent import CareerAgent
from src import config

def test_agent_workflow():
    print("=" * 60)
    print("Multi-Agent Workflow Verification")
    print("=" * 60)

    # Force enable agent workflow for this test
    config.USE_AGENT_WORKFLOW = True
    config.USE_CRITIC_AGENT = True
    
    agent = CareerAgent()
    
    # Analytical query
    query = f"Unique query at {time.time()}: Based on my experience, am I suitable for a backend architect role?"
    print(f"\nRunning analytical query: '{query}'")
    
    start_time = time.time()
    response = ""
    for chunk in agent.chat(query, []):
        response = chunk # Generator yields full string so far
        # Print a dot for each chunk to show progress
        print(".", end="", flush=True)
        
    duration = time.time() - start_time
    print(f"\n\nTotal Time: {duration:.2f}s")
    print(f"\nFinal Response Preview:\n{response[:500]}...")
    
    if "Analyzing" in response:
        print("\n✅ PASS: Intent Classifier correctly triggered analytical flow.")
    else:
        print("\n❌ FAIL: Analytical flow NOT triggered.")

    if len(response) > 300:
        print("\n✅ PASS: Agent provided a detailed reasoned answer.")
    else:
        print("\n❌ FAIL: Response too short.")

    print("=" * 60)

if __name__ == "__main__":
    test_agent_workflow()
