"""
2_rag_bot/tests/verify_factual_flow.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Verification script for the Factual RAG flow.
"""

import os
import sys

# Add the project directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.agent import CareerAgent

def test_factual_flow():
    print("=" * 60)
    print("Factual RAG Flow Verification")
    print("=" * 60)

    agent = CareerAgent()
    
    # Factual query
    query = "What technologies do you use?"
    print(f"\nRunning factual query: '{query}'")
    
    response = ""
    for chunk in agent.chat(query, []):
        response = chunk
        print(".", end="", flush=True)
        
    print(f"\n\nFinal Response Preview:\n{response[:200]}...")
    print("=" * 60)

if __name__ == "__main__":
    test_factual_flow()
