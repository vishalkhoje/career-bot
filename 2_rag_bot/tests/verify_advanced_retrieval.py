"""
2_rag_bot/tests/verify_advanced_retrieval.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Verification script for Hybrid Search + Re-ranking.
"""

import os
import sys
import time

# Add the project directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.retriever import CareerRetriever
from src import config

def test_advanced_retrieval():
    print("=" * 60)
    print("Advanced Retrieval Verification (Hybrid + Re-ranking)")
    print("=" * 60)

    # Force enable advanced retrieval for this test
    config.USE_ADVANCED_RETRIEVAL = True
    
    start_init = time.time()
    retriever = CareerRetriever()
    print(f"Initialization took: {time.time() - start_init:.2f}s")

    query = "What projects did you work on at MathCo?"
    
    print(f"\nQuery: {query}")
    print("-" * 60)
    
    start_ret = time.time()
    context = retriever.retrieve(query)
    duration = time.time() - start_ret
    
    print(f"Retrieval took: {duration:.2f}s")
    print(f"\nContext Preview (first 500 chars):\n{context[:500]}...")
    
    if len(context) > 200:
        print("\n✅ PASS: Context retrieved successfully.")
    else:
        print("\n❌ FAIL: Context too short.")

    if hasattr(retriever, 'compression_retriever'):
        print("✅ PASS: Compression (Rerank) retriever was used.")
    else:
        print("❌ FAIL: Compression retriever NOT used.")

    print("=" * 60)

if __name__ == "__main__":
    test_advanced_retrieval()
