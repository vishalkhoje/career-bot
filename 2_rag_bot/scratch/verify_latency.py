"""
Verify latency reduction with FRESH (non-cached) queries.
Uses unique queries that won't be in the cache.
"""
import sys
import os
import time

sys.path.append(os.path.abspath("2_rag_bot"))

from src.agent import CareerAgent
from src.cache import ResponseCache
from src import config

def run_latency_test():
    print("=" * 60)
    print("⏱️  LATENCY VERIFICATION TEST (COLD START)")
    print("=" * 60)
    
    # Clear cache for these specific test queries
    agent = CareerAgent()
    
    # Unique queries that won't be cached
    test_queries = [
        ("FACTUAL", "What programming languages does Vishal know?"),
        ("FACTUAL", "Which company did Vishal work at most recently?"),
        ("ANALYTICAL", "Compare Vishal's backend skills against a typical Staff Engineer requirements."),
    ]
    
    # Clear cache entries for these queries
    for _, q in test_queries:
        cache_key = agent.cache.make_key(q)
        agent.cache.delete(cache_key)
    
    results = []
    
    for expected_type, query in test_queries:
        print(f"\n--- Testing: [{expected_type}] '{query}' ---")
        start = time.time()
        
        full_response = ""
        for chunk in agent.chat(query, []):
            full_response += chunk
        
        elapsed_ms = (time.time() - start) * 1000
        results.append({
            "type": expected_type,
            "query": query,
            "latency_ms": elapsed_ms,
            "response_length": len(full_response)
        })
        print(f"\n✅ Latency: {elapsed_ms:.0f}ms | Response: {len(full_response)} chars")

    # Summary
    print("\n" + "=" * 60)
    print("📊 LATENCY SUMMARY (COLD START)")
    print("=" * 60)
    avg_latency = sum(r["latency_ms"] for r in results) / len(results)
    for r in results:
        print(f"  [{r['type']}] {r['query'][:40]}... → {r['latency_ms']:.0f}ms")
    print(f"\n  Average Latency: {avg_latency:.0f}ms")
    print(f"  Previous Average: ~5356ms")
    
    if avg_latency < 5356:
        improvement = (5356 - avg_latency) / 5356 * 100
        print(f"  🚀 Improvement: {5356 - avg_latency:.0f}ms FASTER ({improvement:.1f}%)")
    else:
        print(f"  ⚠️  No improvement detected (current: {avg_latency:.0f}ms)")
    print("=" * 60)

if __name__ == "__main__":
    run_latency_test()
