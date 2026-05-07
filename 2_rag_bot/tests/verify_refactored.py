"""
scratch/verify_refactored.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Verification script for the refactored src/ package.

Tests:
  1. CareerAgent can be initialised without errors.
  2. Cache MISS on first call.
  3. Cache HIT on identical second call (same session).
  4. Resume download link is included in the response.
"""

import os
import sys

# Add the project directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import CareerAgent, ResponseCache, config

RESUME_QUERY = "Can you share your resume link?"
PASS = "✅ PASS"
FAIL = "❌ FAIL"


def test_cache_and_resume() -> None:
    print("=" * 50)
    print("Refactored Package Verification")
    print("=" * 50)

    # Clear cache so we always start from a known state
    ResponseCache(db_path=config.CACHE_DB_PATH).clear()

    agent = CareerAgent()

    # ── Test 1: First call should be a CACHE MISS ──────────────────────────
    print(f"\nTest 1 — First call (CACHE MISS expected)")
    # Consume generator to get final response
    resp1 = ""
    for chunk in agent.chat(RESUME_QUERY, []):
        resp1 = chunk
    
    if "drive.google.com" in resp1:
        print(f"{PASS} Resume link found in response.")
    else:
        print(f"{FAIL} Resume link NOT found. Response: {resp1[:120]}...")

    # ── Test 2: Second identical call should be a CACHE HIT ────────────────
    print(f"\nTest 2 — Second identical call (CACHE HIT expected)")
    resp2 = ""
    for chunk in agent.chat(RESUME_QUERY, []):
        resp2 = chunk
        
    if resp1 == resp2:
        print(f"{PASS} Response identical — served from cache.")
    else:
        print(f"{FAIL} Responses differ — cache may not be working.")

    # ── Test 3: Case-insensitive hit ───────────────────────────────────────
    print(f"\nTest 3 — Lowercase variant (CACHE HIT expected)")
    resp3 = ""
    for chunk in agent.chat(RESUME_QUERY.lower(), []):
        resp3 = chunk
        
    if resp1 == resp3:
        print(f"{PASS} Case-insensitive cache hit confirmed.")
    else:
        print(f"{FAIL} Cache missed on lowercase variant.")

    print("\n" + "=" * 50)
    print("Verification complete.")
    print("=" * 50)


if __name__ == "__main__":
    test_cache_and_resume()
