"""
scratch/verify_streaming.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~
Verification script for streaming responses in the refactored package.

Tests:
  1. CareerAgent.chat yields multiple chunks.
  2. The full response is eventually correct.
"""

import os
import sys
import time

# Add the project directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import CareerAgent, ResponseCache, config

QUERY = "Tell me about your career in 3 sentences."

def test_streaming() -> None:
    print("=" * 50)
    print("Streaming Verification")
    print("=" * 50)

    # Clear cache to force a fresh LLM call (streaming doesn't happen on cache hit, it's instant)
    ResponseCache(db_path=config.CACHE_DB_PATH).clear()

    agent = CareerAgent()

    print(f"\nQuery: {QUERY}")
    print("-" * 50)
    print("Bot (streaming): ", end="", flush=True)

    chunk_count = 0
    full_text = ""
    start_time = time.time()

    # The chat method is now a generator
    for chunk in agent.chat(QUERY, []):
        # In our implementation, 'chunk' is the full string so far (Gradio style)
        # To show the 'new' part, we compare with full_text
        new_part = chunk[len(full_text):]
        print(new_part, end="", flush=True)
        full_text = chunk
        chunk_count += 1
        time.sleep(0.01) # Small delay to visualize streaming if it's too fast

    duration = time.time() - start_time
    print(f"\n\nStreaming finished in {duration:.2f}s.")
    print(f"Total chunks yielded: {chunk_count}")

    if chunk_count > 1:
        print("✅ PASS: Multiple chunks yielded.")
    else:
        print("❌ FAIL: Only one chunk yielded (not streaming?).")

    if len(full_text) > 50:
        print("✅ PASS: Response content seems substantial.")
    else:
        print("❌ FAIL: Response too short.")

    print("=" * 50)

if __name__ == "__main__":
    test_streaming()
