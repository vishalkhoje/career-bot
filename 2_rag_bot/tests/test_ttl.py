import os
import time
import sys

# Add the project directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import CareerAgent
Me = CareerAgent

def test_cache_expiration():
    print("Testing Cache Expiration (10 min TTL)...")
    me = Me()
    query = "What is your name?"
    
    # Update cache TTL for testing (default is 24h, we want 10m)
    me.cache.ttl_seconds = 600
    
    # 1. Warm up cache
    print("Step 1: Warming up cache...")
    list(me.chat(query, [])) # Iterate through generator
    
    if os.path.exists(me.cache.db_path):
        print("✅ Cache file created.")
    else:
        print("❌ Cache file NOT created.")
        return

    # 2. Fake the age of the cache file to 11 minutes (660 seconds)
    print("Step 2: Faking cache age to 11 minutes...")
    past_time = time.time() - 660
    os.utime(me.cache.db_path, (past_time, past_time))
    
    # 3. Call chat again
    print("Step 3: Querying again (Should trigger expiration)...")
    start_time = time.time()
    list(me.chat(query, []))
    duration = time.time() - start_time
    
    print(f"Response took: {duration:.2f}s")
    
    # 4. Verify
    if duration > 1.0: # If it's slow, it's a cache miss
        print("✅ Success: Cache expired and fresh response generated.")
    else:
        print("❌ Failure: Cache was still used despite being old.")

if __name__ == "__main__":
    test_cache_expiration()
