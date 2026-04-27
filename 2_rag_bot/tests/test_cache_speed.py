import time
import os
import sys

# Add the project directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.agent import CareerAgent
Me = CareerAgent

def test_speed():
    me = Me()
    query = "What are your key skills?"
    
    print(f"Querying: '{query}'")
    
    # First run (No cache)
    start_time = time.time()
    response1 = me.chat(query, [])
    end_time = time.time()
    print(f"Run 1 Time: {end_time - start_time:.2f}s")
    
    # Second run (Should be cached if implemented)
    start_time = time.time()
    response2 = me.chat(query, [])
    end_time = time.time()
    print(f"Run 2 Time: {end_time - start_time:.2f}s")

if __name__ == "__main__":
    test_speed()
