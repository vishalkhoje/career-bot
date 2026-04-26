import time
import os
import sys

# Add the directory to path so we can import app-rag
sys.path.append(os.path.join(os.getcwd(), "2_rag_bot"))

app_rag = __import__("app-rag")
Me = app_rag.Me

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
