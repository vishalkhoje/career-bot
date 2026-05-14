import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.agents.career_agent import CareerAgent
from src.core import config

def run_test():
    db_path = config.CACHE_DB_PATH
    if os.path.exists(db_path):
        os.remove(db_path)
        print(f"Removed old cache db at {db_path}")

    print("Initializing Agent...")
    agent = CareerAgent()

    query1 = "What are your top skills?"
    print(f"\n--- Running Query 1: '{query1}' ---")
    start = time.time()
    resp1 = ""
    for chunk in agent.chat(query1, []):
        resp1 = chunk
    print(f"Time: {time.time() - start:.2f}s")
    print("Response sample:", resp1[:100].replace('\n', ' '))

    query2 = "Tell me about your main skills."
    print(f"\n--- Running Query 2 (Semantically similar): '{query2}' ---")
    start = time.time()
    resp2 = ""
    for chunk in agent.chat(query2, []):
        resp2 = chunk
    print(f"Time: {time.time() - start:.2f}s")
    print("Response sample:", resp2[:100].replace('\n', ' '))

    query3 = "Where did you go to school?"
    print(f"\n--- Running Query 3 (Different intent): '{query3}' ---")
    start = time.time()
    resp3 = ""
    for chunk in agent.chat(query3, []):
        resp3 = chunk
    print(f"Time: {time.time() - start:.2f}s")
    print("Response sample:", resp3[:100].replace('\n', ' '))

if __name__ == "__main__":
    run_test()
