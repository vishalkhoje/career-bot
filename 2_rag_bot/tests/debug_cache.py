import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.globals import set_llm_cache
from langchain_community.cache import SQLiteCache
from langchain_community.callbacks.manager import get_openai_callback
from langchain_core.messages import HumanMessage

load_dotenv(override=True)

# 1. Setup Cache
cache_path = ".debug_cache.db"
if os.path.exists(cache_path):
    os.remove(cache_path)
set_llm_cache(SQLiteCache(database_path=cache_path))

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, cache=True)

def run_test():
    msg = [HumanMessage(content="Hello, say 'Hi'")]
    
    print("\n--- First Call ---")
    with get_openai_callback() as cb:
        res = llm.invoke(msg)
        print(f"Result: {res.content}")
        print(f"Tokens: {cb.total_tokens}")
        print(f"Cost: ${cb.total_cost}")

    print("\n--- Second Call (Should be cached) ---")
    with get_openai_callback() as cb:
        res = llm.invoke(msg)
        print(f"Result: {res.content}")
        print(f"Tokens: {cb.total_tokens}")
        print(f"Cost: ${cb.total_cost}")

if __name__ == "__main__":
    run_test()
