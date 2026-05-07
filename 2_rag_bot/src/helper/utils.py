import time
import random
from typing import Callable, Any
import functools

def retry_with_backoff(retries: int = 3, backoff_in_seconds: int = 1):
    """
    Decorator for retrying a function with exponential backoff.
    """
    def decorator(func: Callable):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            x = 0
            while True:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if x == retries:
                        print(f"[Retry] ❌ Maximum retries reached for {func.__name__}. Error: {e}")
                        raise
                    
                    sleep = (backoff_in_seconds * 2 ** x + 
                             random.uniform(0, 1))
                    print(f"[Retry] ⚠️ Attempt {x+1} failed for {func.__name__}. Retrying in {sleep:.2f}s... (Error: {e})")
                    time.sleep(sleep)
                    x += 1
        return wrapper
    return decorator

def safe_llm_call(llm_func, *args, fallback_msg="I'm sorry, I'm having trouble processing that right now.", **kwargs):
    """
    Executes an LLM call with a safety net.
    """
    try:
        return llm_func(*args, **kwargs)
    except Exception as e:
        print(f"[Fallback] 🚨 LLM Call failed: {e}. Returning safe response.")
        return type('obj', (object,), {'content': fallback_msg})()
