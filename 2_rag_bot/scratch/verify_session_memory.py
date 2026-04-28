import sys
import os

# Add 2_rag_bot to path
sys.path.append(os.path.abspath("2_rag_bot"))

from src.agent import CareerAgent

def verify_memory():
    print("--- 🧠 Session Memory Verification ---")
    agent = CareerAgent()
    
    # Session state (simulated Gradio history)
    history = []
    
    # Turn 1: Introduce user
    print("\n[Turn 1] User: Hi, I'm Vishal, a recruiter from MathCo.")
    q1 = "Hi, I'm Vishal, a recruiter from MathCo."
    r1 = ""
    for chunk in agent.chat(q1, history):
        r1 += chunk
    print(f"Bot: {r1[:50]}...")
    
    # Update history for Turn 2
    history.append({"role": "user", "content": q1})
    history.append({"role": "assistant", "content": r1})
    
    # Turn 2: Test memory
    print("\n[Turn 2] User: Do you remember which company I'm from?")
    q2 = "Do you remember which company I'm from?"
    r2 = ""
    for chunk in agent.chat(q2, history):
        r2 += chunk
    print(f"Bot: {r2}")
    
    # Verify result
    if "MathCo" in r2:
        print("\n✅ SUCCESS: Bot remembered 'MathCo' from the session history!")
    else:
        print("\n❌ FAILED: Bot forgot the company name.")

if __name__ == "__main__":
    verify_memory()
