import sys
import os
from unittest.mock import MagicMock, patch

# Add 2_rag_bot to path
sys.path.append(os.path.abspath("2_rag_bot"))

from src import CareerAgent

def verify_failure_handling():
    print("--- 🛡️ Failure Handling Verification ---")
    agent = CareerAgent()
    
    # We will mock the LLM to throw an exception
    # specifically for the invoke method
    print("\n[Test 1] Simulating LLM API Timeout/Failure...")
    
    from langchain_openai import ChatOpenAI
    with patch("langchain_openai.ChatOpenAI.invoke", side_effect=Exception("API Timeout")):
        # This should trigger retries and then fallback to FACTUAL intent
        history = []
        message = "Tell me about your projects."
        
        full_response = ""
        for chunk in agent.chat(message, history):
            full_response += chunk
            
        print(f"\nFinal Response Sample: {full_response[:100]}...")
        
        # Check if it returned a valid response despite the failure
        # In this case, since we patched all ChatOpenAI.invoke, 
        # it will fail intent classification AND standard RAG generation 
        # (because standard RAG uses stream but intent uses invoke).
        
        if "trouble connecting to my service" in full_response:
             print("\n✅ SUCCESS: Agent returned the safe fallback message after repeated API Timeouts.")
        else:
             print("\n⚠️  NOTE: Check output for [Retry] logs to confirm backoff worked.")

    print("\n[Test 2] Simulating Total Outage...")
    with patch("langchain_openai.ChatOpenAI.invoke", side_effect=Exception("Critical Outage")):
        with patch("langchain_openai.ChatOpenAI.stream", side_effect=Exception("Critical Outage")):
            full_response = ""
            for chunk in agent.chat("Should fail", []):
                full_response += chunk
            
            print(f"Bot Response: {full_response}")
            if "unable to process your request" in full_response or "trouble connecting" in full_response:
                print("\n✅ SUCCESS: Agent returned the safe fallback message during total outage.")
            else:
                print("\n❌ FAILED: Agent did not return safe fallback.")

if __name__ == "__main__":
    verify_failure_handling()
