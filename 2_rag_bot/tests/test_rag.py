import os
import sys
from dotenv import load_dotenv

# Add the project directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import CareerAgent

def test_rag_bot():
    print("Testing RAG Bot...")
    try:
        me = CareerAgent()
        # Consume the generator
        full_response = ""
        for chunk in me.chat("What are your key skills?", []):
            full_response = chunk
        
        print(f"Response: {full_response[:100]}...")
        if len(full_response) > 0:
            print("✅ RAG Bot logic is working!")
        else:
            print("❌ RAG Bot returned empty response.")
    except Exception as e:
        print(f"❌ RAG Bot failed with error: {e}")

if __name__ == "__main__":
    test_rag_bot()
