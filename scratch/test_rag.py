import os
import sys
from dotenv import load_dotenv

# Add the directory to path so we can import app-rag
sys.path.append(os.path.join(os.getcwd(), "2_rag_bot"))

# We need to rename app-rag to app_rag or similar to import it
# but wait, I can just import from app-rag using __import__
app_rag = __import__("app-rag")
Me = app_rag.Me

def test_rag_bot():
    print("Testing RAG Bot...")
    try:
        me = Me()
        response = me.chat("What are your key skills?", [])
        print(f"Response: {response[:100]}...")
        if len(response) > 0:
            print("✅ RAG Bot logic is working!")
        else:
            print("❌ RAG Bot returned empty response.")
    except Exception as e:
        print(f"❌ RAG Bot failed with error: {e}")

if __name__ == "__main__":
    test_rag_bot()
