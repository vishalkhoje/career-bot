import os
import sys
from dotenv import load_dotenv

# Add the directory to path
sys.path.append(os.path.join(os.getcwd(), "2_rag_bot"))

app_rag = __import__("app-rag")
Me = app_rag.Me

def test_chat_response():
    load_dotenv(override=True)
    me = Me()
    
    # We must ensure we are NOT hitting the cache for this test
    # (Though we just cleared it, so it's fine)
    
    queries = [
        "Can you share your resume link?",
        "Can you share your resume link?", # Repeated to test cache
        "Can you share your resume link?", # Repeated again
    ]
    
    print("Testing Chat Responses for Resume Queries:\n")
    for q in queries:
        print(f"User: {q}")
        response = me.chat(q, [])
        print(f"Bot: {response}")
        if "drive.google.com" in response:
            print("✅ SUCCESS: Link found in response.")
        else:
            print("❌ FAILURE: Link missing in response.")
        print("-" * 30)

if __name__ == "__main__":
    test_chat_response()
