import os
import sys
from dotenv import load_dotenv

# Add the project directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import CareerAgent
Me = CareerAgent

def test_resume_retrieval():
    load_dotenv(override=True)
    me = Me()
    query = "Share your resume download link"
    
    print(f"Testing retrieval for: '{query}'")
    
    if me.vector_store:
        docs = me.vector_store.similarity_search(query, k=10)
        print("\n--- Retrieved Context ---")
        for i, doc in enumerate(docs):
            print(f"\nChunk {i+1}:")
            print(doc.page_content)
            if "drive.google.com" in doc.page_content:
                print("✅ Found resume link in this chunk!")
    else:
        print("❌ Vector store not initialized.")

if __name__ == "__main__":
    test_resume_retrieval()
