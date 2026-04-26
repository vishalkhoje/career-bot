import os
from pinecone import Pinecone
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("PINECONE_API_KEY")
if not api_key:
    print("PINECONE_API_KEY not found")
else:
    pc = Pinecone(api_key=api_key)
    try:
        indices = pc.list_indexes()
        print(f"Indices: {[idx.name for idx in indices]}")
    except Exception as e:
        print(f"Error: {e}")
