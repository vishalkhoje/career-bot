import sys
import os
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone

# Add 2_rag_bot to path
sys.path.append(os.path.abspath("2_rag_bot"))
from src import config

def inspect_chunks():
    print("--- Inspecting Chunks in Pinecone ---")
    embeddings = OpenAIEmbeddings(
        model=config.OPENAI_EMBEDDING_MODEL,
        api_key=config.OPENAI_API_KEY
    )
    vector_store = PineconeVectorStore(
        index_name=config.PINECONE_INDEX_NAME,
        embedding=embeddings,
        pinecone_api_key=config.PINECONE_API_KEY
    )
    
    # Search for HRMS
    docs = vector_store.similarity_search("HRMS", k=5)
    
    for i, doc in enumerate(docs):
        print(f"\n--- CHUNK {i+1} ---")
        print(doc.page_content)
        print("-" * 20)

if __name__ == "__main__":
    inspect_chunks()
