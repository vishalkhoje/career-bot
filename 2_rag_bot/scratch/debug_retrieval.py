import sys
import os
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

# Add 2_rag_bot to path
sys.path.append(os.path.abspath("2_rag_bot"))
from src import config

def debug_retrieval():
    print("--- Debugging Retrieval for HRMS Query ---")
    embeddings = OpenAIEmbeddings(
        model=config.OPENAI_EMBEDDING_MODEL,
        api_key=config.OPENAI_API_KEY
    )
    vector_store = PineconeVectorStore(
        index_name=config.PINECONE_INDEX_NAME,
        embedding=embeddings,
        pinecone_api_key=config.PINECONE_API_KEY
    )
    
    query = "in your project details vishal share more information about HRMS & Payroll SaaS details with organisation and duration"
    
    # 1. Basic Vector Search
    print("\n[1] Basic Vector Search (k=10):")
    docs = vector_store.similarity_search(query, k=10)
    for i, doc in enumerate(docs):
        print(f"\nCHUNK {i+1} (Source: {doc.metadata.get('chunk_id')}):")
        print(doc.page_content[:300] + "...")
        if "HRMS" in doc.page_content:
            print(">> FOUND HRMS IN THIS CHUNK <<")
        if "sumHR" in doc.page_content:
            print(">> FOUND sumHR IN THIS CHUNK <<")

if __name__ == "__main__":
    debug_retrieval()
