import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec
import time

load_dotenv(override=True)

# Configuration
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "career-bot")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

def ingest_data():
    if not OPENAI_API_KEY:
        print("Error: OPENAI_API_KEY missing in .env")
        return

    if not PINECONE_API_KEY:
        print("Error: PINECONE_API_KEY missing in .env")
        return

    print("Starting ingestion process...")
    
    # Clear the LLM response cache to avoid stale answers
    current_dir = os.path.dirname(os.path.abspath(__file__))
    cache_db = os.path.join(current_dir, ".langchain.db")
    if os.path.exists(cache_db):
        print(f"Clearing old cache: {cache_db}...")
        try:
            os.remove(cache_db)
        except Exception as e:
            print(f"Warning: Could not clear cache: {e}")

    # 1. Load Documents
    documents = []
    
    pdf_path = "me/linkedin.pdf"
    if os.path.exists(pdf_path):
        print(f"Loading {pdf_path}...")
        try:
            loader = PyPDFLoader(pdf_path)
            documents.extend(loader.load())
        except Exception as e:
            print(f"Error loading PDF: {e}")
    
    summary_path = "me/summary.txt"
    if os.path.exists(summary_path):
        print(f"Loading {summary_path}...")
        try:
            loader = TextLoader(summary_path)
            documents.extend(loader.load())
        except Exception as e:
            print(f"Error loading summary: {e}")
        
    if not documents:
        print("No documents found to ingest.")
        return

    # 2. Split Text
    print("Splitting documents into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = text_splitter.split_documents(documents)
    print(f"Created {len(docs)} chunks.")

    # 3. Initialize Embeddings
    print("Initializing OpenAI Embeddings...")
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        api_key=OPENAI_API_KEY,
    )

    # 4. Initialize Pinecone
    print("Connecting to Pinecone...")
    try:
        pc = Pinecone(api_key=PINECONE_API_KEY)
        
        existing_indices = [idx.name for idx in pc.list_indexes()]
        if INDEX_NAME not in existing_indices:
            print(f"Creating index {INDEX_NAME}...")
            pc.create_index(
                name=INDEX_NAME,
                dimension=1536, # Standard for text-embedding-3-small
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1")
            )
            # Wait for index to be ready
            while not pc.describe_index(INDEX_NAME).status['ready']:
                time.sleep(1)
        
        # 5. Store in Pinecone
        print(f"Upserting vectors to index '{INDEX_NAME}'...")
        vector_store = PineconeVectorStore.from_documents(
            docs, 
            embeddings, 
            index_name=INDEX_NAME,
            pinecone_api_key=PINECONE_API_KEY
        )
        print("Ingestion complete!")
    except Exception as e:
        print(f"Error during Pinecone operations: {e}")

if __name__ == "__main__":
    ingest_data()
