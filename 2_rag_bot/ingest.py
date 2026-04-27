"""
ingest.py
~~~~~~~~~
Document ingestion pipeline for the RAG career chatbot.

This script loads career documents from the `me/` directory, splits them
into chunks, embeds them with OpenAI, and upserts them into Pinecone.

It also clears the response cache so the bot generates fresh answers based
on the updated knowledge base the next time it is queried.

Usage:
    python3 ingest.py

Run this script every time you update a file in the `me/` directory.
"""

import os
import sys
import time

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pinecone import Pinecone, ServerlessSpec

# Import the ResponseCache so we can clear it via its official API
# (avoids duplicating the db-path logic in two places)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.cache import ResponseCache
from src import config

# ── Configuration ──────────────────────────────────────────────────────────────
# Re-load env explicitly so this script works when run standalone
load_dotenv(override=True)

OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
PINECONE_API_KEY: str = os.getenv("PINECONE_API_KEY", "")
INDEX_NAME: str = os.getenv("PINECONE_INDEX_NAME", "career-bot")

# Chunking parameters tuned for the current dataset (~10k chars total):
#   chunk_size=1000 keeps paragraphs semantically whole.
#   chunk_overlap=200 prevents information loss at chunk boundaries.
CHUNK_SIZE: int = 1000
CHUNK_OVERLAP: int = 200


def ingest_data() -> None:
    """
    Run the full ingestion pipeline:

    1. Validate environment variables.
    2. Clear the SQLite response cache to avoid stale answers.
    3. Load PDF and text documents from the `me/` directory.
    4. Split documents into overlapping chunks.
    5. Generate embeddings via OpenAI.
    6. Create the Pinecone index (if not already existing) and upsert vectors.
    """
    # ── 0. Validate required environment variables ────────────────────────────
    if not OPENAI_API_KEY:
        print("Error: OPENAI_API_KEY missing in .env")
        return

    if not PINECONE_API_KEY:
        print("Error: PINECONE_API_KEY missing in .env")
        return

    print("Starting ingestion process...")

    # ── 1. Clear the response cache ───────────────────────────────────────────
    # Use ResponseCache.clear() so the db-path is always in sync with the app.
    cache = ResponseCache(db_path=config.CACHE_DB_PATH)
    cache.clear()

    # ── 2. Load documents ─────────────────────────────────────────────────────
    documents = []

    pdf_path = "me/linkedin.pdf"
    if os.path.exists(pdf_path):
        print(f"Loading {pdf_path}...")
        try:
            documents.extend(PyPDFLoader(pdf_path).load())
        except Exception as exc:
            print(f"Error loading PDF: {exc}")

    summary_path = "me/summary.txt"
    if os.path.exists(summary_path):
        print(f"Loading {summary_path}...")
        try:
            documents.extend(TextLoader(summary_path).load())
        except Exception as exc:
            print(f"Error loading summary: {exc}")

    if not documents:
        print("No documents found to ingest. Exiting.")
        return

    # ── 3. Split text into chunks ─────────────────────────────────────────────
    print("Splitting documents into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    docs = text_splitter.split_documents(documents)
    print(f"Created {len(docs)} chunks.")

    # ── 4. Initialise embeddings ──────────────────────────────────────────────
    print("Initializing OpenAI Embeddings...")
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        api_key=OPENAI_API_KEY,
    )

    # ── 5. Initialise Pinecone and upsert ─────────────────────────────────────
    print("Connecting to Pinecone...")
    try:
        pc = Pinecone(api_key=PINECONE_API_KEY)

        existing_indices = [idx.name for idx in pc.list_indexes()]
        if INDEX_NAME not in existing_indices:
            print(f"Creating index '{INDEX_NAME}'...")
            pc.create_index(
                name=INDEX_NAME,
                dimension=1536,  # Dimensionality of text-embedding-3-small
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            )
            # Wait until the index is ready before upserting
            while not pc.describe_index(INDEX_NAME).status["ready"]:
                time.sleep(1)

        print(f"Upserting vectors to index '{INDEX_NAME}'...")
        PineconeVectorStore.from_documents(
            docs,
            embeddings,
            index_name=INDEX_NAME,
            pinecone_api_key=PINECONE_API_KEY,
        )
        print("Ingestion complete!")

    except Exception as exc:
        print(f"Error during Pinecone operations: {exc}")


if __name__ == "__main__":
    ingest_data()
