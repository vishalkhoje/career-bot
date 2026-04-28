import hashlib
import os
import sys
import time
from datetime import datetime
from typing import List, Set

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from pinecone import Pinecone, ServerlessSpec

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.cache import ResponseCache
from src import config

# Load env vars
load_dotenv(override=True)

class DataPipeline:
    """
    Production-grade data pipeline for the career bot.
    Handles validation, deduplication, versioning, and indexing.
    """
    
    def __init__(self, force_reindex: bool = False):
        self.force_reindex = force_reindex
        self.embeddings = OpenAIEmbeddings(
            model=config.OPENAI_EMBEDDING_MODEL,
            api_key=config.OPENAI_API_KEY
        )
        self.pc = Pinecone(api_key=config.PINECONE_API_KEY)
        
    def validate_source_files(self, file_paths: List[str]) -> List[str]:
        """
        Check if files exist and are not empty.
        """
        valid_files = []
        for path in file_paths:
            if not os.path.exists(path):
                print(f"[Pipeline] ⚠️ Warning: File not found: {path}")
                continue
            if os.path.getsize(path) == 0:
                print(f"[Pipeline] ⚠️ Warning: File is empty: {path}")
                continue
            valid_files.append(path)
        return valid_files

    def deduplicate_chunks(self, chunks: List[Document]) -> List[Document]:
        """
        Remove duplicate chunks based on content hash.
        """
        seen_hashes: Set[str] = set()
        unique_chunks = []
        
        for chunk in chunks:
            # Create a unique fingerprint for this content
            content_hash = hashlib.sha256(chunk.page_content.encode()).hexdigest()
            if content_hash not in seen_hashes:
                seen_hashes.add(content_hash)
                # Store hash in metadata for future verification
                chunk.metadata["content_hash"] = content_hash
                unique_chunks.append(chunk)
                
        print(f"[Pipeline] Deduplication: {len(chunks)} -> {len(unique_chunks)} unique chunks.")
        return unique_chunks

    def add_versioning_metadata(self, chunks: List[Document]):
        """
        Inject version and timestamp into every chunk.
        """
        timestamp = datetime.now().isoformat()
        for chunk in chunks:
            chunk.metadata.update({
                "embedding_version": config.EMBEDDING_VERSION,
                "ingestion_timestamp": timestamp
            })

    def run(self):
        """
        Execute the full pipeline.
        """
        print(f"\n{'='*50}\nCareer Bot Data Pipeline\n{'='*50}")
        
        # 1. Validation
        source_files = ["me/linkedin.pdf", "me/summary.txt"]
        valid_files = self.validate_source_files(source_files)
        if not valid_files:
            print("[Pipeline] ❌ Error: No valid source files found. Aborting.")
            return

        # 2. Loading
        all_docs = []
        for path in valid_files:
            print(f"[Pipeline] Loading {path}...")
            if path.endswith(".pdf"):
                loader = PyPDFLoader(path)
                pages = loader.load()
                
                # Clean and merge pages
                cleaned_pages = []
                import re
                for p in pages:
                    text = p.page_content
                    # Remove "Page X of Y" and similar footers/headers
                    text = re.sub(r"Page \d+ of \d+", "", text)
                    text = re.sub(r"^\s*\d+\s*$", "", text, flags=re.MULTILINE) # Remove isolated page numbers
                    cleaned_pages.append(text.strip())
                
                full_text = "\n\n".join(cleaned_pages)
                combined_doc = Document(page_content=full_text, metadata={"source": path})
                all_docs.append(combined_doc)
            else:
                loader = TextLoader(path)
                all_docs.extend(loader.load())

        # 3. Chunking
        # For resumes, we want to keep as much context as possible. 
        # Instead of small chunks, we use a very large chunk size (30k) 
        # which effectively keeps the entire resume together as one or two high-context chunks.
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=30000, 
            chunk_overlap=0, # No overlap needed if it's all in one chunk
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        raw_chunks = []
        for doc in all_docs:
            source_label = "resume" if "linkedin.pdf" in doc.metadata.get("source", "") else "summary"
            # Prepend global context
            global_prefix = f"Candidate: {config.BOT_NAME}\nProfile Context: Complete Career History and Project Details\n\n"
            
            doc_chunks = text_splitter.split_documents([doc])
            for i, chunk in enumerate(doc_chunks):
                chunk.page_content = global_prefix + chunk.page_content
                chunk.metadata.update({"source_type": source_label, "chunk_id": i})
                raw_chunks.append(chunk)

        # 4. Deduplication
        unique_chunks = self.deduplicate_chunks(raw_chunks)

        # 5. Versioning
        self.add_versioning_metadata(unique_chunks)

        # 6. Indexing (Pinecone)
        index_name = config.PINECONE_INDEX_NAME
        
        if self.force_reindex:
            print(f"[Pipeline] 🔄 FORCE RE-INDEX: Deleting index '{index_name}'...")
            if index_name in [idx.name for idx in self.pc.list_indexes()]:
                self.pc.delete_index(index_name)
                while index_name in [idx.name for idx in self.pc.list_indexes()]:
                    time.sleep(1)

        if index_name not in [idx.name for idx in self.pc.list_indexes()]:
            print(f"[Pipeline] Creating new index '{index_name}'...")
            self.pc.create_index(
                name=index_name,
                dimension=1536,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1")
            )
            while not self.pc.describe_index(index_name).status["ready"]:
                time.sleep(1)

        print(f"[Pipeline] Upserting {len(unique_chunks)} vectors to Pinecone...")
        PineconeVectorStore.from_documents(
            unique_chunks, 
            self.embeddings, 
            index_name=index_name,
            pinecone_api_key=config.PINECONE_API_KEY
        )

        # 7. Cache Sync
        print("[Pipeline] Clearing response cache to prevent stale answers...")
        ResponseCache(db_path=config.CACHE_DB_PATH).clear()
        
        print(f"{'='*50}\nPipeline Execution Complete!\n{'='*50}")

if __name__ == "__main__":
    # You can pass --force-reindex as a command line arg if needed
    force = "--force-reindex" in sys.argv
    pipeline = DataPipeline(force_reindex=force)
    pipeline.run()
