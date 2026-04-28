import os
from typing import List

from langchain.retrievers import ContextualCompressionRetriever, EnsembleRetriever
from langchain.retrievers.document_compressors import FlashrankRerank
from langchain_community.retrievers import BM25Retriever
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_core.documents import Document

from . import config


class CareerRetriever:
    """
    Advanced RAG Retriever for the career bot.
    
    Implements:
      1. Hybrid Search: Combines Dense (OpenAI) and Sparse (BM25) retrieval.
      2. Re-ranking: Uses FlashRank (local) to optimize the top-k results.
      3. Metadata Awareness: Ensures deterministic ordering and supports filtering.
    """

    def __init__(self) -> None:
        """
        Initialise the retrieval components.
        """
        if not config.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY is not set.")
        if not config.PINECONE_API_KEY:
            raise ValueError("PINECONE_API_KEY is not set.")

        # 1. Base Dense Retriever (Pinecone)
        embeddings = OpenAIEmbeddings(
            model=config.OPENAI_EMBEDDING_MODEL,
            api_key=config.OPENAI_API_KEY,
        )

        self.vector_store = PineconeVectorStore(
            index_name=config.PINECONE_INDEX_NAME,
            embedding=embeddings,
            pinecone_api_key=config.PINECONE_API_KEY,
        )
        
        # 2. Setup Advanced Components if enabled
        self.ensemble_retriever = None
        if config.USE_ADVANCED_RETRIEVAL:
            self._setup_advanced_pipeline()

    def _setup_advanced_pipeline(self):
        """
        Bootstrap the hybrid + re-ranker pipeline.
        Note: BM25 requires all documents in memory for local indexing.
        """
        try:
            print("[Retriever] Initializing Advanced Pipeline (Hybrid + Rerank)...")
            
            # For a career bot, the dataset is small, so we can fetch all docs 
            # to build a local BM25 index for hybrid search.
            # In a massive production app, you'd use Pinecone's native hybrid features.
            
            # Fetch some "representative" chunks to seed the BM25 retriever
            # (In this project, we'll just fetch the top 100 to ensure we cover the whole profile)
            all_docs = self.vector_store.similarity_search("career profile overview", k=100)
            
            if not all_docs:
                print("[Retriever] Warning: No documents found to seed BM25. Falling back to vector search.")
                return

            bm25_retriever = BM25Retriever.from_documents(all_docs)
            bm25_retriever.k = config.RETRIEVAL_K

            # Combine Vector (0.7 weight) and BM25 (0.3 weight)
            self.ensemble_retriever = EnsembleRetriever(
                retrievers=[self.vector_store.as_retriever(search_kwargs={"k": config.RETRIEVAL_K}), 
                           bm25_retriever],
                weights=[0.7, 0.3]
            )
            
            # 3. Add Re-ranker (FlashRank)
            # This will take the ensemble results and re-order them using a Cross-Encoder
            compressor = FlashrankRerank(top_n=config.RERANK_TOP_K)
            self.compression_retriever = ContextualCompressionRetriever(
                base_compressor=compressor, 
                base_retriever=self.ensemble_retriever
            )
            print("[Retriever] Advanced Pipeline Ready.")
            
        except Exception as exc:
            print(f"[Retriever] Failed to setup advanced pipeline: {exc}. Falling back to basic search.")
            self.ensemble_retriever = None

    def retrieve(self, query: str) -> str:
        """
        Retrieve context using the best available pipeline.
        """
        try:
            if config.USE_ADVANCED_RETRIEVAL and hasattr(self, 'compression_retriever'):
                docs = self.compression_retriever.get_relevant_documents(query)
            elif self.ensemble_retriever:
                docs = self.ensemble_retriever.get_relevant_documents(query)
            else:
                docs = self.vector_store.similarity_search(query, k=config.RETRIEVAL_K)

            # Deterministic sorting (CRITICAL for cache hit consistency)
            docs.sort(key=lambda d: d.page_content)
            
            context = "\n\n".join(doc.page_content for doc in docs)
            return context
            
        except Exception as exc:
            print(f"[Retriever] Retrieval error: {exc}")
            return "Context retrieval failed. Please try again."
