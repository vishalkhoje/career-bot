"""
src/retriever.py
~~~~~~~~~~~~~~~~
Pinecone vector-store retrieval wrapper.

Responsibility: given a user query, retrieve the most relevant chunks
from the Pinecone index and return them as a single concatenated string
that the prompt builder can embed directly into the system prompt.

Sorting retrieved documents by content before joining is critical for
cache stability — it ensures the same query always produces the same
context string regardless of Pinecone's internal ordering.
"""

from __future__ import annotations

from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

from . import config


class CareerRetriever:
    """
    Wraps a Pinecone vector store and exposes a single ``retrieve`` method.

    Attributes:
        vector_store: The underlying LangChain ``PineconeVectorStore`` instance.
    """

    def __init__(self) -> None:
        """
        Initialise the embedding model and connect to the Pinecone index.

        Raises:
            ValueError: If required environment variables are missing.
        """
        if not config.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY is not set.")
        if not config.PINECONE_API_KEY:
            raise ValueError("PINECONE_API_KEY is not set.")

        embeddings = OpenAIEmbeddings(
            model=config.OPENAI_EMBEDDING_MODEL,
            api_key=config.OPENAI_API_KEY,
        )

        self.vector_store = PineconeVectorStore(
            index_name=config.PINECONE_INDEX_NAME,
            embedding=embeddings,
            pinecone_api_key=config.PINECONE_API_KEY,
        )

    def retrieve(self, query: str, k: int = config.RETRIEVAL_K) -> str:
        """
        Retrieve the top-k relevant document chunks for a query.

        Documents are sorted alphabetically by content before joining.
        This deterministic ordering is essential for cache key stability —
        the same query must always produce the exact same context string.

        Args:
            query: The user's question or message.
            k:     Number of chunks to retrieve (default from config).

        Returns:
            A single string containing all retrieved chunks separated by
            double newlines.  Returns a safe fallback string on error.
        """
        try:
            docs = self.vector_store.similarity_search(query, k=k)
            # Sort for deterministic ordering (cache stability)
            docs.sort(key=lambda d: d.page_content)
            return "\n\n".join(doc.page_content for doc in docs)
        except Exception as exc:  # pylint: disable=broad-except
            print(f"[Retriever] Error during similarity search: {exc}")
            return "Context retrieval failed — please try again."
