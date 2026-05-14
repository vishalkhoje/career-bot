"""
src/cache.py
~~~~~~~~~~~~
Persistent SQLite-based response cache for the career bot.

Design decisions:
  - Cache key = SHA-256 of the normalized (lowercased + stripped) user message.
    History is intentionally excluded so the same question always returns
    the cached answer regardless of how long the conversation has been.
  - The cache file is deleted (and re-created) when TTL expires or when a
    fresh ingestion is run, ensuring answers never go stale.
  - All database operations use a context manager to guarantee connection
    closure even on exceptions.
"""

from __future__ import annotations

import json
import os
import sqlite3
import time

import numpy as np
from langchain_openai import OpenAIEmbeddings

from ..core import config


class ResponseCache:
    """
    A lightweight, persistent cache backed by a local SQLite database using Semantic Caching.

    Usage::

        cache = ResponseCache(db_path="/path/to/.langchain.db", ttl_seconds=86400)
        hit, query_emb = cache.get_semantic("What are your top skills?")
        if hit is None:
            answer = call_llm(...)
            cache.set_semantic("What are your top skills?", answer, query_emb)
    """

    _TABLE_DDL = """
        CREATE TABLE IF NOT EXISTS responses (
            query_text    TEXT PRIMARY KEY,
            embedding     TEXT NOT NULL,
            response_text TEXT NOT NULL,
            created_at    REAL  NOT NULL
        )
    """

    def __init__(self, db_path: str, ttl_seconds: int = 86_400) -> None:
        """
        Initialise the cache and create the SQLite table if it does not exist.

        Args:
            db_path:     Absolute path to the SQLite database file.
            ttl_seconds: Time-to-live in seconds (default: 24 hours).
        """
        self.db_path = db_path
        self.ttl_seconds = ttl_seconds
        self._init_db()
        self.embeddings = OpenAIEmbeddings(
            model=config.OPENAI_EMBEDDING_MODEL,
            api_key=config.OPENAI_API_KEY,
        )

    # ── Public API ─────────────────────────────────────────────────────────────

    def get_semantic(self, query: str, threshold: float = 0.70) -> tuple[str | None, list[float]]:
        """
        Retrieve a cached response using semantic similarity.
        
        Args:
            query:     The user's raw input string.
            threshold: Cosine similarity threshold (0 to 1).
            
        Returns:
            A tuple of (cached_response, query_embedding).
            cached_response is None if no match > threshold is found.
        """
        query_emb = self.embeddings.embed_query(query)
        query_emb_np = np.array(query_emb, dtype=np.float32)

        with self._connect() as conn:
            rows = conn.execute(
                "SELECT query_text, embedding, response_text FROM responses"
            ).fetchall()

        if not rows:
            return None, query_emb

        cached_embs = []
        cached_responses = []

        for row in rows:
            emb_list = json.loads(row["embedding"])
            cached_embs.append(emb_list)
            cached_responses.append(row["response_text"])

        cached_embs_np = np.array(cached_embs, dtype=np.float32)

        # Compute cosine similarities.
        similarities = np.dot(cached_embs_np, query_emb_np)

        best_idx = int(np.argmax(similarities))
        best_score = float(similarities[best_idx])

        print(f"[Cache] Semantic Check: Best Score={best_score:.4f} (Threshold={threshold}) for query '{query}'")

        if best_score >= threshold:
            print(f"[Cache] ⚡ Semantic HIT — Score: {best_score:.4f} >= {threshold} for query '{query}'")
            return cached_responses[best_idx], query_emb
        
        return None, query_emb

    def set_semantic(self, query: str, response: str, query_emb: list[float]) -> None:
        """
        Store a response in the cache with its semantic embedding.

        Args:
            query:     The user's input string.
            response:  The LLM response string to cache.
            query_emb: The embedding vector of the query.
        """
        with self._connect() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO responses "
                "(query_text, embedding, response_text, created_at) VALUES (?, ?, ?, ?)",
                (query, json.dumps(query_emb), response, time.time()),
            )

    def check_expiry(self) -> None:
        """
        Delete the cache database if it is older than the configured TTL.

        Called at the start of every chat turn.  Because we delete the
        entire file, the next call to :meth:`get` will always be a miss,
        forcing fresh LLM responses until the cache warms up again.
        """
        if not os.path.exists(self.db_path):
            return

        age = time.time() - os.path.getmtime(self.db_path)
        if age > self.ttl_seconds:
            print(
                f"[Cache] TTL exceeded ({age:.0f}s > {self.ttl_seconds}s). "
                "Clearing stale cache..."
            )
            self.clear()

    def clear(self) -> None:
        """
        Delete the cache database file.

        Called automatically by :mod:`ingest` after re-indexing documents
        so that stale answers are never served.
        """
        if os.path.exists(self.db_path):
            try:
                os.remove(self.db_path)
                print(f"[Cache] Cleared: {self.db_path}")
            except OSError as exc:
                print(f"[Cache] Warning — could not delete cache file: {exc}")
        # Re-create the empty schema
        self._init_db()

    def delete(self, query: str) -> None:
        """
        Delete a specific cache entry by query text.
        """
        with self._connect() as conn:
            conn.execute("DELETE FROM responses WHERE query_text = ?", (query,))

    # ── Private helpers ────────────────────────────────────────────────────────

    def _init_db(self) -> None:
        """Create the SQLite table if it does not already exist."""
        with self._connect() as conn:
            conn.execute(self._TABLE_DDL)

    def _connect(self) -> sqlite3.Connection:
        """Open and return a SQLite connection with row-factory set."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn
