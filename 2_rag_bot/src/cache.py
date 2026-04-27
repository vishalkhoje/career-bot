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

import hashlib
import os
import sqlite3
import time


class ResponseCache:
    """
    A lightweight, persistent cache backed by a local SQLite database.

    Usage::

        cache = ResponseCache(db_path="/path/to/.langchain.db", ttl_seconds=86400)
        key = cache.make_key("What are your top skills?")
        hit = cache.get(key)
        if hit is None:
            answer = call_llm(...)
            cache.set(key, answer)
    """

    _TABLE_DDL = """
        CREATE TABLE IF NOT EXISTS responses (
            query_hash   TEXT PRIMARY KEY,
            response_text TEXT NOT NULL,
            created_at   REAL  NOT NULL
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

    # ── Public API ─────────────────────────────────────────────────────────────

    @staticmethod
    def make_key(message: str) -> str:
        """
        Generate a deterministic cache key from a user message.

        The message is lower-cased and stripped before hashing so that
        minor variations in whitespace or capitalisation hit the same
        cache entry.

        Args:
            message: The raw user input string.

        Returns:
            A 64-character hex SHA-256 digest.
        """
        normalised = message.lower().strip()
        return hashlib.sha256(normalised.encode()).hexdigest()

    def get(self, key: str) -> str | None:
        """
        Retrieve a cached response by key.

        Args:
            key: The cache key produced by :meth:`make_key`.

        Returns:
            The cached response string, or ``None`` if not found.
        """
        with self._connect() as conn:
            row = conn.execute(
                "SELECT response_text FROM responses WHERE query_hash = ?", (key,)
            ).fetchone()
        return row[0] if row else None

    def set(self, key: str, value: str) -> None:
        """
        Store a response in the cache.

        Args:
            key:   The cache key produced by :meth:`make_key`.
            value: The LLM response string to cache.
        """
        with self._connect() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO responses "
                "(query_hash, response_text, created_at) VALUES (?, ?, ?)",
                (key, value, time.time()),
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
