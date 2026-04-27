"""
src/config.py
~~~~~~~~~~~~~
Centralised configuration and environment variable loading.

All constants used across the package are defined here.
Import from this module — never call os.getenv() directly
in business-logic files.
"""

import os
from dotenv import load_dotenv

# Load .env file from the parent directory (2_rag_bot/)
_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(dotenv_path=os.path.join(_BASE_DIR, ".env"), override=True)

# ── OpenAI ────────────────────────────────────────────────────────────────────
OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
OPENAI_CHAT_MODEL: str = "gpt-4o-mini"
OPENAI_EMBEDDING_MODEL: str = "text-embedding-3-small"

# ── Pinecone ──────────────────────────────────────────────────────────────────
PINECONE_API_KEY: str = os.getenv("PINECONE_API_KEY", "")
PINECONE_INDEX_NAME: str = os.getenv("PINECONE_INDEX_NAME", "career-bot")

# ── Pushover (lead capture + unknown-question alerts) ─────────────────────────
PUSHOVER_TOKEN: str = os.getenv("PUSHOVER_TOKEN", "")
PUSHOVER_USER: str = os.getenv("PUSHOVER_USER", "")

# ── LangSmith (observability / tracing) ───────────────────────────────────────
LANGSMITH_API_KEY: str = os.getenv("LANGSMITH_API_KEY", "")
LANGSMITH_PROJECT: str = os.getenv("LANGSMITH_PROJECT", "career-bot")

# ── App Runtime ───────────────────────────────────────────────────────────────
# "development" → Gradio 4.x tuples; "production" → Gradio 5.x dicts
ENV: str = os.getenv("ENV", "production")

# ── Retrieval ─────────────────────────────────────────────────────────────────
# Number of Pinecone chunks to retrieve per query
RETRIEVAL_K: int = 10

# ── Cache ─────────────────────────────────────────────────────────────────────
# SQLite cache file location (same dir as this package's parent)
CACHE_DB_PATH: str = os.path.join(_BASE_DIR, ".langchain.db")
# Time-to-live for the cache (seconds).  24 hours.
CACHE_TTL_SECONDS: int = 86_400

# ── Bot Identity ──────────────────────────────────────────────────────────────
BOT_NAME: str = "Vishal Khoje"
