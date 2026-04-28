# Career Conversation AI Bot

This repository contains two versions of the Career Conversation AI Bot.

## Project Structure

### 📂 [1_simple_bot](./1_simple_bot)
A straightforward implementation that passes the entire LinkedIn PDF and summary text directly into the LLM context. Best for small documents and quick testing.
- **Main Script**: `app.py`
- **Key Feature**: Zero-config context passing.

### 📂 [2_rag_bot](./2_rag_bot)
An advanced implementation using **RAG (Retrieval-Augmented Generation)** with Pinecone. Best for production and large knowledge bases.
- **Main Script**: `app-rag.py`
- **Data Pipeline**: `ingest.py` (with validation, deduplication, and versioning).
- **Database**: Pinecone Vector DB.
- **Key Feature**: Production Observability (Latency, Tokens, Error Tracking).
- **Core Upgrade**: Multi-Agent Workflow (Intent Classifier -> Planner -> Critic).
- **New Feature**: Evaluation System (Auto-eval, User Feedback, Offline Benchmarking).

---

## Getting Started

1. Choose a version (`1_simple_bot` or `2_rag_bot`).
2. Navigate into the directory:
   ```bash
   cd 1_simple_bot  # or cd 2_rag_bot
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Configure your `.env` file with OpenAI and Pinecone keys.
5. Run the app:
   ```bash
   python app.py  # in 1_simple_bot
   # OR
   python app-rag.py # in 2_rag_bot (run ingest.py first)
   ```
