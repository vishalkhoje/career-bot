# 🤖 Career Conversation Bot — RAG-Powered AI Career Assistant

> An intelligent, production-grade chatbot that represents your professional profile through a Retrieval-Augmented Generation (RAG) pipeline. Visitors can ask questions about your experience, skills, and projects — and get accurate, factual answers drawn directly from your own career data.

---

## 📋 Table of Contents

1. [What is This?](#-what-is-this)
2. [Tech Stack](#-tech-stack)
3. [Framework Decisions](#-framework-decisions)
4. [Architecture Overview](#-architecture-overview)
5. [Folder Structure](#-folder-structure)
6. [File-by-File Explanation](#-file-by-file-explanation)
7. [Step-by-Step Data Flow](#-step-by-step-data-flow)
8. [Setup & Installation](#-setup--installation)
9. [Running the Project](#-running-the-project)
10. [Updating Your Career Data](#-updating-your-career-data)
11. [Caching System](#-caching-system)
12. [Observability with LangSmith](#-observability-with-langsmith)
13. [Running Tests](#-running-tests)
14. [Environment Variables Reference](#-environment-variables-reference)
15. [Troubleshooting](#-troubleshooting)

---

## 💡 What is This?

This is **not** just a chatbot that reads your resume. It is a full **RAG (Retrieval-Augmented Generation)** system that:

- Stores your career data as mathematical vectors in a cloud vector database (Pinecone).
- On every query, retrieves only the **most semantically relevant** chunks of your profile.
- Injects those chunks into the LLM's context, grounding its answer in real facts.
- Uses **tool calling** to capture leads (email addresses) and log unanswered questions automatically.
- **Streams responses** word-by-word so the UI feels instant, not frozen.
- **Caches responses** in SQLite so repeat questions are answered for free ($0.00 cost).

---

## 🛠 Tech Stack

| Component | Technology | Version |
|---|---|---|
| **LLM** | OpenAI `gpt-4o-mini` | API |
| **Embeddings** | OpenAI `text-embedding-3-small` | API |
| **Vector Database** | Pinecone (Serverless) | API |
| **LLM Orchestration** | LangChain | `≥0.1` |
| **UI Framework** | Gradio | `≥4.0.0` |
| **Response Cache** | SQLite (manual, built-in) | stdlib |
| **Observability** | LangSmith | API |
| **Notifications** | Pushover | API |
| **PDF Parsing** | PyPDF | pip |
| **Language** | Python | `3.10+` |

---

## 🏗 Framework Decisions

### Why LangChain?
LangChain provides a standardised interface over OpenAI's API, making it easy to swap between models, add tool-calling, and integrate with vector stores — all with minimal boilerplate. The `ChatOpenAI` + `.bind_tools()` pattern means adding new tools requires only a schema and a handler function.

### Why Pinecone?
Pinecone is a managed, serverless vector database designed specifically for semantic search. There is no server to provision — the index scales automatically. The `cosine` similarity metric is ideal for comparing embedding vectors, returning the most semantically similar chunks regardless of exact keyword matches.

### Why Gradio?
Gradio allows a production-grade chat UI to be created in fewer than 20 lines of Python. It handles streaming responses natively (detecting generator functions automatically), manages conversation history, and provides a shareable link out of the box.

### Why a Manual SQLite Cache (not LangChain's built-in)?
LangChain's internal cache intercepts calls at the LangChain level, but it still records a trace in LangSmith and reports a cost — because it cannot distinguish between a true LLM call and a cache hit at the callback level. The manual SQLite cache **completely bypasses the LLM call**, resulting in genuinely zero API cost and zero LangSmith trace for repeat questions.

### Why Stream Responses?
Without streaming, the UI is frozen for the entire duration of the LLM call (typically 2–4 seconds). With streaming, text appears word-by-word as it is generated, giving the impression of an instant, human-like response.

---

## 🏛 Architecture Overview

The system is split into two completely independent phases:

![Career Bot Workflow](./career_bot_workflow.png)

---

## 📁 Folder Structure

```
2_rag_bot/
│
├── app-rag.py              # Entry point — wires agent + UI and launches Gradio
├── ingest.py               # Data ingestion pipeline (run once per data update)
├── requirements.txt        # All Python dependencies
├── .env                    # Secret keys and configuration (never commit this)
├── .langchain.db           # Auto-generated SQLite response cache
├── RAG_DOCUMENTATION.md    # Technical RAG architecture reference
│
├── me/                     # Your career data files (the "knowledge base")
│   ├── linkedin.pdf        # Your LinkedIn profile export (PDF)
│   └── summary.txt         # Freeform career summary, resume link, FAQs
│
├── src/                    # All application source code
│   ├── __init__.py         # Package entry point, exposes CareerAgent + build_ui
│   ├── config.py           # Centralised config — all env vars and constants
│   ├── tools.py            # LLM tool schemas + Python handler functions
│   ├── cache.py            # SQLite response cache (ResponseCache class)
│   ├── retriever.py        # Pinecone vector store wrapper (CareerRetriever)
│   ├── prompt.py           # System prompt builder (pure function)
│   ├── agent.py            # Core orchestration — CareerAgent class
│   └── ui.py               # Gradio UI assembly (no business logic)
│
└── tests/                  # Test and verification scripts
    ├── verify_refactored.py  # End-to-end: cache miss → hit → case-insensitive hit
    ├── verify_streaming.py   # Confirms responses stream in multiple chunks
    ├── verify_chat_resume.py # Confirms resume download link is returned
    ├── test_resume_link.py   # Inspects Pinecone chunks for resume link presence
    ├── test_rag.py           # Basic sanity check for the full RAG loop
    ├── test_cache_speed.py   # Measures latency: Run 1 (LLM) vs Run 2 (Cache)
    ├── test_ttl.py           # Validates 24-hour cache expiration behaviour
    ├── check_pinecone.py     # Lists Pinecone indices (connectivity check)
    └── debug_cache.py        # Low-level LangChain cache debug utility
```

---

## 📄 File-by-File Explanation

### `app-rag.py` — Entry Point
The application entry point. Intentionally minimal (~25 lines). It imports `CareerAgent` and `build_ui` from the `src/` package, wires them together, and calls `demo.launch()`. No business logic lives here.

```python
# What it does:
agent = CareerAgent()      # 1. Initialise the bot
demo = build_ui(agent)     # 2. Build the Gradio interface
demo.launch(...)           # 3. Start the web server
```

---

### `ingest.py` — Data Ingestion Pipeline
Reads your career documents, splits them into chunks, embeds them with OpenAI, and upserts them to Pinecone. Also clears the response cache so the bot never serves stale answers after a data update.

**Run this every time you update `me/linkedin.pdf` or `me/summary.txt`.**

```bash
python3 ingest.py
```

**What it does internally:**
1. Validates env vars (`OPENAI_API_KEY`, `PINECONE_API_KEY`).
2. Calls `ResponseCache.clear()` to invalidate stale cached answers.
3. Loads `me/linkedin.pdf` via `PyPDFLoader` and `me/summary.txt` via `TextLoader`.
4. Splits documents into 1000-character chunks with 200-character overlap.
5. Embeds each chunk using `text-embedding-3-small` (1536 dimensions).
6. Creates the Pinecone index if it does not exist, then upserts all vectors.

---

### `src/config.py` — Centralised Configuration
The single source of truth for **all** environment variables and application constants. Every other module imports from here. Never call `os.getenv()` directly in business logic files.

**Key constants:**
| Constant | Default | Description |
|---|---|---|
| `OPENAI_CHAT_MODEL` | `gpt-4o-mini` | LLM model name |
| `OPENAI_EMBEDDING_MODEL` | `text-embedding-3-small` | Embedding model |
| `PINECONE_INDEX_NAME` | `career-bot` | Pinecone index name |
| `RETRIEVAL_K` | `10` | Number of chunks retrieved per query |
| `CACHE_TTL_SECONDS` | `86400` | Cache expiry (24 hours) |
| `ENV` | `production` | `development` = Gradio 4.x, `production` = Gradio 5.x |

---

### `src/tools.py` — LLM Tool Definitions
Defines the two tools the LLM can call during a conversation:

| Tool | Trigger | Action |
|---|---|---|
| `record_user_details` | User provides email | Sends Pushover notification with lead details |
| `record_unknown_question` | LLM cannot answer | Logs the question via Pushover so you can improve your data |

Also exports `TOOL_SCHEMAS` (passed to `ChatOpenAI.bind_tools()`) and `TOOL_MAP` (a `dict[str, callable]` for safe dispatch — replacing the dangerous `globals().get()` anti-pattern).

---

### `src/cache.py` — Response Cache
A lightweight, persistent SQLite cache with a 24-hour TTL.

**Cache key design:** The key is the **SHA-256 hash of the normalised (lowercased + stripped) user message only**. Conversation history is intentionally excluded so the same question always hits the same cache entry — regardless of how many previous turns exist in the conversation.

**Public API:**
```python
cache = ResponseCache(db_path="...", ttl_seconds=86400)
key   = cache.make_key("What are your skills?")  # Deterministic hash
hit   = cache.get(key)                             # None if not cached
cache.set(key, llm_response)                       # Store after LLM call
cache.check_expiry()                               # Delete DB if >24h old
cache.clear()                                      # Called by ingest.py
```

---

### `src/retriever.py` — Pinecone Retriever
Wraps `PineconeVectorStore` and exposes a single `retrieve(query, k)` method. Documents are **sorted alphabetically by content** before being joined into the context string. This deterministic ordering is critical — it ensures the same query always produces the same context, which is essential for cache key stability.

---

### `src/prompt.py` — System Prompt Builder
A pure function `build_system_prompt(context, name)` that returns the full system prompt string. Keeping the prompt in its own file means it can be updated, reviewed, and unit-tested without touching any orchestration code.

The prompt enforces **strict guardrails**:
- Answer only from the provided context — no hallucination.
- Record unanswerable questions via tool.
- Provide resume links when found in context.
- Steer interested visitors towards providing their email.

---

### `src/agent.py` — Core Orchestration (`CareerAgent`)
The heart of the application. The `chat(message, history)` method is a **generator** that:

1. Normalises the message and checks the cache.
2. If **cache hit** → yields the stored response immediately (no API call).
3. If **cache miss** → retrieves Pinecone context, builds messages, calls the LLM via `llm.stream()`.
4. Yields each text chunk as it arrives (streaming).
5. Stores the complete response in the cache after streaming finishes.

Tool calls are handled mid-stream: if the LLM requests a tool, the stream is paused, the tool is executed, and the LLM is called again to generate the final text response.

---

### `src/ui.py` — Gradio UI
Builds and returns the `gr.Blocks` application. No business logic. The `ENV` variable controls Gradio compatibility:
- `ENV=production` → `type="messages"` for Gradio 5.x
- `ENV=development` → Gradio 4.x tuple format (local dev)

---

## ▶ Step-by-Step Data Flow

### On every user message:

```
Step 1:  User types message in Gradio UI
Step 2:  Gradio calls agent.chat(message, history)
Step 3:  CareerAgent normalises message → message.lower().strip()
Step 4:  ResponseCache.make_key(message) → SHA-256 hash
Step 5a: Cache HIT  → yield cached response → DONE ($0.00, <50ms)
Step 5b: Cache MISS → continue to Step 6
Step 6:  CareerRetriever.retrieve(message, k=10)
            → PineconeVectorStore.similarity_search(message, k=10)
            → Sort docs alphabetically for determinism
            → Join into a single context string
Step 7:  build_system_prompt(context) → inject career facts into prompt
Step 8:  Build LangChain message list:
            [SystemMessage] + [History] + [HumanMessage]
Step 9:  ChatOpenAI.stream(messages) → yields chunks
Step 10: For each chunk with text content → yield accumulated text to Gradio
Step 11: If LLM requests a tool call:
            → Pause stream
            → Execute tool (Pushover notification)
            → Append ToolMessage
            → Resume LLM stream
Step 12: Stream complete → store full response in ResponseCache
Step 13: Gradio displays the streamed response word-by-word
```

---

## ⚙ Setup & Installation

### Prerequisites
- Python `3.10+`
- An OpenAI API account with a funded key
- A Pinecone account (free tier is sufficient)
- (Optional) A Pushover account for lead notifications
- (Optional) A LangSmith account for observability

### 1. Clone the repo and navigate to this project

```bash
git clone <your-repo-url>
cd career_conversation/2_rag_bot
```

### 2. Create and activate a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate       # macOS/Linux
# .venv\Scripts\activate        # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

Copy `.env.example` to `.env` and fill in your keys:

```bash
cp .env.example .env
```

See [Environment Variables Reference](#-environment-variables-reference) for all required keys.

### 5. Add your career data

Place your files in the `me/` directory:

```
me/
├── linkedin.pdf    ← Export your LinkedIn profile as PDF
└── summary.txt     ← Write a free-form career summary
                      Include a Resume Download link at the very top!
```

> **Tip:** Add the resume download link as the very first line of `summary.txt`. The RAG system prioritises earlier content, ensuring the link is always in the retrieved context.

### 6. Run the ingestion pipeline

```bash
python3 ingest.py
```

You should see:
```
Starting ingestion process...
[Cache] Cleared: /path/to/.langchain.db
Loading me/linkedin.pdf...
Loading me/summary.txt...
Splitting documents into chunks...
Created 15 chunks.
Initializing OpenAI Embeddings...
Connecting to Pinecone...
Upserting vectors to index 'career-bot'...
Ingestion complete!
```

---

## 🚀 Running the Project

### Development mode (Gradio 4.x compatible)

```bash
ENV=development python3 app-rag.py
```

### Production mode (Gradio 5.x compatible)

```bash
python3 app-rag.py
```

Open your browser at **http://localhost:7860**

---

## 📝 Updating Your Career Data

Whenever you change `me/linkedin.pdf` or `me/summary.txt`:

```bash
python3 ingest.py
```

This automatically:
1. Clears the response cache (so stale answers are never served).
2. Re-indexes all documents in Pinecone.

You do **not** need to restart `app-rag.py` — the next query will pick up the new vectors automatically.

---

## ⚡ Caching System

The cache is a local SQLite file (`.langchain.db`) that stores LLM responses.

### How it works

| Scenario | Behaviour | Cost |
|---|---|---|
| First time a question is asked | LLM is called, response is stored | ~$0.0002 |
| Same question asked again (any session) | Served from SQLite, LLM is **not** called | **$0.00** |
| Cache is 24+ hours old | Entire cache is deleted on next request | n/a |
| `ingest.py` is run | Cache is deleted to prevent stale answers | n/a |

### Cache key behaviour

The cache key is a SHA-256 hash of the **normalised message only** (lowercased + stripped). This means:

```
"Can you share your resume?"          → same key
"can you share your resume?"          → same key ✅
"  Can you share your resume?  "      → same key ✅
```

---

## 🔍 Observability with LangSmith

LangSmith traces every LLM call (but not cache hits — those bypass the LLM entirely).

To enable:
1. Create a project at [smith.langchain.com](https://smith.langchain.com).
2. Set `LANGSMITH_API_KEY` in `.env`.
3. Set `LANGSMITH_PROJECT` to your project name.

In LangSmith you can inspect:
- Token counts and costs per query.
- Latency breakdown (embedding → retrieval → generation).
- Full prompt + response for hallucination analysis.
- Tool call invocations and their arguments.

---

## 🧪 Running Tests

All test scripts are in `tests/`. Run them from the `2_rag_bot/` directory:

```bash
# End-to-end: cache miss → hit → case-insensitive hit + resume link check
python3 tests/verify_refactored.py

# Verify streaming: confirms response arrives in multiple chunks
python3 tests/verify_streaming.py

# Check Pinecone connectivity and list indices
python3 tests/check_pinecone.py

# Measure cache speed improvement (Run 1 vs Run 2 latency)
python3 tests/test_cache_speed.py

# Inspect which Pinecone chunks contain the resume link
python3 tests/test_resume_link.py
```

### Expected output from `verify_refactored.py`

```
==================================================
Refactored Package Verification
==================================================
Test 1 — First call (CACHE MISS expected)
[Cache] key=4a9fe2c5c293  query='Can you share your resume link?'
[Cache] 🔍 MISS — calling LLM (streaming)
✅ PASS Resume link found in response.

Test 2 — Second identical call (CACHE HIT expected)
[Cache] key=4a9fe2c5c293  query='Can you share your resume link?'
[Cache] ⚡ HIT — serving from SQLite, $0.00 cost
✅ PASS Response identical — served from cache.

Test 3 — Lowercase variant (CACHE HIT expected)
[Cache] key=4a9fe2c5c293  query='can you share your resume link?'
[Cache] ⚡ HIT — serving from SQLite, $0.00 cost
✅ PASS Case-insensitive cache hit confirmed.
==================================================
```

---

## 🔑 Environment Variables Reference

Create a `.env` file in `2_rag_bot/` with these values:

```env
# ── Required ──────────────────────────────────────────────────────────────────

# OpenAI API key (used for embeddings and the chat LLM)
OPENAI_API_KEY=sk-...

# Pinecone API key (used for vector search)
PINECONE_API_KEY=...

# ── Optional but Recommended ──────────────────────────────────────────────────

# Pinecone index name (created automatically if it doesn't exist)
PINECONE_INDEX_NAME=career-bot

# LangSmith observability (traces LLM calls for cost + hallucination analysis)
LANGSMITH_API_KEY=lsv2_pt_...
LANGCHAIN_TRACING_V2=true
LANGSMITH_PROJECT=career-bot

# Pushover (real-time push notifications for leads and unknown questions)
PUSHOVER_TOKEN=...
PUSHOVER_USER=...

# ── Runtime ────────────────────────────────────────────────────────────────────

# "development" → Gradio 4.x format (local dev)
# "production"  → Gradio 5.x format (deployment)
ENV=development
```

---

## 🔧 Troubleshooting

### `OPENAI_API_KEY is not set`
Make sure your `.env` file exists in `2_rag_bot/` and contains `OPENAI_API_KEY=sk-...`.

### Bot says "I'm sorry, I can only provide information based on the provided career profile"
The answer was not found in the retrieved context chunks. Try:
1. Run `python3 tests/test_resume_link.py` to inspect what Pinecone is returning.
2. Re-run `python3 ingest.py` to ensure the latest data is indexed.
3. Add more detail to `me/summary.txt` about the topic being asked.

### Cache never hits for the same question
The cache is keyed on the **normalised message**. If the hash changes between identical queries, delete `.langchain.db` and re-test with `python3 tests/verify_refactored.py`.

### Gradio shows "TypeError: 'generator' object is not subscriptable"
You are calling `agent.chat()` outside of Gradio without consuming the generator. Wrap the call: `response = "".join(agent.chat(msg, []))` or use `for chunk in agent.chat(msg, []): ...`.

### Pinecone `NotFoundException` on startup
The index does not exist yet. Run `python3 ingest.py` to create it and populate it with your documents.

### Pushover notifications not arriving
- Verify `PUSHOVER_TOKEN` and `PUSHOVER_USER` are set correctly in `.env`.
- Check the Pushover dashboard for delivery failures.
- The bot will log `[Pushover] Skipping — credentials not configured` if keys are missing.

---

## 📚 Additional Documentation

- [RAG_DOCUMENTATION.md](./RAG_DOCUMENTATION.md) — Detailed technical architecture with Mermaid diagrams.
- [LangChain Docs](https://python.langchain.com/docs/) — Orchestration framework reference.
- [Pinecone Docs](https://docs.pinecone.io/) — Vector database reference.
- [Gradio Docs](https://www.gradio.app/docs/) — UI framework reference.
- [LangSmith Docs](https://docs.smith.langchain.com/) — Observability reference.

---

*Built with ❤️ as a production-grade demonstration of RAG architecture for career representation.*
