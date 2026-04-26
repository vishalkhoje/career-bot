# RAG Workflow Documentation

This document explains the technical architecture and step-by-step flow of the Career Bot's Retrieval-Augmented Generation (RAG) system.

## 1. High-Level Architecture

The system is divided into two main phases: **Data Ingestion** (Offline/One-time) and **Query Retrieval** (Online/Real-time).

![RAG Workflow Infographic](./rag_workflow.png)

> [!TIP]
> **View Visual Diagram**: If the image above or the Mermaid diagram below doesn't render, open [**RAG_WORKFLOW.html**](./RAG_WORKFLOW.html) in your browser to see the full interactive visualization.

```mermaid
graph TD
    subgraph "1. Data Ingestion (Offline)"
        A1[me/linkedin.pdf] --> B1[PyPDFLoader]
        A2[me/summary.txt] --> B2[TextLoader]
        B1 --> C[RecursiveCharacterTextSplitter]
        B2 --> C
        C --> D[OpenAI: text-embedding-3-small]
        D --> E[(Pinecone: career-bot index)]
    end

    subgraph "2. Real-Time Chat (app-rag.py)"
        F[Gradio UI] -->|User Question| G[OpenAIEmbeddings]
        G -->|Query Vector| H[Pinecone Similarity Search]
        E -->|Relevant Snippets| H
        H -->|Context| I[System Prompt Builder]
        I -->|Context + Prompt| J[OpenAI: gpt-4o]
        J -->|Response| F
    end

    subgraph "3. Lead Capture & Notifications"
        J -->|Tool Call| L{Function Router}
        L -->|record_user_details| M[Pushover Notification]
        L -->|record_unknown_question| M
    end
```

---

## 2. Step-by-Step Flow

### Phase A: Data Ingestion (`ingest.py`)

1.  **Document Loading**: The script reads `me/linkedin.pdf` and `me/summary.txt`.
2.  **Chunking**: The text is split into small chunks (currently 500 characters with a 50-character overlap). This ensures that the context provided to the AI is granular and specific.
3.  **Embedding Generation**: Each chunk is sent to OpenAI's `text-embedding-3-small` model, which converts the text into a 1536-dimensional vector (a mathematical representation of meaning).
4.  **Vector Storage**: These vectors, along with the original text (metadata), are stored in the **Pinecone** index (`career-bot`).

### Phase B: Query Retrieval & Generation (`app-rag.py`)

1.  **User Input**: The user asks a question through the Gradio interface.
2.  **Vector Search**: The system converts the user's question into a vector and performs a "Similarity Search" against the Pinecone database.
3.  **Context Retrieval**: Pinecone returns the top 5 most relevant text chunks based on mathematical similarity (Cosine Similarity).
4.  **Prompt Construction**: The retrieved snippets are combined into a `context` block.
5.  **System Prompt Enrichment**: This context is injected into the system prompt, giving the AI specific facts to use for its answer.
6.  **LLM Execution**: The final prompt (History + System Prompt + Context + User Question) is sent to **OpenAI GPT-4o**.
7.  **Response**: The AI generates a professional response staying in character as Vishal Khoje.

---

## 3. Technology Stack

-   **LLM**: OpenAI GPT-4o
-   **Embeddings**: OpenAI `text-embedding-3-small`
-   **Vector Database**: Pinecone (Serverless)
-   **Framework**: LangChain (for orchestration)
-   **UI**: Gradio
-   **PDF Processing**: PyPDF

---

## 4. How to Update Data

Whenever you change your LinkedIn PDF or the summary text:
1.  Replace the files in the `me/` directory.
2.  Run the ingestion script:
    ```bash
    python3 ingest.py
    ```
    *Note: This will re-index your documents in Pinecone.*
