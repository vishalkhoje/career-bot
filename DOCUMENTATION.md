# Career Conversation (AI Resume/Chat Assistant)

## 1) Project Overview

This project is a Gradio-based chat app that lets visitors ask questions about a persona (example: `Vishal Khoje`). The assistant is prompted with:

- A text summary from `me/summary.txt`
- The content extracted from your LinkedIn PDF inside `me/` (see the “Requirements” section for supported filenames)

It also includes two “tools” that notify you via Pushover when:

- A user’s question can’t be answered (lead capture for unknown questions)
- The user shares their email address (lead capture)

## 2) “Resume” (for the project)

Gradio-powered AI career assistant that answers questions about a person using their `me/summary.txt` + LinkedIn PDF context. It can also capture visitor contact info and unresolved questions by sending notifications through Pushover.

## 3) Configuration & Requirements

### Runtime requirements

- Python 3.10+ (any modern Python 3.x should work)
- A working Azure OpenAI setup
- A Pushover account for notifications

### Install dependencies

1. Install Python dependencies:
  ```bash
   pip install -r requirements.txt
  ```

### Required environment variables (`.env`)

Copy `.env.example` to a real `.env` file in the project root with:

- `OPENAI_API_KEY` (used as the Azure OpenAI key)
- `AZURE_OPENAI_ENDPOINT` (your Azure OpenAI endpoint, e.g. `https://...openai.azure.com`)
- `AZURE_OPENAI_DEPLOYMENT` (your Azure OpenAI deployment name)
- `PUSHOVER_TOKEN` (Pushover application token)
- `PUSHOVER_USER` (Pushover user key)

The app loads these via `python-dotenv` (`load_dotenv(override=True)`).

### Azure OpenAI model/deployment requirement

The code calls Azure OpenAI with:

```python
model=os.getenv("AZURE_OPENAI_DEPLOYMENT")
```

When using `AzureOpenAI`, the `model` value is the **Azure deployment name** from `AZURE_OPENAI_DEPLOYMENT`.
Set that environment variable to the exact deployment name you created in Azure.

### Required local files (`me/`)

The app reads:

- `me/summary.txt`
- LinkedIn PDF: `me/linkedin.pdf`

The persona name is currently hard-coded in `app.py` as `Vishal Khoje`. Update the code if you want to change the display name and prompt persona.

### External services used

- Azure OpenAI: generates chat responses and tool calls
- Pushover: receives notifications via HTTP POST
- PDF extraction: `pypdf` reads and extracts LinkedIn text

## 4) Step-by-Step Documentation

### Step 1: Verify repository files

- Confirm `app.py` exists in the project root
- Confirm `me/summary.txt` exists
- Confirm the LinkedIn PDF exists as `me/linkedin.pdf`

### Step 2: Create a virtual environment (recommended)

```bash
python -m venv .venv
source .venv/bin/activate
```

### Step 3: Install requirements

```bash
pip install -r requirements.txt
```

### Step 4: Configure environment variables

Copy `.env.example` to `.env` in the project root:

```bash
AZURE_OPENAI_API_KEY="..."
AZURE_OPENAI_ENDPOINT="https://....openai.azure.com"
AZURE_OPENAI_DEPLOYMENT="..."
PUSHOVER_TOKEN="..."
PUSHOVER_USER="..."
```

### Step 5: Run the app

```bash
python app.py
```

### Step 6: Open the UI

The Gradio server runs on:

- `http://localhost:7860`

The app is configured with:

- `server_name="0.0.0.0"`
- `server_port=7860`

### Step 7: Use the chat

1. Ask questions like: “What are your key skills?” or “Summarize the person’s experience.”
2. If the assistant decides it cannot answer, it will call the `record_unknown_question` tool.
3. If a user provides an email address (the model will request/collect it based on the system prompt), the assistant will call `record_user_details`.

When either tool is invoked, the app sends a Pushover notification to the configured recipient.

## 5) Additional Notes / What’s Missing (and suggested improvements)

### A) Known gotchas

- Ensure the LinkedIn PDF exists as `me/linkedin.pdf` (otherwise startup fails with a `FileNotFoundError`)
- **Azure deployment name**: set `AZURE_OPENAI_DEPLOYMENT` to your exact Azure deployment name.

### B) Privacy and data handling

- The app extracts text from a LinkedIn PDF and includes it in the prompt. Ensure you have permission to use/process that content.
- User emails captured through `record_user_details` are sent through Pushover notifications. Treat Pushover messages as sensitive.

### C) How to customize “resume/persona” content

- This repo is intended as a public template:
  - Replace `me/summary.txt` with your own summary
- Replace the LinkedIn PDF in `me/` with your own file as `me/linkedin.pdf`
  - Update the hard-coded name in `app.py` if you want the UI + prompt to use your name instead of `Vishal Khoje`

### D) Suggested future enhancements

- Add a real persistence layer for leads/unknown questions (database or spreadsheet instead of notifications only).
- Add RAG with chunking and citations (instead of sending the whole extracted text).
- Add an “admin” view to review captured leads/questions.

