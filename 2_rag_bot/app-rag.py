"""
app-rag.py
~~~~~~~~~~
Entry point for the RAG-powered career chatbot.

All business logic lives in the `src/` package.
This file is intentionally minimal — it only wires up the agent and
launches the Gradio UI.

Usage:
    python3 app-rag.py
"""

import os
import sys

# Add the current directory to sys.path so 'src' is found
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.agents.career_agent import CareerAgent
from src.helper.ui import build_ui

if __name__ == "__main__":
    agent = CareerAgent()
    demo = build_ui(agent)

    demo.queue()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        ssr_mode=False,
        share=False,
    )