"""
src/ui.py
~~~~~~~~~
Gradio UI assembly for the career chatbot.

This module has exactly one responsibility: take a ``CareerAgent`` instance
and return a configured ``gr.Blocks`` application.  No business logic lives
here — all intelligence is in the agent.

ENV behaviour (set in .env):
  - "production"  → Gradio 5.x format (type="messages")
  - "development" → Gradio 4.x format (tuples), no type arg
"""

from __future__ import annotations

import gradio as gr

from . import config
from .agent import CareerAgent


def build_ui(agent: CareerAgent) -> gr.Blocks:
    """
    Construct and return the Gradio application.

    Args:
        agent: A fully initialised :class:`~src.agent.CareerAgent`.

    Returns:
        A ``gr.Blocks`` instance ready to be launched via ``.launch()``.
    """
    is_production = config.ENV == "production"

    chat_kwargs: dict = {
        "fn": agent.chat,
        "title": "Chat with Vishal's AI",
        "description": (
            "Skip the standard resume. Ask me directly about Vishal's "
            "technical skills, past projects, and career highlights."
        ),
    }

    if is_production:
        # Gradio 5.x requires type="messages" for the new message format
        chat_kwargs["type"] = "messages"
        print("[UI] Running in PRODUCTION mode (Gradio 5 compatible).")
    else:
        print("[UI] Running in DEVELOPMENT mode (Gradio 4 compatible).")

    with gr.Blocks() as demo:
        gr.ChatInterface(**chat_kwargs)
        gr.DeepLinkButton()

    return demo
