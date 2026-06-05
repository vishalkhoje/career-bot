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

from ..core import config
from ..agents.career_agent import CareerAgent


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

    with gr.Blocks(title="Vishal's AI") as demo:
        # Force type="messages" for Gradio 5+ compatibility (handled inside chat_kwargs) and better history handling
        interface = gr.ChatInterface(
            **chat_kwargs
        )
        
        def handle_feedback(data: gr.LikeData):
            from ..database.evaluation import EvaluationSystem
            import threading
            eval_sys = EvaluationSystem()
            
            feedback = "up" if data.liked else "down"
            
            # Ensure the value is a string (Gradio 5 'messages' mode can pass dicts/lists)
            val = data.value
            if isinstance(val, dict):
                val = val.get("content", str(val))
            elif isinstance(val, list):
                # Extract text if it's a list of message parts
                texts = [item.get("text", str(item)) if isinstance(item, dict) else str(item) for item in val]
                val = " ".join(texts)
            else:
                val = str(val)

            print(f"[UI] User feedback received: {feedback} for message: {val[:50]}...")
            
            import sqlite3
            with sqlite3.connect(eval_sys.db_path) as conn:
                # We search for the most recent evaluation where either the query or the response matches
                # This ensures feedback works whether the user 'likes' their question or the bot's answer.
                cursor = conn.execute(
                    """
                    SELECT id FROM evaluations 
                    WHERE query = ? OR response = ? 
                    ORDER BY id DESC LIMIT 1
                    """,
                    (val, val)
                )
                row = cursor.fetchone()
                if row:
                    eval_id = row[0]
                    conn.execute("UPDATE evaluations SET feedback = ? WHERE id = ?", (feedback, eval_id))
                    
                    # If feedback is negative, trigger self-improvement analysis in background
                    if feedback == "down":
                        print(f"[UI] Negative feedback detected for ID {eval_id}. Triggering analysis...")
                        thread = threading.Thread(target=eval_sys.analyze_and_improve, args=(eval_id,))
                        thread.start()

        # In Gradio 5/6, the 'like' event is on the chatbot component
        interface.chatbot.like(handle_feedback, None, None)

    return demo
