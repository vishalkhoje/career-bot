"""
src/tools.py
~~~~~~~~~~~~
LLM tool definitions and their Python handler functions.

Each tool has two parts:
  1. A JSON schema (OpenAI function-calling format) consumed by the LLM.
  2. A Python function that is actually executed when the LLM calls the tool.

The module also exposes:
  - TOOL_SCHEMAS  – list[dict] passed to ChatOpenAI.bind_tools()
  - TOOL_MAP      – dict[str, Callable] used for deterministic dispatch
                    (replaces the dangerous globals().get() anti-pattern)
"""

from __future__ import annotations

import requests
from . import config


# ── Pushover notification helper ───────────────────────────────────────────────

def push(text: str) -> None:
    """
    Send a push notification via Pushover.

    Args:
        text: The message body to send.
    """
    if not config.PUSHOVER_TOKEN or not config.PUSHOVER_USER:
        print(f"[Pushover] Skipping — credentials not configured. Message: {text}")
        return
    try:
        requests.post(
            "https://api.pushover.net/1/messages.json",
            data={
                "token": config.PUSHOVER_TOKEN,
                "user": config.PUSHOVER_USER,
                "message": text,
            },
            timeout=5,
        )
    except requests.RequestException as exc:
        print(f"[Pushover] Failed to send notification: {exc}")


# ── Tool handler functions ─────────────────────────────────────────────────────

def record_user_details(
    email: str,
    name: str = "Name not provided",
    notes: str = "not provided",
) -> dict:
    """
    Record a visitor's contact details via Pushover.

    Called by the LLM when a user expresses interest in getting in touch
    and provides their email address.

    Args:
        email: The visitor's email address (required).
        name:  The visitor's name (optional).
        notes: Additional context from the conversation (optional).

    Returns:
        A confirmation dict: {"recorded": "ok"}
    """
    push(f"New lead — Name: {name} | Email: {email} | Notes: {notes}")
    return {"recorded": "ok"}


def record_unknown_question(question: str) -> dict:
    """
    Log a question the bot could not answer.

    The LLM calls this whenever it cannot find an answer in the
    provided career context, so the owner can update their profile
    or FAQs accordingly.

    Args:
        question: The unanswered question text.

    Returns:
        A confirmation dict: {"recorded": "ok"}
    """
    push(f"Unknown question: {question}")
    return {"recorded": "ok"}


# ── OpenAI function-calling JSON schemas ───────────────────────────────────────

_RECORD_USER_DETAILS_SCHEMA: dict = {
    "name": "record_user_details",
    "description": (
        "Use this tool to record that a user is interested in being in touch "
        "and has provided an email address."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "email": {
                "type": "string",
                "description": "The email address of this user",
            },
            "name": {
                "type": "string",
                "description": "The user's name, if they provided it",
            },
            "notes": {
                "type": "string",
                "description": (
                    "Any additional information about the conversation "
                    "that's worth recording to give context"
                ),
            },
        },
        "required": ["email"],
        "additionalProperties": False,
    },
}

_RECORD_UNKNOWN_QUESTION_SCHEMA: dict = {
    "name": "record_unknown_question",
    "description": (
        "Always use this tool to record any question that couldn't be answered "
        "because the answer was not found in the career context."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "question": {
                "type": "string",
                "description": "The question that couldn't be answered",
            },
        },
        "required": ["question"],
        "additionalProperties": False,
    },
}


# ── Public exports ─────────────────────────────────────────────────────────────

#: Pass this list to ChatOpenAI.bind_tools()
TOOL_SCHEMAS: list[dict] = [
    {"type": "function", "function": _RECORD_USER_DETAILS_SCHEMA},
    {"type": "function", "function": _RECORD_UNKNOWN_QUESTION_SCHEMA},
]

#: Deterministic function dispatch — used in agent.py
TOOL_MAP: dict[str, callable] = {
    "record_user_details": record_user_details,
    "record_unknown_question": record_unknown_question,
}
