"""
src/agent.py
~~~~~~~~~~~~
Core orchestration layer: the CareerAgent.

This module owns the complete request-response lifecycle:
  1. Normalise + cache-check the incoming message (before hitting any API)
  2. On cache miss: retrieve context from Pinecone
  3. Build the LangChain message list (system + history + user turn)
  4. Invoke the LLM with tool-calling enabled
  5. Dispatch any tool calls, loop until a final text response is produced
  6. Store the response in the cache and return it

Design goals:
  - Each collaborator (cache, retriever, prompt, tools) is injected via
    constructor, making the class easy to test in isolation.
  - No business logic lives in app-rag.py or ui.py.
  - All logging uses print() with a consistent [TAG] prefix so it is easy
    to grep in production logs.
"""

from __future__ import annotations

import json
import time
from typing import Union

from langchain_community.callbacks.manager import get_openai_callback
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_openai import ChatOpenAI

from . import config
from .cache import ResponseCache
from .prompt import build_system_prompt
from .retriever import CareerRetriever
from .tools import TOOL_MAP, TOOL_SCHEMAS

# Conversation history item can be dict (Gradio 5) or 2-tuple (Gradio 4)
_HistoryItem = Union[dict, tuple]


class CareerAgent:
    """
    Orchestrates retrieval, caching, and LLM calls for the career chatbot.

    Attributes:
        name:      The person being represented (from config).
        cache:     :class:`~src.cache.ResponseCache` instance.
        retriever: :class:`~src.retriever.CareerRetriever` instance.
        llm:       LangChain ``ChatOpenAI`` with tools bound.
    """

    def __init__(self) -> None:
        """
        Wire up all collaborators.

        Raises:
            ValueError: If OPENAI_API_KEY is not set.
        """
        if not config.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY is not set in the environment.")

        self.name = config.BOT_NAME

        # Persistent SQLite response cache (24-hour TTL by default)
        self.cache = ResponseCache(
            db_path=config.CACHE_DB_PATH,
            ttl_seconds=config.CACHE_TTL_SECONDS,
        )

        # Pinecone-backed semantic retriever
        self.retriever = CareerRetriever()

        # LangChain chat model with tools bound.
        # cache=False: we manage our own cache above — no need for LangChain's.
        self.llm = ChatOpenAI(
            model=config.OPENAI_CHAT_MODEL,
            api_key=config.OPENAI_API_KEY,
            temperature=0,
            cache=False,
        ).bind_tools(TOOL_SCHEMAS)

    # ── Public interface ────────────────────────────────────────────────────────

    def chat(self, message: str, history: list[_HistoryItem]):
        """
        Process a user message and yield the bot's reply in chunks (streaming).

        This is a generator function. Gradio's ``ChatInterface`` automatically
        detects generators and updates the UI in real-time as chunks are yielded.

        Cache strategy:
            The cache key is derived from the *normalised message text only*
            (lowercase + stripped).  Conversation history is deliberately
            excluded from the key so that the same question asked at any
            point in a conversation always hits the same cache entry.
            If a cache hit occurs, the full response is yielded immediately.

        Args:
            message: The raw text typed by the user.
            history: Conversation history as provided by Gradio.
                     Gradio 5 passes dicts; Gradio 4 passes 2-tuples.

        Yields:
            The bot's response chunks as they are generated.
        """
        message = message.strip()

        # ── Step 0: Expire stale cache if TTL exceeded ─────────────────────
        self.cache.check_expiry()

        # ── Step 1: Check cache BEFORE any API call ─────────────────────────
        cache_key = self.cache.make_key(message)
        print(f"\n[Cache] key={cache_key[:12]}  query='{message}'")

        cached = self.cache.get(cache_key)
        if cached:
            print("[Cache] ⚡ HIT — serving from SQLite, $0.00 cost")
            yield cached
            return

        print("[Cache] 🔍 MISS — calling LLM (streaming)")

        # ── Step 2: Retrieve relevant context from Pinecone ─────────────────
        context = self.retriever.retrieve(message)

        # ── Step 3: Build the message list for the LLM ──────────────────────
        messages = [SystemMessage(content=build_system_prompt(context))]
        messages.extend(self._normalise_history(history))
        messages.append(HumanMessage(content=message))

        # ── Step 4: LLM call with streaming and token tracking ──────────────
        full_response = ""
        for chunk in self._invoke_llm_stream(messages):
            full_response += chunk
            yield full_response

        # ── Step 5: Persist to cache ────────────────────────────────────────
        if full_response:
            self.cache.set(cache_key, full_response)

    # ── Private helpers ─────────────────────────────────────────────────────────

    def _invoke_llm_stream(self, messages: list):
        """
        Invoke the LLM with streaming support, handling any tool calls.

        If the LLM requests a tool call, we execute it and then resume
        streaming from the next turn.

        Args:
            messages: The full message list (system + history + user turn).

        Yields:
            Text chunks from the LLM response.
        """
        start = time.time()

        with get_openai_callback() as cb:
            done = False
            
            while not done:
                content_yielded = ""
                full_message = None
                
                # Use .stream() for real-time output
                for chunk in self.llm.stream(messages):
                    if full_message is None:
                        full_message = chunk
                    else:
                        full_message += chunk
                    
                    if chunk.content:
                        content_yielded += chunk.content
                        yield chunk.content

                if full_message and full_message.tool_calls:
                    # Append AI message so the LLM sees its own tool call
                    messages.append(full_message)
                    tool_results = self._dispatch_tool_calls(full_message.tool_calls)
                    messages.extend(tool_results)
                    # Loop again to get the final text after tool execution
                else:
                    done = True

            duration = time.time() - start
            print(
                f"\n[LLM] Tokens={cb.total_tokens}  "
                f"Cost=${cb.total_cost:.6f}  "
                f"Time={duration:.2f}s (Streaming complete)"
            )

    def _dispatch_tool_calls(self, tool_calls: list) -> list[ToolMessage]:
        """
        Execute each requested tool and return ToolMessage results.

        Uses the explicit TOOL_MAP dict rather than globals() for safe,
        deterministic dispatch.

        Args:
            tool_calls: List of tool-call dicts from the LLM response.

        Returns:
            A list of ``ToolMessage`` objects to append to the message chain.
        """
        results: list[ToolMessage] = []

        for call in tool_calls:
            tool_name = call["name"]
            arguments = call["args"]
            call_id = call["id"]

            print(f"[Tool] Calling '{tool_name}' with args={arguments}")

            handler = TOOL_MAP.get(tool_name)
            if handler is None:
                print(f"[Tool] Warning — unknown tool '{tool_name}', skipping.")
                result = {"error": f"Unknown tool: {tool_name}"}
            else:
                try:
                    result = handler(**arguments)
                except Exception as exc:  # pylint: disable=broad-except
                    print(f"[Tool] '{tool_name}' raised an error: {exc}")
                    result = {"error": str(exc)}

            results.append(
                ToolMessage(
                    content=json.dumps(result),
                    tool_call_id=call_id,
                )
            )

        return results

    @staticmethod
    def _normalise_history(history: list[_HistoryItem]) -> list:
        """
        Convert Gradio conversation history to LangChain message objects.

        Handles both Gradio 4.x (list of 2-tuples) and Gradio 5.x
        (list of dicts with "role" / "content" keys).

        Args:
            history: Raw history from ``gr.ChatInterface``.

        Returns:
            A list of ``HumanMessage`` / ``AIMessage`` objects.
        """
        messages = []
        for item in history:
            if isinstance(item, dict):
                role = item.get("role", "")
                content = item.get("content", "")
                if role == "user":
                    messages.append(HumanMessage(content=content))
                elif role == "assistant":
                    messages.append(AIMessage(content=content))
            elif isinstance(item, (list, tuple)) and len(item) == 2:
                user_text, assistant_text = item
                messages.append(HumanMessage(content=user_text))
                messages.append(AIMessage(content=assistant_text))
        return messages
