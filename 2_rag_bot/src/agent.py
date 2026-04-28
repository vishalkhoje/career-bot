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
        # We use separate instances for streaming vs non-streaming to avoid API errors 
        # related to stream_options in non-streaming calls.
        self.non_streaming_llm = ChatOpenAI(
            model=config.OPENAI_CHAT_MODEL,
            api_key=config.OPENAI_API_KEY,
            temperature=0,
            cache=False,
        ).bind_tools(TOOL_SCHEMAS)

        self.llm = ChatOpenAI(
            model=config.OPENAI_CHAT_MODEL,
            api_key=config.OPENAI_API_KEY,
            temperature=0,
            cache=False,
            model_kwargs={"stream_options": {"include_usage": True}}
        ).bind_tools(TOOL_SCHEMAS)

    # ── Public interface ────────────────────────────────────────────────────────

    def chat(self, message: str, history: list[_HistoryItem]):
        """
        Process a user message using a Multi-Agent Reasoning Workflow.
        """
        from .monitoring import Observability
        from .prompt import (
            build_intent_classifier_prompt,
            build_planner_prompt,
            build_critic_prompt,
            build_system_prompt
        )
        
        # Convert history to LangChain message format for potential use
        chat_history = []
        for item in history:
            if isinstance(item, dict):
                role = item.get("role")
                content = item.get("content")
                if role == "user":
                    chat_history.append(HumanMessage(content=content))
                elif role == "assistant":
                    chat_history.append(AIMessage(content=content))
            elif isinstance(item, (list, tuple)) and len(item) == 2:
                # Gradio 4 format
                chat_history.append(HumanMessage(content=item[0]))
                chat_history.append(AIMessage(content=item[1]))
        start_time = time.time()
        message = message.strip()
        tokens_used = 0
        cache_hit = False
        error_msg = None
        status = "success"

        try:
            # ── Immediate Feedback ─────────────────────────────────────────────
            yield "🔍 *Analyzing your question...*"

            # ── Step 0: Expire stale cache ─────────────────────────────────────
            self.cache.check_expiry()

            # ── Step 1: Cache Check ────────────────────────────────────────────
            print(f"\n[Step 1/5: Cache Check] query='{message[:50]}...'")
            cache_key = self.cache.make_key(message)
            cached = self.cache.get(cache_key)
            if cached:
                print("[Cache] ⚡ HIT — serving from SQLite")
                cache_hit = True
                yield cached
                return

            # ── Step 2: Intent Classification ──────────────────────────────────
            if config.USE_AGENT_WORKFLOW:
                print(f"[Step 2/5: Intent Classifier] Analyzing query type...")
                intent_prompt = build_intent_classifier_prompt(message)
                intent_resp = self.non_streaming_llm.invoke([HumanMessage(content=intent_prompt)]).content.strip()
                intent = "ANALYTICAL" if "ANALYTICAL" in intent_resp.upper() else "FACTUAL"
                print(f"[Agent] Classified as: {intent}")
                yield f"🧠 *Thought: This is an {intent.lower()} query. Searching my memory...*"
            else:
                intent = "FACTUAL"

            # ── Step 3: Retrieval ──────────────────────────────────────────────
            print(f"[Step 3/5: Retriever] Fetching relevant context from Pinecone...")
            context = self.retriever.retrieve(message)
            yield f"📚 *Context found. Finalizing reasoning...*"

            # ── Step 4: Analytical Branch (Reasoning) ─────────────────────────
            if intent == "ANALYTICAL" and config.USE_AGENT_WORKFLOW:
                print("[Step 4/5: Planner Agent] Breaking down the reasoning...")
                yield f"📋 *Strategy: Creating a multi-step plan to answer your complex query...*"
                
                planner_prompt = build_planner_prompt(message, context)
                plan = self.non_streaming_llm.invoke([HumanMessage(content=planner_prompt)]).content
                print(f"[Agent] Strategic Plan: {plan.replace(chr(10), ' | ')}")

                yield f"🎯 *Plan Ready: {plan.split(chr(10))[0]}... Generating detailed answer...*\n\n"
                
                # Reasoning phase
                full_response = ""
                reasoning_messages = [
                    SystemMessage(content=build_system_prompt(context)),
                    HumanMessage(content=f"Original Query: {message}\n\nStrategic Plan: {plan}\n\nPlease execute the plan and provide a comprehensive, reasoned answer.")
                ]
                
                with get_openai_callback() as cb:
                    for chunk in self._invoke_llm_stream(reasoning_messages):
                        full_response += chunk
                        yield full_response
                    tokens_used += cb.total_tokens

                # ── Step 5: Critic Review ─────────────────────────────────────
                if config.USE_CRITIC_AGENT:
                    print("[Step 5/5: Critic Agent] Verifying answer quality...")
                    critic_prompt = build_critic_prompt(message, full_response, context)
                    critic_resp = self.non_streaming_llm.invoke([HumanMessage(content=critic_prompt)]).content
                    
                    if "APPROVED" in critic_resp.upper():
                        print("[Agent] ✅ Critic Status: APPROVED")
                    else:
                        print(f"[Agent] ⚠️ Critic Suggestion: {critic_resp[:100]}...")
            
            else:
                # Standard RAG flow (FACTUAL)
                print("[Step 4/5: LLM Generation] Generating standard RAG response...")
                messages = [SystemMessage(content=build_system_prompt(context))]
                messages.extend(self._normalise_history(history))
                messages.append(HumanMessage(content=message))

                full_response = ""
                with get_openai_callback() as cb:
                    for chunk in self._invoke_llm_stream(messages):
                        full_response += chunk
                        yield full_response
                    tokens_used += cb.total_tokens
                
                print("[Step 5/5: Post-Processing] Finalizing response...")

            # ── Final: Cache Storage ──────────────────────────────────────────
            if full_response:
                self.cache.set(cache_key, full_response)

        except Exception as exc:
            status = "error"
            error_msg = str(exc)
            print(f"[Agent] Critical Error: {exc}")
            yield f"I'm sorry, I encountered an error: {exc}"
        finally:
            latency_ms = (time.time() - start_time) * 1000
            
            # ── Observability ──────────────────────────────────────────────────
            Observability.log_request(
                query=message,
                latency_ms=latency_ms,
                tokens=tokens_used,
                status=status,
                error=error_msg,
                cache_hit=cache_hit
            )

            # ── Evaluation ─────────────────────────────────────────────────────
            if not cache_hit and status == "success":
                from .evaluation import EvaluationSystem
                eval_sys = EvaluationSystem()
                
                # Auto-evaluate (hallucination & relevance)
                scores = eval_sys.auto_evaluate(message, full_response, context)
                
                # Persist to evaluations.db
                eval_id = eval_sys.log_evaluation(
                    query=message,
                    response=full_response,
                    latency_ms=latency_ms,
                    tokens=tokens_used,
                    cache_hit=cache_hit,
                    hallucination_score=scores["hallucination"],
                    relevance_score=scores["relevance"],
                    correctness_score=scores.get("correctness", 0.0),
                    metadata={"intent": intent}
                )
                print(f"[Eval] Entry saved: ID={eval_id} | Hallucination={scores['hallucination']} | Relevance={scores['relevance']} | Correctness={scores.get('correctness', 0.0)}")

    # ── Private helpers ─────────────────────────────────────────────────────────

    def _invoke_llm_stream(self, messages: list):
        """
        Invoke the LLM with streaming support, handling any tool calls.
        """
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
            else:
                done = True

    def _dispatch_tool_calls(self, tool_calls: list) -> list[ToolMessage]:
        """
        Execute each requested tool and return ToolMessage results.
        """
        from .monitoring import Observability
        results: list[ToolMessage] = []

        for call in tool_calls:
            tool_name = call["name"]
            arguments = call["args"]
            call_id = call["id"]

            print(f"[Tool] Calling '{tool_name}' with args={arguments}")
            
            tool_start = time.time()
            success = True
            error_msg = None
            
            handler = TOOL_MAP.get(tool_name)
            if handler is None:
                print(f"[Tool] Warning — unknown tool '{tool_name}', skipping.")
                result = {"error": f"Unknown tool: {tool_name}"}
                success = False
                error_msg = f"Unknown tool: {tool_name}"
            else:
                try:
                    result = handler(**arguments)
                except Exception as exc:
                    print(f"[Tool] '{tool_name}' raised an error: {exc}")
                    result = {"error": str(exc)}
                    success = False
                    error_msg = str(exc)

            tool_duration = (time.time() - tool_start) * 1000
            Observability.log_tool_execution(
                tool_name=tool_name,
                duration_ms=tool_duration,
                success=success,
                error=error_msg
            )

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
