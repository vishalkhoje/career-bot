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

import os
import json
import time
import threading
import uuid
from typing import List, Dict, Any, Optional, Union, Generator
from langchain_community.callbacks.manager import get_openai_callback
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_openai import ChatOpenAI

from ..core import config
from ..database.cache import ResponseCache
from .prompts import build_system_prompt
from ..core.retriever import CareerRetriever
from ..tools.registry import TOOL_MAP, TOOL_SCHEMAS

# Conversation history item can be dict (Gradio 5) or 2-tuple (Gradio 4)
_HistoryItem = Union[dict, tuple]


class FeedbackLearner:
    """
    Retrieves past corrected failures to provide few-shot demonstrations.
    """
    def __init__(self, db_path: str = "evaluations.db"):
        self.db_path = db_path

    def get_relevant_corrections(self, query: str, limit: int = 3) -> Optional[str]:
        """
        Fetch recently corrected failures from the database.
        In a production system, this would use semantic search. 
        Here we fetch the most recent unique gold standards.
        """
        import sqlite3
        if not os.path.exists(self.db_path):
            return None
            
        try:
            with sqlite3.connect(self.db_path) as conn:
                # We look for records that have a gold_response
                cursor = conn.execute("""
                    SELECT query, gold_response, failure_reason 
                    FROM evaluations 
                    WHERE gold_response IS NOT NULL AND gold_response != ''
                    ORDER BY id DESC LIMIT ?
                """, (limit,))
                rows = cursor.fetchall()
                
                if not rows:
                    return None
                
                examples = []
                for q, gold, reason in rows:
                    examples.append(f"Query: {q}\nError Type: {reason}\nCorrection: {gold}\n---")
                
                return "\n".join(examples)
        except Exception as e:
            print(f"[FeedbackLearner] Failed to fetch corrections: {e}")
            return None


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
        
        # Learner to pull from evaluations.db
        self.learner = FeedbackLearner(db_path="evaluations.db")

        # LangChain chat model with tools bound.
        # We use separate instances for streaming vs non-streaming to avoid API errors 
        # related to stream_options in non-streaming calls.
        self.non_streaming_llm = ChatOpenAI(
            model=config.OPENAI_CHAT_MODEL,
            api_key=config.OPENAI_API_KEY,
            temperature=0,
            cache=False,
        )

        self.llm = ChatOpenAI(
            model=config.OPENAI_CHAT_MODEL,
            api_key=config.OPENAI_API_KEY,
            temperature=0,
            cache=False,
            streaming=True,
            model_kwargs={"stream_options": {"include_usage": True}}
        ).bind_tools(TOOL_SCHEMAS)

    # ── Public interface ────────────────────────────────────────────────────────

    def chat(self, message: str, history: list[_HistoryItem]):
        """
        Process a user message using a Multi-Agent Reasoning Workflow.
        """
        from ..helper.monitoring import Observability
        from .prompts import (
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
        tokens_by_agent = {}
        def _track_tokens(agent_name, callback):
            tokens_by_agent[agent_name] = {
                "prompt": callback.prompt_tokens,
                "completion": callback.completion_tokens,
                "total": callback.total_tokens,
                "cost": callback.total_cost
            }

        start_time = time.time()
        request_id = str(uuid.uuid4())
        message = message.strip()
        tokens_used = 0
        cache_hit = False
        error_msg = None
        status = "success"
        execution_steps = []
        retrieved_contexts = []
        eval_scores = None # Will store scores from Critic if available
        latency_breakdown = {}
        tool_calls_counts = []
        iterations_counts = []
        full_response = ""
        context = ""
        intent = "UNKNOWN"

        from ..helper.utils import retry_with_backoff, safe_llm_call

        try:
            # ── Immediate Feedback ─────────────────────────────────────────────
            yield "🔍 *Analyzing your question...*"

            # ── Step 0.5: Enforce Daily Cost Limit ─────────────────────────────
            current_cost = Observability.get_daily_cost()
            if current_cost >= config.DAILY_COST_LIMIT:
                print(f"[Agent] 🚨 Daily cost limit reached (${current_cost:.2f} / ${config.DAILY_COST_LIMIT:.2f})")
                yield f"I've reached my maximum daily limit for answering questions. Please reach out to me directly at **{config.CONTACT_EMAIL}**!"
                status = "rejected"
                return

            # ── Step 0: Expire stale cache ─────────────────────────────────────
            self.cache.check_expiry()

            # ── Step 1: Cache Check ────────────────────────────────────────────
            print(f"\n[Step 1/5: Cache Check] query='{message[:50]}...'")
            cached, query_emb = self.cache.get_semantic(message, threshold=0.85)
            if cached:
                print("[Cache] ⚡ HIT — serving from SQLite")
                cache_hit = True
                execution_steps.append("Cache Hit")
                yield cached
                return
            
            execution_steps.append("Cache Miss")

            # ── Steps 2+3: Intent Classification + Retrieval (PARALLEL) ──────────
            # These two operations are independent — run them concurrently
            from concurrent.futures import ThreadPoolExecutor, as_completed
            
            step_start = time.time()
            yield "🔍 *Classifying your query and searching my memory simultaneously...*"
            
            def _classify_intent():
                """LLM-based intent classification."""
                try:
                    from .prompts import build_intent_classifier_prompt
                    intent_prompt = build_intent_classifier_prompt(message)
                    with get_openai_callback() as cb:
                        resp = self.non_streaming_llm.invoke([HumanMessage(content=intent_prompt)], max_tokens=10).content.strip()
                        _track_tokens("Intent Classifier", cb)
                    print(f"[Tokens] Intent Classifier: {cb.total_tokens} tokens")
                    resp_upper = resp.upper()
                    if "ANALYTICAL" in resp_upper:
                        return "ANALYTICAL"
                    elif "GENERIC" in resp_upper:
                        return "GENERIC"
                    return "FACTUAL"
                except Exception as e:
                    print(f"[Fallback] Intent classification failed: {e}. Defaulting to FACTUAL.")
                    return "FACTUAL"
            
            def _retrieve_context():
                """Pinecone retrieval."""
                try:
                    return self.retriever.retrieve(message)
                except Exception as e:
                    print(f"[Fallback] Retrieval failed: {e}. Using empty context.")
                    return ""
            
            if config.USE_AGENT_WORKFLOW:
                print(f"[Steps 2+3: PARALLEL] Intent Classifier + Retriever + Learner launching...")
                
                with ThreadPoolExecutor(max_workers=3) as executor:
                    intent_future = executor.submit(_classify_intent)
                    retrieval_future = executor.submit(_retrieve_context)
                    learner_future = executor.submit(self.learner.get_relevant_corrections, message)
                    
                    intent = intent_future.result()
                    context = retrieval_future.result()
                    learned_examples = learner_future.result()
                
                parallel_ms = (time.time() - step_start) * 1000
                latency_breakdown['intent+retrieval+learner_ms (parallel)'] = parallel_ms
                print(f"[Agent] Classified as: {intent} | Retrieval done | Parallel time: {parallel_ms:.0f}ms")
                
                execution_steps.append(f"Intent: {intent}")
                execution_steps.append("Retrieval")
                yield f"🧠 *Thought: This is an {intent.lower()} query. Context found!*"
            else:
                intent = "FACTUAL"
                execution_steps.append("Intent: FACTUAL (Default)")
                context = _retrieve_context()
                learned_examples = self.learner.get_relevant_corrections(message)
                latency_breakdown['retrieval_ms'] = (time.time() - step_start) * 1000
                execution_steps.append("Retrieval")
            
            retrieved_contexts = [context] if context else []
            yield f"📚 *Finalizing reasoning...*"

            # ── Step 4: Analytical Branch (Reasoning with Retry) ────────────────
            if intent == "ANALYTICAL" and config.USE_AGENT_WORKFLOW:
                step_start = time.time()
                print("[Step 4/5: Planner Agent] Breaking down the reasoning...")
                execution_steps.append("Planning")
                yield f"📋 *Strategy: Creating a multi-step plan to answer your complex query...*\n\n> "
                
                plan = ""
                try:
                    planner_prompt = build_planner_prompt(message, context)
                    with get_openai_callback() as cb:
                        for chunk in self.llm.stream([HumanMessage(content=planner_prompt)]):
                            if chunk.content:
                                plan += chunk.content
                                yield chunk.content
                        _track_tokens("Planner Agent", cb)
                    print(f"[Tokens] Planner Agent: {cb.total_tokens} tokens")
                    yield f"\n\n🎯 *Generating detailed answer based on plan...*\n\n"
                except Exception:
                    print("[Fallback] Planning failed. Falling back to simple RAG response.")
                    intent = "FACTUAL"
                latency_breakdown['planner_ms'] = (time.time() - step_start) * 1000
                
                if intent == "ANALYTICAL":
                    
                    full_response = ""
                    history_messages = self._normalise_history(history)
                    
                    reasoning_messages = [
                        SystemMessage(content=build_system_prompt(context, learned_examples=learned_examples)),
                    ]
                    reasoning_messages.extend(history_messages)
                    reasoning_messages.append(
                        HumanMessage(content=f"Original Query: {message}\n\nStrategic Plan: {plan}\n\nPlease execute the plan and provide a comprehensive, reasoned answer.")
                    )
                    
                    step_start = time.time()
                    try:
                        with get_openai_callback() as cb:
                            for chunk in self._invoke_llm_stream(reasoning_messages, tool_calls_sink=tool_calls_counts, iterations_sink=iterations_counts, request_id=request_id):
                                full_response += chunk
                                yield full_response
                            _track_tokens("Reasoning Agent", cb)
                            tokens_used += cb.total_tokens
                        print(f"[Tokens] Reasoning Agent: {cb.total_tokens} tokens")
                        execution_steps.append("LLM Reasoning")
                    except Exception as e:
                        print(f"[Fallback] Reasoning stream failed: {e}. Falling back to simple RAG.")
                        intent = "FACTUAL"
                    latency_breakdown['llm_reasoning_ms'] = (time.time() - step_start) * 1000

                # ── Step 5: Critic Review (CONDITIONAL — only for long context) ──
                context_len = len(context) if context else 0
                if intent == "ANALYTICAL" and config.USE_CRITIC_AGENT and context_len > 3000:
                    step_start = time.time()
                    print("[Step 5/5: Critic Agent] Verifying answer quality (context is large)...")
                    
                    @retry_with_backoff(retries=1)
                    def run_critic():
                        # Only pass first 2000 chars of context to Critic to reduce prompt size
                        critic_prompt = build_critic_prompt(message, full_response, context[:2000])
                        with get_openai_callback() as cb:
                            res = self.non_streaming_llm.invoke([HumanMessage(content=critic_prompt)]).content
                            _track_tokens("Critic Agent", cb)
                        print(f"[Tokens] Critic Agent: {cb.total_tokens} tokens")
                        return res
                    
                    try:
                        critic_resp = run_critic()
                        # Parse JSON response
                        try:
                            import json
                            if "```json" in critic_resp:
                                critic_resp = critic_resp.split("```json")[1].split("```")[0].strip()
                            elif "```" in critic_resp:
                                critic_resp = critic_resp.split("```")[1].split("```")[0].strip()
                            
                            critic_data = json.loads(critic_resp)
                            eval_scores = critic_data.get("scores")
                            c_status = critic_data.get("status", "APPROVED")
                            
                            if c_status == "APPROVED":
                                print(f"[Agent] ✅ Critic Status: APPROVED (G={eval_scores.get('groundedness')})")
                            else:
                                print(f"[Agent] ⚠️ Critic Suggestion: {critic_data.get('feedback')[:100]}...")
                        except Exception as parse_err:
                            print(f"[Agent] Critic JSON parse error: {parse_err}")
                            if "APPROVED" in critic_resp.upper():
                                print("[Agent] ✅ Critic Status: APPROVED (Text fallback)")
                        
                        execution_steps.append("Critic Review")
                    except Exception:
                        print("[Fallback] Critic review failed. Proceeding with unverified answer.")
                    latency_breakdown['critic_ms'] = (time.time() - step_start) * 1000
                elif intent == "ANALYTICAL":
                    print("[Step 5/5: Critic Agent] ⏭ Skipped (context is short, low risk)")
            
            # ── Generic Branch: Fast Rejection ────────────────────────────────
            if intent == "GENERIC":
                step_start = time.time()
                print("[Step 4/5: Generic Agent] Rejecting generic query...")
                generic_response = "I'm sorry, but I am an AI assistant specifically built to answer questions about Vishal's career profile and professional background. I cannot answer generic queries, write code, or engage in unrelated discussions."
                yield generic_response
                full_response = generic_response
                execution_steps.append("Generic Agent")
                latency_breakdown['generic_ms'] = (time.time() - step_start) * 1000

            # ── Fallback Branch: Standard RAG flow ────────────────────────────
            elif intent == "FACTUAL":
                step_start = time.time()
                print("[Step 4/5: LLM Generation] Generating standard RAG response...")
                messages = [SystemMessage(content=build_system_prompt(context, learned_examples=learned_examples))]
                messages.extend(self._normalise_history(history))
                messages.append(HumanMessage(content=message))

                full_response = ""
                with get_openai_callback() as cb:
                    try:
                        for chunk in self._invoke_llm_stream(messages, tool_calls_sink=tool_calls_counts, iterations_sink=iterations_counts, request_id=request_id):
                            full_response += chunk
                            yield full_response
                        _track_tokens("Generation Agent", cb)
                        tokens_used += cb.total_tokens
                    except Exception as e:
                        print(f"[Critical Fallback] Final LLM Stream failed: {e}")
                        yield "I apologize, but I am having trouble connecting to my service right now. Please try again in a few seconds."
                        return
                print(f"[Tokens] Generation Agent: {cb.total_tokens} tokens")
                latency_breakdown['llm_generation_ms'] = (time.time() - step_start) * 1000
                
                execution_steps.append("LLM Generation")
                print("[Step 5/5: Post-Processing] Finalizing response...")

            # ── Final: Cache Storage ──────────────────────────────────────────
            if full_response:
                self.cache.set_semantic(message, full_response, query_emb)

        except Exception as exc:
            status = "error"
            error_msg = str(exc)
            print(f"[Agent] Critical Error: {exc}")
            yield "I'm sorry, I'm currently unable to process your request. This might be a temporary connection issue. Please try again shortly."
        finally:
            latency_ms = (time.time() - start_time) * 1000
            
            # ── Latency Breakdown ──────────────────────────────────────────────
            print(f"\n--- ⏱️ Latency Breakdown ---")
            for component, ms in latency_breakdown.items():
                print(f"  {component}: {ms:.0f}ms")
            print(f"  TOTAL (user-facing): {latency_ms:.0f}ms")
            print(f"----------------------------")
            
            print(f"\n--- 💰 Token & Cost Summary ---")
            total_cost = 0
            for agent, stats in tokens_by_agent.items():
                print(f"  {agent:18}: {stats['total']:4} tokens (${stats['cost']:.6f})")
                total_cost += stats['cost']
                
            # Recalculate tokens_used to include the entire workflow's tokens
            tokens_used = sum([stats['total'] for stats in tokens_by_agent.values()])
            
            print(f"  {'TOTAL':18}: {tokens_used:4} tokens (${total_cost:.6f})")
            print(f"------------------------------")

            if total_cost > config.COST_ALERT_THRESHOLD:
                print(f"\n[ALERT] 💰 HIGH COST DETECTED: This request cost ${total_cost:.4f} (Threshold: ${config.COST_ALERT_THRESHOLD})")
                print(f"Consider reducing context size or disabling the Critic Agent for simple queries.")

            # ── Observability (log immediately, WITHOUT auto-eval) ────────────
            total_tool_calls = sum(tool_calls_counts)
            Observability.log_request(
                query=message,
                latency_ms=latency_ms,
                tokens=tokens_used,
                status=status,
                error=error_msg,
                cache_hit=cache_hit,
                steps=execution_steps,
                retrieved_docs=retrieved_contexts,
                tools_called=total_tool_calls,
                hallucination_score=0.0,
                iterations=sum(iterations_counts),
                request_id=request_id
            )

            # ── Background Evaluation (ASYNC — does NOT block the user) ───────
            if not cache_hit and status == "success":
                def _background_eval(q, resp, ctx, lat, tok, intent_val, existing_scores, req_id):
                    try:
                        from ..database.evaluation import EvaluationSystem
                        eval_sys = EvaluationSystem()
                        
                        if existing_scores:
                            print("[Eval] ⏭ Skipping Auto-Eval LLM call (using Critic scores)")
                            scores = existing_scores
                        else:
                            # Re-run auto-eval if critic didn't run
                            scores = eval_sys.auto_evaluate(q, resp, ctx)
                        
                        eval_id = eval_sys.log_evaluation(
                            query=q,
                            response=resp,
                            latency_ms=lat,
                            tokens=tok,
                            cache_hit=False,
                            hallucination_score=scores.get("hallucination", 0.0),
                            relevance_score=scores.get("relevance", 0.0),
                            correctness_score=scores.get("correctness", 0.0),
                            groundedness_score=scores.get("groundedness", 0.0),
                            metadata={"intent": intent_val, "request_id": req_id}
                        )
                        print(f"[Eval] ✅ Background log complete: ID={eval_id} | G={scores.get('groundedness',0):.2f} | R={scores.get('relevance',0):.2f}")
                    except Exception as e:
                        print(f"[Eval] Background eval failed: {e}")
                
                thread = threading.Thread(
                    target=_background_eval,
                    args=(message, full_response, context, latency_ms, tokens_used, intent, eval_scores, request_id),
                    daemon=True
                )
                thread.start()
                print("[Eval] 🔄 Background evaluation task queued.")

    # ── Private helpers ─────────────────────────────────────────────────────────

    def _invoke_llm_stream(self, messages: list, tool_calls_sink: list = None, iterations_sink: list = None, request_id: str = None):
        """
        Invoke the LLM with streaming support, handling any tool calls.
        
        Args:
            messages: List of LangChain messages.
            tool_calls_sink: Optional list to append tool call counts to.
        """
        done = False
        iterations = 0
        max_iterations = 3
        
        while not done and iterations < max_iterations:
            iterations += 1
            if iterations_sink is not None:
                iterations_sink.append(1)
            
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
                if tool_calls_sink is not None:
                    # Each iteration that has tool_calls counts as 1 "tool call event"
                    # or we can count the actual number of tools in full_message.tool_calls
                    tool_calls_sink.append(len(full_message.tool_calls))
                
                tool_results = self._dispatch_tool_calls(full_message.tool_calls, request_id=request_id)
                messages.extend(tool_results)
            else:
                done = True

    def _dispatch_tool_calls(self, tool_calls: list, request_id: str = None) -> list[ToolMessage]:
        """
        Execute each requested tool and return ToolMessage results.
        """
        from ..helper.monitoring import Observability
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
                error=error_msg,
                request_id=request_id
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
        # Optimization: Only process the last N turns to keep context window small
        # In Gradio 5 (dicts), 1 turn = 2 messages. In Gradio 4 (tuples), 1 turn = 1 tuple.
        is_dict_format = len(history) > 0 and isinstance(history[0], dict)
        slice_size = config.MAX_MEMORY_TURNS * 2 if is_dict_format else config.MAX_MEMORY_TURNS
        relevant_history = history[-slice_size:]
        
        messages = []
        for item in relevant_history:
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
