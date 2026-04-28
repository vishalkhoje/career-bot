"""
src/evaluation.py
~~~~~~~~~~~~~~~~~
Production Evaluation System for the Career Bot.

Handles:
- Auto-evaluation (LLM-based scoring for hallucination and relevance)
- Storage of feedback and evaluation metrics in evaluations.db
- Offline dataset evaluation
"""

import sqlite3
import json
import os
from datetime import datetime
from typing import Dict, Any, List, Optional
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

from . import config

class EvaluationSystem:
    def __init__(self, db_path: str = "evaluations.db"):
        self.db_path = db_path
        self._init_db()
        self.llm = ChatOpenAI(
            model=config.OPENAI_CHAT_MODEL,
            api_key=config.OPENAI_API_KEY,
            temperature=0
        )

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS evaluations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT,
                    query TEXT,
                    response TEXT,
                    latency_ms REAL,
                    tokens INTEGER,
                    cache_hit INTEGER,
                    hallucination_score REAL,
                    relevance_score REAL,
                    correctness_score REAL,
                    feedback TEXT, -- 'up', 'down', or NULL
                    metadata TEXT
                )
            """)

    def log_evaluation(self, 
                       query: str, 
                       response: str, 
                       latency_ms: float, 
                       tokens: int, 
                       cache_hit: bool,
                       hallucination_score: float = 0.0,
                       relevance_score: float = 0.0,
                       correctness_score: float = 0.0,
                       metadata: Optional[Dict] = None):
        """
        Store a new evaluation entry.
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO evaluations 
                (timestamp, query, response, latency_ms, tokens, cache_hit, hallucination_score, relevance_score, correctness_score, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                datetime.utcnow().isoformat(),
                query,
                response,
                latency_ms,
                tokens,
                1 if cache_hit else 0,
                hallucination_score,
                relevance_score,
                correctness_score,
                json.dumps(metadata) if metadata else "{}"
            ))
            return conn.execute("SELECT last_insert_rowid()").fetchone()[0]

    def update_feedback(self, eval_id: int, feedback: str):
        """
        Update user feedback (up/down).
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("UPDATE evaluations SET feedback = ? WHERE id = ?", (feedback, eval_id))

    def auto_evaluate(self, query: str, response: str, context: str) -> Dict[str, float]:
        """
        LLM-based auto-evaluation of the response.
        Returns hallucination_score (0-1, lower is better), relevance_score (0-1, higher is better), and correctness (0-1, higher is better).
        """
        prompt = f"""
        You are an AI Evaluation Agent. Grade the following RAG response based on the provided query and context.
        
        QUERY: {query}
        CONTEXT: {context}
        RESPONSE: {response}
        
        Grade these three metrics from 0.0 to 1.0:
        1. Hallucination Risk: (1.0 = contains facts NOT in context, 0.0 = perfectly grounded)
        2. Relevance: (1.0 = perfectly answers the query, 0.0 = irrelevant)
        3. Correctness: (1.0 = factually accurate based on context, 0.0 = completely incorrect)
        
        Response format: JSON only. Example: {{"hallucination": 0.1, "relevance": 0.9, "correctness": 0.8}}
        """
        
        try:
            resp = self.llm.invoke([HumanMessage(content=prompt)]).content
            # Strip markdown code blocks if present
            if "```json" in resp:
                resp = resp.split("```json")[1].split("```")[0].strip()
            elif "```" in resp:
                resp = resp.split("```")[1].split("```")[0].strip()
            
            scores = json.loads(resp)
            return {
                "hallucination": float(scores.get("hallucination", 0.0)),
                "relevance": float(scores.get("relevance", 0.0)),
                "correctness": float(scores.get("correctness", 0.0))
            }
        except Exception as e:
            print(f"[Eval] Auto-evaluation failed to parse: {e} | Raw: {resp[:100]}")
            return {"hallucination": 0.5, "relevance": 0.5, "correctness": 0.5}

    def run_offline_eval(self, dataset: List[Dict]):
        """
        Run evaluation on a fixed dataset.
        """
        results = []
        for item in dataset:
            query = item["query"]
            expected = item["expected"]
            print(f"[Eval] Testing: {query}")
            # This would be called from a script that has access to the agent
            results.append({"query": query, "expected": expected})
        return results
