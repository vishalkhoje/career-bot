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

from ..core import config

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
            # Main evaluations table
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
                    groundedness_score REAL,
                    feedback TEXT,
                    failure_reason TEXT,
                    gold_response TEXT,
                    metadata TEXT
                )
            """)
            
            # New benchmarks table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS benchmarks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT,
                    version TEXT,
                    avg_groundedness REAL,
                    avg_relevance REAL,
                    avg_hallucination REAL,
                    total_queries INTEGER,
                    report_json TEXT
                )
            """)
            
            # Migration: Add new columns if they don't exist
            columns = [
                ("correctness_score", "REAL DEFAULT 0.0"),
                ("groundedness_score", "REAL DEFAULT 0.0"),
                ("failure_reason", "TEXT"),
                ("gold_response", "TEXT")
            ]
            for col_name, col_type in columns:
                try:
                    conn.execute(f"ALTER TABLE evaluations ADD COLUMN {col_name} {col_type}")
                except sqlite3.OperationalError:
                    pass

    def log_evaluation(self, 
                       query: str, 
                       response: str, 
                       latency_ms: float, 
                       tokens: int, 
                       cache_hit: bool,
                       hallucination_score: float = 0.0,
                       relevance_score: float = 0.0,
                       correctness_score: float = 0.0,
                       groundedness_score: float = 0.0,
                       metadata: Optional[Dict] = None):
        """
        Store a new evaluation entry.
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO evaluations 
                (timestamp, query, response, latency_ms, tokens, cache_hit, hallucination_score, relevance_score, correctness_score, groundedness_score, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                groundedness_score,
                json.dumps(metadata) if metadata else "{}"
            ))
            return conn.execute("SELECT last_insert_rowid()").fetchone()[0]

    def log_benchmark(self, version: str, results: Dict):
        """
        Save a benchmark run summary.
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO benchmarks (timestamp, version, avg_groundedness, avg_relevance, avg_hallucination, total_queries, report_json)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                datetime.utcnow().isoformat(),
                version,
                results.get("avg_groundedness", 0),
                results.get("avg_relevance", 0),
                results.get("avg_hallucination", 0),
                results.get("total_queries", 0),
                json.dumps(results)
            ))

    def update_feedback(self, eval_id: int, feedback: str):
        """
        Update user feedback (up/down).
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("UPDATE evaluations SET feedback = ? WHERE id = ?", (feedback, eval_id))

    def analyze_and_improve(self, eval_id: int):
        """
        Failure analysis for negative feedback.
        Generates a gold standard response and identifies failure reasons.
        """
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute("SELECT query, response, metadata FROM evaluations WHERE id = ?", (eval_id,)).fetchone()
            if not row:
                return
            
            query, bad_response, metadata_json = row
            metadata = json.loads(metadata_json)
            context = metadata.get("context_preview", "No context provided.")

        prompt = f"""
        You are a Senior RAG Architect. A user gave a Thumbs Down (👎) to the following response.
        Your goal is to analyze the failure and provide a 'Gold Standard' response that we can use for future learning.

        QUERY: {query}
        CONTEXT: {context}
        BAD RESPONSE: {bad_response}

        Tasks:
        1. Identify the primary reason for failure (e.g., Missing Information, Hallucination, Tone, Irrelevant).
        2. Write a perfect, factually correct 'Gold Standard' response based ONLY on the context.

        Response format: JSON only.
        Example: {{"reason": "Hallucination", "gold_response": "The correct answer based on context is..."}}
        """

        try:
            resp = self.llm.invoke([HumanMessage(content=prompt)]).content
            if "```json" in resp:
                resp = resp.split("```json")[1].split("```")[0].strip()
            elif "```" in resp:
                resp = resp.split("```")[1].split("```")[0].strip()
            
            analysis = json.loads(resp)
            reason = analysis.get("reason", "Unknown")
            gold = analysis.get("gold_response", "")

            if gold:
                with sqlite3.connect(self.db_path) as conn:
                    conn.execute(
                        "UPDATE evaluations SET failure_reason = ?, gold_response = ? WHERE id = ?",
                        (reason, gold, eval_id)
                    )
                print(f"[Self-Improvement] Analyzed failure for ID {eval_id}: {reason}")
        except Exception as e:
            print(f"[Self-Improvement] Failed to analyze failure: {e}")

    def auto_evaluate(self, query: str, response: str, context: str) -> Dict[str, float]:
        """
        Rigorous LLM-based auto-evaluation using Chain-of-Thought.
        """
        prompt = f"""
        You are a Quality Assurance Agent for a RAG-based AI system.
        Your task is to grade the AI's response based on the provided context and the user query.

        QUERY: {query}
        CONTEXT: {context}
        RESPONSE: {response}

        Grade the response on the following criteria (0.0 to 1.0):
        1. GROUNDEDNESS: Is every claim in the response supported by the context? (1.0 = fully grounded, 0.0 = completely made up)
        2. RELEVANCE: Does the response directly address the user's query? (1.0 = perfectly relevant, 0.0 = off-topic)
        3. HALLUCINATION: Does the response contain info NOT in the context? (1.0 = extreme hallucination, 0.0 = zero hallucination)

        Evaluation Process:
        - Step 1: Fact-check the response against the context. List any claims not found in context.
        - Step 2: Check if the response answers the specific question asked.
        - Step 3: Assign scores.

        Return ONLY a JSON object:
        {{"thought": "...", "groundedness": 0.0, "relevance": 0.0, "hallucination": 0.0, "correctness": 0.0}}
        """
        
        try:
            resp = self.llm.invoke([HumanMessage(content=prompt)]).content
            if "```json" in resp:
                resp = resp.split("```json")[1].split("```")[0].strip()
            elif "```" in resp:
                resp = resp.split("```")[1].split("```")[0].strip()
            
            scores = json.loads(resp)
            # Ensure all keys exist and are floats
            result = {}
            for key in ["hallucination", "relevance", "correctness", "groundedness"]:
                result[key] = float(scores.get(key, 0.5))
            return result
        except Exception as e:
            print(f"[Evaluation] Auto-eval failed: {e}")
            return {"hallucination": 0.5, "relevance": 0.5, "correctness": 0.5, "groundedness": 0.5}

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
