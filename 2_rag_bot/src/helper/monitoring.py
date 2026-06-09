"""
src/monitoring.py
~~~~~~~~~~~~~~~~~
System-level observability and metrics for the career bot.

Responsibility: Track request latency, token usage, error rates, and 
tool performance. This provides production-level visibility into 
the bot's performance and cost.
"""

import time
import json
import logging
from datetime import datetime
from typing import Dict, Any, Optional, List
from logging.handlers import TimedRotatingFileHandler

# Setup a dedicated logger for production metrics
metrics_logger = logging.getLogger("career_bot_metrics")
metrics_logger.setLevel(logging.INFO)

# File handler for structured JSON logs with daily rotation (cleans up old logs)
fh = TimedRotatingFileHandler(
    "career_bot_metrics.jsonl",
    when="midnight",
    interval=1,
    backupCount=7  # Keep 7 days of history, older logs are automatically deleted
)
metrics_logger.addHandler(fh)

class Observability:
    """
    Singleton-style metrics collector.
    """
    
    @staticmethod
    def calculate_cost(tokens: int, model: str = "gpt-4o-mini") -> float:
        """
        Estimate cost based on token usage.
        Using gpt-4o-mini rates as a baseline (~$0.15 per 1M input tokens).
        """
        rate_per_token = 0.00015 / 1000  # $0.15 per 1M tokens
        if "gpt-4o" in model and "mini" not in model:
            rate_per_token = 0.005 / 1000 # $5.00 per 1M tokens (GPT-4o)
        return tokens * rate_per_token

    @staticmethod
    def get_daily_cost() -> float:
        """
        Calculates the total cost incurred today by parsing the active metrics file.
        Because TimedRotatingFileHandler rotates at midnight, 'career_bot_metrics.jsonl' 
        always contains only today's logs.
        """
        import os
        log_file = "career_bot_metrics.jsonl"
        total_cost = 0.0
        if not os.path.exists(log_file):
            return total_cost
            
        try:
            with open(log_file, "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    try:
                        data = json.loads(line)
                        total_cost += float(data.get("cost_usd", 0.0))
                    except json.JSONDecodeError:
                        continue
        except Exception as e:
            print(f"[Monitoring] Error reading daily cost: {e}")
            
        return total_cost

    @staticmethod
    def check_alerts(event: Dict[str, Any]):
        """
        Check for conditions that should trigger an alert.
        """
        from ..core import config
        latency_threshold = config.LATENCY_ALERT_THRESHOLD
        hallucination_threshold = 0.7
        
        alerts = []
        if event.get("latency_ms", 0) > latency_threshold:
            alerts.append(f"⚠️ SLOW RESPONSE: {event['latency_ms']:.0f}ms")
        
        if event.get("hallucination_score", 0) > hallucination_threshold:
            alerts.append(f"🚨 HIGH HALLUCINATION RISK: {event['hallucination_score']}")
            
        if alerts:
            for alert in alerts:
                print(f"\033[93m[ALERT] {alert}\033[0m")
        return alerts

    @staticmethod
    def log_request(
        query: str,
        latency_ms: float,
        tokens: int = 0,
        status: str = "success",
        error: Optional[str] = None,
        cache_hit: bool = False,
        tools_called: int = 0,
        steps: Optional[List[str]] = None,
        retrieved_docs: Optional[List[str]] = None,
        hallucination_score: float = 0.0,
        iterations: int = 0,
        request_id: Optional[str] = None
    ):
        """
        Record a complete chat request event with deep observability.
        """
        cost = Observability.calculate_cost(tokens)
        
        event = {
            "timestamp": datetime.utcnow().isoformat(),
            "query_preview": query[:100],
            "latency_ms": round(latency_ms, 2),
            "token_usage": tokens,
            "cost_usd": round(cost, 6),
            "status": status,
            "error": error,
            "cache_hit": cache_hit,
            "tools_called": tools_called,
            "steps": steps or [],
            "retrieved_docs_preview": [d[:100] for d in (retrieved_docs or [])],
            "hallucination_score": hallucination_score,
            "iterations": iterations,
            "request_id": request_id
        }
        
        # Log as structured JSON
        metrics_logger.info(json.dumps(event))
        
        # Check for alerts
        Observability.check_alerts(event)
        
        # Print a summary
        color = "\033[92m" if status == "success" else "\033[91m"
        reset = "\033[0m"
        print(f"\n--- {color}Deep Observability Report{reset} ---")
        print(f"Status:    {status.upper()}")
        print(f"Latency:   {latency_ms:.0f}ms")
        print(f"Tokens:    {tokens} (${cost:.6f})")
        print(f"Cache:     {'HIT' if cache_hit else 'MISS'}")
        if steps:
            print(f"Steps:     {' -> '.join(steps)}")
        if error:
            print(f"Error:     {error}")
        print("-" * 30)

    @staticmethod
    def log_tool_execution(tool_name: str, duration_ms: float, success: bool, error: Optional[str] = None, request_id: Optional[str] = None):
        """
        Specific tracking for tool failures and performance.
        """
        event = {
            "timestamp": datetime.utcnow().isoformat(),
            "event_type": "tool_execution",
            "tool_name": tool_name,
            "duration_ms": round(duration_ms, 2),
            "success": success,
            "error": error,
            "request_id": request_id
        }
        metrics_logger.info(json.dumps(event))
        
        if not success:
            print(f"\033[91m[Observability] Tool Failure: {tool_name} | {error}\033[0m")
