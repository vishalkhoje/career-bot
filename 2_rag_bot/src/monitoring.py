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
from typing import Dict, Any, Optional

# Setup a dedicated logger for production metrics
metrics_logger = logging.getLogger("career_bot_metrics")
metrics_logger.setLevel(logging.INFO)

# File handler for structured JSON logs (good for Datadog/ELK)
fh = logging.FileHandler("career_bot_metrics.jsonl")
metrics_logger.addHandler(fh)

class Observability:
    """
    Singleton-style metrics collector.
    """
    
    @staticmethod
    def log_request(
        query: str,
        latency_ms: float,
        tokens: int = 0,
        status: str = "success",
        error: Optional[str] = None,
        cache_hit: bool = False,
        tools_called: int = 0
    ):
        """
        Record a complete chat request event.
        """
        event = {
            "timestamp": datetime.utcnow().isoformat(),
            "query_preview": query[:50],
            "latency_ms": round(latency_ms, 2),
            "token_usage": tokens,
            "status": status,
            "error": error,
            "cache_hit": cache_hit,
            "tools_called": tools_called
        }
        
        # Log as structured JSON for easy parsing
        metrics_logger.info(json.dumps(event))
        
        # Print a summary to console for immediate visibility
        color = "\033[92m" if status == "success" else "\033[91m"
        reset = "\033[0m"
        print(f"\n--- {color}Observability Report{reset} ---")
        print(f"Status:    {status.upper()}")
        print(f"Latency:   {latency_ms:.0f}ms")
        print(f"Tokens:    {tokens}")
        print(f"Cache:     {'HIT' if cache_hit else 'MISS'}")
        if error:
            print(f"Error:     {error}")
        print("-" * 30)

    @staticmethod
    def log_tool_execution(tool_name: str, duration_ms: float, success: bool, error: Optional[str] = None):
        """
        Specific tracking for tool failures and performance.
        """
        event = {
            "timestamp": datetime.utcnow().isoformat(),
            "event_type": "tool_execution",
            "tool_name": tool_name,
            "duration_ms": round(duration_ms, 2),
            "success": success,
            "error": error
        }
        metrics_logger.info(json.dumps(event))
        
        if not success:
            print(f"\033[91m[Observability] Tool Failure: {tool_name} | {error}\033[0m")
