# Copyright (c) 2025. Claude Code Provider for Microsoft Agent Framework.

"""Request logger for tracking all agent interactions.

Logs every request/response to a JSONL file with timestamps, tokens, costs, etc.
"""

import json
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Any


# Model pricing (USD per 1M tokens) - Updated Dec 2024
MODEL_PRICING = {
    # Sonnet
    "sonnet": {"input": 3.0, "output": 15.0},
    "claude-sonnet-4-20250514": {"input": 3.0, "output": 15.0},
    "claude-3-5-sonnet-20241022": {"input": 3.0, "output": 15.0},
    # Opus
    "opus": {"input": 15.0, "output": 75.0},
    "claude-opus-4-20250514": {"input": 15.0, "output": 75.0},
    # Haiku
    "haiku": {"input": 0.25, "output": 1.25},
    "claude-3-5-haiku-20241022": {"input": 0.25, "output": 1.25},
}


def get_model_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    """Calculate cost for a request.

    Args:
        model: Model name or ID
        input_tokens: Number of input tokens
        output_tokens: Number of output tokens

    Returns:
        Cost in USD
    """
    # Normalize model name
    model_lower = model.lower() if model else "sonnet"

    # Find pricing
    pricing = None
    for key, value in MODEL_PRICING.items():
        if key in model_lower or model_lower in key:
            pricing = value
            break

    if pricing is None:
        # Default to sonnet pricing
        pricing = MODEL_PRICING["sonnet"]

    input_cost = (input_tokens / 1_000_000) * pricing["input"]
    output_cost = (output_tokens / 1_000_000) * pricing["output"]

    return input_cost + output_cost


@dataclass
class RequestLogEntry:
    """A single request/response log entry."""

    # Timing
    timestamp: str
    duration_seconds: float

    # Agent info
    agent_name: str
    model: str

    # Request
    prompt: str
    prompt_tokens: int

    # Response
    response: str
    response_tokens: int
    success: bool
    error: Optional[str]

    # Cost
    cost_usd: float

    # Session
    session_id: Optional[str]

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


class RequestLogger:
    """Logger that writes all requests to a JSONL file.

    Thread-safe for asyncio (single process, single thread with coroutines).

    Example:
        logger = RequestLogger("/path/to/requests.jsonl")

        # Log a request
        logger.log(
            agent_name="researcher",
            model="sonnet",
            prompt="What is 2+2?",
            response="2+2 equals 4.",
            prompt_tokens=10,
            response_tokens=8,
            duration_seconds=1.5,
            success=True,
        )

        # Get summary
        summary = logger.get_summary()
        print(f"Total cost: ${summary['total_cost_usd']:.4f}")
    """

    def __init__(self, log_file: str | Path):
        """Initialize the request logger.

        Args:
            log_file: Path to the JSONL log file. Will be created if doesn't exist.
                     Appends to existing file.
        """
        self._log_file = Path(log_file)
        self._log_file.parent.mkdir(parents=True, exist_ok=True)

        # Accumulated stats
        self._total_requests = 0
        self._total_input_tokens = 0
        self._total_output_tokens = 0
        self._total_cost_usd = 0.0
        self._total_duration = 0.0
        self._by_agent: dict[str, dict] = {}
        self._by_model: dict[str, dict] = {}

    def log(
        self,
        *,
        agent_name: str,
        model: str,
        prompt: str,
        response: str,
        prompt_tokens: int,
        response_tokens: int,
        duration_seconds: float,
        success: bool,
        error: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> RequestLogEntry:
        """Log a request/response.

        Args:
            agent_name: Name of the agent making the request
            model: Model used for the request
            prompt: The prompt sent
            response: The response received
            prompt_tokens: Number of input tokens
            response_tokens: Number of output tokens
            duration_seconds: Time taken for the request
            success: Whether the request succeeded
            error: Error message if failed
            session_id: Optional session ID

        Returns:
            The log entry that was written
        """
        # Calculate cost
        cost = get_model_cost(model, prompt_tokens, response_tokens)

        # Create entry
        entry = RequestLogEntry(
            timestamp=datetime.now(timezone.utc).isoformat(),
            duration_seconds=round(duration_seconds, 3),
            agent_name=agent_name,
            model=model,
            prompt=prompt,
            prompt_tokens=prompt_tokens,
            response=response,
            response_tokens=response_tokens,
            success=success,
            error=error,
            cost_usd=round(cost, 6),
            session_id=session_id,
        )

        # Write to file (append mode)
        with open(self._log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry.to_dict()) + "\n")

        # Update stats
        self._total_requests += 1
        self._total_input_tokens += prompt_tokens
        self._total_output_tokens += response_tokens
        self._total_cost_usd += cost
        self._total_duration += duration_seconds

        # Update by-agent stats
        if agent_name not in self._by_agent:
            self._by_agent[agent_name] = {
                "requests": 0,
                "input_tokens": 0,
                "output_tokens": 0,
                "cost_usd": 0.0,
            }
        self._by_agent[agent_name]["requests"] += 1
        self._by_agent[agent_name]["input_tokens"] += prompt_tokens
        self._by_agent[agent_name]["output_tokens"] += response_tokens
        self._by_agent[agent_name]["cost_usd"] += cost

        # Update by-model stats
        if model not in self._by_model:
            self._by_model[model] = {
                "requests": 0,
                "input_tokens": 0,
                "output_tokens": 0,
                "cost_usd": 0.0,
            }
        self._by_model[model]["requests"] += 1
        self._by_model[model]["input_tokens"] += prompt_tokens
        self._by_model[model]["output_tokens"] += response_tokens
        self._by_model[model]["cost_usd"] += cost

        return entry

    def get_summary(self) -> dict[str, Any]:
        """Get summary of all logged requests.

        Returns:
            Dictionary with aggregated statistics
        """
        return {
            "log_file": str(self._log_file),
            "total_requests": self._total_requests,
            "total_input_tokens": self._total_input_tokens,
            "total_output_tokens": self._total_output_tokens,
            "total_tokens": self._total_input_tokens + self._total_output_tokens,
            "total_cost_usd": round(self._total_cost_usd, 4),
            "total_duration_seconds": round(self._total_duration, 2),
            "by_agent": self._by_agent.copy(),
            "by_model": self._by_model.copy(),
        }

    @property
    def log_file(self) -> Path:
        """Get the log file path."""
        return self._log_file

    @property
    def total_cost(self) -> float:
        """Get total cost in USD."""
        return self._total_cost_usd

    @property
    def total_requests(self) -> int:
        """Get total number of requests."""
        return self._total_requests
