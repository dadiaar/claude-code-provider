# Copyright (c) 2025. Claude Code Provider for Microsoft Agent Framework.

"""Cost tracking for Claude Code usage."""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger("claude_code_provider")


# Pricing per million tokens (as of December 2025)
# https://platform.claude.com/docs/en/about-claude/pricing
MODEL_PRICING: dict[str, dict[str, float]] = {
    # Claude 4.5 models (latest)
    "claude-opus-4-5-20251101": {"input": 5.00, "output": 25.00},
    "claude-opus-4.5": {"input": 5.00, "output": 25.00},
    "opus": {"input": 5.00, "output": 25.00},  # Default to 4.5 pricing
    "claude-sonnet-4-5-20241022": {"input": 3.00, "output": 15.00},
    "claude-sonnet-4.5": {"input": 3.00, "output": 15.00},
    "sonnet": {"input": 3.00, "output": 15.00},
    "claude-haiku-4-5-20241022": {"input": 1.00, "output": 5.00},
    "claude-haiku-4.5": {"input": 1.00, "output": 5.00},
    "haiku": {"input": 1.00, "output": 5.00},  # Default to 4.5 pricing
    # Claude 4.x models
    "claude-opus-4-20250514": {"input": 15.00, "output": 75.00},
    "claude-opus-4.1": {"input": 15.00, "output": 75.00},
    "claude-sonnet-4-20250514": {"input": 3.00, "output": 15.00},
    "claude-sonnet-4": {"input": 3.00, "output": 15.00},
    # Claude 3.5 models (legacy)
    "claude-3-5-haiku-latest": {"input": 0.80, "output": 4.00},
    "claude-3-5-haiku-20241022": {"input": 0.80, "output": 4.00},
    "claude-3-5-sonnet-latest": {"input": 3.00, "output": 15.00},
    "claude-3-5-sonnet-20241022": {"input": 3.00, "output": 15.00},
    # Claude 3 models (deprecated)
    "claude-3-opus-latest": {"input": 15.00, "output": 75.00},
    "claude-3-opus-20240229": {"input": 15.00, "output": 75.00},
    "claude-3-haiku": {"input": 0.25, "output": 1.25},
}

# Default pricing if model not found
DEFAULT_PRICING = {"input": 3.00, "output": 15.00}


@dataclass
class RequestCost:
    """Cost information for a single request.

    Attributes:
        input_tokens: Number of input tokens.
        output_tokens: Number of output tokens.
        input_cost: Cost for input tokens in USD.
        output_cost: Cost for output tokens in USD.
        total_cost: Total cost in USD.
        model: Model used for the request.
        timestamp: When the request was made.
    """
    input_tokens: int
    output_tokens: int
    input_cost: float
    output_cost: float
    total_cost: float
    model: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "input_cost": self.input_cost,
            "output_cost": self.output_cost,
            "total_cost": self.total_cost,
            "model": self.model,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class CostSummary:
    """Summary of costs over multiple requests.

    Attributes:
        total_requests: Number of requests.
        total_input_tokens: Total input tokens used.
        total_output_tokens: Total output tokens used.
        total_input_cost: Total input cost in USD.
        total_output_cost: Total output cost in USD.
        total_cost: Total cost in USD.
        by_model: Cost breakdown by model.
    """
    total_requests: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_input_cost: float = 0.0
    total_output_cost: float = 0.0
    total_cost: float = 0.0
    by_model: dict[str, dict[str, float]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "total_requests": self.total_requests,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_input_cost": self.total_input_cost,
            "total_output_cost": self.total_output_cost,
            "total_cost": self.total_cost,
            "by_model": self.by_model,
        }


class CostTracker:
    """Tracks costs for Claude Code usage.

    Example:
        ```python
        tracker = CostTracker()

        # Record a request
        cost = tracker.record_request(
            model="sonnet",
            input_tokens=1000,
            output_tokens=500,
        )
        print(f"Request cost: ${cost.total_cost:.4f}")

        # Get summary
        summary = tracker.get_summary()
        print(f"Total spent: ${summary.total_cost:.4f}")

        # Set budget alert
        tracker.set_budget(max_cost=10.0)
        if tracker.is_over_budget():
            print("Over budget!")
        ```
    """

    def __init__(self, custom_pricing: dict[str, dict[str, float]] | None = None) -> None:
        """Initialize cost tracker.

        Args:
            custom_pricing: Custom pricing overrides.
        """
        self._pricing = {**MODEL_PRICING}
        if custom_pricing:
            self._pricing.update(custom_pricing)

        self._requests: list[RequestCost] = []
        self._budget: float | None = None
        self._budget_alert_callback: Any = None

    def get_pricing(self, model: str) -> dict[str, float]:
        """Get pricing for a model.

        Args:
            model: Model name.

        Returns:
            Pricing dictionary with 'input' and 'output' costs per million tokens.
        """
        # Try exact match first
        if model in self._pricing:
            return self._pricing[model]

        # Try partial match
        model_lower = model.lower()
        for key, pricing in self._pricing.items():
            if key in model_lower or model_lower in key:
                return pricing

        logger.warning(f"Unknown model pricing for '{model}', using default")
        return DEFAULT_PRICING

    def calculate_cost(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int,
    ) -> tuple[float, float, float]:
        """Calculate cost for a request.

        Args:
            model: Model used.
            input_tokens: Number of input tokens.
            output_tokens: Number of output tokens.

        Returns:
            Tuple of (input_cost, output_cost, total_cost) in USD.
        """
        pricing = self.get_pricing(model)

        input_cost = (input_tokens / 1_000_000) * pricing["input"]
        output_cost = (output_tokens / 1_000_000) * pricing["output"]
        total_cost = input_cost + output_cost

        return input_cost, output_cost, total_cost

    def record_request(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int,
    ) -> RequestCost:
        """Record a request and calculate its cost.

        Args:
            model: Model used.
            input_tokens: Number of input tokens.
            output_tokens: Number of output tokens.

        Returns:
            RequestCost with cost information.
        """
        input_cost, output_cost, total_cost = self.calculate_cost(
            model, input_tokens, output_tokens
        )

        request = RequestCost(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            input_cost=input_cost,
            output_cost=output_cost,
            total_cost=total_cost,
            model=model,
        )

        self._requests.append(request)

        # Check budget
        if self._budget is not None:
            summary = self.get_summary()
            if summary.total_cost > self._budget and self._budget_alert_callback:
                self._budget_alert_callback(summary)

        logger.debug(
            f"Request cost: ${total_cost:.6f} "
            f"(in: {input_tokens}, out: {output_tokens}, model: {model})"
        )

        return request

    def get_summary(self) -> CostSummary:
        """Get cost summary for all recorded requests.

        Returns:
            CostSummary with aggregated costs.
        """
        summary = CostSummary()
        by_model: dict[str, dict[str, float]] = {}

        for req in self._requests:
            summary.total_requests += 1
            summary.total_input_tokens += req.input_tokens
            summary.total_output_tokens += req.output_tokens
            summary.total_input_cost += req.input_cost
            summary.total_output_cost += req.output_cost
            summary.total_cost += req.total_cost

            if req.model not in by_model:
                by_model[req.model] = {
                    "requests": 0,
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "cost": 0.0,
                }

            by_model[req.model]["requests"] += 1
            by_model[req.model]["input_tokens"] += req.input_tokens
            by_model[req.model]["output_tokens"] += req.output_tokens
            by_model[req.model]["cost"] += req.total_cost

        summary.by_model = by_model
        return summary

    def get_requests(self) -> list[RequestCost]:
        """Get all recorded requests.

        Returns:
            List of RequestCost objects.
        """
        return self._requests.copy()

    def set_budget(
        self,
        max_cost: float,
        alert_callback: Any = None,
    ) -> None:
        """Set a budget limit.

        Args:
            max_cost: Maximum cost in USD.
            alert_callback: Function to call when budget exceeded.
        """
        self._budget = max_cost
        self._budget_alert_callback = alert_callback
        logger.info(f"Budget set to ${max_cost:.2f}")

    def is_over_budget(self) -> bool:
        """Check if over budget.

        Returns:
            True if total cost exceeds budget.
        """
        if self._budget is None:
            return False
        return self.get_summary().total_cost > self._budget

    def get_remaining_budget(self) -> float | None:
        """Get remaining budget.

        Returns:
            Remaining budget in USD, or None if no budget set.
        """
        if self._budget is None:
            return None
        return self._budget - self.get_summary().total_cost

    def reset(self) -> None:
        """Reset all tracked requests."""
        self._requests = []
        logger.info("Cost tracker reset")

    def update_pricing(self, model: str, input_price: float, output_price: float) -> None:
        """Update pricing for a model.

        Args:
            model: Model name.
            input_price: Price per million input tokens.
            output_price: Price per million output tokens.
        """
        self._pricing[model] = {"input": input_price, "output": output_price}
        logger.info(f"Updated pricing for {model}: in=${input_price}, out=${output_price}")
