# Copyright (c) 2025. Claude Code Provider for Microsoft Agent Framework.

"""Multi-model routing for cost/quality optimization."""

import logging
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable

logger = logging.getLogger("claude_code_provider")


class ModelTier(str, Enum):
    """Model tier for routing decisions."""
    FAST = "haiku"      # Fast and cheap
    BALANCED = "sonnet"  # Balanced
    POWERFUL = "opus"    # Most capable


@dataclass
class RoutingContext:
    """Context for routing decisions.

    Attributes:
        prompt: The user prompt.
        system_prompt: Optional system prompt.
        estimated_tokens: Estimated token count.
        conversation_length: Number of messages in conversation.
        has_code: Whether the prompt contains code.
        has_complex_reasoning: Whether complex reasoning is likely needed.
        custom_metadata: Additional metadata for routing.
    """
    prompt: str
    system_prompt: str | None = None
    estimated_tokens: int = 0
    conversation_length: int = 0
    has_code: bool = False
    has_complex_reasoning: bool = False
    custom_metadata: dict[str, Any] = field(default_factory=dict)


class RoutingStrategy(ABC):
    """Base class for routing strategies."""

    @abstractmethod
    def select_model(self, context: RoutingContext) -> str:
        """Select a model based on the routing context.

        Args:
            context: The routing context.

        Returns:
            Model name to use.
        """
        pass

    @abstractmethod
    def get_name(self) -> str:
        """Get the strategy name."""
        pass


class SimpleRouter(RoutingStrategy):
    """Simple router that always returns the same model."""

    def __init__(self, model: str = "sonnet") -> None:
        self.model = model

    def select_model(self, context: RoutingContext) -> str:
        return self.model

    def get_name(self) -> str:
        return f"simple:{self.model}"


class ComplexityRouter(RoutingStrategy):
    """Routes based on prompt complexity.

    Uses heuristics to estimate task complexity:
    - Short, simple prompts -> haiku
    - Medium complexity -> sonnet
    - Complex reasoning/code -> opus
    """

    def __init__(
        self,
        simple_threshold: int = 100,
        complex_threshold: int = 500,
        code_patterns: list[str] | None = None,
        reasoning_keywords: list[str] | None = None,
    ) -> None:
        """Initialize complexity router.

        Args:
            simple_threshold: Token threshold for simple tasks.
            complex_threshold: Token threshold for complex tasks.
            code_patterns: Regex patterns indicating code.
            reasoning_keywords: Keywords indicating complex reasoning.
        """
        self.simple_threshold = simple_threshold
        self.complex_threshold = complex_threshold

        # Pre-compile regex patterns for performance (fix for #27)
        # This avoids recompiling on every call to _has_code()
        pattern_strings = code_patterns or [
            r"```\w*\n",  # Code blocks
            r"def\s+\w+",  # Python functions
            r"function\s+\w+",  # JavaScript functions
            r"class\s+\w+",  # Classes
            r"import\s+\w+",  # Imports
        ]
        self._compiled_code_patterns = [
            re.compile(p, re.IGNORECASE) for p in pattern_strings
        ]

        self.reasoning_keywords = reasoning_keywords or [
            "analyze", "compare", "evaluate", "explain why",
            "design", "architect", "optimize", "debug",
            "step by step", "reasoning", "proof", "derive",
            "complex", "challenging", "difficult",
        ]

    def _estimate_tokens(self, text: str) -> int:
        """Rough token estimation (4 chars per token)."""
        return len(text) // 4

    def _has_code(self, text: str) -> bool:
        """Check if text contains code.

        Uses pre-compiled patterns for performance (fix for #27).
        """
        for pattern in self._compiled_code_patterns:
            if pattern.search(text):
                return True
        return False

    def _has_complex_reasoning(self, text: str) -> bool:
        """Check if text requires complex reasoning."""
        text_lower = text.lower()
        return any(keyword in text_lower for keyword in self.reasoning_keywords)

    def select_model(self, context: RoutingContext) -> str:
        prompt = context.prompt
        system = context.system_prompt or ""
        combined = f"{system} {prompt}"

        # Estimate tokens if not provided
        tokens = context.estimated_tokens or self._estimate_tokens(combined)

        # Check for code
        has_code = context.has_code or self._has_code(combined)

        # Check for complex reasoning
        has_complex = context.has_complex_reasoning or self._has_complex_reasoning(combined)

        # Decision logic
        if has_code and has_complex:
            logger.debug("Routing to opus: code + complex reasoning")
            return ModelTier.POWERFUL.value

        if has_complex or tokens > self.complex_threshold:
            logger.debug(f"Routing to sonnet: complex={has_complex}, tokens={tokens}")
            return ModelTier.BALANCED.value

        if tokens < self.simple_threshold and not has_code:
            logger.debug(f"Routing to haiku: simple task, tokens={tokens}")
            return ModelTier.FAST.value

        logger.debug(f"Routing to sonnet: default, tokens={tokens}")
        return ModelTier.BALANCED.value

    def get_name(self) -> str:
        return "complexity"


class CostOptimizedRouter(RoutingStrategy):
    """Routes to minimize cost while maintaining quality.

    Starts with the cheapest model and upgrades if needed.
    """

    def __init__(
        self,
        budget_remaining: float | None = None,
        quality_threshold: float = 0.8,
    ) -> None:
        """Initialize cost-optimized router.

        Args:
            budget_remaining: Remaining budget in USD.
            quality_threshold: Minimum acceptable quality (0-1).
        """
        self.budget_remaining = budget_remaining
        self.quality_threshold = quality_threshold
        self._complexity_router = ComplexityRouter()

    def select_model(self, context: RoutingContext) -> str:
        # If budget is very low, always use cheapest
        if self.budget_remaining is not None and self.budget_remaining < 0.01:
            logger.debug("Routing to haiku: budget constrained")
            return ModelTier.FAST.value

        # Use complexity router for base decision
        base_model = self._complexity_router.select_model(context)

        # If budget is moderate, potentially downgrade
        if self.budget_remaining is not None and self.budget_remaining < 1.0:
            if base_model == ModelTier.POWERFUL.value:
                logger.debug("Downgrading from opus to sonnet: budget constrained")
                return ModelTier.BALANCED.value

        return base_model

    def update_budget(self, remaining: float) -> None:
        """Update remaining budget."""
        self.budget_remaining = remaining

    def get_name(self) -> str:
        return "cost_optimized"


class TaskTypeRouter(RoutingStrategy):
    """Routes based on task type keywords.

    Maps task types to models:
    - Simple Q&A, summarization -> haiku
    - Code generation, analysis -> sonnet
    - Complex reasoning, research -> opus
    """

    def __init__(
        self,
        task_mappings: dict[str, str] | None = None,
    ) -> None:
        """Initialize task type router.

        Args:
            task_mappings: Custom task type to model mappings.
        """
        self.task_mappings = task_mappings or {
            # Simple tasks -> haiku
            "summarize": "haiku",
            "translate": "haiku",
            "format": "haiku",
            "list": "haiku",
            "define": "haiku",
            "simple": "haiku",

            # Medium tasks -> sonnet
            "code": "sonnet",
            "write": "sonnet",
            "explain": "sonnet",
            "analyze": "sonnet",
            "review": "sonnet",
            "default": "sonnet",

            # Complex tasks -> opus
            "research": "opus",
            "design": "opus",
            "architect": "opus",
            "debug complex": "opus",
            "multi-step": "opus",
        }

    def select_model(self, context: RoutingContext) -> str:
        prompt_lower = context.prompt.lower()

        # Check each task type
        for task_type, model in self.task_mappings.items():
            if task_type in prompt_lower:
                logger.debug(f"Routing to {model}: matched task type '{task_type}'")
                return model

        # Default to sonnet
        return self.task_mappings.get("default", "sonnet")

    def get_name(self) -> str:
        return "task_type"


class CustomRouter(RoutingStrategy):
    """Custom router using a user-provided function."""

    def __init__(
        self,
        routing_function: Callable[[RoutingContext], str],
        name: str = "custom",
    ) -> None:
        """Initialize custom router.

        Args:
            routing_function: Function that takes context and returns model name.
            name: Name for this router.
        """
        self.routing_function = routing_function
        self._name = name

    def select_model(self, context: RoutingContext) -> str:
        return self.routing_function(context)

    def get_name(self) -> str:
        return self._name


class ModelRouter:
    """Main model router that manages routing strategies.

    Example:
        ```python
        router = ModelRouter()

        # Use complexity-based routing
        router.set_strategy(ComplexityRouter())

        # Route a prompt
        model = router.route("Explain quantum computing in detail")
        print(f"Using model: {model}")

        # Or use cost-optimized routing
        router.set_strategy(CostOptimizedRouter(budget_remaining=5.0))
        model = router.route("Simple greeting")
        ```
    """

    def __init__(
        self,
        default_strategy: RoutingStrategy | None = None,
    ) -> None:
        """Initialize model router.

        Args:
            default_strategy: Default routing strategy.
        """
        self._strategy = default_strategy or ComplexityRouter()
        self._fallback_model = "sonnet"

    def set_strategy(self, strategy: RoutingStrategy) -> None:
        """Set the routing strategy.

        Args:
            strategy: The routing strategy to use.
        """
        self._strategy = strategy
        logger.info(f"Set routing strategy: {strategy.get_name()}")

    def get_strategy(self) -> RoutingStrategy:
        """Get the current routing strategy."""
        return self._strategy

    def set_fallback_model(self, model: str) -> None:
        """Set the fallback model if routing fails.

        Args:
            model: Fallback model name.
        """
        self._fallback_model = model

    def route(
        self,
        prompt: str,
        system_prompt: str | None = None,
        **kwargs: Any,
    ) -> str:
        """Route a prompt to the appropriate model.

        Args:
            prompt: The user prompt.
            system_prompt: Optional system prompt.
            **kwargs: Additional context for routing.

        Returns:
            Selected model name.
        """
        context = RoutingContext(
            prompt=prompt,
            system_prompt=system_prompt,
            **kwargs,
        )

        try:
            model = self._strategy.select_model(context)
            logger.debug(
                f"Routed to {model} using {self._strategy.get_name()} strategy"
            )
            return model
        except Exception as e:
            logger.warning(f"Routing failed, using fallback: {e}")
            return self._fallback_model

    def create_context(
        self,
        prompt: str,
        system_prompt: str | None = None,
        **kwargs: Any,
    ) -> RoutingContext:
        """Create a routing context for inspection.

        Args:
            prompt: The user prompt.
            system_prompt: Optional system prompt.
            **kwargs: Additional context.

        Returns:
            RoutingContext object.
        """
        return RoutingContext(
            prompt=prompt,
            system_prompt=system_prompt,
            **kwargs,
        )
