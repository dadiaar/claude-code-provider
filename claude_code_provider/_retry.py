# Copyright (c) 2025. Claude Code Provider for Microsoft Agent Framework.

"""Retry and resilience utilities for Claude Code provider."""

import asyncio
import logging
import random
import time
from dataclasses import dataclass, field
from typing import Awaitable, Callable, TypeVar, Any

logger = logging.getLogger("claude_code_provider")

T = TypeVar("T")
R = TypeVar("R")  # Return type for retry functions


@dataclass
class RetryConfig:
    """Configuration for retry behavior.

    Attributes:
        max_retries: Maximum number of retry attempts (0 = no retries).
        base_delay: Initial delay between retries in seconds.
        max_delay: Maximum delay between retries in seconds.
        exponential_base: Base for exponential backoff (2 = double each time).
        jitter: Whether to add random jitter to delays.
        retryable_exceptions: Exception types that should trigger a retry.
    """

    max_retries: int = 3
    base_delay: float = 1.0
    max_delay: float = 30.0
    exponential_base: float = 2.0
    jitter: bool = True
    retryable_exceptions: tuple[type[Exception], ...] = field(
        default_factory=lambda: (
            ConnectionError,
            TimeoutError,
            OSError,
        )
    )

    def get_delay(self, attempt: int) -> float:
        """Calculate delay for a given attempt number.

        Args:
            attempt: The attempt number (0-indexed).

        Returns:
            Delay in seconds.
        """
        delay = min(
            self.base_delay * (self.exponential_base ** attempt),
            self.max_delay,
        )

        if self.jitter:
            # Add up to 25% random jitter
            jitter_amount = delay * 0.25 * random.random()
            delay += jitter_amount

        return delay


async def retry_async(
    func: Callable[..., Awaitable[R]],
    *args: Any,
    config: RetryConfig | None = None,
    on_retry: Callable[[int, Exception], None] | None = None,
    **kwargs: Any,
) -> R:
    """Execute an async function with retry logic.

    Args:
        func: The async function to execute.
        *args: Positional arguments for the function.
        config: Retry configuration. Uses defaults if not provided.
        on_retry: Optional callback called before each retry with (attempt, exception).
        **kwargs: Keyword arguments for the function.

    Returns:
        The result of the function.

    Raises:
        The last exception if all retries are exhausted.
    """
    if config is None:
        config = RetryConfig()

    last_exception: Exception | None = None

    for attempt in range(config.max_retries + 1):
        try:
            return await func(*args, **kwargs)
        except config.retryable_exceptions as e:
            last_exception = e

            if attempt >= config.max_retries:
                logger.error(
                    f"All {config.max_retries + 1} attempts failed. Last error: {e}"
                )
                raise

            delay = config.get_delay(attempt)
            logger.warning(
                f"Attempt {attempt + 1}/{config.max_retries + 1} failed: {e}. "
                f"Retrying in {delay:.2f}s..."
            )

            if on_retry:
                # Wrap callback in try/except to prevent callback errors
                # from breaking retry logic (fix for #22)
                try:
                    result = on_retry(attempt, e)
                    # Handle async callbacks properly
                    if asyncio.iscoroutine(result):
                        await result
                except Exception as callback_error:
                    logger.warning(
                        f"on_retry callback raised an exception (ignored): {callback_error}"
                    )

            await asyncio.sleep(delay)
        except Exception:
            # Non-retryable exception - raise immediately
            raise

    # This should never be reached, but just in case
    if last_exception:
        raise last_exception
    raise RuntimeError("Unexpected state in retry logic")


@dataclass
class CircuitBreaker:
    """Circuit breaker pattern for preventing cascade failures.

    States:
        CLOSED: Normal operation, requests pass through.
        OPEN: Failing, requests are rejected immediately.
        HALF_OPEN: Testing if service recovered.

    Attributes:
        failure_threshold: Number of failures before opening circuit.
        recovery_timeout: Seconds to wait before trying again.
        success_threshold: Successes needed in HALF_OPEN to close circuit.

    Thread-Safety:
        This implementation uses asyncio.Lock for concurrency safety within
        a single event loop. All state mutations are protected by the lock.

        Important: Each CircuitBreaker instance is bound to the event loop
        where it's first used. Do NOT share instances across different event
        loops (e.g., across threads in multi-worker deployments). Instead,
        create separate CircuitBreaker instances per event loop or thread.

        For multi-worker applications (e.g., uvicorn with multiple workers),
        each worker process has its own memory space and CircuitBreaker.
    """

    failure_threshold: int = 5
    recovery_timeout: float = 60.0
    success_threshold: int = 2

    # Internal state
    _failures: int = field(default=0, repr=False)
    _successes: int = field(default=0, repr=False)
    _state: str = field(default="CLOSED", repr=False)
    _last_failure_time: float | None = field(default=None, repr=False)
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock, repr=False)

    async def record_success(self) -> None:
        """Record a successful call (async for thread safety)."""
        async with self._lock:
            if self._state == "HALF_OPEN":
                self._successes += 1
                if self._successes >= self.success_threshold:
                    self._close()
            elif self._state == "CLOSED":
                # Reset failure count on success
                self._failures = 0

    async def record_failure(self) -> None:
        """Record a failed call (async for thread safety)."""
        async with self._lock:
            self._failures += 1
            self._last_failure_time = time.monotonic()

            if self._state == "HALF_OPEN":
                self._open()
            elif self._state == "CLOSED" and self._failures >= self.failure_threshold:
                self._open()

    async def can_execute(self) -> bool:
        """Check if a call can be executed (async for thread safety).

        Returns:
            True if the call should proceed, False if circuit is open.
        """
        async with self._lock:
            if self._state == "CLOSED":
                return True

            if self._state == "OPEN":
                # Check if recovery timeout has passed
                if self._last_failure_time is not None:
                    elapsed = time.monotonic() - self._last_failure_time
                    if elapsed >= self.recovery_timeout:
                        self._half_open()
                        return True
                return False

            # HALF_OPEN - allow the request
            return True

    def _open(self) -> None:
        """Open the circuit."""
        self._state = "OPEN"
        self._successes = 0
        logger.warning(f"Circuit breaker OPENED after {self._failures} failures")

    def _close(self) -> None:
        """Close the circuit."""
        self._state = "CLOSED"
        self._failures = 0
        self._successes = 0
        logger.info("Circuit breaker CLOSED - service recovered")

    def _half_open(self) -> None:
        """Set circuit to half-open state."""
        self._state = "HALF_OPEN"
        self._successes = 0
        logger.info("Circuit breaker HALF_OPEN - testing recovery")

    @property
    def state(self) -> str:
        """Get current circuit state."""
        return self._state

    @property
    def is_open(self) -> bool:
        """Check if circuit is currently open."""
        return self._state == "OPEN"
