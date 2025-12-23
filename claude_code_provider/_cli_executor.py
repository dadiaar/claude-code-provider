# Copyright (c) 2025. Claude Code Provider for Microsoft Agent Framework.

"""Async subprocess wrapper for Claude Code CLI execution."""

import asyncio
import json
import logging
from dataclasses import dataclass
from typing import Any, AsyncIterator

try:
    from ._settings import ClaudeCodeSettings
    from ._exceptions import (
        ClaudeCodeExecutionError,
        ClaudeCodeParseError,
        ClaudeCodeTimeoutError,
    )
    from ._retry import RetryConfig, retry_async, CircuitBreaker
except ImportError:
    from _settings import ClaudeCodeSettings
    from _exceptions import (
        ClaudeCodeExecutionError,
        ClaudeCodeParseError,
        ClaudeCodeTimeoutError,
    )
    from _retry import RetryConfig, retry_async, CircuitBreaker

# Default timeout for CLI execution (5 minutes)
DEFAULT_TIMEOUT_SECONDS = 300.0

logger = logging.getLogger("claude_code_provider")


@dataclass
class CLIResult:
    """Result from a Claude CLI execution.

    Attributes:
        success: Whether the execution was successful.
        result: The text result from Claude.
        session_id: The session ID for conversation continuity.
        usage: Token usage information.
        raw_response: The full JSON response from the CLI.
        error: Error message if execution failed.
    """

    success: bool
    result: str
    session_id: str | None = None
    usage: dict[str, Any] | None = None
    raw_response: dict[str, Any] | None = None
    error: str | None = None

    @property
    def input_tokens(self) -> int:
        """Get input token count."""
        if self.usage:
            return self.usage.get("input_tokens", 0)
        return 0

    @property
    def output_tokens(self) -> int:
        """Get output token count."""
        if self.usage:
            return self.usage.get("output_tokens", 0)
        return 0


@dataclass
class StreamEvent:
    """A streaming event from Claude CLI.

    Attributes:
        event_type: Type of event ('system', 'assistant', 'result', etc.)
        data: The event data.
    """

    event_type: str
    data: dict[str, Any]

    @property
    def is_assistant_message(self) -> bool:
        """Check if this is an assistant message event."""
        return self.event_type == "assistant"

    @property
    def is_result(self) -> bool:
        """Check if this is a final result event."""
        return self.event_type == "result"

    @property
    def text(self) -> str | None:
        """Extract text content from the event."""
        if self.is_assistant_message:
            message = self.data.get("message", {})
            content = message.get("content", [])
            texts = [c.get("text", "") for c in content if c.get("type") == "text"]
            return "".join(texts) if texts else None
        elif self.is_result:
            return self.data.get("result")
        return None


class CLIExecutor:
    """Executes Claude CLI commands asynchronously with retry and resilience."""

    def __init__(
        self,
        settings: ClaudeCodeSettings,
        *,
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
        retry_config: RetryConfig | None = None,
        enable_circuit_breaker: bool = True,
    ) -> None:
        """Initialize the CLI executor.

        Args:
            settings: Claude Code settings.
            timeout: Timeout for CLI execution in seconds.
            retry_config: Configuration for retry behavior. None = no retries.
            enable_circuit_breaker: Whether to use circuit breaker pattern.
        """
        self.settings = settings
        self.timeout = timeout
        self.retry_config = retry_config
        self.circuit_breaker = CircuitBreaker() if enable_circuit_breaker else None

    async def execute(
        self,
        prompt: str,
        *,
        session_id: str | None = None,
        system_prompt: str | None = None,
        model: str | None = None,
        max_turns: int | None = None,
        extra_args: list[str] | None = None,
        timeout: float | None = None,
    ) -> CLIResult:
        """Execute a Claude CLI command and return the result.

        Args:
            prompt: The prompt to send to Claude.
            session_id: Optional session ID to resume a conversation.
            system_prompt: Optional system prompt to prepend.
            model: Optional model override.
            max_turns: Optional max turns override.
            extra_args: Additional CLI arguments.
            timeout: Optional timeout override in seconds.

        Returns:
            CLIResult with the execution result.

        Raises:
            ClaudeCodeExecutionError: If CLI execution fails.
            ClaudeCodeTimeoutError: If execution times out.
            ClaudeCodeParseError: If response parsing fails.
        """
        # Check circuit breaker
        if self.circuit_breaker and not self.circuit_breaker.can_execute():
            raise ClaudeCodeExecutionError(
                message="Circuit breaker is open - too many recent failures. "
                        f"Recovery in {self.circuit_breaker.recovery_timeout}s.",
                exit_code=None,
                stderr=None,
            )

        # Use retry wrapper if configured
        if self.retry_config:
            return await retry_async(
                self._execute_once,
                prompt,
                session_id=session_id,
                system_prompt=system_prompt,
                model=model,
                max_turns=max_turns,
                extra_args=extra_args,
                timeout=timeout,
                config=self.retry_config,
            )
        else:
            return await self._execute_once(
                prompt,
                session_id=session_id,
                system_prompt=system_prompt,
                model=model,
                max_turns=max_turns,
                extra_args=extra_args,
                timeout=timeout,
            )

    async def _execute_once(
        self,
        prompt: str,
        *,
        session_id: str | None = None,
        system_prompt: str | None = None,
        model: str | None = None,
        max_turns: int | None = None,
        extra_args: list[str] | None = None,
        timeout: float | None = None,
    ) -> CLIResult:
        """Execute CLI once (internal method used by retry wrapper)."""
        effective_timeout = timeout if timeout is not None else self.timeout

        args = self._build_args(
            prompt=prompt,
            session_id=session_id,
            system_prompt=system_prompt,
            model=model,
            max_turns=max_turns,
            streaming=False,
            extra_args=extra_args,
        )

        try:
            process = await asyncio.create_subprocess_exec(
                self.settings.cli_path,
                *args,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=self.settings.working_directory,
            )

            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(),
                    timeout=effective_timeout,
                )
            except asyncio.TimeoutError:
                # Kill the process on timeout
                process.kill()
                await process.wait()
                if self.circuit_breaker:
                    self.circuit_breaker.record_failure()
                raise ClaudeCodeTimeoutError(timeout_seconds=effective_timeout)

            if process.returncode != 0:
                error_msg = stderr.decode() if stderr else f"Exit code: {process.returncode}"
                logger.error(f"Claude CLI failed: {error_msg}")
                if self.circuit_breaker:
                    self.circuit_breaker.record_failure()
                raise ClaudeCodeExecutionError(
                    message=f"Claude CLI failed: {error_msg}",
                    exit_code=process.returncode,
                    stderr=stderr.decode() if stderr else None,
                )

            # Parse JSON response
            try:
                response = json.loads(stdout.decode())
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse CLI response: {e}")
                if self.circuit_breaker:
                    self.circuit_breaker.record_failure()
                raise ClaudeCodeParseError(
                    message=f"Failed to parse CLI JSON response: {e}",
                    raw_output=stdout.decode(),
                )

            # Check if the response itself indicates an error
            is_error = response.get("is_error", False)
            if is_error and self.circuit_breaker:
                self.circuit_breaker.record_failure()
            elif self.circuit_breaker:
                self.circuit_breaker.record_success()

            return CLIResult(
                success=not is_error,
                result=response.get("result", ""),
                session_id=response.get("session_id"),
                usage=response.get("usage"),
                raw_response=response,
                error=response.get("error") if is_error else None,
            )

        except ClaudeCodeTimeoutError:
            raise  # Already handled above
        except (ClaudeCodeExecutionError, ClaudeCodeParseError):
            raise  # Re-raise our own exceptions
        except Exception as e:
            logger.exception(f"Failed to execute Claude CLI: {e}")
            if self.circuit_breaker:
                self.circuit_breaker.record_failure()
            raise ClaudeCodeExecutionError(
                message=f"Unexpected error executing Claude CLI: {e}",
                exit_code=None,
                stderr=None,
            )

    async def execute_stream(
        self,
        prompt: str,
        *,
        session_id: str | None = None,
        system_prompt: str | None = None,
        model: str | None = None,
        max_turns: int | None = None,
        extra_args: list[str] | None = None,
    ) -> AsyncIterator[StreamEvent]:
        """Execute a Claude CLI command with streaming output.

        Args:
            prompt: The prompt to send to Claude.
            session_id: Optional session ID to resume a conversation.
            system_prompt: Optional system prompt to prepend.
            model: Optional model override.
            max_turns: Optional max turns override.
            extra_args: Additional CLI arguments.

        Yields:
            StreamEvent objects as they arrive.
        """
        args = self._build_args(
            prompt=prompt,
            session_id=session_id,
            system_prompt=system_prompt,
            model=model,
            max_turns=max_turns,
            streaming=True,
            extra_args=extra_args,
        )

        try:
            process = await asyncio.create_subprocess_exec(
                self.settings.cli_path,
                *args,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=self.settings.working_directory,
            )

            if process.stdout is None:
                return

            # Read line by line for stream-json format
            while True:
                line = await process.stdout.readline()
                if not line:
                    break

                line_str = line.decode().strip()
                if not line_str:
                    continue

                try:
                    data = json.loads(line_str)
                    event_type = data.get("type", "unknown")
                    yield StreamEvent(event_type=event_type, data=data)
                except json.JSONDecodeError as e:
                    logger.warning(f"Failed to parse streaming line: {e}")
                    continue

            await process.wait()

        except Exception as e:
            logger.exception(f"Failed to execute Claude CLI stream: {e}")
            yield StreamEvent(
                event_type="error",
                data={"error": str(e)},
            )

    def _build_args(
        self,
        prompt: str,
        *,
        session_id: str | None = None,
        system_prompt: str | None = None,
        model: str | None = None,
        max_turns: int | None = None,
        streaming: bool = False,
        extra_args: list[str] | None = None,
    ) -> list[str]:
        """Build CLI arguments for execution.

        Args:
            prompt: The prompt to send.
            session_id: Optional session ID.
            system_prompt: Optional system prompt.
            model: Optional model override.
            max_turns: Optional max turns.
            streaming: Whether to use streaming output.
            extra_args: Additional CLI arguments.

        Returns:
            List of CLI arguments.
        """
        args = ["-p", prompt]

        # Output format
        if streaming:
            args.extend(["--output-format", "stream-json", "--verbose"])
        else:
            args.extend(["--output-format", "json"])

        # Model (use override, then settings)
        effective_model = model or self.settings.model
        if effective_model:
            args.extend(["--model", effective_model])

        # Session resumption
        if session_id:
            args.extend(["--resume", session_id])

        # System prompt
        if system_prompt:
            args.extend(["--system-prompt", system_prompt])

        # Max turns (use override, then settings)
        effective_max_turns = max_turns if max_turns is not None else self.settings.default_max_turns
        if effective_max_turns is not None:
            args.extend(["--max-turns", str(effective_max_turns)])

        # Add settings-based args (tools, permissions, etc.)
        args.extend(self.settings.to_cli_args())

        # Extra args
        if extra_args:
            args.extend(extra_args)

        return args
