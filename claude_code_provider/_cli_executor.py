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

# Default timeout for streaming read operations (60 seconds)
# Shorter than execution timeout since streaming should have regular data
DEFAULT_STREAM_READ_TIMEOUT = 60.0

# Maximum prompt size (1MB - generous but prevents memory issues)
MAX_PROMPT_SIZE = 1_000_000

# Maximum CLI argument size before switching to stdin (100KB)
# Linux ARG_MAX is typically 128KB-2MB, use 100KB to be safe
MAX_CLI_ARG_SIZE = 100_000

# Allowed CLI flags that can be passed via extra_args
# This whitelist prevents injection of dangerous flags
# Note: --dangerously-skip-permissions is NOT included here for security.
# Use the skip_permissions parameter in execute() if absolutely needed.
ALLOWED_EXTRA_ARGS = frozenset({
    "--verbose", "-v",
    "--no-cache",
})

logger = logging.getLogger("claude_code_provider")


def _validate_prompt(prompt: str) -> str:
    """Validate and sanitize a prompt string.

    Args:
        prompt: The prompt to validate.

    Returns:
        The sanitized prompt.

    Raises:
        ValueError: If prompt is invalid.
    """
    if not isinstance(prompt, str):
        raise ValueError("Prompt must be a string")

    if len(prompt) > MAX_PROMPT_SIZE:
        raise ValueError(f"Prompt exceeds maximum size of {MAX_PROMPT_SIZE} bytes")

    if len(prompt) == 0:
        raise ValueError("Prompt cannot be empty")

    # Reject null bytes (security risk - could truncate strings in C libraries)
    if '\x00' in prompt:
        raise ValueError("Prompt contains null bytes which are not allowed")

    return prompt


def _validate_extra_args(extra_args: list[str] | None) -> list[str]:
    """Validate extra CLI arguments against whitelist.

    Only whitelisted flag arguments (starting with -) are allowed.
    Non-flag arguments are rejected to prevent injection attacks.

    Args:
        extra_args: List of extra arguments.

    Returns:
        Validated list of arguments.

    Raises:
        ValueError: If any argument is not in the whitelist or is not a flag.
    """
    if not extra_args:
        return []

    validated = []
    for arg in extra_args:
        if not isinstance(arg, str):
            raise ValueError(f"Extra argument must be a string, got {type(arg)}")

        # Reject non-flag arguments to prevent injection
        if not arg.startswith("-"):
            raise ValueError(
                f"Only flag arguments (starting with -) are allowed, got '{arg}'"
            )

        # Check if it's an allowed flag
        if arg not in ALLOWED_EXTRA_ARGS:
            raise ValueError(
                f"CLI argument '{arg}' is not allowed. "
                f"Allowed args: {', '.join(sorted(ALLOWED_EXTRA_ARGS))}"
            )
        validated.append(arg)

    return validated


def _validate_cli_response(response: Any) -> dict[str, Any]:
    """Validate CLI response structure.

    Args:
        response: The parsed JSON response.

    Returns:
        The validated response dict.

    Raises:
        ClaudeCodeParseError: If response structure is invalid.
    """
    if not isinstance(response, dict):
        raise ClaudeCodeParseError(
            message="CLI response must be a JSON object",
            raw_output=str(response),
        )

    # Check for required fields based on response type
    if "error" in response and response.get("is_error"):
        # Error response - error field should be a string
        if not isinstance(response.get("error"), str):
            raise ClaudeCodeParseError(
                message="Error response must have 'error' string field",
                raw_output=str(response),
            )
    else:
        # Success response - should have result
        if "result" not in response:
            raise ClaudeCodeParseError(
                message="Success response must have 'result' field",
                raw_output=str(response),
            )

    return response


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
        """Get total input token count including cache tokens.

        Claude Code CLI reports input tokens in three separate fields:
        - input_tokens: New tokens not from cache
        - cache_creation_input_tokens: Tokens written to cache
        - cache_read_input_tokens: Tokens read from cache

        This property returns the sum of all three.
        For individual components, use raw_input_tokens, cache_creation_tokens,
        and cache_read_tokens properties.
        """
        if self.usage:
            return (
                self.usage.get("input_tokens", 0) +
                self.usage.get("cache_creation_input_tokens", 0) +
                self.usage.get("cache_read_input_tokens", 0)
            )
        return 0

    @property
    def raw_input_tokens(self) -> int:
        """Get raw input tokens (new tokens, not from cache)."""
        if self.usage:
            return self.usage.get("input_tokens", 0)
        return 0

    @property
    def cache_creation_tokens(self) -> int:
        """Get tokens written to cache during this request."""
        if self.usage:
            return self.usage.get("cache_creation_input_tokens", 0)
        return 0

    @property
    def cache_read_tokens(self) -> int:
        """Get tokens read from cache during this request."""
        if self.usage:
            return self.usage.get("cache_read_input_tokens", 0)
        return 0

    @property
    def output_tokens(self) -> int:
        """Get output token count."""
        if self.usage:
            return self.usage.get("output_tokens", 0)
        return 0

    @property
    def token_breakdown(self) -> dict[str, int]:
        """Get detailed token breakdown.

        Returns:
            Dictionary with all token counts:
            - input_tokens: Total input (sum of raw + cache)
            - raw_input_tokens: New tokens not from cache
            - cache_creation_tokens: Tokens written to cache
            - cache_read_tokens: Tokens read from cache
            - output_tokens: Output tokens
        """
        return {
            "input_tokens": self.input_tokens,
            "raw_input_tokens": self.raw_input_tokens,
            "cache_creation_tokens": self.cache_creation_tokens,
            "cache_read_tokens": self.cache_read_tokens,
            "output_tokens": self.output_tokens,
        }


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
        if self.circuit_breaker and not await self.circuit_breaker.can_execute():
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

        args, stdin_prompt = self._build_args(
            prompt=prompt,
            session_id=session_id,
            system_prompt=system_prompt,
            model=model,
            max_turns=max_turns,
            streaming=False,
            extra_args=extra_args,
        )

        try:
            # Use stdin for large prompts to avoid ARG_MAX limit
            stdin_pipe = asyncio.subprocess.PIPE if stdin_prompt else None

            process = await asyncio.create_subprocess_exec(
                self.settings.cli_path,
                *args,
                stdin=stdin_pipe,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=self.settings.working_directory,
            )

            try:
                # Pass prompt via stdin if needed
                stdin_data = stdin_prompt.encode() if stdin_prompt else None
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(input=stdin_data),
                    timeout=effective_timeout,
                )
            except asyncio.TimeoutError:
                # Kill the process on timeout
                process.kill()
                await process.wait()
                if self.circuit_breaker:
                    await self.circuit_breaker.record_failure()
                raise ClaudeCodeTimeoutError(timeout_seconds=effective_timeout)

            if process.returncode != 0:
                error_msg = stderr.decode() if stderr else f"Exit code: {process.returncode}"
                logger.error(f"Claude CLI failed: {error_msg}")
                if self.circuit_breaker:
                    await self.circuit_breaker.record_failure()
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
                    await self.circuit_breaker.record_failure()
                raise ClaudeCodeParseError(
                    message=f"Failed to parse CLI JSON response: {e}",
                    raw_output=stdout.decode(),
                )

            # Validate response structure
            response = _validate_cli_response(response)

            # Check if the response itself indicates an error
            is_error = response.get("is_error", False)
            if is_error and self.circuit_breaker:
                await self.circuit_breaker.record_failure()
            elif self.circuit_breaker:
                await self.circuit_breaker.record_success()

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
                await self.circuit_breaker.record_failure()
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
        stream_timeout: float | None = None,
    ) -> AsyncIterator[StreamEvent]:
        """Execute a Claude CLI command with streaming output.

        Args:
            prompt: The prompt to send to Claude.
            session_id: Optional session ID to resume a conversation.
            system_prompt: Optional system prompt to prepend.
            model: Optional model override.
            max_turns: Optional max turns override.
            extra_args: Additional CLI arguments.
            stream_timeout: Timeout for each streaming read operation in seconds.
                Defaults to DEFAULT_STREAM_READ_TIMEOUT (60 seconds).

        Yields:
            StreamEvent objects as they arrive.

        Raises:
            ClaudeCodeTimeoutError: If streaming read times out.
        """
        effective_stream_timeout = (
            stream_timeout if stream_timeout is not None
            else DEFAULT_STREAM_READ_TIMEOUT
        )
        args, stdin_prompt = self._build_args(
            prompt=prompt,
            session_id=session_id,
            system_prompt=system_prompt,
            model=model,
            max_turns=max_turns,
            streaming=True,
            extra_args=extra_args,
        )

        process = None
        try:
            # Use stdin for large prompts to avoid ARG_MAX limit
            stdin_pipe = asyncio.subprocess.PIPE if stdin_prompt else None

            process = await asyncio.create_subprocess_exec(
                self.settings.cli_path,
                *args,
                stdin=stdin_pipe,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=self.settings.working_directory,
            )

            # Write stdin prompt if needed and close stdin
            if stdin_prompt and process.stdin:
                process.stdin.write(stdin_prompt.encode())
                await process.stdin.drain()
                process.stdin.close()
                await process.stdin.wait_closed()

            if process.stdout is None:
                return

            # Read line by line for stream-json format
            while True:
                try:
                    line = await asyncio.wait_for(
                        process.stdout.readline(),
                        timeout=effective_stream_timeout,
                    )
                except asyncio.TimeoutError:
                    logger.error(
                        f"Streaming read timed out after {effective_stream_timeout}s"
                    )
                    raise ClaudeCodeTimeoutError(
                        timeout_seconds=effective_stream_timeout
                    )

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

        except ClaudeCodeTimeoutError:
            # Re-raise timeout errors without wrapping
            raise
        except Exception as e:
            logger.exception(f"Failed to execute Claude CLI stream: {e}")
            yield StreamEvent(
                event_type="error",
                data={"error": str(e)},
            )
        finally:
            # Ensure process is cleaned up even if exception occurs
            if process is not None and process.returncode is None:
                try:
                    process.terminate()
                    # Give it a moment to terminate gracefully
                    await asyncio.wait_for(process.wait(), timeout=2.0)
                except asyncio.TimeoutError:
                    # Force kill if it doesn't terminate
                    process.kill()
                    await process.wait()
                except Exception:
                    # Last resort - try to kill
                    try:
                        process.kill()
                    except Exception:
                        pass

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
    ) -> tuple[list[str], str | None]:
        """Build CLI arguments for execution.

        Args:
            prompt: The prompt to send.
            session_id: Optional session ID.
            system_prompt: Optional system prompt.
            model: Optional model override.
            max_turns: Optional max turns.
            streaming: Whether to use streaming output.
            extra_args: Additional CLI arguments (whitelisted only).

        Returns:
            Tuple of (list of CLI arguments, optional stdin prompt).
            If the prompt is large, it will be returned as stdin_prompt
            instead of being included in args.

        Raises:
            ValueError: If prompt or extra_args are invalid.
        """
        # Validate inputs
        validated_prompt = _validate_prompt(prompt)
        validated_extra_args = _validate_extra_args(extra_args)

        # Determine if prompt should be passed via stdin (for large prompts)
        use_stdin = len(validated_prompt) > MAX_CLI_ARG_SIZE
        stdin_prompt = validated_prompt if use_stdin else None

        if use_stdin:
            # Use stdin for large prompt - don't include -p flag, prompt comes from stdin
            args = []
            logger.debug(f"Using stdin for large prompt ({len(validated_prompt)} bytes)")
        else:
            args = ["-p", validated_prompt]

        # Output format
        if streaming:
            args.extend(["--output-format", "stream-json", "--verbose"])
        else:
            args.extend(["--output-format", "json"])

        # Model (use override, then settings)
        effective_model = model or self.settings.model
        if effective_model:
            args.extend(["--model", effective_model])

        # Session resumption (validate like checkpoint_id)
        if session_id:
            if not session_id or not all(c.isalnum() or c in '_-' for c in session_id):
                raise ValueError(
                    f"Invalid session_id format: '{session_id}'. "
                    "Only alphanumeric characters, underscores, and hyphens allowed."
                )
            args.extend(["--resume", session_id])

        # System prompt (also check for size)
        if system_prompt:
            if len(system_prompt) > MAX_CLI_ARG_SIZE:
                # For large system prompts, prepend to the user prompt in stdin
                if stdin_prompt:
                    # Both are large - combine them
                    stdin_prompt = f"{system_prompt}\n\n---\n\n{validated_prompt}"
                else:
                    # Only system prompt is large - add user prompt to stdin too
                    stdin_prompt = f"{system_prompt}\n\n---\n\n{validated_prompt}"
                    # Remove the -p arg we added earlier since prompt is now in stdin
                    args = [a for a in args if a != "-p" and a != validated_prompt]
                logger.debug(f"Using stdin for large system prompt ({len(system_prompt)} bytes)")
            else:
                args.extend(["--system-prompt", system_prompt])

        # Max turns (use override, then settings)
        effective_max_turns = max_turns if max_turns is not None else self.settings.default_max_turns
        if effective_max_turns is not None:
            args.extend(["--max-turns", str(effective_max_turns)])

        # Add settings-based args (tools, permissions, etc.)
        # Exclude model and max_turns since we already handle them above with overrides
        args.extend(self.settings.to_cli_args(exclude={"model", "max_turns"}))

        # Extra args (already validated)
        if validated_extra_args:
            args.extend(validated_extra_args)

        return args, stdin_prompt
