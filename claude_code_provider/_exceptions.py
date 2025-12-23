# Copyright (c) 2025. Claude Code Provider for Microsoft Agent Framework.

"""Exceptions for Claude Code CLI provider.

These exceptions follow MAF's exception hierarchy for consistency.
"""

from agent_framework.exceptions import (
    ChatClientException,
    ChatClientInitializationError,
    ServiceException,
    ServiceResponseException,
    ServiceContentFilterException,
)


class ClaudeCodeException(ChatClientException):
    """Base exception for Claude Code provider errors."""
    pass


class ClaudeCodeCLINotFoundError(ChatClientInitializationError):
    """Raised when the Claude CLI executable is not found."""

    def __init__(self, cli_path: str = "claude"):
        super().__init__(
            f"Claude CLI not found at '{cli_path}'. "
            "Please install Claude Code CLI or set the correct path."
        )
        self.cli_path = cli_path


class ClaudeCodeExecutionError(ServiceResponseException):
    """Raised when CLI execution fails."""

    def __init__(self, message: str, exit_code: int | None = None, stderr: str | None = None):
        super().__init__(message)
        self.exit_code = exit_code
        self.stderr = stderr


class ClaudeCodeParseError(ServiceResponseException):
    """Raised when CLI output cannot be parsed."""

    def __init__(self, message: str, raw_output: str | None = None):
        super().__init__(message)
        self.raw_output = raw_output


class ClaudeCodeTimeoutError(ServiceException):
    """Raised when CLI execution times out."""

    def __init__(self, timeout_seconds: float):
        super().__init__(f"Claude CLI execution timed out after {timeout_seconds} seconds")
        self.timeout_seconds = timeout_seconds


class ClaudeCodeContentFilterError(ServiceContentFilterException):
    """Raised when content is filtered by Claude."""

    def __init__(self, message: str = "Content was filtered by Claude"):
        super().__init__(message)


class ClaudeCodeSessionError(ClaudeCodeException):
    """Raised when there's an issue with session management."""

    def __init__(self, message: str, session_id: str | None = None):
        super().__init__(message)
        self.session_id = session_id
