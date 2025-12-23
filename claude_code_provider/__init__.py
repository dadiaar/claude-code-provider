# Copyright (c) 2025. Claude Code Provider for Microsoft Agent Framework.

"""Claude Code CLI Provider for Microsoft Agent Framework.

This package provides a chat client that uses the Claude Code CLI (`claude`)
instead of direct API calls, allowing MAF agents to use a Claude subscription.

Example:
    from claude_code_provider import ClaudeCodeClient

    client = ClaudeCodeClient(model="sonnet")
    agent = client.create_agent(
        name="assistant",
        instructions="You are a helpful assistant.",
    )
    response = await agent.run("Hello!")
"""

try:
    from ._settings import ClaudeCodeSettings
    from ._chat_client import ClaudeCodeClient
    from ._agent import ClaudeAgent, CompactResult, UsageStats, ContextInfo
    from ._exceptions import (
        ClaudeCodeException,
        ClaudeCodeCLINotFoundError,
        ClaudeCodeExecutionError,
        ClaudeCodeParseError,
        ClaudeCodeTimeoutError,
        ClaudeCodeContentFilterError,
        ClaudeCodeSessionError,
    )
    from ._retry import RetryConfig, CircuitBreaker
except ImportError:
    from _settings import ClaudeCodeSettings
    from _chat_client import ClaudeCodeClient
    from _agent import ClaudeAgent, CompactResult, UsageStats, ContextInfo
    from _exceptions import (
        ClaudeCodeException,
        ClaudeCodeCLINotFoundError,
        ClaudeCodeExecutionError,
        ClaudeCodeParseError,
        ClaudeCodeTimeoutError,
        ClaudeCodeContentFilterError,
        ClaudeCodeSessionError,
    )
    from _retry import RetryConfig, CircuitBreaker

__all__ = [
    "ClaudeCodeClient",
    "ClaudeCodeSettings",
    "ClaudeAgent",
    "CompactResult",
    "UsageStats",
    "ContextInfo",
    "ClaudeCodeException",
    "ClaudeCodeCLINotFoundError",
    "ClaudeCodeExecutionError",
    "ClaudeCodeParseError",
    "ClaudeCodeTimeoutError",
    "ClaudeCodeContentFilterError",
    "ClaudeCodeSessionError",
    "RetryConfig",
    "CircuitBreaker",
]
