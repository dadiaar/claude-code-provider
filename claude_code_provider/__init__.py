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
    from ._mcp import MCPServer, MCPTransport, MCPManager, MCPServerInfo
    from ._cost import CostTracker, RequestCost, CostSummary
    from ._routing import (
        ModelRouter,
        RoutingStrategy,
        RoutingContext,
        ComplexityRouter,
        CostOptimizedRouter,
        TaskTypeRouter,
        CustomRouter,
        SimpleRouter,
        ModelTier,
    )
    from ._logging import DebugLogger, setup_logging, LogLevel, LogEntry
    from ._sessions import SessionManager, SessionInfo, SessionExport
    from ._batch import BatchProcessor, BatchResult, BatchItem, BatchStatus
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
    from _mcp import MCPServer, MCPTransport, MCPManager, MCPServerInfo
    from _cost import CostTracker, RequestCost, CostSummary
    from _routing import (
        ModelRouter,
        RoutingStrategy,
        RoutingContext,
        ComplexityRouter,
        CostOptimizedRouter,
        TaskTypeRouter,
        CustomRouter,
        SimpleRouter,
        ModelTier,
    )
    from _logging import DebugLogger, setup_logging, LogLevel, LogEntry
    from _sessions import SessionManager, SessionInfo, SessionExport
    from _batch import BatchProcessor, BatchResult, BatchItem, BatchStatus

__all__ = [
    # Core
    "ClaudeCodeClient",
    "ClaudeCodeSettings",
    "ClaudeAgent",
    "CompactResult",
    "UsageStats",
    "ContextInfo",
    # Exceptions
    "ClaudeCodeException",
    "ClaudeCodeCLINotFoundError",
    "ClaudeCodeExecutionError",
    "ClaudeCodeParseError",
    "ClaudeCodeTimeoutError",
    "ClaudeCodeContentFilterError",
    "ClaudeCodeSessionError",
    # Retry/Resilience
    "RetryConfig",
    "CircuitBreaker",
    # MCP
    "MCPServer",
    "MCPTransport",
    "MCPManager",
    "MCPServerInfo",
    # Cost Tracking
    "CostTracker",
    "RequestCost",
    "CostSummary",
    # Model Routing
    "ModelRouter",
    "RoutingStrategy",
    "RoutingContext",
    "ComplexityRouter",
    "CostOptimizedRouter",
    "TaskTypeRouter",
    "CustomRouter",
    "SimpleRouter",
    "ModelTier",
    # Logging
    "DebugLogger",
    "setup_logging",
    "LogLevel",
    "LogEntry",
    # Sessions
    "SessionManager",
    "SessionInfo",
    "SessionExport",
    # Batch Processing
    "BatchProcessor",
    "BatchResult",
    "BatchItem",
    "BatchStatus",
]
