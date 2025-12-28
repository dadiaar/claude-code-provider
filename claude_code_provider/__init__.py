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
    from ._settings import ClaudeCodeSettings, ConfigurationError, VALID_MODELS, ClaudeModel
    from ._validation import ValidationError
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
    from ._process_pool import ClaudeProcessPool, PersistentProcess
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
    from ._logging import (
        DebugLogger,
        StructuredLogger,
        setup_logging,
        configure_logging,
        get_logger,
        LogLevel,
        LogFormat,
        LogEntry,
    )
    from ._sessions import SessionManager, SessionInfo, SessionExport
    from ._batch import BatchProcessor, BatchResult, BatchItem, BatchStatus
    from ._orchestration import (
        # Result types
        OrchestrationResult,
        # Config classes
        GroupChatConfig,
        HandoffConfig,
        MagenticConfig,
        # Orchestrators
        GroupChatOrchestrator,
        HandoffOrchestrator,
        SequentialOrchestrator,
        ConcurrentOrchestrator,
        MagenticOrchestrator,
        FeedbackLoopOrchestrator,
        # Factory functions
        create_review_loop,
        create_pipeline,
        create_parallel_analysis,
        # Limit profiles
        LIMIT_PROFILES,
        MODEL_TIMEOUTS,
        get_limit_profile,
        get_model_timeout,
        # Checkpointing
        Checkpoint,
        CheckpointManager,
        get_checkpoint_manager,
        clear_checkpoints,
        # Graceful stop
        GracefulStopHandler,
        get_stop_handler,
        # Availability flags
        MAF_ORCHESTRATION_AVAILABLE,
        MAF_MAGENTIC_AVAILABLE,
    )
except ImportError:
    from _settings import ClaudeCodeSettings, ConfigurationError, VALID_MODELS, ClaudeModel
    from _validation import ValidationError
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
    from _process_pool import ClaudeProcessPool, PersistentProcess
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
    from _logging import (
        DebugLogger,
        StructuredLogger,
        setup_logging,
        configure_logging,
        get_logger,
        LogLevel,
        LogFormat,
        LogEntry,
    )
    from _sessions import SessionManager, SessionInfo, SessionExport
    from _batch import BatchProcessor, BatchResult, BatchItem, BatchStatus
    from _orchestration import (
        # Result types
        OrchestrationResult,
        # Config classes
        GroupChatConfig,
        HandoffConfig,
        MagenticConfig,
        # Orchestrators
        GroupChatOrchestrator,
        HandoffOrchestrator,
        SequentialOrchestrator,
        ConcurrentOrchestrator,
        MagenticOrchestrator,
        FeedbackLoopOrchestrator,
        # Factory functions
        create_review_loop,
        create_pipeline,
        create_parallel_analysis,
        # Limit profiles
        LIMIT_PROFILES,
        MODEL_TIMEOUTS,
        get_limit_profile,
        get_model_timeout,
        # Checkpointing
        Checkpoint,
        CheckpointManager,
        get_checkpoint_manager,
        clear_checkpoints,
        # Graceful stop
        GracefulStopHandler,
        get_stop_handler,
        # Availability flags
        MAF_ORCHESTRATION_AVAILABLE,
        MAF_MAGENTIC_AVAILABLE,
    )

__all__ = [
    # Core
    "ClaudeCodeClient",
    "ClaudeCodeSettings",
    "ConfigurationError",
    "ValidationError",
    "VALID_MODELS",
    "ClaudeModel",
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
    # Process Pool
    "ClaudeProcessPool",
    "PersistentProcess",
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
    "StructuredLogger",
    "setup_logging",
    "configure_logging",
    "get_logger",
    "LogLevel",
    "LogFormat",
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
    # Orchestration
    "OrchestrationResult",
    "GroupChatConfig",
    "HandoffConfig",
    "MagenticConfig",
    "GroupChatOrchestrator",
    "HandoffOrchestrator",
    "SequentialOrchestrator",
    "ConcurrentOrchestrator",
    "MagenticOrchestrator",
    "FeedbackLoopOrchestrator",
    "create_review_loop",
    "create_pipeline",
    "create_parallel_analysis",
    # Limit Profiles
    "LIMIT_PROFILES",
    "MODEL_TIMEOUTS",
    "get_limit_profile",
    "get_model_timeout",
    # Checkpointing
    "Checkpoint",
    "CheckpointManager",
    "get_checkpoint_manager",
    "clear_checkpoints",
    # Graceful Stop
    "GracefulStopHandler",
    "get_stop_handler",
    # Availability Flags
    "MAF_ORCHESTRATION_AVAILABLE",
    "MAF_MAGENTIC_AVAILABLE",
]
