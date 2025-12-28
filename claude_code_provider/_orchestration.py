# Copyright (c) 2025. Claude Code Provider for Microsoft Agent Framework.

"""Orchestration builders for multi-agent workflows.

This module exposes MAF's orchestration patterns with Claude Code enhancements:
- GroupChat: Dynamic multi-agent with manager-directed selection
- Handoff: Swarm-like coordinator → specialist routing
- Magentic: Complex autonomous orchestration with replanning
- Sequential: Simple linear agent chains
- Concurrent: Parallel fan-out/fan-in patterns

Example:
    ```python
    from claude_code_provider import ClaudeCodeClient, GroupChatOrchestrator

    client = ClaudeCodeClient(model="sonnet")

    # Create agents
    researcher = client.create_agent(name="researcher", instructions="...")
    writer = client.create_agent(name="writer", instructions="...")
    reviewer = client.create_agent(name="reviewer", instructions="...")

    # Create group chat with manager
    def select_speaker(state):
        # Logic to pick next speaker based on conversation
        last = state.conversation[-1].text if state.conversation else ""
        if "research complete" in last.lower():
            return "writer"
        elif "draft complete" in last.lower():
            return "reviewer"
        elif "approved" in last.lower():
            return None  # Finish
        return "researcher"

    orchestrator = GroupChatOrchestrator(
        participants=[researcher, writer, reviewer],
        manager=select_speaker,
        max_rounds=10,
    )

    result = await orchestrator.run("Write an article about AI safety")
    ```
"""

from dataclasses import dataclass, field
from typing import Any, Callable, TypeVar, TYPE_CHECKING
from collections.abc import Sequence
import asyncio
import json
import os
import re
import secrets
import signal
import stat
import hashlib
from pathlib import Path
from datetime import datetime

from agent_framework import ChatMessage, Role

# Import MAF orchestration builders
try:
    from agent_framework._workflows import (
        GroupChatBuilder,
        HandoffBuilder,
        SequentialBuilder,
        ConcurrentBuilder,
    )
    from agent_framework._workflows._group_chat import GroupChatStateSnapshot
    from agent_framework._workflows._events import (
        AgentRunUpdateEvent,
        AgentRunEvent,
        WorkflowOutputEvent,
        ExecutorCompletedEvent,
    )
    MAF_ORCHESTRATION_AVAILABLE = True
except ImportError:
    MAF_ORCHESTRATION_AVAILABLE = False
    GroupChatBuilder = None
    HandoffBuilder = None
    SequentialBuilder = None
    ConcurrentBuilder = None
    GroupChatStateSnapshot = None
    AgentRunUpdateEvent = None
    AgentRunEvent = None
    WorkflowOutputEvent = None
    ExecutorCompletedEvent = None

# Try to import MagenticBuilder (may be in separate location)
try:
    from agent_framework._workflows import MagenticBuilder
    from agent_framework._workflows._magentic import StandardMagenticManager
    MAF_MAGENTIC_AVAILABLE = True
except ImportError:
    MAF_MAGENTIC_AVAILABLE = False
    MagenticBuilder = None
    StandardMagenticManager = None

if TYPE_CHECKING:
    from ._agent import ClaudeAgent
    from ._cost import CostTracker


T = TypeVar("T")


# =============================================================================
# LIMIT PROFILES
# =============================================================================

# Model-aware per-agent timeouts (in seconds)
MODEL_TIMEOUTS = {
    "haiku": 600,      # 10 minutes
    "sonnet": 1800,    # 30 minutes
    "opus": 3600,      # 60 minutes
    "default": 1800,   # 30 minutes fallback
}

# Predefined limit profiles for different use cases
# Note: checkpoint_enabled is False by default for security in multi-user scenarios.
# Enable explicitly when needed: checkpoint_enabled=True
LIMIT_PROFILES = {
    "demo": {
        "description": "Quick demos and testing",
        "max_iterations": 5,
        "timeout_seconds": 300,           # 5 minutes
        "per_agent_timeout": 120,         # 2 minutes
        "checkpoint_enabled": False,      # Off by default for security
    },
    "standard": {
        "description": "Typical production tasks",
        "max_iterations": 20,
        "timeout_seconds": 3600,          # 1 hour
        "per_agent_timeout": None,        # Use MODEL_TIMEOUTS
        "checkpoint_enabled": False,      # Off by default for security
    },
    "extended": {
        "description": "Complex multi-step workflows",
        "max_iterations": 100,
        "timeout_seconds": 7200,          # 2 hours
        "per_agent_timeout": None,        # Use MODEL_TIMEOUTS
        "checkpoint_enabled": False,      # Off by default for security
    },
    "unlimited": {
        "description": "Long-running tasks, generous limits",
        "max_iterations": 500,
        "timeout_seconds": 14400,         # 4 hours
        "per_agent_timeout": None,        # Use MODEL_TIMEOUTS
        "checkpoint_enabled": False,      # Off by default for security
    },
}

# Default profile
DEFAULT_PROFILE = "standard"


def get_limit_profile(name: str | None = None) -> dict[str, Any]:
    """Get a limit profile by name.

    Args:
        name: Profile name ("demo", "standard", "extended", "unlimited").
              If None, returns the default profile.

    Returns:
        Dict with limit configuration.

    Example:
        ```python
        profile = get_limit_profile("extended")
        orchestrator = FeedbackLoopOrchestrator(
            worker=dev,
            reviewer=rev,
            max_iterations=profile["max_iterations"],
            timeout_seconds=profile["timeout_seconds"],
        )
        ```
    """
    profile_name = name or DEFAULT_PROFILE
    if profile_name not in LIMIT_PROFILES:
        raise ValueError(f"Unknown profile: {profile_name}. Available: {list(LIMIT_PROFILES.keys())}")
    return LIMIT_PROFILES[profile_name].copy()


def get_model_timeout(model: str) -> int:
    """Get timeout for a specific model.

    Args:
        model: Model name (haiku, sonnet, opus).

    Returns:
        Timeout in seconds.
    """
    model_lower = model.lower()
    for key in MODEL_TIMEOUTS:
        if key in model_lower:
            return MODEL_TIMEOUTS[key]
    return MODEL_TIMEOUTS["default"]


# =============================================================================
# CHECKPOINTING SYSTEM
# =============================================================================

@dataclass
class Checkpoint:
    """Checkpoint state for orchestration resumption."""

    checkpoint_id: str
    """Unique identifier for this checkpoint."""

    orchestration_type: str
    """Type of orchestration (feedback_loop, group_chat, etc.)."""

    task: str
    """Original task being performed."""

    conversation: list[dict[str, Any]]
    """Serialized conversation history."""

    current_iteration: int
    """Current iteration/round number."""

    current_work: str
    """Latest work output."""

    feedback: str
    """Latest feedback (if any)."""

    participants_used: list[str]
    """Names of participants that have contributed."""

    metadata: dict[str, Any]
    """Additional orchestration-specific state."""

    created_at: str
    """ISO timestamp when checkpoint was created."""

    updated_at: str
    """ISO timestamp when checkpoint was last updated."""

    status: str
    """Status: 'in_progress', 'completed', 'timeout', 'stopped'."""


class CheckpointManager:
    """Manages saving and loading orchestration checkpoints.

    Checkpoints are saved as JSON files in the checkpoint directory.
    Each orchestration run gets a unique checkpoint ID based on the task hash
    combined with cryptographic randomness to prevent collisions.

    Security features:
    - Cryptographic random IDs prevent collision attacks
    - Symlink detection prevents path traversal via symlinks
    - File size limits prevent OOM via malicious checkpoint files
    - Secure file permissions (0o600) protect checkpoint data

    Example:
        ```python
        manager = CheckpointManager()

        # Save checkpoint
        manager.save(checkpoint)

        # Load checkpoint (returns None if not found)
        checkpoint = manager.load(checkpoint_id)

        # Clear all checkpoints
        manager.clear_all()

        # Clear specific checkpoint
        manager.clear(checkpoint_id)
        ```
    """

    # Maximum checkpoint file size (10 MB) to prevent OOM
    MAX_CHECKPOINT_SIZE = 10 * 1024 * 1024

    def __init__(self, checkpoint_dir: str | Path = "./checkpoints"):
        """Initialize checkpoint manager.

        Args:
            checkpoint_dir: Directory to store checkpoints.
        """
        self.checkpoint_dir = Path(checkpoint_dir).resolve()
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def _get_checkpoint_path(self, checkpoint_id: str) -> Path:
        """Get file path for a checkpoint.

        Args:
            checkpoint_id: The checkpoint ID.

        Returns:
            Path to the checkpoint file.

        Raises:
            ValueError: If checkpoint_id contains invalid characters, path traversal,
                       or the path is a symlink.
        """
        # Validate checkpoint ID format (alphanumeric, underscores, hyphens only)
        if not checkpoint_id or not all(c.isalnum() or c in '_-' for c in checkpoint_id):
            raise ValueError(
                f"Invalid checkpoint ID format: '{checkpoint_id}'. "
                "Only alphanumeric characters, underscores, and hyphens are allowed."
            )

        path = self.checkpoint_dir / f"{checkpoint_id}.json"

        # Ensure path is within checkpoint directory (prevent path traversal)
        # Note: self.checkpoint_dir is already resolved in __init__
        try:
            path.resolve().relative_to(self.checkpoint_dir)
        except ValueError:
            raise ValueError(f"Checkpoint path traversal detected: {checkpoint_id}")

        # Prevent symlink attacks: if the file exists, ensure it's not a symlink
        # Use lstat() to check the link itself, not the target
        if path.exists() and path.is_symlink():
            raise ValueError(
                f"Checkpoint file is a symlink (potential symlink attack): {checkpoint_id}"
            )

        return path

    def generate_checkpoint_id(
        self,
        task: str,
        orchestration_type: str,
        unique: bool = True,
    ) -> str:
        """Generate a checkpoint ID based on task and type.

        Args:
            task: The task description.
            orchestration_type: Type of orchestration.
            unique: If True (default), appends cryptographic random suffix
                   to ensure uniqueness. If False, returns deterministic ID
                   based on task hash (useful for resumption by task).

        Returns:
            Checkpoint ID (unique by default, deterministic if unique=False).

        Security:
            Uses secrets.token_hex() for cryptographic randomness when unique=True,
            preventing checkpoint ID collision attacks.
        """
        content = f"{orchestration_type}:{task}"
        hash_part = hashlib.sha256(content.encode()).hexdigest()[:12]

        if unique:
            # Add cryptographic random suffix to prevent collisions
            random_suffix = secrets.token_hex(4)  # 8 hex chars = 32 bits
            return f"{orchestration_type}_{hash_part}_{random_suffix}"
        else:
            # Deterministic ID for task-based resumption
            return f"{orchestration_type}_{hash_part}"

    def save(self, checkpoint: Checkpoint) -> Path:
        """Save a checkpoint to disk with secure permissions.

        Uses atomic write pattern to prevent data corruption on crashes:
        1. Write to temp file with secure permissions
        2. Verify permissions
        3. Atomic rename to target

        Args:
            checkpoint: Checkpoint to save.

        Returns:
            Path to saved checkpoint file.
        """
        import tempfile

        checkpoint.updated_at = datetime.now().isoformat()

        # Serialize checkpoint to JSON-compatible dict
        data = {
            "checkpoint_id": checkpoint.checkpoint_id,
            "orchestration_type": checkpoint.orchestration_type,
            "task": checkpoint.task,
            "conversation": checkpoint.conversation,
            "current_iteration": checkpoint.current_iteration,
            "current_work": checkpoint.current_work,
            "feedback": checkpoint.feedback,
            "participants_used": checkpoint.participants_used,
            "metadata": checkpoint.metadata,
            "created_at": checkpoint.created_at,
            "updated_at": checkpoint.updated_at,
            "status": checkpoint.status,
        }

        path = self._get_checkpoint_path(checkpoint.checkpoint_id)

        # Ensure directory has secure permissions (owner only)
        try:
            os.chmod(self.checkpoint_dir, stat.S_IRWXU)  # 0o700
        except OSError:
            pass  # May fail on some filesystems

        # Use atomic write pattern to prevent corruption on crash
        json_content = json.dumps(data, indent=2)
        temp_fd = None
        temp_path = None

        try:
            # Create temp file with secure permissions atomically
            temp_fd, temp_path = tempfile.mkstemp(
                dir=str(self.checkpoint_dir),
                prefix='.checkpoint_',
                suffix='.tmp'
            )
            # mkstemp creates with 0o600 by default, but ensure it
            os.fchmod(temp_fd, stat.S_IRUSR | stat.S_IWUSR)  # 0o600

            # Write content
            os.write(temp_fd, json_content.encode('utf-8'))
            os.fsync(temp_fd)  # Ensure data is on disk
            os.close(temp_fd)
            temp_fd = None

            # Atomic rename to target
            os.replace(temp_path, str(path))
            temp_path = None  # Successfully moved

        finally:
            # Cleanup on failure
            if temp_fd is not None:
                try:
                    os.close(temp_fd)
                except OSError:
                    pass
            if temp_path is not None:
                try:
                    os.unlink(temp_path)
                except OSError:
                    pass

        return path

    def load(self, checkpoint_id: str) -> Checkpoint | None:
        """Load a checkpoint from disk.

        Args:
            checkpoint_id: ID of checkpoint to load.

        Returns:
            Checkpoint if found, None otherwise.

        Raises:
            ValueError: If checkpoint file exceeds size limit (prevents OOM).
        """
        path = self._get_checkpoint_path(checkpoint_id)
        if not path.exists():
            return None

        # Check file size before loading to prevent OOM attacks
        file_size = path.stat().st_size
        if file_size > self.MAX_CHECKPOINT_SIZE:
            raise ValueError(
                f"Checkpoint file exceeds size limit: {file_size} bytes "
                f"(max: {self.MAX_CHECKPOINT_SIZE} bytes). "
                "This may indicate a malicious or corrupted checkpoint file."
            )

        with open(path) as f:
            data = json.load(f)

        return Checkpoint(
            checkpoint_id=data["checkpoint_id"],
            orchestration_type=data["orchestration_type"],
            task=data["task"],
            conversation=data["conversation"],
            current_iteration=data["current_iteration"],
            current_work=data["current_work"],
            feedback=data["feedback"],
            participants_used=data["participants_used"],
            metadata=data["metadata"],
            created_at=data["created_at"],
            updated_at=data["updated_at"],
            status=data["status"],
        )

    def exists(self, checkpoint_id: str) -> bool:
        """Check if a checkpoint exists.

        Args:
            checkpoint_id: ID to check.

        Returns:
            True if checkpoint exists.
        """
        return self._get_checkpoint_path(checkpoint_id).exists()

    def clear(self, checkpoint_id: str) -> bool:
        """Delete a specific checkpoint.

        Args:
            checkpoint_id: ID of checkpoint to delete.

        Returns:
            True if deleted, False if not found.
        """
        path = self._get_checkpoint_path(checkpoint_id)
        if path.exists():
            path.unlink()
            return True
        return False

    def clear_all(self) -> int:
        """Delete all checkpoints.

        Returns:
            Number of checkpoints deleted.
        """
        count = 0
        for path in self.checkpoint_dir.glob("*.json"):
            path.unlink()
            count += 1
        return count

    async def async_save(self, checkpoint: Checkpoint) -> Path:
        """Async version of save() - runs blocking I/O in thread pool.

        Use this in async contexts to avoid blocking the event loop.

        Args:
            checkpoint: Checkpoint to save.

        Returns:
            Path to saved checkpoint file.
        """
        return await asyncio.to_thread(self.save, checkpoint)

    async def async_load(self, checkpoint_id: str) -> Checkpoint | None:
        """Async version of load() - runs blocking I/O in thread pool.

        Use this in async contexts to avoid blocking the event loop.

        Args:
            checkpoint_id: ID of checkpoint to load.

        Returns:
            Checkpoint if found, None otherwise.
        """
        return await asyncio.to_thread(self.load, checkpoint_id)

    async def async_clear(self, checkpoint_id: str) -> bool:
        """Async version of clear() - runs blocking I/O in thread pool.

        Args:
            checkpoint_id: ID of checkpoint to delete.

        Returns:
            True if deleted, False if not found.
        """
        return await asyncio.to_thread(self.clear, checkpoint_id)

    def list_checkpoints(self) -> list[Checkpoint]:
        """List all available checkpoints.

        Returns:
            List of all checkpoints.
        """
        checkpoints = []
        for path in self.checkpoint_dir.glob("*.json"):
            checkpoint_id = path.stem
            checkpoint = self.load(checkpoint_id)
            if checkpoint:
                checkpoints.append(checkpoint)
        return checkpoints


# Cache of checkpoint managers by directory (resolved path as key)
_checkpoint_managers: dict[Path, CheckpointManager] = {}


def get_checkpoint_manager(checkpoint_dir: str | Path = "./checkpoints") -> CheckpointManager:
    """Get or create a checkpoint manager for the specified directory.

    This function caches managers per directory to avoid creating multiple
    managers for the same directory while correctly handling different directories.

    Args:
        checkpoint_dir: Directory for checkpoints.

    Returns:
        CheckpointManager instance for the specified directory.

    Note:
        Unlike a simple singleton, this properly respects the checkpoint_dir
        parameter. Each unique directory gets its own cached manager.
    """
    resolved_dir = Path(checkpoint_dir).resolve()
    if resolved_dir not in _checkpoint_managers:
        _checkpoint_managers[resolved_dir] = CheckpointManager(checkpoint_dir)
    return _checkpoint_managers[resolved_dir]


def clear_checkpoints(checkpoint_dir: str | Path = "./checkpoints") -> int:
    """Convenience function to clear all checkpoints.

    Args:
        checkpoint_dir: Directory containing checkpoints.

    Returns:
        Number of checkpoints deleted.
    """
    manager = CheckpointManager(checkpoint_dir)
    return manager.clear_all()


# =============================================================================
# GRACEFUL STOP HANDLING
# =============================================================================

class GracefulStopHandler:
    """Handler for graceful stop on SIGINT (Ctrl+C) or SIGTERM.

    When triggered, sets a flag that orchestrators can check to stop
    gracefully and save their checkpoint.

    Thread-Safety Note:
        Each orchestrator should create its own GracefulStopHandler instance
        to avoid conflicts when multiple orchestrators run concurrently.
        The global get_stop_handler() is provided for simple single-orchestrator
        use cases but should be avoided in multi-orchestrator scenarios.

    Signal Handler Safety:
        The signal handler only sets an atomic flag and does NOT perform I/O
        (print, logging, file operations). This ensures async-signal-safety.
        Any logging of the stop event should be done by the main code when
        it detects should_stop is True.

    Example:
        ```python
        handler = GracefulStopHandler()
        handler.register()

        while not handler.should_stop:
            # ... do work ...
            pass

        if handler.should_stop:
            # Log and save checkpoint in main code (not signal handler)
            print("Graceful stop requested, saving checkpoint...")
            # Save checkpoint and exit
            pass
        ```
    """

    def __init__(self):
        self.should_stop = False
        self._original_sigint = None
        self._original_sigterm = None
        self._registered = False

    def _signal_handler(self, signum, frame):
        """Handle stop signal.

        IMPORTANT: This handler ONLY sets a flag. It does NOT perform any I/O
        operations (print, logging, file writes) because those are not
        async-signal-safe and can cause deadlocks or undefined behavior.
        """
        # Only set the flag - no I/O operations allowed in signal handlers
        self.should_stop = True

    def register(self):
        """Register signal handlers.

        Note: If called multiple times without unregister(), subsequent calls
        are no-ops. This prevents accidentally overwriting the original handlers.
        """
        if self._registered:
            return  # Already registered, don't overwrite

        self._original_sigint = signal.signal(signal.SIGINT, self._signal_handler)
        try:
            self._original_sigterm = signal.signal(signal.SIGTERM, self._signal_handler)
        except (ValueError, OSError):
            # SIGTERM might not be available on all platforms
            pass
        self._registered = True

    def unregister(self):
        """Restore original signal handlers."""
        if not self._registered:
            return  # Nothing to unregister

        if self._original_sigint is not None:
            signal.signal(signal.SIGINT, self._original_sigint)
            self._original_sigint = None
        if self._original_sigterm is not None:
            try:
                signal.signal(signal.SIGTERM, self._original_sigterm)
            except (ValueError, OSError):
                pass
            self._original_sigterm = None
        self._registered = False

    def reset(self):
        """Reset the stop flag."""
        self.should_stop = False

    @property
    def is_registered(self) -> bool:
        """Check if signal handlers are currently registered."""
        return self._registered


# Thread-local storage for stop handlers (one per thread)
import threading
_stop_handler_local = threading.local()


def get_stop_handler() -> GracefulStopHandler:
    """Get the thread-local stop handler.

    Returns a thread-local GracefulStopHandler instance. Each thread gets
    its own handler to avoid conflicts in multi-threaded applications.

    For multi-orchestrator scenarios within the same thread, consider
    creating explicit GracefulStopHandler instances instead.
    """
    if not hasattr(_stop_handler_local, 'handler'):
        _stop_handler_local.handler = GracefulStopHandler()
    return _stop_handler_local.handler


# =============================================================================
# ORCHESTRATION RESULT
# =============================================================================

@dataclass
class OrchestrationResult:
    """Result from an orchestration run."""

    final_output: str
    """The final synthesized output."""

    conversation: list[ChatMessage]
    """Full conversation history."""

    rounds: int
    """Number of orchestration rounds executed."""

    participants_used: set[str]
    """Names of participants that contributed."""

    metadata: dict[str, Any] = field(default_factory=dict)
    """Additional metadata from the orchestration."""


@dataclass
class GroupChatConfig:
    """Configuration for GroupChat orchestration."""

    max_rounds: int = 15
    """Maximum number of manager selection rounds."""

    termination_condition: Callable[[Any], bool] | None = None
    """Optional condition to terminate early."""

    manager_display_name: str = "manager"
    """Display name for the manager in logs."""

    final_message: str | None = None
    """Optional message to append when finishing."""


@dataclass
class HandoffConfig:
    """Configuration for Handoff orchestration."""

    autonomous: bool = True
    """Run in autonomous mode (agents iterate without user input)."""

    autonomous_turn_limit: int = 50
    """Maximum turns in autonomous mode."""

    termination_condition: Callable[[Any], bool] | None = None
    """Optional condition to terminate early."""

    enable_return_to_previous: bool = False
    """Allow returning to previous agent instead of coordinator."""


@dataclass
class MagenticConfig:
    """Configuration for Magentic orchestration."""

    max_stall_count: int = 3
    """Maximum times to replan when stalled."""

    max_reset_count: int | None = None
    """Maximum conversation resets (None = unlimited)."""

    max_round_count: int | None = None
    """Maximum total rounds (None = unlimited)."""

    progress_ledger_retry_count: int = 3
    """Retries for parsing progress ledger JSON."""


class GroupChatOrchestrator:
    """Orchestrator using MAF's GroupChatBuilder for multi-agent coordination.

    The manager (function or agent) decides which participant speaks next
    based on the full conversation history. Supports feedback loops where
    agents can be redirected back for revisions.

    Example with function manager:
        ```python
        def select_speaker(state):
            last_msg = state.conversation[-1].text.lower() if state.conversation else ""
            if "needs revision" in last_msg:
                return "developer"  # Send back for fixes
            elif "approved" in last_msg:
                return None  # Finish
            return "reviewer"  # Continue to review

        orchestrator = GroupChatOrchestrator(
            participants=[developer, reviewer],
            manager=select_speaker,
            max_rounds=10,
        )
        ```

    Example with agent manager:
        ```python
        manager_agent = client.create_agent(
            name="coordinator",
            instructions="Select next speaker: developer, reviewer, or 'finish'",
        )

        orchestrator = GroupChatOrchestrator(
            participants=[developer, reviewer],
            manager=manager_agent,
        )
        ```
    """

    def __init__(
        self,
        participants: Sequence["ClaudeAgent"],
        manager: Callable | "ClaudeAgent",
        *,
        config: GroupChatConfig | None = None,
        cost_tracker: "CostTracker | None" = None,
    ):
        """Initialize GroupChat orchestrator.

        Args:
            participants: Agents that can be selected to speak.
            manager: Function or agent that selects the next speaker.
            config: Configuration options.
            cost_tracker: Optional cost tracker for monitoring usage.
        """
        if not MAF_ORCHESTRATION_AVAILABLE:
            raise ImportError(
                "MAF orchestration builders not available. "
                "Ensure agent_framework is properly installed."
            )

        self.participants = participants
        self.manager = manager
        self.config = config or GroupChatConfig()
        self.cost_tracker = cost_tracker
        self._workflow = None

    def _build_workflow(self):
        """Build the MAF workflow."""
        # Get inner agents from ClaudeAgent wrappers
        inner_agents = {
            agent.name: agent._agent
            for agent in self.participants
        }

        builder = GroupChatBuilder()
        builder = builder.participants(inner_agents)

        # Set manager
        if callable(self.manager) and not hasattr(self.manager, '_agent'):
            # Function-based manager
            builder = builder.set_select_speakers_func(
                self.manager,
                display_name=self.config.manager_display_name,
                final_message=self.config.final_message,
            )
        else:
            # Agent-based manager
            manager_agent = self.manager._agent if hasattr(self.manager, '_agent') else self.manager
            builder = builder.set_manager(
                manager_agent,
                display_name=self.config.manager_display_name,
            )

        # Set limits
        builder = builder.with_max_rounds(self.config.max_rounds)

        # Set termination condition
        if self.config.termination_condition:
            builder = builder.with_termination_condition(self.config.termination_condition)

        return builder.build()

    async def run(self, task: str) -> OrchestrationResult:
        """Run the orchestration with a task.

        Args:
            task: The task/goal for the multi-agent team.

        Returns:
            OrchestrationResult with final output and metadata.
        """
        if self._workflow is None:
            self._workflow = self._build_workflow()

        # Run the workflow
        conversation = []
        participants_used = set()
        rounds = 0
        last_text = ""

        async for event in self._workflow.run_stream(task):
            # Handle AgentRunUpdateEvent - captures streaming agent responses
            if AgentRunUpdateEvent and isinstance(event, AgentRunUpdateEvent):
                if event.executor_id:
                    participants_used.add(event.executor_id)
                if event.data and hasattr(event.data, 'text') and event.data.text:
                    last_text = event.data.text
                rounds += 1

            # Handle WorkflowOutputEvent - captures final conversation
            elif WorkflowOutputEvent and isinstance(event, WorkflowOutputEvent):
                if event.data and isinstance(event.data, list):
                    conversation = event.data

        # Get final output from last text or conversation
        final_output = last_text
        if not final_output and conversation:
            last_msg = conversation[-1]
            if hasattr(last_msg, 'text'):
                final_output = last_msg.text
            elif isinstance(last_msg, dict) and 'text' in last_msg:
                final_output = last_msg['text']
            else:
                final_output = str(last_msg)

        return OrchestrationResult(
            final_output=final_output,
            conversation=conversation,
            rounds=rounds,
            participants_used=participants_used,
            metadata={"orchestration_type": "group_chat"},
        )


class HandoffOrchestrator:
    """Orchestrator using MAF's HandoffBuilder for coordinator → specialist patterns.

    A coordinator agent routes tasks to specialists via tool calls. Supports
    both human-in-the-loop and autonomous modes.

    Example:
        ```python
        coordinator = client.create_agent(
            name="coordinator",
            instructions="Route to billing, technical, or account specialist",
        )

        orchestrator = HandoffOrchestrator(
            coordinator=coordinator,
            specialists=[billing_agent, technical_agent, account_agent],
            autonomous=True,
            autonomous_turn_limit=20,
        )

        result = await orchestrator.run("Customer: I was charged twice")
        ```
    """

    def __init__(
        self,
        coordinator: "ClaudeAgent",
        specialists: Sequence["ClaudeAgent"],
        *,
        config: HandoffConfig | None = None,
        cost_tracker: "CostTracker | None" = None,
    ):
        """Initialize Handoff orchestrator.

        Args:
            coordinator: The routing/coordinator agent.
            specialists: Specialist agents to route to.
            config: Configuration options.
            cost_tracker: Optional cost tracker.
        """
        if not MAF_ORCHESTRATION_AVAILABLE:
            raise ImportError(
                "MAF orchestration builders not available. "
                "Ensure agent_framework is properly installed."
            )

        self.coordinator = coordinator
        self.specialists = specialists
        self.config = config or HandoffConfig()
        self.cost_tracker = cost_tracker
        self._workflow = None

    def _build_workflow(self):
        """Build the MAF workflow."""
        # Collect all participants
        all_participants = [self.coordinator._agent] + [s._agent for s in self.specialists]

        builder = HandoffBuilder(participants=all_participants)
        builder = builder.set_coordinator(self.coordinator._agent)

        # Configure interaction mode
        if self.config.autonomous:
            builder = builder.with_interaction_mode(
                "autonomous",
                autonomous_turn_limit=self.config.autonomous_turn_limit,
            )

        # Set termination condition
        if self.config.termination_condition:
            builder = builder.with_termination_condition(self.config.termination_condition)

        # Enable return to previous
        if self.config.enable_return_to_previous:
            builder = builder.enable_return_to_previous()

        return builder.build()

    async def run(self, task: str) -> OrchestrationResult:
        """Run the orchestration with a task.

        Args:
            task: The task/request to handle.

        Returns:
            OrchestrationResult with final output and metadata.
        """
        if self._workflow is None:
            self._workflow = self._build_workflow()

        conversation = []
        participants_used = set()
        rounds = 0
        last_text = ""

        async for event in self._workflow.run_stream(task):
            # Handle AgentRunUpdateEvent - captures streaming agent responses
            if AgentRunUpdateEvent and isinstance(event, AgentRunUpdateEvent):
                if event.executor_id:
                    participants_used.add(event.executor_id)
                if event.data and hasattr(event.data, 'text') and event.data.text:
                    last_text = event.data.text
                rounds += 1

            # Handle WorkflowOutputEvent - captures final conversation
            elif WorkflowOutputEvent and isinstance(event, WorkflowOutputEvent):
                if event.data and isinstance(event.data, list):
                    conversation = event.data

        # Get final output from last text or conversation
        final_output = last_text
        if not final_output and conversation:
            last_msg = conversation[-1]
            if hasattr(last_msg, 'text'):
                final_output = last_msg.text
            elif isinstance(last_msg, dict) and 'text' in last_msg:
                final_output = last_msg['text']
            else:
                final_output = str(last_msg)

        return OrchestrationResult(
            final_output=final_output,
            conversation=conversation,
            rounds=rounds,
            participants_used=participants_used,
            metadata={"orchestration_type": "handoff"},
        )


class SequentialOrchestrator:
    """Orchestrator using MAF's SequentialBuilder for linear agent chains.

    Agents process in sequence, each building on the previous output.
    No feedback loops - purely linear progression.

    Example:
        ```python
        orchestrator = SequentialOrchestrator(
            agents=[researcher, writer, editor],
        )

        result = await orchestrator.run("Write about climate change")
        # researcher → writer → editor
        ```
    """

    def __init__(
        self,
        agents: Sequence["ClaudeAgent"],
        *,
        cost_tracker: "CostTracker | None" = None,
    ):
        """Initialize Sequential orchestrator.

        Args:
            agents: Agents to run in sequence.
            cost_tracker: Optional cost tracker.
        """
        if not MAF_ORCHESTRATION_AVAILABLE:
            raise ImportError(
                "MAF orchestration builders not available."
            )

        self.agents = agents
        self.cost_tracker = cost_tracker
        self._workflow = None

    def _build_workflow(self):
        """Build the MAF workflow."""
        inner_agents = [agent._agent for agent in self.agents]
        return SequentialBuilder().participants(inner_agents).build()

    async def run(self, task: str) -> OrchestrationResult:
        """Run the orchestration with a task."""
        if self._workflow is None:
            self._workflow = self._build_workflow()

        conversation = []
        participants_used = set()
        last_text = ""

        async for event in self._workflow.run_stream(task):
            # Handle AgentRunUpdateEvent - captures streaming agent responses
            if AgentRunUpdateEvent and isinstance(event, AgentRunUpdateEvent):
                if event.executor_id:
                    participants_used.add(event.executor_id)
                if event.data and hasattr(event.data, 'text') and event.data.text:
                    last_text = event.data.text

            # Handle WorkflowOutputEvent - captures final conversation
            elif WorkflowOutputEvent and isinstance(event, WorkflowOutputEvent):
                if event.data and isinstance(event.data, list):
                    conversation = event.data

        # Get final output from last text or conversation
        final_output = last_text
        if not final_output and conversation:
            last_msg = conversation[-1]
            if hasattr(last_msg, 'text'):
                final_output = last_msg.text
            elif isinstance(last_msg, dict) and 'text' in last_msg:
                final_output = last_msg['text']
            else:
                final_output = str(last_msg)

        return OrchestrationResult(
            final_output=final_output,
            conversation=conversation,
            rounds=len(self.agents),
            participants_used=participants_used,
            metadata={"orchestration_type": "sequential"},
        )


class ConcurrentOrchestrator:
    """Orchestrator using MAF's ConcurrentBuilder for parallel execution.

    Fans out input to multiple agents in parallel, then aggregates results.

    Example:
        ```python
        orchestrator = ConcurrentOrchestrator(
            agents=[analyst1, analyst2, analyst3],
            aggregator=lambda results: "\\n---\\n".join(r.text for r in results),
        )

        result = await orchestrator.run("Analyze this market trend")
        # All analysts work in parallel, results combined
        ```
    """

    def __init__(
        self,
        agents: Sequence["ClaudeAgent"],
        *,
        aggregator: Callable | None = None,
        cost_tracker: "CostTracker | None" = None,
    ):
        """Initialize Concurrent orchestrator.

        Args:
            agents: Agents to run in parallel.
            aggregator: Optional function to combine results.
            cost_tracker: Optional cost tracker.
        """
        if not MAF_ORCHESTRATION_AVAILABLE:
            raise ImportError(
                "MAF orchestration builders not available."
            )

        self.agents = agents
        self.aggregator = aggregator
        self.cost_tracker = cost_tracker
        self._workflow = None

    def _build_workflow(self):
        """Build the MAF workflow."""
        inner_agents = [agent._agent for agent in self.agents]
        builder = ConcurrentBuilder().participants(inner_agents)

        if self.aggregator:
            builder = builder.with_aggregator(self.aggregator)

        return builder.build()

    async def run(self, task: str) -> OrchestrationResult:
        """Run the orchestration with a task."""
        if self._workflow is None:
            self._workflow = self._build_workflow()

        conversation = []
        participants_used = set(agent.name for agent in self.agents)
        last_text = ""

        async for event in self._workflow.run_stream(task):
            # Handle AgentRunUpdateEvent - captures streaming agent responses
            if AgentRunUpdateEvent and isinstance(event, AgentRunUpdateEvent):
                if event.executor_id:
                    participants_used.add(event.executor_id)
                if event.data and hasattr(event.data, 'text') and event.data.text:
                    last_text = event.data.text

            # Handle WorkflowOutputEvent - captures final conversation
            elif WorkflowOutputEvent and isinstance(event, WorkflowOutputEvent):
                if event.data and isinstance(event.data, list):
                    conversation = event.data

        # Get final output from last text or conversation
        final_output = last_text
        if not final_output and conversation:
            last_msg = conversation[-1]
            if hasattr(last_msg, 'text'):
                final_output = last_msg.text
            elif isinstance(last_msg, dict) and 'text' in last_msg:
                final_output = last_msg['text']
            else:
                final_output = str(last_msg)

        return OrchestrationResult(
            final_output=final_output,
            conversation=conversation,
            rounds=1,  # All run in parallel
            participants_used=participants_used,
            metadata={"orchestration_type": "concurrent"},
        )


class MagenticOrchestrator:
    """Orchestrator using MAF's MagenticBuilder for autonomous complex tasks.

    Uses sophisticated task/progress ledgers with stall/loop detection and
    automatic replanning. Best for complex tasks requiring adaptive behavior.

    Example:
        ```python
        orchestrator = MagenticOrchestrator(
            participants=[researcher, coder, reviewer],
            manager_agent=coordinator,
            max_stall_count=3,
        )

        result = await orchestrator.run("Build a web scraper for news sites")
        # Manager maintains facts/plan, detects stalls, replans as needed
        ```
    """

    def __init__(
        self,
        participants: Sequence["ClaudeAgent"],
        manager_agent: "ClaudeAgent",
        *,
        config: MagenticConfig | None = None,
        cost_tracker: "CostTracker | None" = None,
    ):
        """Initialize Magentic orchestrator.

        Args:
            participants: Worker agents.
            manager_agent: Manager agent for orchestration decisions.
            config: Configuration options.
            cost_tracker: Optional cost tracker.
        """
        if not MAF_MAGENTIC_AVAILABLE:
            raise ImportError(
                "MAF Magentic orchestration not available. "
                "Ensure agent_framework is properly installed with Magentic support."
            )

        self.participants = participants
        self.manager_agent = manager_agent
        self.config = config or MagenticConfig()
        self.cost_tracker = cost_tracker
        self._workflow = None

    def _build_workflow(self):
        """Build the MAF workflow."""
        # MagenticBuilder.participants() expects keyword arguments
        inner_agents = {agent.name: agent._agent for agent in self.participants}

        builder = MagenticBuilder()
        builder = builder.participants(**inner_agents)
        # Use with_standard_manager with the agent and configuration
        builder = builder.with_standard_manager(
            agent=self.manager_agent._agent,
            max_stall_count=self.config.max_stall_count,
            max_reset_count=self.config.max_reset_count,
            max_round_count=self.config.max_round_count,
        )

        return builder.build()

    async def run(self, task: str) -> OrchestrationResult:
        """Run the orchestration with a task."""
        if self._workflow is None:
            self._workflow = self._build_workflow()

        conversation = []
        participants_used = set()
        rounds = 0
        last_text = ""

        async for event in self._workflow.run_stream(task):
            # Handle AgentRunUpdateEvent - captures streaming agent responses
            if AgentRunUpdateEvent and isinstance(event, AgentRunUpdateEvent):
                if event.executor_id:
                    participants_used.add(event.executor_id)
                if event.data and hasattr(event.data, 'text') and event.data.text:
                    last_text = event.data.text
                rounds += 1

            # Handle WorkflowOutputEvent - captures final conversation
            elif WorkflowOutputEvent and isinstance(event, WorkflowOutputEvent):
                if event.data and isinstance(event.data, list):
                    conversation = event.data

        # Get final output from last text or conversation
        final_output = last_text
        if not final_output and conversation:
            last_msg = conversation[-1]
            if hasattr(last_msg, 'text'):
                final_output = last_msg.text
            elif isinstance(last_msg, dict) and 'text' in last_msg:
                final_output = last_msg['text']
            else:
                final_output = str(last_msg)

        return OrchestrationResult(
            final_output=final_output,
            conversation=conversation,
            rounds=rounds,
            participants_used=participants_used,
            metadata={"orchestration_type": "magentic"},
        )


# Simple orchestrator for feedback loops without full MAF workflow
class FeedbackLoopOrchestrator:
    """Orchestrator with feedback loop, checkpointing, and graceful stop.

    Features:
    - Worker → Reviewer feedback loop until approved or max iterations
    - Auto-resume from checkpoint if previous run was interrupted
    - Graceful stop on Ctrl+C (saves checkpoint, returns partial result)
    - Configurable via limit profiles or explicit parameters
    - Never loses work - always returns result with status

    Example:
        ```python
        # Using limit profile
        profile = get_limit_profile("standard")
        orchestrator = FeedbackLoopOrchestrator(
            worker=developer,
            reviewer=reviewer,
            **profile,
        )

        # Or explicit configuration
        orchestrator = FeedbackLoopOrchestrator(
            worker=developer,
            reviewer=reviewer,
            max_iterations=10,
            timeout_seconds=3600,
            checkpoint_enabled=True,
        )

        result = await orchestrator.run("Write a function to parse JSON")
        # result.metadata["status"] is "completed", "stopped", "timeout", or "max_iterations"
        ```
    """

    def __init__(
        self,
        worker: "ClaudeAgent",
        reviewer: "ClaudeAgent",
        *,
        max_iterations: int = 10,
        timeout_seconds: float = 3600.0,
        approval_check: Callable[[str], bool] | None = None,
        on_interaction: Callable[[str, str, str], None] | None = None,
        synthesizer: "ClaudeAgent | None" = None,
        cost_tracker: "CostTracker | None" = None,
        checkpoint_enabled: bool = False,
        checkpoint_dir: str | Path = "./checkpoints",
    ):
        """Initialize feedback loop orchestrator.

        Args:
            worker: Agent that produces work.
            reviewer: Agent that reviews and provides feedback.
            max_iterations: Maximum review cycles (default 10).
            timeout_seconds: Maximum total time in seconds (default 3600 = 1 hour).
            approval_check: Function to check if review approves the work.
                Default checks for "approved" in text (case-insensitive).
            on_interaction: Callback(agent_name, role, content) for each interaction.
            synthesizer: Optional agent to create final output.
            cost_tracker: Optional cost tracker.
            checkpoint_enabled: Enable auto-save/resume checkpoints (default False for security).
            checkpoint_dir: Directory for checkpoint files (default "./checkpoints").
        """
        self.worker = worker
        self.reviewer = reviewer
        self.max_iterations = max_iterations
        self.timeout_seconds = timeout_seconds
        # Use word boundary to avoid matching "disapproved", "unapproved", etc.
        self.approval_check = approval_check or (
            lambda t: bool(re.search(r'\bapproved\b', t.lower()))
        )
        self.on_interaction = on_interaction
        self.synthesizer = synthesizer
        self.cost_tracker = cost_tracker
        self.checkpoint_enabled = checkpoint_enabled
        self.checkpoint_manager = CheckpointManager(checkpoint_dir) if checkpoint_enabled else None

    async def run(self, task: str) -> OrchestrationResult:
        """Run the feedback loop orchestration.

        This method:
        1. Checks for existing checkpoint and resumes if found
        2. Registers graceful stop handler (Ctrl+C)
        3. Runs the feedback loop with timeout protection
        4. Saves checkpoint after each iteration
        5. Returns result with status (never raises on timeout/stop)

        Args:
            task: The task to complete.

        Returns:
            OrchestrationResult with final output and status in metadata.
            Status can be: "completed", "stopped", "timeout", "max_iterations"
        """
        # Generate checkpoint ID for this task
        # Use unique=False for deterministic IDs that enable task-based resumption
        checkpoint_id = None
        if self.checkpoint_manager:
            checkpoint_id = self.checkpoint_manager.generate_checkpoint_id(
                task, "feedback_loop", unique=False
            )

            # Check for existing checkpoint (auto-resume)
            # Use async_load to avoid blocking the event loop
            existing = await self.checkpoint_manager.async_load(checkpoint_id)
            if existing and existing.status == "in_progress":
                print(f"[Orchestrator] Resuming from checkpoint (iteration {existing.current_iteration})...")
                return await self._run_internal(
                    task,
                    checkpoint_id=checkpoint_id,
                    resume_from=existing,
                )

        # Fresh run
        return await self._run_internal(task, checkpoint_id=checkpoint_id)

    def _serialize_conversation(self, conversation: list[ChatMessage]) -> list[dict]:
        """Serialize conversation for JSON checkpoint."""
        return [
            {"role": msg.role.value if hasattr(msg.role, 'value') else str(msg.role),
             "text": msg.text,
             "author_name": msg.author_name if hasattr(msg, 'author_name') else None}
            for msg in conversation
        ]

    def _deserialize_conversation(self, data: list[dict]) -> list[ChatMessage]:
        """Deserialize conversation from JSON checkpoint."""
        return [
            ChatMessage(role=Role.ASSISTANT, text=item["text"], author_name=item.get("author_name"))
            for item in data
        ]

    async def _save_checkpoint(
        self,
        checkpoint_id: str,
        task: str,
        conversation: list[ChatMessage],
        iteration: int,
        current_work: str,
        feedback: str,
        status: str,
    ) -> None:
        """Save current state to checkpoint (async to avoid blocking event loop)."""
        if not self.checkpoint_manager:
            return

        participants = [self.worker.name, self.reviewer.name]
        if self.synthesizer:
            participants.append(self.synthesizer.name)

        checkpoint = Checkpoint(
            checkpoint_id=checkpoint_id,
            orchestration_type="feedback_loop",
            task=task,
            conversation=self._serialize_conversation(conversation),
            current_iteration=iteration,
            current_work=current_work,
            feedback=feedback,
            participants_used=participants,
            metadata={"max_iterations": self.max_iterations},
            created_at=datetime.now().isoformat(),
            updated_at=datetime.now().isoformat(),
            status=status,
        )
        await self.checkpoint_manager.async_save(checkpoint)

    async def _run_internal(
        self,
        task: str,
        checkpoint_id: str | None = None,
        resume_from: Checkpoint | None = None,
    ) -> OrchestrationResult:
        """Internal run method with checkpointing and graceful stop."""

        # Initialize state (resume or fresh)
        if resume_from:
            conversation = self._deserialize_conversation(resume_from.conversation)
            start_iteration = resume_from.current_iteration
            current_work = resume_from.current_work
            feedback = resume_from.feedback
        else:
            conversation = []
            start_iteration = 0
            current_work = ""
            feedback = ""

        iterations = start_iteration
        status = "in_progress"
        approved = False

        # Register graceful stop handler
        stop_handler = get_stop_handler()
        stop_handler.reset()
        stop_handler.register()

        start_time = asyncio.get_event_loop().time()

        try:
            for i in range(start_iteration, self.max_iterations):
                iterations = i + 1

                # Check for graceful stop (logging done here, not in signal handler)
                if stop_handler.should_stop:
                    status = "stopped"
                    print(f"\n[Orchestrator] Graceful stop requested. Saving checkpoint...")
                    print(f"[Orchestrator] Stopped at iteration {iterations}")
                    break

                # Check timeout
                elapsed = asyncio.get_event_loop().time() - start_time
                if elapsed >= self.timeout_seconds:
                    status = "timeout"
                    print(f"[Orchestrator] Timeout at iteration {iterations} ({elapsed:.0f}s elapsed)")
                    break

                # Worker produces/revises work
                if feedback:
                    worker_prompt = (
                        f"Original task: {task}\n\n"
                        f"Your previous work:\n{current_work}\n\n"
                        f"Feedback to address:\n{feedback}\n\n"
                        f"Please revise your work based on the feedback."
                    )
                else:
                    worker_prompt = task

                # Log input
                if self.on_interaction:
                    self.on_interaction(self.worker.name, "input", worker_prompt)

                worker_response = await self.worker.run(worker_prompt)
                current_work = worker_response.text
                conversation.append(ChatMessage(
                    role=Role.ASSISTANT, text=current_work, author_name=self.worker.name
                ))

                # Log output
                if self.on_interaction:
                    self.on_interaction(self.worker.name, "output", current_work)

                # Track cost
                if self.cost_tracker and hasattr(worker_response, 'usage_details') and worker_response.usage_details:
                    self.cost_tracker.record_request(
                        "worker",
                        worker_response.usage_details.input_token_count or 0,
                        worker_response.usage_details.output_token_count or 0,
                    )

                # Check stop again before reviewer (logging done here, not in signal handler)
                if stop_handler.should_stop:
                    status = "stopped"
                    print(f"\n[Orchestrator] Graceful stop requested. Saving checkpoint...")
                    print(f"[Orchestrator] Stopped after worker, before reviewer (iteration {iterations})")
                    break

                # Reviewer evaluates work
                review_prompt = (
                    f"Task: {task}\n\n"
                    f"Work to review:\n{current_work}\n\n"
                    f"Provide specific feedback. If the work fully meets requirements, "
                    f"say APPROVED. Otherwise, explain what needs improvement."
                )

                # Log input
                if self.on_interaction:
                    self.on_interaction(self.reviewer.name, "input", review_prompt)

                review_response = await self.reviewer.run(review_prompt)
                feedback = review_response.text
                conversation.append(ChatMessage(
                    role=Role.ASSISTANT, text=feedback, author_name=self.reviewer.name
                ))

                # Log output
                if self.on_interaction:
                    self.on_interaction(self.reviewer.name, "output", feedback)

                # Track cost
                if self.cost_tracker and hasattr(review_response, 'usage_details') and review_response.usage_details:
                    self.cost_tracker.record_request(
                        "reviewer",
                        review_response.usage_details.input_token_count or 0,
                        review_response.usage_details.output_token_count or 0,
                    )

                # Save checkpoint after each iteration (async to avoid blocking)
                if checkpoint_id:
                    await self._save_checkpoint(
                        checkpoint_id, task, conversation, iterations,
                        current_work, feedback, "in_progress"
                    )

                # Check for approval
                if self.approval_check(feedback):
                    approved = True
                    status = "completed"
                    break

            # Check if we hit max iterations without approval
            if status == "in_progress":
                status = "max_iterations"

        finally:
            # Unregister stop handler
            stop_handler.unregister()

        # Optional synthesis step (only if completed or we have work)
        final_output = current_work
        if self.synthesizer and current_work:
            synth_prompt = (
                f"Original task: {task}\n\n"
                f"Final work:\n{current_work}\n\n"
                f"Create a polished final output."
            )

            if self.on_interaction:
                self.on_interaction(self.synthesizer.name, "input", synth_prompt)

            try:
                synth_response = await self.synthesizer.run(synth_prompt)
                final_output = synth_response.text
                conversation.append(ChatMessage(
                    role=Role.ASSISTANT, text=final_output, author_name=self.synthesizer.name
                ))

                if self.on_interaction:
                    self.on_interaction(self.synthesizer.name, "output", final_output)
            except Exception as e:
                # Synthesis failed, use current_work as final output
                print(f"[Orchestrator] Synthesis failed: {e}, using last work as output")

        # Build participant set
        participants = {self.worker.name, self.reviewer.name}
        if self.synthesizer:
            participants.add(self.synthesizer.name)

        # Final checkpoint save with terminal status (async to avoid blocking)
        if checkpoint_id and status in ("completed", "max_iterations"):
            await self._save_checkpoint(
                checkpoint_id, task, conversation, iterations,
                current_work, feedback, status
            )
            # Clear checkpoint on successful completion
            if status == "completed" and self.checkpoint_manager:
                await self.checkpoint_manager.async_clear(checkpoint_id)

        return OrchestrationResult(
            final_output=final_output,
            conversation=conversation,
            rounds=iterations,
            participants_used=participants,
            metadata={
                "orchestration_type": "feedback_loop",
                "status": status,
                "iterations": iterations,
                "max_iterations": self.max_iterations,
                "approved": approved,
                "checkpoint_id": checkpoint_id,
            },
        )


# Utility functions for common orchestration patterns
def create_review_loop(
    agents: dict[str, "ClaudeAgent"],
    max_iterations: int = 5,
) -> FeedbackLoopOrchestrator:
    """Create a simple worker → reviewer feedback loop.

    Args:
        agents: Dict with "worker" and "reviewer" keys.
        max_iterations: Maximum review cycles.

    Returns:
        Configured FeedbackLoopOrchestrator.
    """
    return FeedbackLoopOrchestrator(
        worker=agents["worker"],
        reviewer=agents["reviewer"],
        max_iterations=max_iterations,
    )


def create_pipeline(
    agents: Sequence["ClaudeAgent"],
) -> SequentialOrchestrator:
    """Create a simple sequential pipeline.

    Args:
        agents: Agents to chain in order.

    Returns:
        Configured SequentialOrchestrator.
    """
    return SequentialOrchestrator(agents=agents)


def create_parallel_analysis(
    analysts: Sequence["ClaudeAgent"],
    aggregator: Callable | None = None,
) -> ConcurrentOrchestrator:
    """Create parallel analysis with optional aggregation.

    Args:
        analysts: Agents to run in parallel.
        aggregator: Optional function to combine results.

    Returns:
        Configured ConcurrentOrchestrator.
    """
    return ConcurrentOrchestrator(agents=analysts, aggregator=aggregator)
