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
import signal
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
    MAF_ORCHESTRATION_AVAILABLE = True
except ImportError:
    MAF_ORCHESTRATION_AVAILABLE = False
    GroupChatBuilder = None
    HandoffBuilder = None
    SequentialBuilder = None
    ConcurrentBuilder = None
    GroupChatStateSnapshot = None

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
    Each orchestration run gets a unique checkpoint ID based on the task hash.

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

    def __init__(self, checkpoint_dir: str | Path = "./checkpoints"):
        """Initialize checkpoint manager.

        Args:
            checkpoint_dir: Directory to store checkpoints.
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def _get_checkpoint_path(self, checkpoint_id: str) -> Path:
        """Get file path for a checkpoint."""
        return self.checkpoint_dir / f"{checkpoint_id}.json"

    def generate_checkpoint_id(self, task: str, orchestration_type: str) -> str:
        """Generate a unique checkpoint ID based on task and type.

        Args:
            task: The task description.
            orchestration_type: Type of orchestration.

        Returns:
            Unique checkpoint ID.
        """
        content = f"{orchestration_type}:{task}"
        hash_part = hashlib.sha256(content.encode()).hexdigest()[:12]
        return f"{orchestration_type}_{hash_part}"

    def save(self, checkpoint: Checkpoint) -> Path:
        """Save a checkpoint to disk.

        Args:
            checkpoint: Checkpoint to save.

        Returns:
            Path to saved checkpoint file.
        """
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
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

        return path

    def load(self, checkpoint_id: str) -> Checkpoint | None:
        """Load a checkpoint from disk.

        Args:
            checkpoint_id: ID of checkpoint to load.

        Returns:
            Checkpoint if found, None otherwise.
        """
        path = self._get_checkpoint_path(checkpoint_id)
        if not path.exists():
            return None

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


# Global checkpoint manager (can be overridden)
_default_checkpoint_manager: CheckpointManager | None = None


def get_checkpoint_manager(checkpoint_dir: str | Path = "./checkpoints") -> CheckpointManager:
    """Get or create the default checkpoint manager.

    Args:
        checkpoint_dir: Directory for checkpoints.

    Returns:
        CheckpointManager instance.
    """
    global _default_checkpoint_manager
    if _default_checkpoint_manager is None:
        _default_checkpoint_manager = CheckpointManager(checkpoint_dir)
    return _default_checkpoint_manager


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

    Example:
        ```python
        handler = GracefulStopHandler()
        handler.register()

        while not handler.should_stop:
            # ... do work ...
            pass

        if handler.should_stop:
            # Save checkpoint and exit
            pass
        ```
    """

    def __init__(self):
        self.should_stop = False
        self._original_sigint = None
        self._original_sigterm = None

    def _signal_handler(self, signum, frame):
        """Handle stop signal."""
        print("\n[Orchestrator] Graceful stop requested. Saving checkpoint...")
        self.should_stop = True

    def register(self):
        """Register signal handlers."""
        self._original_sigint = signal.signal(signal.SIGINT, self._signal_handler)
        try:
            self._original_sigterm = signal.signal(signal.SIGTERM, self._signal_handler)
        except (ValueError, OSError):
            # SIGTERM might not be available on all platforms
            pass

    def unregister(self):
        """Restore original signal handlers."""
        if self._original_sigint is not None:
            signal.signal(signal.SIGINT, self._original_sigint)
        if self._original_sigterm is not None:
            try:
                signal.signal(signal.SIGTERM, self._original_sigterm)
            except (ValueError, OSError):
                pass

    def reset(self):
        """Reset the stop flag."""
        self.should_stop = False


# Global stop handler
_stop_handler = GracefulStopHandler()


def get_stop_handler() -> GracefulStopHandler:
    """Get the global stop handler."""
    return _stop_handler


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

        async for response in self._workflow.run_stream(task):
            # Collect conversation and track participants
            if hasattr(response, 'messages'):
                for msg in response.messages:
                    conversation.append(msg)
                    if hasattr(msg, 'name') and msg.name:
                        participants_used.add(msg.name)

            if hasattr(response, 'agent_name'):
                participants_used.add(response.agent_name)

            rounds += 1

        # Get final output
        final_output = ""
        if conversation:
            final_output = conversation[-1].text if hasattr(conversation[-1], 'text') else str(conversation[-1])

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

        async for response in self._workflow.run_stream(task):
            if hasattr(response, 'messages'):
                for msg in response.messages:
                    conversation.append(msg)
                    if hasattr(msg, 'name') and msg.name:
                        participants_used.add(msg.name)
            rounds += 1

        final_output = ""
        if conversation:
            final_output = conversation[-1].text if hasattr(conversation[-1], 'text') else str(conversation[-1])

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

        async for response in self._workflow.run_stream(task):
            if hasattr(response, 'messages'):
                for msg in response.messages:
                    conversation.append(msg)
                    if hasattr(msg, 'name') and msg.name:
                        participants_used.add(msg.name)

        final_output = ""
        if conversation:
            final_output = conversation[-1].text if hasattr(conversation[-1], 'text') else str(conversation[-1])

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

        async for response in self._workflow.run_stream(task):
            if hasattr(response, 'messages'):
                for msg in response.messages:
                    conversation.append(msg)

        final_output = ""
        if conversation:
            final_output = conversation[-1].text if hasattr(conversation[-1], 'text') else str(conversation[-1])

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
        inner_agents = [agent._agent for agent in self.participants]

        # Create the standard magentic manager
        manager = StandardMagenticManager(
            agent=self.manager_agent._agent,
            max_stall_count=self.config.max_stall_count,
            max_reset_count=self.config.max_reset_count,
            max_round_count=self.config.max_round_count,
            progress_ledger_retry_count=self.config.progress_ledger_retry_count,
        )

        builder = MagenticBuilder()
        builder = builder.participants(inner_agents)
        builder = builder.set_manager(manager, display_name="Magentic")

        return builder.build()

    async def run(self, task: str) -> OrchestrationResult:
        """Run the orchestration with a task."""
        if self._workflow is None:
            self._workflow = self._build_workflow()

        conversation = []
        participants_used = set()
        rounds = 0

        async for response in self._workflow.run_stream(task):
            if hasattr(response, 'messages'):
                for msg in response.messages:
                    conversation.append(msg)
                    if hasattr(msg, 'name') and msg.name:
                        participants_used.add(msg.name)
            rounds += 1

        final_output = ""
        if conversation:
            final_output = conversation[-1].text if hasattr(conversation[-1], 'text') else str(conversation[-1])

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
        self.approval_check = approval_check or (lambda t: "approved" in t.lower())
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
        checkpoint_id = None
        if self.checkpoint_manager:
            checkpoint_id = self.checkpoint_manager.generate_checkpoint_id(task, "feedback_loop")

            # Check for existing checkpoint (auto-resume)
            existing = self.checkpoint_manager.load(checkpoint_id)
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

    def _save_checkpoint(
        self,
        checkpoint_id: str,
        task: str,
        conversation: list[ChatMessage],
        iteration: int,
        current_work: str,
        feedback: str,
        status: str,
    ) -> None:
        """Save current state to checkpoint."""
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
        self.checkpoint_manager.save(checkpoint)

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

                # Check for graceful stop
                if stop_handler.should_stop:
                    status = "stopped"
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

                # Check stop again before reviewer
                if stop_handler.should_stop:
                    status = "stopped"
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

                # Save checkpoint after each iteration
                if checkpoint_id:
                    self._save_checkpoint(
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

        # Final checkpoint save with terminal status
        if checkpoint_id and status in ("completed", "max_iterations"):
            self._save_checkpoint(
                checkpoint_id, task, conversation, iterations,
                current_work, feedback, status
            )
            # Clear checkpoint on successful completion
            if status == "completed" and self.checkpoint_manager:
                self.checkpoint_manager.clear(checkpoint_id)

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
