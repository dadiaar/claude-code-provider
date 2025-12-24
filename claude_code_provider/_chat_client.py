# Copyright (c) 2025. Claude Code Provider for Microsoft Agent Framework.

"""Claude Code CLI Chat Client for Microsoft Agent Framework."""

import logging
from collections.abc import AsyncIterable, MutableSequence, Callable
from typing import Any, ClassVar

from agent_framework import (
    BaseChatClient,
    ChatMessage,
    ChatOptions,
    ChatResponse,
    ChatResponseUpdate,
)
from agent_framework.observability import use_instrumentation
from agent_framework._middleware import use_chat_middleware
from agent_framework._tools import use_function_invocation

try:
    from ._cli_executor import CLIExecutor
    from ._message_converter import prepare_cli_execution
    from ._response_parser import (
        parse_cli_result_to_response,
        parse_stream_event_to_update,
    )
    from ._settings import ClaudeCodeSettings
    from ._retry import RetryConfig
    from ._agent import ClaudeAgent, DEFAULT_AUTOCOMPACT_THRESHOLD
    from ._validation import (
        ValidationError,
        validate_string,
        validate_positive_int,
        validate_positive_float,
        validate_agents_list,
        validate_agent,
        validate_callable,
    )
    from ._orchestration import (
        GroupChatOrchestrator,
        GroupChatConfig,
        HandoffOrchestrator,
        HandoffConfig,
        SequentialOrchestrator,
        ConcurrentOrchestrator,
        MagenticOrchestrator,
        MagenticConfig,
        FeedbackLoopOrchestrator,
        MAF_ORCHESTRATION_AVAILABLE,
    )
except ImportError:
    from _cli_executor import CLIExecutor
    from _message_converter import prepare_cli_execution
    from _response_parser import (
        parse_cli_result_to_response,
        parse_stream_event_to_update,
    )
    from _settings import ClaudeCodeSettings
    from _retry import RetryConfig
    from _agent import ClaudeAgent, DEFAULT_AUTOCOMPACT_THRESHOLD
    from _validation import (
        ValidationError,
        validate_string,
        validate_positive_int,
        validate_positive_float,
        validate_agents_list,
        validate_agent,
        validate_callable,
    )
    from _orchestration import (
        GroupChatOrchestrator,
        GroupChatConfig,
        HandoffOrchestrator,
        HandoffConfig,
        SequentialOrchestrator,
        ConcurrentOrchestrator,
        MagenticOrchestrator,
        MagenticConfig,
        FeedbackLoopOrchestrator,
        MAF_ORCHESTRATION_AVAILABLE,
    )

logger = logging.getLogger("claude_code_provider")


@use_function_invocation
@use_instrumentation
@use_chat_middleware
class ClaudeCodeClient(BaseChatClient):
    """Chat client that uses Claude Code CLI instead of direct API calls.

    This client allows Microsoft Agent Framework agents to use a Claude
    subscription account through the Claude Code CLI tool.

    Claude Code Built-in Tools:
        The following tools are available and executed by Claude Code internally:
        - Bash: Execute shell commands
        - Read: Read file contents
        - Edit: Edit files with search/replace
        - Write: Write new files
        - Glob: Find files by pattern
        - Grep: Search file contents
        - WebFetch: Fetch web content
        - WebSearch: Search the web
        - Task: Launch sub-agents

        Pass tool names via the `tools` parameter to control which are available.

    Example:
        ```python
        from claude_code_provider import ClaudeCodeClient

        # Create client with default settings (all tools)
        client = ClaudeCodeClient()

        # Or with specific model and limited tools
        client = ClaudeCodeClient(
            model="sonnet",
            tools=["Read", "Bash", "Glob"],  # Only these tools
        )

        # Create an agent
        agent = client.create_agent(
            name="assistant",
            instructions="You are a helpful coding assistant.",
        )

        # Run the agent
        response = await agent.run("What files are in this directory?")
        print(response.text)
        ```
    """

    OTEL_PROVIDER_NAME: ClassVar[str] = "claude_code"

    def __init__(
        self,
        *,
        model: str | None = None,
        cli_path: str = "claude",
        max_turns: int | None = None,
        tools: list[str] | None = None,
        allowed_tools: list[str] | None = None,
        disallowed_tools: list[str] | None = None,
        working_directory: str | None = None,
        timeout: float = 300.0,
        retry_config: RetryConfig | None = None,
        enable_retries: bool = False,
        enable_circuit_breaker: bool = True,
        settings: ClaudeCodeSettings | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the Claude Code client.

        Args:
            model: Default model to use ('sonnet', 'opus', 'haiku', or full name).
            cli_path: Path to the claude CLI executable.
            max_turns: Default maximum agentic turns.
            tools: List of Claude Code tools to enable.
            allowed_tools: Tools that run without prompts.
            disallowed_tools: Tools that are blocked.
            working_directory: Working directory for CLI execution.
            timeout: Timeout for CLI execution in seconds (default: 300).
            retry_config: Custom retry configuration. If None and enable_retries=True,
                uses default RetryConfig.
            enable_retries: Enable automatic retries for transient failures.
            enable_circuit_breaker: Enable circuit breaker pattern for failure protection.
            settings: Pre-configured settings object (overrides other args).
            **kwargs: Additional arguments passed to BaseChatClient.

        Raises:
            ValidationError: If any parameter is invalid.
        """
        # Validate parameters
        validate_positive_float(timeout, "timeout", min_value=1.0, max_value=3600.0)
        validate_positive_int(max_turns, "max_turns", allow_none=True, min_value=1)
        validate_string(model, "model", allow_none=True)
        validate_string(cli_path, "cli_path")

        super().__init__(**kwargs)

        # Use provided settings or create from arguments
        if settings:
            self.settings = settings
        else:
            self.settings = ClaudeCodeSettings(
                cli_path=cli_path,
                model=model,
                default_max_turns=max_turns,
                tools=tools,
                allowed_tools=allowed_tools,
                disallowed_tools=disallowed_tools,
                working_directory=working_directory,
            )

        # Determine retry config
        effective_retry_config = retry_config
        if effective_retry_config is None and enable_retries:
            effective_retry_config = RetryConfig()

        self.executor = CLIExecutor(
            self.settings,
            timeout=timeout,
            retry_config=effective_retry_config,
            enable_circuit_breaker=enable_circuit_breaker,
        )
        self.model_id = model or self.settings.model
        self.timeout = timeout

        # Track session IDs for conversation continuity
        self._session_id: str | None = None

        # Track if we're in a context manager
        self._context_entered = False

    async def __aenter__(self) -> "ClaudeCodeClient":
        """Enter async context manager.

        Returns:
            The client instance.

        Example:
            ```python
            async with ClaudeCodeClient(model="sonnet") as client:
                agent = client.create_agent(name="assistant", instructions="...")
                response = await agent.run("Hello!")
            # Cleanup happens automatically here
            ```
        """
        self._context_entered = True
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Exit async context manager and cleanup resources.

        Args:
            exc_type: Exception type if an error occurred.
            exc_val: Exception value if an error occurred.
            exc_tb: Exception traceback if an error occurred.
        """
        self._context_entered = False
        self._session_id = None
        # Additional cleanup can be added here as needed
        # e.g., closing any open connections, flushing logs, etc.

    async def _inner_get_response(
        self,
        *,
        messages: MutableSequence[ChatMessage],
        chat_options: ChatOptions,
        **kwargs: Any,
    ) -> ChatResponse:
        """Execute a non-streaming request via Claude CLI.

        Args:
            messages: The chat messages to send.
            chat_options: Options for the request.
            **kwargs: Additional arguments.

        Returns:
            ChatResponse with the result.
        """
        # Prepare CLI execution parameters
        cli_params = prepare_cli_execution(
            messages=messages,
            chat_options=chat_options,
            session_id=self._session_id,
        )

        # Execute CLI
        result = await self.executor.execute(
            prompt=cli_params["prompt"],
            session_id=cli_params["session_id"],
            system_prompt=cli_params["system_prompt"],
            model=cli_params.get("model"),
            extra_args=cli_params.get("extra_args"),
        )

        # Update session ID for conversation continuity
        if result.session_id:
            self._session_id = result.session_id

        # Convert to ChatResponse
        response = parse_cli_result_to_response(result)

        # Set conversation_id for thread management
        if result.session_id:
            response.conversation_id = result.session_id

        return response

    async def _inner_get_streaming_response(
        self,
        *,
        messages: MutableSequence[ChatMessage],
        chat_options: ChatOptions,
        **kwargs: Any,
    ) -> AsyncIterable[ChatResponseUpdate]:
        """Execute a streaming request via Claude CLI.

        Args:
            messages: The chat messages to send.
            chat_options: Options for the request.
            **kwargs: Additional arguments.

        Yields:
            ChatResponseUpdate objects as they arrive.
        """
        # Prepare CLI execution parameters
        cli_params = prepare_cli_execution(
            messages=messages,
            chat_options=chat_options,
            session_id=self._session_id,
        )

        # Execute CLI with streaming
        async for event in self.executor.execute_stream(
            prompt=cli_params["prompt"],
            session_id=cli_params["session_id"],
            system_prompt=cli_params["system_prompt"],
            model=cli_params.get("model"),
            extra_args=cli_params.get("extra_args"),
        ):
            # Update session ID from events
            if event.data.get("session_id"):
                self._session_id = event.data["session_id"]

            # Convert to ChatResponseUpdate
            update = parse_stream_event_to_update(event)
            if update:
                yield update

    def service_url(self) -> str:
        """Get the service URL identifier.

        Returns:
            A string identifying this as the Claude Code CLI service.
        """
        return f"claude-code-cli://{self.settings.cli_path}"

    def reset_session(self) -> None:
        """Reset the session ID to start a new conversation.

        Call this when you want to start a fresh conversation
        without any prior context.
        """
        self._session_id = None

    @property
    def current_session_id(self) -> str | None:
        """Get the current session ID.

        Returns:
            The current session ID if a conversation is in progress.
        """
        return self._session_id

    def create_agent(
        self,
        *,
        autocompact: bool = True,
        autocompact_threshold: int = DEFAULT_AUTOCOMPACT_THRESHOLD,
        keep_last_n_messages: int = 2,
        **kwargs: Any,
    ) -> ClaudeAgent:
        """Create an agent with optional autocompact support.

        Args:
            autocompact: Whether to automatically compact when threshold is reached.
                Defaults to True to prevent context overflow errors.
            autocompact_threshold: Token threshold for autocompact (default: 100,000).
            keep_last_n_messages: Recent messages to keep when compacting (default: 2).
            **kwargs: Arguments passed to BaseChatClient.create_agent().

        Returns:
            ClaudeAgent with compact functionality.

        Example:
            ```python
            # Agent with autocompact enabled
            agent = client.create_agent(
                name="assistant",
                instructions="You are helpful.",
                autocompact=True,
                autocompact_threshold=50_000,
            )

            # Long conversation - autocompact happens automatically
            for i in range(100):
                response = await agent.run(f"Message {i}")

            # Or manually compact anytime
            result = await agent.compact()
            print(f"Compacted: {result.original_tokens_estimate} -> {result.summary_tokens_estimate} tokens")
            ```
        """
        # Create the inner MAF agent
        inner_agent = super().create_agent(**kwargs)

        # Wrap with our enhanced agent
        return ClaudeAgent(
            inner_agent=inner_agent,
            client=self,
            autocompact=autocompact,
            autocompact_threshold=autocompact_threshold,
            keep_last_n_messages=keep_last_n_messages,
        )

    # =========================================================================
    # Orchestration Methods
    # =========================================================================

    def create_group_chat(
        self,
        participants: list[ClaudeAgent],
        manager: Any,
        *,
        max_rounds: int = 15,
        termination_condition: Any = None,
        manager_display_name: str = "manager",
        final_message: str | None = None,
        cost_tracker: Any = None,
    ) -> GroupChatOrchestrator:
        """Create a GroupChat orchestrator for multi-agent coordination.

        The manager (function or agent) selects which participant speaks next
        based on the full conversation history.

        Args:
            participants: Agents that can be selected to speak.
            manager: Function(state) -> str|None or ClaudeAgent that selects next speaker.
            max_rounds: Maximum manager selection rounds.
            termination_condition: Optional callable(conversation) -> bool.
            manager_display_name: Display name for manager in logs.
            final_message: Optional message when finishing.
            cost_tracker: Optional CostTracker instance.

        Returns:
            GroupChatOrchestrator ready to run.

        Example:
            ```python
            def select_speaker(state):
                last = state.conversation[-1].text.lower() if state.conversation else ""
                if "needs revision" in last:
                    return "developer"
                elif "approved" in last:
                    return None  # Finish
                return "reviewer"

            orchestrator = client.create_group_chat(
                participants=[developer, reviewer],
                manager=select_speaker,
                max_rounds=10,
            )

            result = await orchestrator.run("Build a REST API")
            ```

        Raises:
            ValidationError: If any parameter is invalid.
        """
        if not MAF_ORCHESTRATION_AVAILABLE:
            raise ImportError("MAF orchestration not available")

        # Validate parameters
        validate_agents_list(participants, "participants", min_length=1)
        validate_positive_int(max_rounds, "max_rounds", min_value=1)
        if manager is not None and not callable(manager) and not hasattr(manager, "run"):
            raise ValidationError(
                "manager",
                "must be a callable or ClaudeAgent"
            )

        config = GroupChatConfig(
            max_rounds=max_rounds,
            termination_condition=termination_condition,
            manager_display_name=manager_display_name,
            final_message=final_message,
        )

        return GroupChatOrchestrator(
            participants=participants,
            manager=manager,
            config=config,
            cost_tracker=cost_tracker,
        )

    def create_handoff(
        self,
        coordinator: ClaudeAgent,
        specialists: list[ClaudeAgent],
        *,
        autonomous: bool = True,
        autonomous_turn_limit: int = 50,
        termination_condition: Any = None,
        enable_return_to_previous: bool = False,
        cost_tracker: Any = None,
    ) -> HandoffOrchestrator:
        """Create a Handoff orchestrator for coordinator → specialist routing.

        A coordinator agent routes tasks to specialists via tool calls.

        Args:
            coordinator: The routing/coordinator agent.
            specialists: Specialist agents to route to.
            autonomous: Run without user input between turns.
            autonomous_turn_limit: Max turns in autonomous mode.
            termination_condition: Optional callable(conversation) -> bool.
            enable_return_to_previous: Allow returning to previous agent.
            cost_tracker: Optional CostTracker instance.

        Returns:
            HandoffOrchestrator ready to run.

        Example:
            ```python
            coordinator = client.create_agent(
                name="coordinator",
                instructions="Route to billing, technical, or account specialist",
            )

            orchestrator = client.create_handoff(
                coordinator=coordinator,
                specialists=[billing, technical, account],
                autonomous=True,
            )

            result = await orchestrator.run("I was charged twice")
            ```

        Raises:
            ValidationError: If any parameter is invalid.
        """
        if not MAF_ORCHESTRATION_AVAILABLE:
            raise ImportError("MAF orchestration not available")

        # Validate parameters
        validate_agent(coordinator, "coordinator")
        validate_agents_list(specialists, "specialists", min_length=1)
        validate_positive_int(autonomous_turn_limit, "autonomous_turn_limit", min_value=1)

        config = HandoffConfig(
            autonomous=autonomous,
            autonomous_turn_limit=autonomous_turn_limit,
            termination_condition=termination_condition,
            enable_return_to_previous=enable_return_to_previous,
        )

        return HandoffOrchestrator(
            coordinator=coordinator,
            specialists=specialists,
            config=config,
            cost_tracker=cost_tracker,
        )

    def create_sequential(
        self,
        agents: list[ClaudeAgent],
        *,
        cost_tracker: Any = None,
    ) -> SequentialOrchestrator:
        """Create a Sequential orchestrator for linear agent chains.

        Agents process in sequence, each building on previous output.

        Args:
            agents: Agents to run in sequence.
            cost_tracker: Optional CostTracker instance.

        Returns:
            SequentialOrchestrator ready to run.

        Example:
            ```python
            orchestrator = client.create_sequential(
                agents=[researcher, writer, editor],
            )

            result = await orchestrator.run("Write about AI")
            # researcher → writer → editor
            ```

        Raises:
            ValidationError: If any parameter is invalid.
        """
        if not MAF_ORCHESTRATION_AVAILABLE:
            raise ImportError("MAF orchestration not available")

        # Validate parameters
        validate_agents_list(agents, "agents", min_length=1)

        return SequentialOrchestrator(
            agents=agents,
            cost_tracker=cost_tracker,
        )

    def create_concurrent(
        self,
        agents: list[ClaudeAgent],
        *,
        aggregator: Any = None,
        cost_tracker: Any = None,
    ) -> ConcurrentOrchestrator:
        """Create a Concurrent orchestrator for parallel execution.

        Fans out input to multiple agents in parallel, then aggregates.

        Args:
            agents: Agents to run in parallel.
            aggregator: Optional function to combine results.
            cost_tracker: Optional CostTracker instance.

        Returns:
            ConcurrentOrchestrator ready to run.

        Example:
            ```python
            orchestrator = client.create_concurrent(
                agents=[analyst1, analyst2, analyst3],
                aggregator=lambda r: "\\n---\\n".join(x.text for x in r),
            )

            result = await orchestrator.run("Analyze this data")
            ```

        Raises:
            ValidationError: If any parameter is invalid.
        """
        if not MAF_ORCHESTRATION_AVAILABLE:
            raise ImportError("MAF orchestration not available")

        # Validate parameters
        validate_agents_list(agents, "agents", min_length=1)
        validate_callable(aggregator, "aggregator", allow_none=True)

        return ConcurrentOrchestrator(
            agents=agents,
            aggregator=aggregator,
            cost_tracker=cost_tracker,
        )

    def create_feedback_loop(
        self,
        worker: ClaudeAgent,
        reviewer: ClaudeAgent,
        *,
        max_iterations: int = 5,
        approval_check: Any = None,
        synthesizer: ClaudeAgent | None = None,
        cost_tracker: Any = None,
    ) -> FeedbackLoopOrchestrator:
        """Create a FeedbackLoop orchestrator for iterative refinement.

        Worker produces work, reviewer provides feedback, repeat until approved.

        Args:
            worker: Agent that produces/revises work.
            reviewer: Agent that reviews and provides feedback.
            max_iterations: Maximum review cycles.
            approval_check: Function(text) -> bool to check approval.
                Default checks for "approved" in text.
            synthesizer: Optional agent for final polish.
            cost_tracker: Optional CostTracker instance.

        Returns:
            FeedbackLoopOrchestrator ready to run.

        Example:
            ```python
            orchestrator = client.create_feedback_loop(
                worker=developer,
                reviewer=code_reviewer,
                max_iterations=5,
                approval_check=lambda t: "LGTM" in t or "approved" in t.lower(),
            )

            result = await orchestrator.run("Write a sorting function")
            print(f"Approved after {result.rounds} iterations")
            ```

        Raises:
            ValidationError: If any parameter is invalid.
        """
        # Validate parameters
        validate_agent(worker, "worker")
        validate_agent(reviewer, "reviewer")
        validate_positive_int(max_iterations, "max_iterations", min_value=1)
        validate_callable(approval_check, "approval_check", allow_none=True)
        validate_agent(synthesizer, "synthesizer", allow_none=True)

        return FeedbackLoopOrchestrator(
            worker=worker,
            reviewer=reviewer,
            max_iterations=max_iterations,
            approval_check=approval_check,
            synthesizer=synthesizer,
            cost_tracker=cost_tracker,
        )
