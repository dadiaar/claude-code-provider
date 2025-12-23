# Copyright (c) 2025. Claude Code Provider for Microsoft Agent Framework.

"""Claude Code CLI Chat Client for Microsoft Agent Framework."""

import logging
from collections.abc import AsyncIterable, MutableSequence
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
        """
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
        autocompact: bool = False,
        autocompact_threshold: int = DEFAULT_AUTOCOMPACT_THRESHOLD,
        keep_last_n_messages: int = 2,
        **kwargs: Any,
    ) -> ClaudeAgent:
        """Create an agent with optional autocompact support.

        Args:
            autocompact: Whether to automatically compact when threshold is reached.
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
