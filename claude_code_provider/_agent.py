# Copyright (c) 2025. Claude Code Provider for Microsoft Agent Framework.

"""Enhanced Agent with compact functionality."""

from typing import Any, TYPE_CHECKING, Literal
from dataclasses import dataclass, field
from collections.abc import AsyncIterable, Callable, Sequence, MutableMapping

from agent_framework import ChatMessage, Role
from agent_framework._agents import ChatAgent
from agent_framework._threads import AgentThread
from agent_framework._tools import ToolProtocol, AIFunction
from agent_framework._types import ToolMode

try:
    from pydantic import BaseModel
except ImportError:
    BaseModel = None  # type: ignore

if TYPE_CHECKING:
    from ._chat_client import ClaudeCodeClient
    from mcp.server import Server


# Approximate tokens per character (conservative estimate)
CHARS_PER_TOKEN = 4

# Default threshold for autocompact (in estimated tokens)
DEFAULT_AUTOCOMPACT_THRESHOLD = 100_000

# Claude's context window (approximate)
CLAUDE_CONTEXT_LIMIT = 200_000


@dataclass
class UsageStats:
    """Accumulated usage statistics."""
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_requests: int = 0
    compactions: int = 0

    @property
    def total_tokens(self) -> int:
        return self.total_input_tokens + self.total_output_tokens


@dataclass
class ContextInfo:
    """Information about current context usage."""
    estimated_tokens: int
    context_limit: int
    usage_percent: float
    messages_count: int
    has_summary: bool
    autocompact_enabled: bool
    autocompact_threshold: int


@dataclass
class CompactResult:
    """Result of a compact operation."""

    original_messages: int
    original_tokens_estimate: int
    summary_tokens_estimate: int
    summary: str


@dataclass
class ConversationMessage:
    """A message in the conversation history."""
    role: str  # "user" or "assistant"
    text: str


class ClaudeAgent:
    """Enhanced agent with compact and autocompact functionality.

    Wraps MAF's ChatAgent to provide Claude Code CLI integration.

    MAF-Standard Methods:
        - run(message, **options): Send a message and get a response
        - run_stream(message, **options): Stream a response token by token
        - as_tool(): Convert agent to a tool for other agents
        - as_mcp_server(): Expose agent as an MCP server
        - get_new_thread(): Get a new conversation thread
        - deserialize_thread(): Restore a thread from serialized state
        - to_dict() / to_json(): Serialize agent state
        - name, instructions, display_name: Agent properties

    MAF Parameters for run/run_stream (passed through to inner agent):
        - model_id: Override the model for this request
        - max_tokens: Maximum tokens in response
        - temperature, top_p: Sampling parameters
        - response_format: Pydantic model for structured output
        - tools, tool_choice: Custom tool configuration
        - See MAF documentation for full parameter list

    CLI Limitations:
        Some parameters may not be fully supported by Claude Code CLI:
        - temperature, top_p, frequency_penalty, presence_penalty
        - logit_bias, seed
        These are passed through but may be ignored by the CLI.

    Extension Methods (Claude Code Provider specific):
        - compact(): Summarize conversation history to reduce context
        - get_usage(): Get accumulated token usage statistics
        - get_context_info(): Get context usage details and limits
        - get_messages(): Get conversation history
        - get_token_estimate(): Estimate current token count
        - reset(): Reset conversation state
        - reset_usage(): Reset usage statistics

    Example:
        ```python
        client = ClaudeCodeClient(model="haiku")
        agent = client.create_agent(
            name="assistant",
            instructions="You are helpful.",
            autocompact=True,  # Enabled by default
        )

        # MAF-standard usage
        response = await agent.run("Hello!")
        response = await agent.run("Complex task", max_tokens=1000)

        # Convert to tool for multi-agent workflows
        agent_tool = agent.as_tool(name="helper")

        # Extension: check usage
        usage = agent.get_usage()
        print(f"Tokens used: {usage.total_tokens}")
        ```
    """

    def __init__(
        self,
        inner_agent: ChatAgent,
        client: "ClaudeCodeClient",
        *,
        autocompact: bool = True,  # ON by default to prevent context overflow
        autocompact_threshold: int = DEFAULT_AUTOCOMPACT_THRESHOLD,
        keep_last_n_messages: int = 2,
    ) -> None:
        """Initialize the enhanced agent.

        Args:
            inner_agent: The MAF ChatAgent to wrap.
            client: The ClaudeCodeClient for making compact requests.
            autocompact: Whether to automatically compact when threshold is reached.
                Defaults to True to prevent context overflow errors.
            autocompact_threshold: Token threshold for autocompact (default: 100,000).
            keep_last_n_messages: Number of recent messages to keep uncompacted.
        """
        self._agent = inner_agent
        self._client = client
        self._autocompact = autocompact
        self._autocompact_threshold = autocompact_threshold
        self._keep_last_n = keep_last_n_messages

        # Track conversation ourselves since CLI uses session-based memory
        self._messages: list[ConversationMessage] = []
        self._compact_summary: str | None = None
        self._needs_context_injection: bool = False  # True after compact

        # Usage tracking
        self._usage = UsageStats()

    @property
    def name(self) -> str | None:
        """Get the agent name."""
        return self._agent.name

    @property
    def instructions(self) -> str | None:
        """Get the agent instructions."""
        return self._agent.chat_options.instructions

    @property
    def display_name(self) -> str:
        """Get the agent display name."""
        return self._agent.display_name

    # =========================================================================
    # MAF-Standard Methods (pass-through to inner agent)
    # =========================================================================

    def as_tool(
        self,
        *,
        name: str | None = None,
        description: str | None = None,
        arg_name: str = "task",
        arg_description: str | None = None,
        stream_callback: Callable | None = None,
    ) -> AIFunction:
        """Create an AIFunction tool that wraps this agent.

        This allows the agent to be used as a tool by other agents,
        enabling multi-agent workflows.

        Args:
            name: Tool name (defaults to agent name).
            description: Tool description (defaults to agent instructions).
            arg_name: Name of the argument the tool accepts.
            arg_description: Description of the argument.
            stream_callback: Optional callback for streaming responses.

        Returns:
            AIFunction that can be passed to other agents as a tool.

        Example:
            ```python
            helper = client.create_agent(name="helper", instructions="...")
            main_agent = client.create_agent(
                name="main",
                tools=[helper.as_tool(name="ask_helper")],
            )
            ```
        """
        return self._agent.as_tool(
            name=name,
            description=description,
            arg_name=arg_name,
            arg_description=arg_description,
            stream_callback=stream_callback,
        )

    def as_mcp_server(
        self,
        *,
        server_name: str = "Agent",
        version: str | None = None,
        instructions: str | None = None,
        lifespan: Callable | None = None,
        **kwargs: Any,
    ) -> "Server[Any]":
        """Create an MCP server from this agent.

        Args:
            server_name: Name of the MCP server.
            version: Server version.
            instructions: Server instructions.
            lifespan: Optional lifespan context manager.
            **kwargs: Additional server options.

        Returns:
            MCP Server instance.
        """
        return self._agent.as_mcp_server(
            server_name=server_name,
            version=version,
            instructions=instructions,
            lifespan=lifespan,
            **kwargs,
        )

    def get_new_thread(
        self,
        *,
        service_thread_id: str | None = None,
        **kwargs: Any,
    ) -> AgentThread:
        """Get a new conversation thread for the agent.

        Args:
            service_thread_id: Optional service-specific thread ID.
            **kwargs: Additional thread options.

        Returns:
            New AgentThread instance.
        """
        return self._agent.get_new_thread(
            service_thread_id=service_thread_id,
            **kwargs,
        )

    def deserialize_thread(
        self,
        serialized_thread: Any,
        **kwargs: Any,
    ) -> AgentThread:
        """Deserialize a thread from its serialized state.

        Args:
            serialized_thread: Serialized thread data.
            **kwargs: Additional options.

        Returns:
            Restored AgentThread instance.
        """
        return self._agent.deserialize_thread(serialized_thread, **kwargs)

    def to_dict(
        self,
        *,
        exclude: set[str] | None = None,
        exclude_none: bool = True,
    ) -> dict[str, Any]:
        """Convert the agent to a dictionary.

        Args:
            exclude: Fields to exclude.
            exclude_none: Whether to exclude None values.

        Returns:
            Dictionary representation of the agent.
        """
        return self._agent.to_dict(exclude=exclude, exclude_none=exclude_none)

    def to_json(
        self,
        *,
        exclude: set[str] | None = None,
        exclude_none: bool = True,
        **kwargs: Any,
    ) -> str:
        """Convert the agent to a JSON string.

        Args:
            exclude: Fields to exclude.
            exclude_none: Whether to exclude None values.
            **kwargs: Additional JSON encoding options.

        Returns:
            JSON string representation of the agent.
        """
        return self._agent.to_json(exclude=exclude, exclude_none=exclude_none, **kwargs)

    # =========================================================================
    # Extension Methods (Claude Code Provider specific)
    # =========================================================================

    def _estimate_tokens(self, messages: list[ConversationMessage]) -> int:
        """Estimate token count from messages."""
        total_chars = sum(len(msg.text) for msg in messages)
        # Add tokens for summary if present
        if self._compact_summary:
            total_chars += len(self._compact_summary)
        return total_chars // CHARS_PER_TOKEN

    def get_messages(self) -> list[ConversationMessage]:
        """Get all messages in the conversation."""
        return self._messages.copy()

    def get_token_estimate(self) -> int:
        """Get estimated token count of conversation."""
        return self._estimate_tokens(self._messages)

    def get_usage(self) -> UsageStats:
        """Get accumulated usage statistics.

        Returns:
            UsageStats with total tokens used across all requests.
        """
        return self._usage

    def get_context_info(self) -> ContextInfo:
        """Get information about current context usage.

        Returns:
            ContextInfo with details about context usage and limits.
        """
        estimated = self.get_token_estimate()
        return ContextInfo(
            estimated_tokens=estimated,
            context_limit=CLAUDE_CONTEXT_LIMIT,
            usage_percent=round((estimated / CLAUDE_CONTEXT_LIMIT) * 100, 1),
            messages_count=len(self._messages),
            has_summary=self._compact_summary is not None,
            autocompact_enabled=self._autocompact,
            autocompact_threshold=self._autocompact_threshold,
        )

    async def compact(
        self,
        *,
        keep_last_n: int | None = None,
    ) -> CompactResult:
        """Compact the conversation by summarizing older messages.

        Args:
            keep_last_n: Number of recent messages to keep. Defaults to init value.

        Returns:
            CompactResult with statistics about the compaction.
        """
        keep_n = keep_last_n if keep_last_n is not None else self._keep_last_n

        messages = self._messages
        original_count = len(messages)
        original_tokens = self._estimate_tokens(messages)

        if original_count <= keep_n:
            # Not enough messages to compact
            return CompactResult(
                original_messages=original_count,
                original_tokens_estimate=original_tokens,
                summary_tokens_estimate=original_tokens,
                summary=self._compact_summary or "",
            )

        # Split messages: old ones to summarize, recent ones to keep
        messages_to_summarize = messages[:-keep_n] if keep_n > 0 else messages
        messages_to_keep = messages[-keep_n:] if keep_n > 0 else []

        # Include previous summary if exists
        context_parts = []
        if self._compact_summary:
            context_parts.append(f"[Previous summary]: {self._compact_summary}")

        # Format messages for summarization
        context_parts.append(self._format_messages_for_summary(messages_to_summarize))
        conversation_text = "\n\n".join(context_parts)

        # Create a temporary agent for summarization (new session, no tools)
        temp_client = type(self._client)(model="haiku")
        summarizer = temp_client.create_agent(
            name="summarizer",
            instructions="""You summarize conversations concisely.
Keep all important facts, decisions, code snippets, file names, and context.
Include specific details like names, numbers, paths, and code.
Output only the summary, no preamble.""",
        )

        summary_response = await summarizer._agent.run(
            f"Summarize this conversation, preserving all important details:\n\n{conversation_text}"
        )
        summary = summary_response.text or ""

        # Reset client session to start fresh
        self._client.reset_session()

        # Update our state
        self._compact_summary = summary
        self._messages = messages_to_keep
        self._needs_context_injection = True  # Inject context on next run
        self._usage.compactions += 1

        summary_tokens = len(summary) // CHARS_PER_TOKEN
        kept_tokens = self._estimate_tokens(messages_to_keep)

        return CompactResult(
            original_messages=original_count,
            original_tokens_estimate=original_tokens,
            summary_tokens_estimate=summary_tokens + kept_tokens,
            summary=summary,
        )

    def _format_messages_for_summary(self, messages: list[ConversationMessage]) -> str:
        """Format messages into readable text for summarization."""
        parts = []
        for msg in messages:
            if msg.role == "user":
                parts.append(f"User: {msg.text}")
            elif msg.role == "assistant":
                parts.append(f"Assistant: {msg.text}")
        return "\n\n".join(parts)

    def _build_prompt_with_context(self, message: str) -> str:
        """Build prompt including compact summary context."""
        if self._compact_summary:
            return f"[Context from previous conversation]: {self._compact_summary}\n\nUser: {message}"
        return message

    async def _maybe_autocompact(self) -> CompactResult | None:
        """Autocompact if threshold is reached."""
        if not self._autocompact:
            return None

        tokens = self.get_token_estimate()
        if tokens >= self._autocompact_threshold:
            return await self.compact()
        return None

    async def run(
        self,
        message: str,
        *,
        thread: AgentThread | None = None,
        model_id: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        frequency_penalty: float | None = None,
        presence_penalty: float | None = None,
        stop: str | Sequence[str] | None = None,
        response_format: type | None = None,
        tools: ToolProtocol | Callable | MutableMapping | Sequence | None = None,
        tool_choice: ToolMode | Literal["auto", "required", "none"] | dict | None = None,
        seed: int | None = None,
        user: str | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ):
        """Run the agent with a message.

        Args:
            message: The user message.
            thread: Optional conversation thread.
            model_id: Override model for this request.
            max_tokens: Maximum tokens in response.
            temperature: Sampling temperature (0-2). Note: May not be supported by CLI.
            top_p: Nucleus sampling parameter. Note: May not be supported by CLI.
            frequency_penalty: Frequency penalty (-2 to 2). Note: May not be supported by CLI.
            presence_penalty: Presence penalty (-2 to 2). Note: May not be supported by CLI.
            stop: Stop sequences.
            response_format: Pydantic model for structured output.
            tools: Additional tools for this request.
            tool_choice: Tool selection mode.
            seed: Random seed for reproducibility. Note: May not be supported by CLI.
            user: User identifier.
            metadata: Additional metadata.
            **kwargs: Additional arguments passed to the inner agent.

        Returns:
            AgentRunResponse from the inner agent.

        Note:
            Some parameters (temperature, top_p, frequency_penalty, presence_penalty,
            seed) may not be fully supported by Claude Code CLI. They are passed
            through but the CLI may ignore them.
        """
        # Check for autocompact before running
        await self._maybe_autocompact()

        # Build prompt with context if we need to inject it after compact
        if self._needs_context_injection and self._compact_summary:
            prompt = self._build_prompt_with_context(message)
            self._needs_context_injection = False  # Only inject once
        else:
            prompt = message

        # Track user message
        # DESIGN DECISION: Message tracking is single-stream by design.
        # This agent is intended for single-stream conversation use (one run() at a time).
        # Concurrent run() calls on the same agent instance are not supported.
        # If concurrent conversations are needed, create separate agent instances.
        # Reviewed: 2025-01 - Not a bug, documented API contract.
        self._messages.append(ConversationMessage(role="user", text=message))

        # Run the inner agent with all parameters
        response = await self._agent.run(
            prompt,
            thread=thread,
            model_id=model_id,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            stop=stop,
            response_format=response_format,
            tools=tools,
            tool_choice=tool_choice,
            seed=seed,
            user=user,
            metadata=metadata,
            **kwargs,
        )

        # Track assistant response
        if response.text:
            self._messages.append(ConversationMessage(role="assistant", text=response.text))

        # Track usage
        self._usage.total_requests += 1
        if hasattr(response, 'usage_details') and response.usage_details:
            ud = response.usage_details
            if hasattr(ud, 'input_token_count'):
                self._usage.total_input_tokens += ud.input_token_count or 0
            if hasattr(ud, 'output_token_count'):
                self._usage.total_output_tokens += ud.output_token_count or 0

        return response

    async def run_stream(
        self,
        message: str,
        *,
        thread: AgentThread | None = None,
        model_id: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        frequency_penalty: float | None = None,
        presence_penalty: float | None = None,
        stop: str | Sequence[str] | None = None,
        response_format: type | None = None,
        tools: ToolProtocol | Callable | MutableMapping | Sequence | None = None,
        tool_choice: ToolMode | Literal["auto", "required", "none"] | dict | None = None,
        seed: int | None = None,
        user: str | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ):
        """Run the agent with streaming response.

        Args:
            message: The user message.
            thread: Optional conversation thread.
            model_id: Override model for this request.
            max_tokens: Maximum tokens in response.
            temperature: Sampling temperature (0-2). Note: May not be supported by CLI.
            top_p: Nucleus sampling parameter. Note: May not be supported by CLI.
            frequency_penalty: Frequency penalty (-2 to 2). Note: May not be supported by CLI.
            presence_penalty: Presence penalty (-2 to 2). Note: May not be supported by CLI.
            stop: Stop sequences.
            response_format: Pydantic model for structured output.
            tools: Additional tools for this request.
            tool_choice: Tool selection mode.
            seed: Random seed for reproducibility. Note: May not be supported by CLI.
            user: User identifier.
            metadata: Additional metadata.
            **kwargs: Additional arguments passed to the inner agent.

        Yields:
            Response chunks from the inner agent.

        Note:
            Some parameters (temperature, top_p, frequency_penalty, presence_penalty,
            seed) may not be fully supported by Claude Code CLI.
        """
        # Check for autocompact before running
        await self._maybe_autocompact()

        # Build prompt with context if we need to inject it after compact
        if self._needs_context_injection and self._compact_summary:
            prompt = self._build_prompt_with_context(message)
            self._needs_context_injection = False  # Only inject once
        else:
            prompt = message

        # Track user message
        self._messages.append(ConversationMessage(role="user", text=message))

        # Collect response text for tracking
        response_text_parts = []

        # Stream from the inner agent with all parameters
        async for chunk in self._agent.run_stream(
            prompt,
            thread=thread,
            model_id=model_id,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            stop=stop,
            response_format=response_format,
            tools=tools,
            tool_choice=tool_choice,
            seed=seed,
            user=user,
            metadata=metadata,
            **kwargs,
        ):
            if hasattr(chunk, 'text') and chunk.text:
                response_text_parts.append(chunk.text)
            yield chunk

        # Track assistant response
        full_response = "".join(response_text_parts)
        if full_response:
            self._messages.append(ConversationMessage(role="assistant", text=full_response))

    def reset(self) -> None:
        """Reset the conversation, starting fresh.

        Note: This does NOT reset usage stats. Call reset_usage() for that.
        """
        self._messages = []
        self._compact_summary = None
        self._needs_context_injection = False
        self._client.reset_session()

    def reset_usage(self) -> None:
        """Reset usage statistics to zero."""
        self._usage = UsageStats()
