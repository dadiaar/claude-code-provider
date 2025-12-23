# Copyright (c) 2025. Claude Code Provider for Microsoft Agent Framework.

"""Enhanced Agent with compact functionality."""

from typing import Any, TYPE_CHECKING
from dataclasses import dataclass, field

from agent_framework import ChatMessage, Role
from agent_framework._agents import ChatAgent

if TYPE_CHECKING:
    from ._chat_client import ClaudeCodeClient


# Approximate tokens per character (conservative estimate)
CHARS_PER_TOKEN = 4

# Default threshold for autocompact (in estimated tokens)
DEFAULT_AUTOCOMPACT_THRESHOLD = 100_000


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

    Wraps MAF's ChatAgent and adds:
    - compact(): Summarize conversation history
    - autocompact: Automatically compact when threshold is reached
    - Token usage estimation

    Example:
        ```python
        client = ClaudeCodeClient(model="haiku")
        agent = client.create_agent(
            name="assistant",
            instructions="You are helpful.",
            autocompact=True,  # Enable autocompact
            autocompact_threshold=50_000,  # Tokens
        )

        # Use normally - autocompact happens automatically
        response = await agent.run("Hello!")

        # Or manually compact
        result = await agent.compact()
        print(f"Reduced from {result.original_tokens_estimate} to {result.summary_tokens_estimate} tokens")
        ```
    """

    def __init__(
        self,
        inner_agent: ChatAgent,
        client: "ClaudeCodeClient",
        *,
        autocompact: bool = False,
        autocompact_threshold: int = DEFAULT_AUTOCOMPACT_THRESHOLD,
        keep_last_n_messages: int = 2,
    ) -> None:
        """Initialize the enhanced agent.

        Args:
            inner_agent: The MAF ChatAgent to wrap.
            client: The ClaudeCodeClient for making compact requests.
            autocompact: Whether to automatically compact when threshold is reached.
            autocompact_threshold: Token threshold for autocompact.
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

    @property
    def name(self) -> str | None:
        """Get the agent name."""
        return self._agent.name

    @property
    def instructions(self) -> str | None:
        """Get the agent instructions."""
        return self._agent.instructions

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
        **kwargs: Any,
    ):
        """Run the agent with a message.

        Args:
            message: The user message.
            **kwargs: Additional arguments passed to the inner agent.

        Returns:
            AgentRunResponse from the inner agent.
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

        # Run the inner agent
        response = await self._agent.run(prompt, **kwargs)

        # Track assistant response
        if response.text:
            self._messages.append(ConversationMessage(role="assistant", text=response.text))

        return response

    async def run_stream(
        self,
        message: str,
        **kwargs: Any,
    ):
        """Run the agent with streaming response.

        Args:
            message: The user message.
            **kwargs: Additional arguments passed to the inner agent.

        Yields:
            Response chunks from the inner agent.
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

        # Stream from the inner agent
        async for chunk in self._agent.run_stream(prompt, **kwargs):
            if hasattr(chunk, 'text') and chunk.text:
                response_text_parts.append(chunk.text)
            yield chunk

        # Track assistant response
        full_response = "".join(response_text_parts)
        if full_response:
            self._messages.append(ConversationMessage(role="assistant", text=full_response))

    def reset(self) -> None:
        """Reset the conversation, starting fresh."""
        self._messages = []
        self._compact_summary = None
        self._needs_context_injection = False
        self._client.reset_session()
