# Copyright (c) 2025. Claude Code Provider for Microsoft Agent Framework.

"""Convert MAF messages to Claude CLI format."""

from typing import Any, Sequence

from agent_framework import ChatMessage, ChatOptions, Role


def extract_system_prompt(messages: Sequence[ChatMessage]) -> str | None:
    """Extract system prompt from messages.

    The first message with SYSTEM role is used as the system prompt.

    Args:
        messages: List of chat messages.

    Returns:
        System prompt text if found, None otherwise.
    """
    for msg in messages:
        if msg.role == Role.SYSTEM:
            return msg.text
    return None


def extract_user_prompt(messages: Sequence[ChatMessage]) -> str:
    """Extract the user prompt from messages.

    Concatenates all non-system messages into a single prompt.
    For conversation history, formats as a dialogue.

    Args:
        messages: List of chat messages.

    Returns:
        The formatted prompt string.
    """
    parts: list[str] = []

    for msg in messages:
        if msg.role == Role.SYSTEM:
            continue  # System prompt handled separately

        text = msg.text or ""
        if not text:
            continue

        if msg.role == Role.USER:
            if len(parts) == 0:
                # First user message - just the text
                parts.append(text)
            else:
                # Subsequent user message in conversation
                parts.append(f"\n\nUser: {text}")
        elif msg.role == Role.ASSISTANT:
            parts.append(f"\n\nAssistant: {text}")
        elif msg.role == Role.TOOL:
            # Tool results - format as context
            parts.append(f"\n\n[Tool Result]: {text}")

    return "".join(parts).strip()


def format_conversation_for_resume(
    messages: Sequence[ChatMessage],
    session_id: str | None,
) -> tuple[str, str | None]:
    """Format messages for a conversation that may be resumed.

    If we have a session_id, we only need the latest user message.
    Otherwise, we need to include the full conversation.

    Args:
        messages: List of chat messages.
        session_id: Existing session ID if resuming.

    Returns:
        Tuple of (prompt, session_id_to_use).
    """
    if session_id:
        # Resuming - only need latest user message
        for msg in reversed(messages):
            if msg.role == Role.USER and msg.text:
                return msg.text, session_id

    # No session or no user message - format full conversation
    return extract_user_prompt(messages), None


def chat_options_to_cli_args(options: ChatOptions) -> list[str]:
    """Convert ChatOptions to CLI arguments.

    Args:
        options: MAF chat options.

    Returns:
        List of CLI arguments.
    """
    args: list[str] = []

    # Note: Model is passed separately via the 'model' parameter, not extra_args.
    # The model_id from chat_options is returned directly in prepare_cli_execution
    # and handled by _build_args in the CLI executor.

    # Note: Claude CLI doesn't support these directly:
    # - temperature
    # - top_p
    # - max_tokens (uses max_turns for agentic behavior)
    # - frequency_penalty
    # - presence_penalty

    # Tools - if specified as a list of strings
    if options.tools:
        tool_names = []
        for tool in options.tools:
            if isinstance(tool, str):
                tool_names.append(tool)
            elif hasattr(tool, "name"):
                tool_names.append(tool.name)
            elif isinstance(tool, dict) and "name" in tool:
                tool_names.append(tool["name"])
        if tool_names:
            args.extend(["--tools", ",".join(tool_names)])

    return args


def prepare_cli_execution(
    messages: Sequence[ChatMessage],
    chat_options: ChatOptions,
    session_id: str | None = None,
) -> dict[str, Any]:
    """Prepare all parameters for CLI execution.

    Args:
        messages: List of chat messages.
        chat_options: Chat options.
        session_id: Existing session ID if resuming.

    Returns:
        Dictionary with 'prompt', 'system_prompt', 'session_id', 'extra_args'.
    """
    system_prompt = extract_system_prompt(messages)

    # Also check instructions in chat_options
    if chat_options.instructions and not system_prompt:
        system_prompt = chat_options.instructions
    elif chat_options.instructions and system_prompt:
        system_prompt = f"{chat_options.instructions}\n\n{system_prompt}"

    prompt, effective_session_id = format_conversation_for_resume(messages, session_id)

    extra_args = chat_options_to_cli_args(chat_options)

    return {
        "prompt": prompt,
        "system_prompt": system_prompt,
        "session_id": effective_session_id,
        "extra_args": extra_args,
        "model": chat_options.model_id,
    }
