# Copyright (c) 2025. Claude Code Provider for Microsoft Agent Framework.

"""Parse Claude CLI responses into MAF types."""

from typing import Any
from uuid import uuid4

from agent_framework import (
    ChatMessage,
    ChatResponse,
    ChatResponseUpdate,
    FinishReason,
    Role,
    TextContent,
    UsageDetails,
)

try:
    from ._cli_executor import CLIResult, StreamEvent
except ImportError:
    from _cli_executor import CLIResult, StreamEvent


def _calculate_total_input_tokens(usage: dict) -> int:
    """Calculate total input tokens including cache tokens.

    Claude Code CLI reports input tokens in three separate fields:
    - input_tokens: New tokens not from cache
    - cache_creation_input_tokens: Tokens written to cache
    - cache_read_input_tokens: Tokens read from cache

    The total input tokens is the sum of all three.

    Args:
        usage: Usage dictionary from CLI response.

    Returns:
        Total input token count.
    """
    return (
        usage.get("input_tokens", 0) +
        usage.get("cache_creation_input_tokens", 0) +
        usage.get("cache_read_input_tokens", 0)
    )


def parse_cli_result_to_response(result: CLIResult) -> ChatResponse:
    """Convert a CLIResult to a MAF ChatResponse.

    Args:
        result: The CLI execution result.

    Returns:
        A ChatResponse object.
    """
    # Build contents
    contents = []
    if result.result:
        contents.append(TextContent(text=result.result))

    # Build message
    message = ChatMessage(
        role=Role.ASSISTANT,
        contents=contents,
        raw_representation=result.raw_response,
    )

    # Parse usage - include all input token types (regular + cache)
    usage = None
    if result.usage:
        input_tokens = _calculate_total_input_tokens(result.usage)
        output_tokens = result.usage.get("output_tokens", 0)
        usage = UsageDetails(
            input_token_count=input_tokens,
            output_token_count=output_tokens,
            total_token_count=input_tokens + output_tokens,
        )
        # Also store raw token breakdown for cost calculation
        # (cache tokens may have different pricing)
        if "input_tokens" in result.usage:
            usage.additional_counts["raw_input_tokens"] = result.usage["input_tokens"]
        if "cache_creation_input_tokens" in result.usage:
            usage.additional_counts["cache_creation_input_tokens"] = result.usage[
                "cache_creation_input_tokens"
            ]
        if "cache_read_input_tokens" in result.usage:
            usage.additional_counts["cache_read_input_tokens"] = result.usage[
                "cache_read_input_tokens"
            ]

    # Determine finish reason
    finish_reason = FinishReason.STOP
    if result.error:
        finish_reason = FinishReason.CONTENT_FILTER  # or could use a custom reason

    return ChatResponse(
        response_id=result.session_id or str(uuid4()),
        messages=[message],
        usage_details=usage,
        finish_reason=finish_reason,
        raw_response=result.raw_response,
    )


def parse_stream_event_to_update(event: StreamEvent) -> ChatResponseUpdate | None:
    """Convert a StreamEvent to a MAF ChatResponseUpdate.

    Args:
        event: The streaming event from CLI.

    Returns:
        A ChatResponseUpdate object, or None if the event should be skipped.
    """
    if event.event_type == "system":
        # Initial system event - could extract session_id
        session_id = event.data.get("session_id")
        return ChatResponseUpdate(
            response_id=session_id,
            contents=[],
            raw_response=event.data,
        )

    elif event.event_type == "assistant":
        # Assistant message with content
        message_data = event.data.get("message", {})
        content_blocks = message_data.get("content", [])

        contents = []
        for block in content_blocks:
            if block.get("type") == "text":
                text = block.get("text", "")
                if text:
                    contents.append(TextContent(text=text))

        if not contents:
            return None

        # Parse usage from message - include all input token types
        usage = None
        usage_data = message_data.get("usage")
        if usage_data:
            usage = UsageDetails(
                input_token_count=_calculate_total_input_tokens(usage_data),
                output_token_count=usage_data.get("output_tokens", 0),
            )

        session_id = event.data.get("session_id")
        message_id = message_data.get("id")

        return ChatResponseUpdate(
            response_id=session_id,
            message_id=message_id,
            contents=contents,
            role=Role.ASSISTANT,
            raw_response=event.data,
        )

    elif event.event_type == "result":
        # Final result event - don't include text here as it duplicates assistant message
        # Only include metadata (usage, finish_reason, session_id)
        session_id = event.data.get("session_id")
        contents = []  # Don't add result_text - it's already in assistant events

        # Parse final usage - include all input token types
        usage = None
        usage_data = event.data.get("usage")
        if usage_data:
            usage = UsageDetails(
                input_token_count=_calculate_total_input_tokens(usage_data),
                output_token_count=usage_data.get("output_tokens", 0),
            )

        finish_reason = FinishReason.STOP
        if event.data.get("is_error"):
            finish_reason = FinishReason.CONTENT_FILTER

        return ChatResponseUpdate(
            response_id=session_id,
            contents=contents,
            finish_reason=finish_reason,
            raw_response=event.data,
        )

    elif event.event_type == "error":
        # Error event
        error_msg = event.data.get("error", "Unknown error")
        return ChatResponseUpdate(
            contents=[TextContent(text=f"Error: {error_msg}")],
            finish_reason=FinishReason.CONTENT_FILTER,
            raw_response=event.data,
        )

    # Skip unknown event types
    return None


def extract_session_id_from_response(response: ChatResponse) -> str | None:
    """Extract session ID from a ChatResponse for conversation continuity.

    Args:
        response: The chat response.

    Returns:
        The session ID if available.
    """
    if response.raw_response and isinstance(response.raw_response, dict):
        return response.raw_response.get("session_id")
    return response.response_id
