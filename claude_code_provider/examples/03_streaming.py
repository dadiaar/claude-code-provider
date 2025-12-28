#!/usr/bin/env python3
# Copyright (c) 2025. Claude Code Provider for Microsoft Agent Framework.

"""Example 03: Streaming Responses

Get responses token-by-token as they're generated, rather than waiting
for the complete response.

Streaming is useful for:
    - Showing progress to users in real-time
    - Long responses where you want immediate feedback
    - Detecting hung connections faster (via per-chunk timeout)

Concepts demonstrated:
    - Using agent.run_stream() instead of agent.run()
    - Processing streaming chunks
    - Real-time output

Run:
    python -m claude_code_provider.examples.03_streaming
"""

import asyncio

from claude_code_provider import ClaudeCodeClient


async def main() -> None:
    """Run the streaming example."""
    print("Example 03: Streaming Responses")
    print("=" * 40)

    client = ClaudeCodeClient(model="haiku")

    agent = client.create_agent(
        name="poet",
        instructions="You write poetry. Be creative but concise.",
    )

    print("\nStreaming response:\n")
    print("-" * 40)

    # Use run_stream() for streaming - yields chunks as they arrive
    async for chunk in agent.run_stream("Write a haiku about programming."):
        # Each chunk may have text content
        if hasattr(chunk, 'text') and chunk.text:
            # Print without newline, flush immediately for real-time output
            print(chunk.text, end="", flush=True)

    print("\n" + "-" * 40)
    print("\nDone!")


if __name__ == "__main__":
    asyncio.run(main())
