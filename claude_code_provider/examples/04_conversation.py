#!/usr/bin/env python3
# Copyright (c) 2025. Claude Code Provider for Microsoft Agent Framework.

"""Example 04: Multi-turn Conversation

Maintain context across multiple messages using the same agent.
The agent remembers previous messages in the conversation.

This works because Claude Code CLI maintains session state.

Concepts demonstrated:
    - Multi-turn conversations with memory
    - Context retention across messages
    - Building on previous responses

Run:
    python -m claude_code_provider.examples.04_conversation
"""

import asyncio

from claude_code_provider import ClaudeCodeClient


async def main() -> None:
    """Run the conversation example."""
    print("Example 04: Multi-turn Conversation")
    print("=" * 40)

    # Create client and agent
    client = ClaudeCodeClient(model="haiku")
    agent = client.create_agent(
        name="memory-test",
        instructions="You are a helpful assistant. Remember what the user tells you.",
    )

    # Turn 1: Introduce information
    print("\nYou: My name is Alice and I like Python.")
    response1 = await agent.run("My name is Alice and I like Python.")
    print(f"Claude: {response1.text}\n")

    # Turn 2: Ask agent to recall information
    print("You: What's my name and what do I like?")
    response2 = await agent.run("What's my name and what do I like?")
    print(f"Claude: {response2.text}\n")

    # Turn 3: Build on the context
    print("You: Recommend me a Python library based on my interests.")
    response3 = await agent.run(
        "Recommend me a Python library based on my interests."
    )
    print(f"Claude: {response3.text}")

    print("\nDone!")


if __name__ == "__main__":
    asyncio.run(main())
