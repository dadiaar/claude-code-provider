#!/usr/bin/env python3
# Copyright (c) 2025. Claude Code Provider for Microsoft Agent Framework.

"""Example 01: Hello World

The simplest possible example - create a client, make an agent, get a response.
No streaming, no tools, no sessions - just the basics.

Concepts demonstrated:
    - Creating a ClaudeCodeClient
    - Creating an agent with instructions
    - Running a prompt and getting the response

Run:
    python -m claude_code_provider.examples.01_hello_world
"""

import asyncio

from claude_code_provider import ClaudeCodeClient


async def main() -> None:
    """Run the hello world example."""
    print("Example 01: Hello World")
    print("=" * 40)

    # Step 1: Create a client with a model
    # Options: "haiku" (fast/cheap), "sonnet" (balanced), "opus" (powerful)
    client = ClaudeCodeClient(model="haiku")

    # Step 2: Create an agent with a name and instructions
    agent = client.create_agent(
        name="greeter",
        instructions="You are a friendly assistant. Be brief and concise.",
    )

    # Step 3: Send a message and get the response
    response = await agent.run("Say hello in exactly 5 words.")

    # Step 4: Print the response
    print(f"\nResponse: {response.text}")
    print("\nDone!")


if __name__ == "__main__":
    asyncio.run(main())
