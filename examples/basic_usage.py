#!/usr/bin/env python3
# Copyright (c) 2025. Claude Code Provider for Microsoft Agent Framework.

"""Basic usage example for Claude Code Provider.

This example demonstrates how to use the ClaudeCodeClient to create
agents that use Claude Code CLI instead of direct API calls.

Run with:
    python -m examples.basic_usage

Or from the swarm directory:
    python claude-code-provider/examples/basic_usage.py
"""

import asyncio
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "microsoft-agent-framework" / "python" / "packages" / "core"))

from claude_code_provider import ClaudeCodeClient


async def example_simple_query():
    """Simple query example."""
    print("=" * 60)
    print("Example 1: Simple Query")
    print("=" * 60)

    # Create client
    client = ClaudeCodeClient(model="haiku")  # Use haiku for speed/cost

    # Simple query without agent
    response = await client.get_response("What is 2 + 2? Reply with just the number.")

    print(f"Response: {response.messages[0].text}")
    print(f"Session ID: {client.current_session_id}")
    if response.usage_details:
        print(f"Tokens: {response.usage_details.input_token_count} in, {response.usage_details.output_token_count} out")
    print()


async def example_agent():
    """Agent example with instructions."""
    print("=" * 60)
    print("Example 2: Agent with Instructions")
    print("=" * 60)

    # Create client
    client = ClaudeCodeClient(model="haiku")

    # Create agent with instructions
    agent = client.create_agent(
        name="math_tutor",
        instructions="You are a helpful math tutor. Give brief, clear explanations.",
    )

    # Run agent
    response = await agent.run("Explain what a prime number is in one sentence.")

    print(f"Response: {response.text}")
    print()


async def example_conversation():
    """Multi-turn conversation example."""
    print("=" * 60)
    print("Example 3: Multi-turn Conversation")
    print("=" * 60)

    # Create client
    client = ClaudeCodeClient(model="haiku")

    # First turn
    response1 = await client.get_response("My name is Alice. Remember it.")
    print(f"Turn 1: {response1.messages[0].text}")
    print(f"Session: {client.current_session_id}")

    # Second turn - should remember context via session
    response2 = await client.get_response("What is my name?")
    print(f"Turn 2: {response2.messages[0].text}")
    print()


async def example_streaming():
    """Streaming response example."""
    print("=" * 60)
    print("Example 4: Streaming Response")
    print("=" * 60)

    # Create client
    client = ClaudeCodeClient(model="haiku")

    print("Streaming: ", end="", flush=True)
    async for update in client.get_streaming_response("Count from 1 to 5, one number per line."):
        if update.text:
            print(update.text, end="", flush=True)
    print("\n")


async def main():
    """Run all examples."""
    print("\nClaude Code Provider - Basic Usage Examples\n")

    try:
        await example_simple_query()
        await example_agent()
        await example_conversation()
        await example_streaming()
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Make sure the 'claude' CLI is installed and in your PATH.")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    print("All examples completed successfully!")


if __name__ == "__main__":
    asyncio.run(main())
