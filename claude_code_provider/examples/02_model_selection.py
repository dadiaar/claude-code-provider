#!/usr/bin/env python3
# Copyright (c) 2025. Claude Code Provider for Microsoft Agent Framework.

"""Example 02: Model Selection

Shows how to use different Claude models and the ClaudeModel enum for
type-safe model selection.

Models available:
    - haiku: Fastest, cheapest, good for simple tasks
    - sonnet: Balanced speed/capability, good for most tasks
    - opus: Most capable, best for complex reasoning

Concepts demonstrated:
    - Using different models by string name
    - Using ClaudeModel enum for type-safe selection
    - Comparing model responses

Run:
    python -m claude_code_provider.examples.02_model_selection
"""

import asyncio

from claude_code_provider import ClaudeCodeClient, ClaudeModel


async def main() -> None:
    """Run the model selection example."""
    print("Example 02: Model Selection")
    print("=" * 40)

    # The same prompt to compare models
    prompt = "What is 2 + 2? Reply with just the number."

    # Method 1: Using string model names
    print("\n--- Using string model names ---")
    client_haiku = ClaudeCodeClient(model="haiku")
    agent_haiku = client_haiku.create_agent(
        name="math_haiku",
        instructions="Be extremely brief.",
    )
    response = await agent_haiku.run(prompt)
    print(f"Haiku: {response.text}")

    # Method 2: Using ClaudeModel enum (type-safe, IDE autocomplete)
    print("\n--- Using ClaudeModel enum ---")
    client_sonnet = ClaudeCodeClient(model=ClaudeModel.SONNET)
    agent_sonnet = client_sonnet.create_agent(
        name="math_sonnet",
        instructions="Be extremely brief.",
    )
    response = await agent_sonnet.run(prompt)
    print(f"Sonnet: {response.text}")

    # Show all available model options
    print("\n--- Available ClaudeModel options ---")
    print("Aliases (recommended):")
    print(f"  ClaudeModel.HAIKU  = '{ClaudeModel.HAIKU}'")
    print(f"  ClaudeModel.SONNET = '{ClaudeModel.SONNET}'")
    print(f"  ClaudeModel.OPUS   = '{ClaudeModel.OPUS}'")
    print("\nFull model names:")
    print(f"  ClaudeModel.SONNET_3_5 = '{ClaudeModel.SONNET_3_5}'")
    print(f"  ClaudeModel.HAIKU_3_5  = '{ClaudeModel.HAIKU_3_5}'")
    print(f"  ClaudeModel.OPUS_3     = '{ClaudeModel.OPUS_3}'")
    print(f"  ClaudeModel.SONNET_4   = '{ClaudeModel.SONNET_4}'")
    print(f"  ClaudeModel.OPUS_4     = '{ClaudeModel.OPUS_4}'")

    print("\nDone!")


if __name__ == "__main__":
    asyncio.run(main())
