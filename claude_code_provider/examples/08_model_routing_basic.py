#!/usr/bin/env python3
# Copyright (c) 2025. Claude Code Provider for Microsoft Agent Framework.

"""Example 08: Basic Model Routing

Automatically select models based on prompt complexity using the
SimpleRouter and basic ModelRouter functionality.

This is the simpler routing example. See 09_model_routing_advanced.py
for more sophisticated routing strategies.

Concepts demonstrated:
    - Creating a ModelRouter
    - Using SimpleRouter for fixed model selection
    - Using the default ComplexityRouter
    - Routing prompts to models

Run:
    python -m claude_code_provider.examples.08_model_routing_basic
"""

import asyncio

from claude_code_provider import (
    ClaudeCodeClient,
    ModelRouter,
    SimpleRouter,
)


async def main() -> None:
    """Run the basic model routing example."""
    print("Example 08: Basic Model Routing")
    print("=" * 40)

    # Create a router with default complexity-based strategy
    router = ModelRouter()

    # Test prompts of varying complexity
    prompts = [
        ("Hi!", "Simple greeting"),
        ("What is 2+2?", "Simple math"),
        ("Explain quantum computing in detail", "Complex topic"),
    ]

    print("\n--- Default ComplexityRouter ---")
    for prompt, description in prompts:
        model = router.route(prompt)
        print(f"  [{model:7}] {description}")

    # Use SimpleRouter for fixed model
    print("\n--- SimpleRouter (fixed model) ---")
    router.set_strategy(SimpleRouter(model="haiku"))

    for prompt, description in prompts:
        model = router.route(prompt)
        print(f"  [{model:7}] {description}")

    # Actually use the routed model
    print("\n--- Live Routing Demo ---")
    router.set_strategy(SimpleRouter(model="haiku"))

    prompt = "What is Python? One sentence."
    model = router.route(prompt)
    print(f"Prompt: '{prompt}'")
    print(f"Routed to: {model}")

    client = ClaudeCodeClient(model=model)
    agent = client.create_agent(
        name="assistant",
        instructions="Be brief.",
    )
    response = await agent.run(prompt)
    print(f"Response: {response.text}")

    print("\nDone!")


if __name__ == "__main__":
    asyncio.run(main())
