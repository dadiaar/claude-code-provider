#!/usr/bin/env python3
# Copyright (c) 2025. Claude Code Provider for Microsoft Agent Framework.

"""Example 09: Advanced Model Routing

Demonstrate sophisticated routing strategies that automatically select
the optimal model based on task characteristics.

Available strategies:
    - ComplexityRouter: Routes based on prompt length/complexity
    - TaskTypeRouter: Routes based on task keywords (summarize, code, design)
    - CostOptimizedRouter: Routes considering budget constraints
    - CustomRouter: Use your own routing function

Concepts demonstrated:
    - Switching between routing strategies
    - Task-type based routing
    - Live routing with actual API calls

Run:
    python -m claude_code_provider.examples.09_model_routing_advanced
"""

import asyncio

from claude_code_provider import (
    ClaudeCodeClient,
    ModelRouter,
    ComplexityRouter,
    TaskTypeRouter,
)


async def main() -> None:
    """Run the advanced model routing example."""
    print("Example 09: Advanced Model Routing")
    print("=" * 40)

    router = ModelRouter()

    # Test prompts of varying complexity
    prompts = [
        ("Hi", "Simple greeting"),
        ("What is 2+2?", "Simple math"),
        ("Explain quantum entanglement in detail", "Complex explanation"),
        ("Design a microservices architecture for e-commerce", "Complex design"),
        ("Summarize: The cat sat.", "Simple summary"),
    ]

    # Strategy 1: Complexity-based routing
    print("\n--- ComplexityRouter ---")
    print("Routes based on prompt length and complexity indicators")
    router.set_strategy(ComplexityRouter())

    for prompt, description in prompts:
        model = router.route(prompt)
        print(f"  [{model:7}] {description}: {prompt[:40]}...")

    # Strategy 2: Task-type routing
    print("\n--- TaskTypeRouter ---")
    print("Routes based on task keywords in the prompt")
    router.set_strategy(TaskTypeRouter())

    task_prompts = [
        "Summarize this article about AI",
        "Write a poem about nature",
        "Design a database schema for users",
        "Translate 'hello' to French",
        "Research the history of computing",
    ]

    for prompt in task_prompts:
        model = router.route(prompt)
        print(f"  [{model:7}] {prompt}")

    # Live demo with actual routing
    print("\n" + "=" * 40)
    print("Live Routing Demo")
    print("=" * 40)

    router.set_strategy(TaskTypeRouter())

    # Simple task -> should route to haiku
    simple_prompt = "Define: Python"
    model = router.route(simple_prompt)
    print(f"\nPrompt: '{simple_prompt}'")
    print(f"Selected model: {model}")

    client = ClaudeCodeClient(model=model)
    agent = client.create_agent(
        name="assistant",
        instructions="Be extremely brief. One sentence max.",
    )
    response = await agent.run(simple_prompt)
    print(f"Response: {response.text}")

    print("\nDone!")


if __name__ == "__main__":
    asyncio.run(main())
