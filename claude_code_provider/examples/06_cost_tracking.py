#!/usr/bin/env python3
# Copyright (c) 2025. Claude Code Provider for Microsoft Agent Framework.

"""Example 06: Cost Tracking

Monitor token usage and estimated costs across requests.
Useful for budget management and usage analysis.

Note: Cost tracking is informational. With a Claude subscription,
you pay the same regardless of usage.

Concepts demonstrated:
    - Creating a CostTracker
    - Recording request costs
    - Setting budget limits
    - Getting usage summaries

Run:
    python -m claude_code_provider.examples.06_cost_tracking
"""

import asyncio

from claude_code_provider import ClaudeCodeClient, CostTracker


async def main() -> None:
    """Run the cost tracking example."""
    print("Example 06: Cost Tracking")
    print("=" * 40)

    client = ClaudeCodeClient(model="haiku")
    tracker = CostTracker()

    # Set a budget limit (informational only)
    tracker.set_budget(max_cost=1.0)

    agent = client.create_agent(
        name="assistant",
        instructions="Be concise. One sentence answers only.",
    )

    # Process multiple prompts
    prompts = [
        "What is Python?",
        "What is JavaScript?",
        "What is Rust?",
    ]

    print("\nProcessing prompts with cost tracking...\n")

    for prompt in prompts:
        response = await agent.run(prompt)

        # Record the cost based on usage
        usage = response.usage_details
        if usage:
            cost = tracker.record_request(
                model="haiku",
                input_tokens=usage.input_token_count or 0,
                output_tokens=usage.output_token_count or 0,
            )
            print(f"Q: {prompt}")
            print(f"A: {response.text[:80]}...")
            print(f"Cost: ${cost.total_cost:.6f}")
            print()

    # Show summary
    summary = tracker.get_summary()
    print("=" * 40)
    print("COST SUMMARY")
    print("=" * 40)
    print(f"Total requests: {summary.total_requests}")
    print(f"Total input tokens: {summary.total_input_tokens:,}")
    print(f"Total output tokens: {summary.total_output_tokens:,}")
    print(f"Total cost: ${summary.total_cost:.6f}")

    remaining = tracker.get_remaining_budget()
    if remaining is not None:
        print(f"Remaining budget: ${remaining:.6f}")

    if tracker.is_over_budget():
        print("WARNING: Over budget!")

    print("\nDone!")


if __name__ == "__main__":
    asyncio.run(main())
