#!/usr/bin/env python3
# Copyright (c) 2025. Claude Code Provider for Microsoft Agent Framework.

"""Example 11: Batch Processing

Process multiple prompts efficiently with concurrency control.
The BatchProcessor handles parallel execution and aggregates results.

Patterns demonstrated:
    - Simple batch: Process list of prompts in parallel
    - Map-reduce: Process in parallel, then combine results

Concepts demonstrated:
    - Creating a BatchProcessor
    - Setting concurrency limits
    - Progress callbacks
    - Map-reduce pattern

Run:
    python -m claude_code_provider.examples.11_batch_processing
"""

import asyncio

from claude_code_provider import ClaudeCodeClient, BatchProcessor


async def main() -> None:
    """Run the batch processing example."""
    print("Example 11: Batch Processing")
    print("=" * 40)

    client = ClaudeCodeClient(model="haiku")
    processor = BatchProcessor(client, default_concurrency=3)

    # Simple batch processing
    prompts = [
        "Define: Python",
        "Define: JavaScript",
        "Define: Rust",
        "Define: Go",
        "Define: TypeScript",
    ]

    def progress(done: int, total: int) -> None:
        print(f"  Progress: {done}/{total}")

    print("\n--- Simple Batch ---")
    print("Processing batch of definitions...\n")

    result = await processor.process_batch(
        prompts=prompts,
        system_prompt="Give a one-sentence definition.",
        concurrency=3,
        progress_callback=progress,
    )

    print(f"\nResults (success rate: {result.success_rate:.0%}):\n")
    for item in result.items:
        if item.result:
            print(f"  {item.prompt}: {item.result[:60]}...")

    # Map-reduce pattern
    print("\n" + "=" * 40)
    print("--- Map-Reduce Pattern ---")
    print("=" * 40 + "\n")

    topics = ["Python", "JavaScript", "Rust"]
    map_prompts = [f"List 3 key features of {t}" for t in topics]

    map_result, final = await processor.map_reduce(
        prompts=map_prompts,
        reduce_prompt="Compare these programming languages based on the features listed:\n{results}",
        system_prompt="Be concise.",
    )

    print("Individual results:")
    for item, topic in zip(map_result.items, topics):
        if item.result:
            print(f"\n{topic}:\n{item.result[:150]}...")

    print("\n" + "-" * 40)
    print("Combined comparison:")
    print(final[:500] if final else "No result")

    print("\nDone!")


if __name__ == "__main__":
    asyncio.run(main())
