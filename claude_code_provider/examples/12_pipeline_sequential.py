#!/usr/bin/env python3
# Copyright (c) 2025. Claude Code Provider for Microsoft Agent Framework.

"""Example 12: Sequential Pipeline

A three-stage pipeline where each agent builds on the previous one's work.
Can use SequentialOrchestrator or manual handoff.

Pipeline stages:
    1. Idea Generator - creates product concept
    2. Feature Designer - lists key features
    3. Marketing Writer - creates promotional copy

Concepts demonstrated:
    - SequentialOrchestrator for clean orchestration
    - Manual pipeline as fallback
    - Multi-stage processing

Run:
    python -m claude_code_provider.examples.12_pipeline_sequential

Output:
    Results saved to ./results/12_pipeline_sequential/result.json
"""

import asyncio
import json
from datetime import datetime
from pathlib import Path

from claude_code_provider import (
    ClaudeCodeClient,
    SequentialOrchestrator,
    MAF_ORCHESTRATION_AVAILABLE,
)


async def run_with_orchestrator() -> dict:
    """Run pipeline using SequentialOrchestrator."""
    client = ClaudeCodeClient(model="haiku")

    print("Using SequentialOrchestrator\n")

    # Create pipeline agents
    idea_generator = client.create_agent(
        name="idea_generator",
        instructions="""You generate creative product ideas.
Given a category, suggest ONE innovative product idea.
Format: Product name and a one-sentence description.""",
    )

    feature_designer = client.create_agent(
        name="feature_designer",
        instructions="""You design product features.
Based on the product idea in the conversation, list exactly 3 key features.
Format: Bullet points.""",
    )

    marketing_writer = client.create_agent(
        name="marketing_writer",
        instructions="""You write compelling marketing copy.
Based on the product and features in the conversation,
write a short tagline and 2-sentence description.""",
    )

    # Create sequential orchestrator
    orchestrator = SequentialOrchestrator(
        agents=[idea_generator, feature_designer, marketing_writer],
    )

    category = "smart home devices"
    print(f"Category: {category}\n")
    print("Running sequential pipeline...")
    print("-" * 40)

    result = await orchestrator.run(
        f"Generate an innovative product idea for: {category}"
    )

    print(f"\nPipeline completed in {result.rounds} stages")
    print(f"Agents used: {result.participants_used}")
    print("\n" + "-" * 40)
    print("FINAL OUTPUT:")
    print("-" * 40)
    print(result.final_output)

    return {
        "approach": "orchestrator",
        "category": category,
        "stages": result.rounds,
        "agents": result.participants_used,
        "final_output": result.final_output,
    }


async def run_manual() -> dict:
    """Run pipeline manually (fallback approach)."""
    client = ClaudeCodeClient(model="haiku")

    print("Using manual pipeline approach\n")

    # Create agents
    idea_generator = client.create_agent(
        name="idea_generator",
        instructions="""You generate creative product ideas.
Given a category, suggest ONE innovative product idea.
Format: Just the product name and a one-sentence description.""",
    )

    feature_designer = client.create_agent(
        name="feature_designer",
        instructions="""You design product features.
Given a product idea, list exactly 3 key features.
Format: Bullet points starting with "- ".""",
    )

    marketing_writer = client.create_agent(
        name="marketing_writer",
        instructions="""You write compelling marketing copy.
Given a product and its features, write a short tagline and
2-sentence description. Be enthusiastic but professional.""",
    )

    category = "smart home devices"

    # Stage 1
    print(f"Category: {category}\n")
    print("Stage 1: Generating idea...")
    idea_response = await idea_generator.run(
        f"Generate an innovative product idea for: {category}"
    )
    idea = idea_response.text
    print(f"\nIdea:\n{idea}\n")

    # Stage 2
    print("Stage 2: Designing features...")
    features_response = await feature_designer.run(
        f"Design 3 key features for this product:\n{idea}"
    )
    features = features_response.text
    print(f"\nFeatures:\n{features}\n")

    # Stage 3
    print("Stage 3: Creating marketing copy...")
    marketing_response = await marketing_writer.run(
        f"""Create marketing copy for this product:

Product: {idea}

Features:
{features}"""
    )
    marketing = marketing_response.text
    print(f"\nMarketing Copy:\n{marketing}\n")

    return {
        "approach": "manual",
        "category": category,
        "stages": [
            {"stage": 1, "agent": "idea_generator", "output": idea},
            {"stage": 2, "agent": "feature_designer", "output": features},
            {"stage": 3, "agent": "marketing_writer", "output": marketing},
        ],
        "final_output": marketing,
    }


def save_results(results: dict) -> Path:
    """Save results to JSON file."""
    output_dir = Path.cwd() / "results" / "12_pipeline_sequential"
    output_dir.mkdir(parents=True, exist_ok=True)

    results["timestamp"] = datetime.now().isoformat()
    results["example"] = "12_pipeline_sequential"

    output_file = output_dir / "result.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2, default=str)

    return output_file


async def main() -> None:
    """Run the sequential pipeline example."""
    print("Example 12: Sequential Pipeline")
    print("=" * 40)

    results = None
    if MAF_ORCHESTRATION_AVAILABLE:
        try:
            results = await run_with_orchestrator()
        except Exception as e:
            print(f"Orchestrator failed ({e}), using manual approach...\n")
            results = await run_manual()
    else:
        print("MAF orchestration not available, using manual approach...\n")
        results = await run_manual()

    # Save results
    output_file = save_results(results)
    print(f"\nResults saved to: {output_file}")
    print("Pipeline complete!")


if __name__ == "__main__":
    asyncio.run(main())
