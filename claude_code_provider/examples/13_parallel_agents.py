#!/usr/bin/env python3
# Copyright (c) 2025. Claude Code Provider for Microsoft Agent Framework.

"""Example 13: Parallel Multi-Agent Execution

Multiple agents work in parallel using ConcurrentOrchestrator or asyncio.gather.
Results are then combined by a synthesizer agent.

Pattern: Fan-out / Fan-in (Concurrent)
    - Input broadcasts to multiple agents simultaneously
    - Each agent provides their unique perspective
    - Synthesizer aggregates into unified output

Concepts demonstrated:
    - ConcurrentOrchestrator for parallel execution
    - Manual parallel execution with asyncio.gather
    - Fan-out / fan-in pattern
    - Synthesis of parallel results

Run:
    python -m claude_code_provider.examples.13_parallel_agents

Output:
    Results saved to ./results/13_parallel_agents/result.json
"""

import asyncio
import json
from datetime import datetime
from pathlib import Path

from claude_code_provider import (
    ClaudeCodeClient,
    ConcurrentOrchestrator,
    MAF_ORCHESTRATION_AVAILABLE,
)


async def run_with_orchestrator() -> dict:
    """Run parallel analysis using ConcurrentOrchestrator."""
    client = ClaudeCodeClient(model="haiku")

    print("Using ConcurrentOrchestrator\n")

    # Create specialized analysts (all run in parallel)
    technical_analyst = client.create_agent(
        name="technical_analyst",
        instructions="""You are a technical analyst. Analyze from a TECHNICAL perspective:
- Technology requirements and challenges
- Implementation feasibility
Be concise (2-3 bullet points).""",
    )

    market_analyst = client.create_agent(
        name="market_analyst",
        instructions="""You are a market analyst. Analyze from a MARKET perspective:
- Target market size
- Competition landscape
Be concise (2-3 bullet points).""",
    )

    financial_analyst = client.create_agent(
        name="financial_analyst",
        instructions="""You are a financial analyst. Analyze from a FINANCIAL perspective:
- Cost structure
- Revenue potential
Be concise (2-3 bullet points).""",
    )

    # Create concurrent orchestrator
    orchestrator = ConcurrentOrchestrator(
        agents=[technical_analyst, market_analyst, financial_analyst],
    )

    topic = "A startup building an AI-powered personal finance assistant"

    print(f"Topic: {topic}\n")
    print("Running parallel analysis...")
    print("-" * 40)

    result = await orchestrator.run(f"Analyze this opportunity:\n\n{topic}")

    print(f"\nCompleted with {len(result.participants_used)} analysts in parallel")
    print(f"Analysts: {result.participants_used}")
    print("\n" + "-" * 40)
    print("COMBINED OUTPUT:")
    print("-" * 40)
    print(result.final_output[:1000] + "..." if len(result.final_output) > 1000 else result.final_output)

    return {
        "approach": "orchestrator",
        "topic": topic,
        "analysts": result.participants_used,
        "combined_output": result.final_output,
    }


async def run_manual() -> dict:
    """Run parallel analysis manually with asyncio.gather."""
    client = ClaudeCodeClient(model="haiku")

    print("Using manual asyncio.gather approach\n")

    # Create analysts
    analysts = {
        "technical": client.create_agent(
            name="technical_analyst",
            instructions="Analyze from TECHNICAL perspective. Be very concise.",
        ),
        "market": client.create_agent(
            name="market_analyst",
            instructions="Analyze from MARKET perspective. Be very concise.",
        ),
        "financial": client.create_agent(
            name="financial_analyst",
            instructions="Analyze from FINANCIAL perspective. Be very concise.",
        ),
    }

    topic = "A startup building an AI-powered personal finance assistant"

    print(f"Topic: {topic}\n")
    print("Running analysts in parallel...")

    # Run all in parallel with asyncio.gather
    tasks = [
        agent.run(f"Analyze: {topic}")
        for agent in analysts.values()
    ]

    results = await asyncio.gather(*tasks)

    # Display results
    analyses = {}
    for (name, agent), result in zip(analysts.items(), results):
        analyses[name] = result.text
        print(f"\n--- {name.upper()} ---")
        print(result.text[:200] + "..." if len(result.text) > 200 else result.text)

    # Synthesize (optional step)
    print("\n" + "-" * 40)
    print("SYNTHESIS")
    print("-" * 40)

    synthesizer = client.create_agent(
        name="synthesizer",
        instructions="Create a brief executive summary from multiple perspectives.",
    )

    all_analyses = "\n\n".join(f"{k.upper()}: {v}" for k, v in analyses.items())
    synthesis = await synthesizer.run(f"Synthesize:\n\n{all_analyses}")
    print(synthesis.text)

    return {
        "approach": "manual",
        "topic": topic,
        "analyses": analyses,
        "synthesis": synthesis.text,
    }


def save_results(results: dict) -> Path:
    """Save results to JSON file."""
    output_dir = Path.cwd() / "results" / "13_parallel_agents"
    output_dir.mkdir(parents=True, exist_ok=True)

    results["timestamp"] = datetime.now().isoformat()
    results["example"] = "13_parallel_agents"

    output_file = output_dir / "result.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2, default=str)

    return output_file


async def main() -> None:
    """Run the parallel agents example."""
    print("Example 13: Parallel Multi-Agent Execution")
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
    print("Done!")


if __name__ == "__main__":
    asyncio.run(main())
