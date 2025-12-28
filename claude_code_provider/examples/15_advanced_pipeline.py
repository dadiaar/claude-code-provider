#!/usr/bin/env python3
# Copyright (c) 2025. Claude Code Provider for Microsoft Agent Framework.

"""Example 15: Advanced Pipeline with Pool, Streaming, and Progress Reporting

This is the capstone example demonstrating production-grade patterns:
    - Connection pool for rate limit management
    - Streaming mode for faster hung detection
    - Haiku as external progress reporter
    - Cost tracking across all agents
    - Multi-model orchestration (haiku, sonnet)

Architecture:
    +-----------------------------------------------------------+
    |  PHASE 1: Parallel Analysis (Haiku x3)                    |
    |  - Technical, Market, Financial analysts run in parallel  |
    |  - Connection pool limits concurrency to avoid rate limits|
    |  - Haiku reports progress after each completes           |
    +-----------------------------------------------------------+
    |  PHASE 2: Synthesis (Sonnet)                              |
    |  - Combines all analyses into unified report             |
    +-----------------------------------------------------------+

Concepts demonstrated:
    - Connection pool with semaphore
    - Streaming responses
    - Progress reporting with separate Haiku agent
    - Cost tracking
    - Multi-phase pipeline

Run:
    python -m claude_code_provider.examples.15_advanced_pipeline

Output:
    Results saved to ./results/15_advanced_pipeline/
"""

import asyncio
import json
import subprocess
import time
from datetime import datetime
from pathlib import Path

from claude_code_provider import ClaudeCodeClient, CostTracker


# =============================================================================
# CONNECTION POOL - Limits concurrent API calls
# =============================================================================

class ConnectionPool:
    """Manages concurrent connections with rate limiting."""

    def __init__(self, max_concurrent: int = 3) -> None:
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._active = 0

    async def acquire(self) -> None:
        await self._semaphore.acquire()
        self._active += 1

    async def release(self) -> None:
        self._active -= 1
        self._semaphore.release()

    @property
    def active_count(self) -> int:
        return self._active


# =============================================================================
# PROGRESS REPORTER - Haiku reports on completion
# =============================================================================

class ProgressReporter:
    """Reports progress using Haiku (claude -p for one-shot response)."""

    def __init__(self) -> None:
        self._start_time = time.time()
        self._report_count = 0

    async def report(
        self,
        agent_name: str,
        phase: str,
        completed: int,
        total: int,
    ) -> None:
        """Report progress using Haiku."""
        self._report_count += 1
        elapsed = time.time() - self._start_time

        prompt = f"""You are a progress reporter. Create a brief status (1 sentence).

Agent: {agent_name}
Phase: {phase}
Progress: {completed}/{total} complete
Elapsed: {elapsed/60:.1f} minutes

Be factual. No follow-up questions."""

        try:
            result = subprocess.run(
                ["claude", "-p", prompt, "--model", "haiku"],
                capture_output=True,
                text=True,
                timeout=30,
            )
            status = result.stdout.strip() if result.returncode == 0 else f"Progress: {completed}/{total}"

            print(f"\n[PROGRESS] {status}")
        except Exception:
            print(f"\n[PROGRESS] {agent_name} complete ({completed}/{total})")


# =============================================================================
# STREAMING WRAPPER - Faster hung detection
# =============================================================================

async def run_with_streaming(agent, prompt: str) -> str:
    """Run agent with streaming for faster hung detection.

    Streaming has a 60s per-chunk timeout vs 600s+ for non-streaming.
    """
    chunks = []
    async for chunk in agent.run_stream(prompt):
        if hasattr(chunk, 'text') and chunk.text:
            chunks.append(chunk.text)
    return "".join(chunks)


# =============================================================================
# MAIN PIPELINE
# =============================================================================

async def main() -> None:
    """Run the advanced pipeline example."""
    print("Example 15: Advanced Pipeline")
    print("=" * 60)
    print("""
Features demonstrated:
  - Connection pool (max 3 concurrent)
  - Streaming mode (faster hung detection)
  - Haiku progress reporter
  - Cost tracking
  - Multi-model (haiku for analysis, sonnet for synthesis)
""")

    # Initialize components
    haiku = ClaudeCodeClient(model="haiku")
    sonnet = ClaudeCodeClient(model="sonnet")
    tracker = CostTracker()
    pool = ConnectionPool(max_concurrent=3)
    reporter = ProgressReporter()

    topic = "A startup building an AI-powered code review tool for enterprises"
    total_tasks = 4  # 3 analysts + 1 synthesizer

    print(f"Topic: {topic}\n")

    # Create analysts
    analysts = {
        "technical": haiku.create_agent(
            name="technical_analyst",
            instructions="""Analyze from TECHNICAL perspective:
- Technology stack requirements
- Implementation challenges
- Scalability considerations
Be concise (3 bullets).""",
        ),
        "market": haiku.create_agent(
            name="market_analyst",
            instructions="""Analyze from MARKET perspective:
- Target market size
- Competition
- Customer needs
Be concise (3 bullets).""",
        ),
        "financial": haiku.create_agent(
            name="financial_analyst",
            instructions="""Analyze from FINANCIAL perspective:
- Revenue model
- Cost structure
- Funding requirements
Be concise (3 bullets).""",
        ),
    }

    # ==========================================================================
    # PHASE 1: Parallel Analysis with Pool
    # ==========================================================================
    print("-" * 60)
    print("PHASE 1: Parallel Analysis (3 Haiku agents)")
    print("-" * 60)

    completed = 0
    analyses = {}

    async def analyze_with_pool(name: str, agent) -> tuple[str, str]:
        """Run analysis with connection pool and streaming."""
        nonlocal completed

        await pool.acquire()
        try:
            print(f"  [{name}] Starting analysis...")
            result = await run_with_streaming(agent, f"Analyze: {topic}")

            # Record cost (estimate since streaming doesn't return usage)
            tracker.record_request("haiku", input_tokens=500, output_tokens=200)

            completed += 1
            await reporter.report(name, "Analysis", completed, total_tasks)

            return name, result
        finally:
            await pool.release()

    # Run all analysts in parallel (pool limits to 3 concurrent)
    tasks = [analyze_with_pool(name, agent) for name, agent in analysts.items()]
    results = await asyncio.gather(*tasks)

    for name, result in results:
        analyses[name] = result
        print(f"\n  [{name.upper()}]")
        print(f"  {result[:150]}...")

    # ==========================================================================
    # PHASE 2: Synthesis (Sonnet)
    # ==========================================================================
    print("\n" + "-" * 60)
    print("PHASE 2: Synthesis (Sonnet)")
    print("-" * 60)

    synthesizer = sonnet.create_agent(
        name="synthesizer",
        instructions="""Create an executive summary:
1. Key findings (3 bullets)
2. Overall recommendation
3. Critical next step
Be concise.""",
    )

    all_analyses = "\n\n".join(f"{k.upper()}:\n{v}" for k, v in analyses.items())

    print("  [synthesizer] Creating executive summary...")
    synthesis = await run_with_streaming(
        synthesizer,
        f"""Synthesize these analyses:

{all_analyses}

Create a unified executive summary.""",
    )

    tracker.record_request("sonnet", input_tokens=1500, output_tokens=500)
    completed += 1
    await reporter.report("synthesizer", "Synthesis", completed, total_tasks)

    # ==========================================================================
    # Final Output
    # ==========================================================================
    print("\n" + "=" * 60)
    print("EXECUTIVE SUMMARY")
    print("=" * 60)
    print(synthesis)

    # Cost summary
    summary = tracker.get_summary()
    print("\n" + "=" * 60)
    print("COST SUMMARY")
    print("=" * 60)
    print(f"Total requests: {summary.total_requests}")
    print(f"Total tokens: {summary.total_input_tokens + summary.total_output_tokens:,}")
    print(f"Estimated cost: ${summary.total_cost:.4f}")
    print(f"Progress reports: {reporter._report_count}")

    # Save results
    output_dir = Path.cwd() / "results" / "15_advanced_pipeline"
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {
        "example": "15_advanced_pipeline",
        "timestamp": datetime.now().isoformat(),
        "topic": topic,
        "analyses": analyses,
        "synthesis": synthesis,
        "cost_summary": {
            "total_requests": summary.total_requests,
            "total_tokens": summary.total_input_tokens + summary.total_output_tokens,
            "total_cost": summary.total_cost,
            "progress_reports": reporter._report_count,
        },
    }

    result_file = output_dir / "result.json"
    with open(result_file, "w") as f:
        json.dump(results, f, indent=2, default=str)

    # Also save synthesis as markdown
    synthesis_file = output_dir / "executive_summary.md"
    synthesis_file.write_text(f"# Executive Summary\n\n{synthesis}")

    print(f"\nResults saved to: {output_dir}")
    print("Done!")


if __name__ == "__main__":
    asyncio.run(main())
