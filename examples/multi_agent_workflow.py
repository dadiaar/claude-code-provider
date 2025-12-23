#!/usr/bin/env python3
# Copyright (c) 2025. Claude Code Provider for Microsoft Agent Framework.

"""Multi-Agent Workflow Example

This example demonstrates how to use ClaudeCodeClient to orchestrate
multiple specialized agents working together on a complex task.

The workflow:
1. Analyst Agent - Analyzes the codebase and identifies issues
2. Planner Agent - Creates a plan to address the issues
3. Coder Agent - Implements the fixes
4. Reviewer Agent - Reviews the changes

Run with:
    python examples/multi_agent_workflow.py
"""

import asyncio
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import Any

# Add paths for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "microsoft-agent-framework" / "python" / "packages" / "core"))

from _chat_client import ClaudeCodeClient


@dataclass
class WorkflowResult:
    """Result from a workflow step."""
    agent_name: str
    output: str
    success: bool
    metadata: dict[str, Any] | None = None


class MultiAgentWorkflow:
    """Orchestrates multiple agents for complex tasks."""

    def __init__(self, model: str = "sonnet"):
        """Initialize the workflow with a base model.

        Args:
            model: The Claude model to use (haiku, sonnet, opus).
        """
        self.model = model

        # Create specialized agents
        self.analyst = self._create_analyst()
        self.planner = self._create_planner()
        self.coder = self._create_coder()
        self.reviewer = self._create_reviewer()

    def _create_analyst(self):
        """Create an analyst agent for code analysis."""
        client = ClaudeCodeClient(
            model=self.model,
            tools=["Read", "Glob", "Grep"],  # Read-only tools
        )
        return client.create_agent(
            name="analyst",
            instructions="""You are a code analyst. Your job is to:
1. Examine code structure and patterns
2. Identify potential issues, bugs, or improvements
3. Provide clear, actionable observations

Focus on: security issues, performance problems, code quality, and maintainability.
Be specific and cite file locations.""",
        )

    def _create_planner(self):
        """Create a planner agent for creating action plans."""
        client = ClaudeCodeClient(
            model=self.model,
            tools=["Read"],  # Minimal tools
        )
        return client.create_agent(
            name="planner",
            instructions="""You are a software architect and planner. Your job is to:
1. Review analysis findings
2. Create a prioritized action plan
3. Break down complex changes into manageable steps

Output a numbered list of specific, actionable tasks.
Consider dependencies between tasks.""",
        )

    def _create_coder(self):
        """Create a coder agent for implementing changes."""
        client = ClaudeCodeClient(
            model=self.model,
            tools=["Read", "Edit", "Write", "Bash", "Glob", "Grep"],
        )
        return client.create_agent(
            name="coder",
            instructions="""You are an expert software developer. Your job is to:
1. Implement code changes based on the plan
2. Follow best practices and coding standards
3. Write clean, maintainable code

Always read files before editing. Make minimal, focused changes.
Test your changes when possible.""",
        )

    def _create_reviewer(self):
        """Create a reviewer agent for code review."""
        client = ClaudeCodeClient(
            model=self.model,
            tools=["Read", "Glob", "Grep", "Bash"],
        )
        return client.create_agent(
            name="reviewer",
            instructions="""You are a senior code reviewer. Your job is to:
1. Review implemented changes
2. Check for bugs, security issues, and code quality
3. Verify the changes match the original plan

Provide specific feedback. If issues are found, list them clearly.
If changes look good, confirm approval.""",
        )

    async def run(self, task: str) -> list[WorkflowResult]:
        """Run the multi-agent workflow.

        Args:
            task: The task description (e.g., "Review and improve error handling")

        Returns:
            List of results from each workflow step.
        """
        results: list[WorkflowResult] = []

        print(f"\n{'='*60}")
        print(f"STARTING MULTI-AGENT WORKFLOW")
        print(f"Task: {task}")
        print(f"{'='*60}\n")

        # Step 1: Analysis
        print("Step 1: ANALYSIS")
        print("-" * 40)
        analysis = await self.analyst.run(
            f"Analyze the codebase for: {task}\n\n"
            "Look at the current directory and subdirectories. "
            "Provide specific findings with file locations."
        )
        print(f"Analyst output:\n{analysis.text[:500]}...")
        results.append(WorkflowResult(
            agent_name="analyst",
            output=analysis.text,
            success=True,
        ))

        # Step 2: Planning
        print("\nStep 2: PLANNING")
        print("-" * 40)
        plan = await self.planner.run(
            f"Based on this analysis, create an action plan:\n\n"
            f"{analysis.text}\n\n"
            "Create a numbered list of specific tasks to address the findings."
        )
        print(f"Planner output:\n{plan.text[:500]}...")
        results.append(WorkflowResult(
            agent_name="planner",
            output=plan.text,
            success=True,
        ))

        # Step 3: Implementation (optional - can be skipped for safety)
        print("\nStep 3: IMPLEMENTATION")
        print("-" * 40)
        print("(Skipped in this demo - would implement changes from the plan)")
        results.append(WorkflowResult(
            agent_name="coder",
            output="Implementation skipped in demo mode",
            success=True,
            metadata={"skipped": True},
        ))

        # Step 4: Review
        print("\nStep 4: REVIEW")
        print("-" * 40)
        review = await self.reviewer.run(
            f"Review the proposed plan for this task:\n\n"
            f"Original task: {task}\n\n"
            f"Analysis findings:\n{analysis.text[:1000]}\n\n"
            f"Proposed plan:\n{plan.text}\n\n"
            "Evaluate if the plan adequately addresses the findings."
        )
        print(f"Reviewer output:\n{review.text[:500]}...")
        results.append(WorkflowResult(
            agent_name="reviewer",
            output=review.text,
            success=True,
        ))

        print(f"\n{'='*60}")
        print("WORKFLOW COMPLETE")
        print(f"{'='*60}\n")

        return results


class ParallelAgentPool:
    """Run multiple agents in parallel for independent tasks."""

    def __init__(self, model: str = "haiku"):
        """Initialize the parallel pool.

        Args:
            model: The Claude model to use.
        """
        self.model = model

    async def analyze_files(self, file_patterns: list[str]) -> dict[str, str]:
        """Analyze multiple files in parallel.

        Args:
            file_patterns: List of glob patterns to analyze.

        Returns:
            Dictionary mapping patterns to analysis results.
        """
        async def analyze_one(pattern: str) -> tuple[str, str]:
            client = ClaudeCodeClient(
                model=self.model,
                tools=["Read", "Glob"],
            )
            agent = client.create_agent(
                name=f"analyzer_{pattern}",
                instructions="Briefly describe what you find. Be concise.",
            )
            response = await agent.run(f"Find and describe files matching: {pattern}")
            return pattern, response.text

        # Run all analyses in parallel
        tasks = [analyze_one(pattern) for pattern in file_patterns]
        results = await asyncio.gather(*tasks)

        return dict(results)


async def demo_sequential_workflow():
    """Demo: Sequential multi-agent workflow."""
    print("\n" + "="*60)
    print("DEMO 1: Sequential Multi-Agent Workflow")
    print("="*60)

    workflow = MultiAgentWorkflow(model="haiku")  # Use haiku for speed
    results = await workflow.run("Review error handling patterns")

    print("\nWorkflow Summary:")
    for result in results:
        status = "OK" if result.success else "FAILED"
        print(f"  {result.agent_name}: {status} ({len(result.output)} chars)")


async def demo_parallel_analysis():
    """Demo: Parallel agent execution."""
    print("\n" + "="*60)
    print("DEMO 2: Parallel Agent Execution")
    print("="*60)

    pool = ParallelAgentPool(model="haiku")

    patterns = ["*.py", "*.md", "*.toml"]
    print(f"\nAnalyzing patterns in parallel: {patterns}")

    results = await pool.analyze_files(patterns)

    print("\nResults:")
    for pattern, analysis in results.items():
        print(f"\n{pattern}:")
        print(f"  {analysis[:200]}...")


async def demo_conversation_chain():
    """Demo: Agents building on each other's work."""
    print("\n" + "="*60)
    print("DEMO 3: Conversation Chain")
    print("="*60)

    # Agent 1: Researcher
    researcher_client = ClaudeCodeClient(model="haiku", tools=["Read", "Glob"])
    researcher = researcher_client.create_agent(
        name="researcher",
        instructions="You research codebases. Be brief and factual.",
    )

    # Agent 2: Summarizer
    summarizer_client = ClaudeCodeClient(model="haiku", tools=[])
    summarizer = summarizer_client.create_agent(
        name="summarizer",
        instructions="You summarize technical findings in plain English. Be concise.",
    )

    # Chain: Researcher -> Summarizer
    print("\nResearcher analyzing...")
    research = await researcher.run("What is the main purpose of this project? Look at README.md")
    print(f"Research: {research.text[:200]}...")

    print("\nSummarizer processing...")
    summary = await summarizer.run(
        f"Summarize this in one sentence for a non-technical person:\n\n{research.text}"
    )
    print(f"Summary: {summary.text}")


async def main():
    """Run all demos."""
    print("\n" + "#"*60)
    print("# Claude Code Provider - Multi-Agent Examples")
    print("#"*60)

    try:
        # Run demos (comment out any you don't want)
        await demo_conversation_chain()
        # await demo_parallel_analysis()  # Uncomment to run
        # await demo_sequential_workflow()  # Uncomment to run (slower)

        print("\n" + "="*60)
        print("All demos completed successfully!")
        print("="*60 + "\n")

    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
