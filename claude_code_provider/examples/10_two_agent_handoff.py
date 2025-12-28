#!/usr/bin/env python3
# Copyright (c) 2025. Claude Code Provider for Microsoft Agent Framework.

"""Example 10: Two-Agent Handoff

Demonstrate the manual handoff pattern where one agent's output
becomes the input for another agent.

This is the simplest multi-agent pattern:
    Agent A produces -> Agent B consumes

Concepts demonstrated:
    - Creating specialized agents with different roles
    - Passing one agent's output to another
    - Building pipelines manually

Run:
    python -m claude_code_provider.examples.10_two_agent_handoff
"""

import asyncio

from claude_code_provider import ClaudeCodeClient


async def main() -> None:
    """Run the two-agent handoff example."""
    print("Example 10: Two-Agent Handoff")
    print("=" * 40)

    client = ClaudeCodeClient(model="haiku")

    # Create two specialized agents
    researcher = client.create_agent(
        name="researcher",
        instructions="""You are a research assistant. Your job is to gather key facts
about a topic. Always respond with a bullet-point list of 3-5 key facts.
Format: Start each fact with "- ".""",
    )

    writer = client.create_agent(
        name="writer",
        instructions="""You are a skilled writer. Your job is to take research facts
and turn them into an engaging paragraph. Write in a conversational,
informative style. One paragraph only.""",
    )

    topic = "the history of the Internet"

    # Step 1: Researcher gathers facts
    print(f"\nTopic: {topic}\n")
    print("Step 1: Researcher gathering facts...")

    research_response = await researcher.run(f"Research key facts about {topic}")
    facts = research_response.text

    print(f"\nResearcher's Facts:\n{facts}\n")

    # Step 2: Writer creates content from facts (handoff)
    print("Step 2: Writer creating content from facts...")

    writer_prompt = f"""Based on these research facts, write an engaging paragraph:

{facts}

Write a single paragraph that incorporates these facts naturally."""

    writer_response = await writer.run(writer_prompt)
    content = writer_response.text

    print(f"\nWriter's Output:\n{content}")

    print("\n" + "=" * 40)
    print("Handoff complete!")


if __name__ == "__main__":
    asyncio.run(main())
