#!/usr/bin/env python3
# Copyright (c) 2025. Claude Code Provider for Microsoft Agent Framework.

"""Example 14: Retry and Resilience

Demonstrate retry logic and circuit breaker patterns for handling
transient failures gracefully.

Resilience patterns:
    - RetryConfig: Configures exponential backoff with jitter
    - CircuitBreaker: Prevents cascade failures when service is down

Concepts demonstrated:
    - Configuring retry behavior
    - Exponential backoff with jitter
    - Circuit breaker states (CLOSED, OPEN, HALF_OPEN)
    - Handling transient errors

Run:
    python -m claude_code_provider.examples.14_retry_resilience

Output:
    Results saved to ./results/14_retry_resilience/result.json
"""

import asyncio
import json
from datetime import datetime
from pathlib import Path

from claude_code_provider import (
    ClaudeCodeClient,
    RetryConfig,
    CircuitBreaker,
)


async def demo_retry_config() -> dict:
    """Demonstrate RetryConfig usage."""
    print("\n--- RetryConfig Demo ---")

    # Create a retry configuration
    config = RetryConfig(
        max_retries=3,           # Retry up to 3 times (4 total attempts)
        base_delay=1.0,          # Start with 1 second delay
        max_delay=30.0,          # Never delay more than 30 seconds
        exponential_base=2.0,    # Double delay each time (1s, 2s, 4s)
        jitter=True,             # Add randomness to prevent thundering herd
    )

    print(f"Configuration:")
    print(f"  max_retries: {config.max_retries}")
    print(f"  base_delay: {config.base_delay}s")
    print(f"  exponential_base: {config.exponential_base}")
    print(f"  jitter: {config.jitter}")

    # Show computed delays
    delays = []
    print(f"\nComputed delays per attempt:")
    for attempt in range(4):
        delay = config.get_delay(attempt)
        delays.append(delay)
        print(f"  Attempt {attempt + 1}: ~{delay:.2f}s")

    return {
        "config": {
            "max_retries": config.max_retries,
            "base_delay": config.base_delay,
            "exponential_base": config.exponential_base,
            "jitter": config.jitter,
        },
        "computed_delays": delays,
    }


async def demo_circuit_breaker() -> dict:
    """Demonstrate CircuitBreaker usage."""
    print("\n--- CircuitBreaker Demo ---")

    # Create a circuit breaker
    breaker = CircuitBreaker(
        failure_threshold=3,     # Open after 3 failures
        recovery_timeout=10.0,   # Wait 10s before trying again
        success_threshold=2,     # Need 2 successes to fully close
    )

    print(f"Configuration:")
    print(f"  failure_threshold: {breaker.failure_threshold}")
    print(f"  recovery_timeout: {breaker.recovery_timeout}s")
    print(f"  success_threshold: {breaker.success_threshold}")

    initial_state = breaker.state
    print(f"\nInitial state: {initial_state}")

    # Simulate failures
    states = []
    print("\nSimulating 3 failures...")
    for i in range(3):
        await breaker.record_failure()
        can_exec = await breaker.can_execute()
        states.append({"failure": i + 1, "state": breaker.state, "can_execute": can_exec})
        print(f"  After failure {i + 1}: state={breaker.state}, can_execute={can_exec}")

    # Circuit should be open now
    print(f"\nCircuit is now OPEN - requests will be rejected")
    print(f"  is_open: {breaker.is_open}")

    return {
        "config": {
            "failure_threshold": breaker.failure_threshold,
            "recovery_timeout": breaker.recovery_timeout,
            "success_threshold": breaker.success_threshold,
        },
        "initial_state": initial_state,
        "states_after_failures": states,
        "final_is_open": breaker.is_open,
    }


async def demo_real_usage() -> dict:
    """Show retry with real API call."""
    print("\n--- Real Usage Demo ---")

    client = ClaudeCodeClient(model="haiku")
    agent = client.create_agent(
        name="assistant",
        instructions="Be brief.",
    )

    # For this demo, we'll just make a normal call
    # In real usage, transient errors would trigger retries
    print("Making API call (no errors expected in this demo)...")

    response_text = ""
    success = False
    error_msg = None

    try:
        response = await agent.run("Say 'Hello!'")
        response_text = response.text
        success = True
        print(f"Response: {response_text}")
        print(f"Total attempts: 1 (no retries needed)")
    except Exception as e:
        error_msg = str(e)
        print(f"Failed after retries: {e}")

    return {
        "success": success,
        "response": response_text,
        "error": error_msg,
    }


def save_results(results: dict) -> Path:
    """Save results to JSON file."""
    output_dir = Path.cwd() / "results" / "14_retry_resilience"
    output_dir.mkdir(parents=True, exist_ok=True)

    results["timestamp"] = datetime.now().isoformat()
    results["example"] = "14_retry_resilience"

    output_file = output_dir / "result.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2, default=str)

    return output_file


async def main() -> None:
    """Run the retry and resilience example."""
    print("Example 14: Retry and Resilience")
    print("=" * 40)

    retry_results = await demo_retry_config()
    circuit_results = await demo_circuit_breaker()
    usage_results = await demo_real_usage()

    # Combine results
    results = {
        "retry_config": retry_results,
        "circuit_breaker": circuit_results,
        "real_usage": usage_results,
    }

    # Save results
    output_file = save_results(results)
    print(f"\nResults saved to: {output_file}")
    print("Done!")


if __name__ == "__main__":
    asyncio.run(main())
