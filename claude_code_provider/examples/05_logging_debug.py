#!/usr/bin/env python3
# Copyright (c) 2025. Claude Code Provider for Microsoft Agent Framework.

"""Example 05: Logging and Debugging

Use structured logging to troubleshoot and monitor operations.
The logging system supports both human-readable and JSON output.

Concepts demonstrated:
    - Setting up the logger with setup_logging()
    - Different log levels (DEBUG, INFO, WARNING, ERROR)
    - Structured logging with extra fields
    - Log capture for testing/analysis
    - Request/response debugging

Run:
    python -m claude_code_provider.examples.05_logging_debug
"""

import asyncio

from claude_code_provider import ClaudeCodeClient, setup_logging


async def main() -> None:
    """Run the logging example."""
    print("Example 05: Logging and Debugging")
    print("=" * 40)

    # Setup logging with color output
    # Options: level="DEBUG", "INFO", "WARNING", "ERROR"
    #          json_output=True for JSON format
    logger = setup_logging(level="DEBUG", json_output=False)

    # Log some operations with extra context
    logger.info("Starting demo", demo="logging")
    logger.debug("Debug information", extra_data={"key": "value"})

    # Create client and make a request
    client = ClaudeCodeClient(model="haiku")

    logger.info("Creating request", model="haiku")
    logger.debug_request("What is logging?", model="haiku")

    response = await client.get_response("What is logging in software? One sentence.")

    # Log the response details
    logger.debug_response({
        "result": response.messages[0].text,
        "usage": {
            "input_tokens": response.usage_details.input_token_count if response.usage_details else 0,
            "output_tokens": response.usage_details.output_token_count if response.usage_details else 0,
        }
    })

    logger.info("Request completed successfully")

    print(f"\nResponse: {response.messages[0].text}")

    # Demonstrate log capture (useful for testing)
    print("\n" + "-" * 40)
    print("Log Capture Demo")
    print("-" * 40)

    # Start capturing logs
    logger.start_capture()

    logger.info("Captured log 1")
    logger.debug("Captured log 2")
    logger.warning("Captured warning")

    # Stop capture and get entries
    entries = logger.stop_capture()

    print(f"\nCaptured {len(entries)} log entries:")
    for entry in entries:
        print(f"  [{entry.level.value}] {entry.message}")

    print("\nDone!")


if __name__ == "__main__":
    asyncio.run(main())
