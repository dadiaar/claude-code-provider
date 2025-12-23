# Claude Code Provider for Microsoft Agent Framework

Use Claude Code CLI as a provider for Microsoft Agent Framework (MAF), enabling MAF agents to use your Claude subscription instead of direct API calls.

## Installation

```bash
# From the swarm directory
pip install -e claude-code-provider/

# Or install dependencies manually
pip install pydantic>=2.0
```

**Prerequisites:**
- Python 3.10+
- Claude Code CLI installed and authenticated (`claude` command available)
- Microsoft Agent Framework core package

## Quick Start

```python
import asyncio
from claude_code_provider import ClaudeCodeClient

async def main():
    # Create client
    client = ClaudeCodeClient(model="sonnet")

    # Simple query
    response = await client.get_response("What is 2+2?")
    print(response.messages[0].text)

    # Create agent with instructions
    agent = client.create_agent(
        name="coder",
        instructions="You are a helpful coding assistant.",
    )
    response = await agent.run("Explain Python decorators briefly.")
    print(response.text)

asyncio.run(main())
```

## Features

- **Full Claude Code capabilities** - Bash, Read, Edit, Glob, Grep, WebFetch, etc.
- **Streaming** - Real-time response streaming
- **Conversation memory** - Session-based continuity
- **Resilience** - Retry logic, circuit breaker, configurable timeouts
- **MAF compatible** - Works with MAF's agent patterns

## Configuration

```python
from claude_code_provider import ClaudeCodeClient, RetryConfig

client = ClaudeCodeClient(
    model="sonnet",              # Model: haiku, sonnet, opus
    tools=["Read", "Bash"],      # Limit available tools
    timeout=120.0,               # Timeout in seconds
    enable_retries=True,         # Auto-retry on failures
    enable_circuit_breaker=True, # Prevent cascade failures
)
```

## License

Proprietary - All Rights Reserved. See LICENSE file.
