# Claude Code Provider for Microsoft Agent Framework

Use Claude Code CLI as a provider for Microsoft Agent Framework (MAF), enabling MAF agents to use your Claude subscription instead of direct API calls.

## Installation

```bash
# From the swarm directory
pip install -e ~/claude-code-provider/

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
- **Autocompact** - Automatic context management to prevent overflow
- **Usage tracking** - Token usage and context monitoring
- **Resilience** - Retry logic, circuit breaker, configurable timeouts
- **MAF compatible** - Full MAF ChatAgent API compatibility

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

## Agent Creation

```python
# Create agent with autocompact (default: enabled)
agent = client.create_agent(
    name="assistant",
    instructions="You are a helpful assistant.",
    autocompact=True,                    # Auto-compact on threshold (default)
    autocompact_threshold=100_000,       # Token threshold for autocompact
    keep_last_n_messages=2,              # Recent messages to preserve
)
```

## MAF-Standard Methods

ClaudeAgent provides full compatibility with MAF's ChatAgent interface:

### Core Methods

```python
# Run a conversation
response = await agent.run("Hello, how are you?")
print(response.text)

# Stream responses
async for update in agent.run_stream("Tell me a story"):
    if update.text:
        print(update.text, end="")
```

### Agent Properties

```python
agent.name           # Agent name
agent.instructions   # System instructions
agent.display_name   # Display name for UI
```

### Serialization

```python
# Serialize to dictionary
data = agent.to_dict()

# Serialize to JSON
json_str = agent.to_json()
```

### Tool Conversion

```python
# Convert agent to a tool for use by other agents
tool = agent.as_tool(
    name="helper_tool",
    description="A helpful assistant tool",
)
```

### MCP Server

```python
# Expose agent as an MCP server
server = agent.as_mcp_server(
    server_name="MyAgent",
    version="1.0.0",
)
```

### Threading

```python
# Get a new conversation thread
thread = agent.get_new_thread()

# Deserialize a saved thread
thread = agent.deserialize_thread(saved_thread_data)

# Use thread in conversation
response = await agent.run("Hello", thread=thread)
```

## Extension Methods (Claude Code Provider Specific)

These methods extend the standard MAF API with Claude Code specific functionality:

### Context Management

```python
# Manually compact conversation to reduce context
result = await agent.compact()
print(f"Compacted: {result.original_tokens_estimate} -> {result.summary_tokens_estimate} tokens")

# Get context usage information
ctx = agent.get_context_info()
print(f"Tokens: {ctx.estimated_tokens}/{ctx.context_limit} ({ctx.usage_percent:.1f}%)")
print(f"Messages: {ctx.messages_count}")
print(f"Has summary: {ctx.has_summary}")
print(f"Autocompact enabled: {ctx.autocompact_enabled}")
```

### Usage Tracking

```python
# Get accumulated usage statistics
usage = agent.get_usage()
print(f"Total requests: {usage.total_requests}")
print(f"Input tokens: {usage.total_input_tokens}")
print(f"Output tokens: {usage.total_output_tokens}")
print(f"Compactions: {usage.compactions}")
print(f"Tokens saved: {usage.tokens_saved_by_compact}")
```

### State Management

```python
# Get conversation history
messages = agent.get_messages()

# Estimate current token count
tokens = agent.get_token_estimate()

# Reset conversation (preserves usage stats)
agent.reset()

# Reset usage statistics
agent.reset_usage()
```

## CLI Limitations

Some MAF parameters are passed through but may not be fully supported by Claude Code CLI:

| Parameter | Status |
|-----------|--------|
| `model_id` | Supported (haiku, sonnet, opus) |
| `max_tokens` | Supported |
| `temperature` | Passed through, may be ignored |
| `top_p` | Passed through, may be ignored |
| `frequency_penalty` | Passed through, may be ignored |
| `presence_penalty` | Passed through, may be ignored |
| `stop` | Passed through, may be ignored |
| `seed` | Passed through, may be ignored |
| `logit_bias` | Not supported |

## Examples

See the `demos/` directory for progressive examples:

1. `01_hello_world.py` - Basic usage
2. `02_streaming.py` - Streaming responses
3. `03_conversation.py` - Conversation continuity
4. `04_tools.py` - Using Claude Code tools
5. `05_agent.py` - Agent with instructions
6. `06_multi_agent.py` - Multiple agents

## Testing

```bash
# Run all tests
python -m pytest tests/test_claude_code_provider.py -v

# Run only unit tests (no CLI required)
python -m pytest tests/test_claude_code_provider.py -v -k "not Integration"
```

## API Reference

### ClaudeCodeClient

The main client class for Claude Code CLI integration.

```python
ClaudeCodeClient(
    model: str | None = None,           # Default model
    cli_path: str = "claude",           # Path to CLI
    max_turns: int | None = None,       # Max agentic turns
    tools: list[str] | None = None,     # Enabled tools
    allowed_tools: list[str] | None = None,    # Auto-approved tools
    disallowed_tools: list[str] | None = None, # Blocked tools
    working_directory: str | None = None,       # Working directory
    timeout: float = 300.0,             # Timeout in seconds
    retry_config: RetryConfig | None = None,   # Custom retry config
    enable_retries: bool = False,       # Enable retries
    enable_circuit_breaker: bool = True,       # Enable circuit breaker
)
```

### ClaudeAgent

Enhanced agent wrapper with compact and tracking functionality.

### Exceptions

```python
from claude_code_provider import (
    ClaudeCodeException,           # Base exception
    ClaudeCodeCLINotFoundError,    # CLI not found
    ClaudeCodeExecutionError,      # CLI execution failed
    ClaudeCodeParseError,          # Response parse error
    ClaudeCodeTimeoutError,        # Timeout exceeded
    ClaudeCodeContentFilterError,  # Content filtered
    ClaudeCodeSessionError,        # Session error
)
```

### Data Classes

```python
from claude_code_provider import (
    CompactResult,    # Result of compact operation
    UsageStats,       # Accumulated usage statistics
    ContextInfo,      # Context usage information
    RetryConfig,      # Retry configuration
    CircuitBreaker,   # Circuit breaker for resilience
)
```

## License

Proprietary - All Rights Reserved. See LICENSE file.
