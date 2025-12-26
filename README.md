# Claude Code Provider for Microsoft Agent Framework

Use Claude Code CLI as a provider for Microsoft Agent Framework (MAF), enabling MAF agents to use your Claude subscription instead of direct API calls.

## Installation

```bash
pip install -e ~/claude-code-provider/
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
    client = ClaudeCodeClient(model="sonnet")
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
- **MCP Server Connections** - Connect external tools via Model Context Protocol
- **Cost Tracking** - Monitor token usage and estimated costs
- **Multi-Model Routing** - Route tasks to optimal models based on complexity
- **Session Management** - Track, export, and cleanup sessions
- **Batch Processing** - Process multiple prompts efficiently
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
agent = client.create_agent(
    name="assistant",
    instructions="You are a helpful assistant.",
    autocompact=True,                    # Auto-compact on threshold (default)
    autocompact_threshold=100_000,       # Token threshold for autocompact
    keep_last_n_messages=2,              # Recent messages to preserve
)
```

---

## MCP Server Connections

Connect external tools via Model Context Protocol.

```python
from claude_code_provider import MCPServer, MCPTransport, MCPManager

# Create MCP server configuration
server = MCPServer(
    name="my-tool",
    command_or_url="npx",
    transport=MCPTransport.STDIO,
    args=["-y", "my-mcp-server"],
    env={"API_KEY": "secret"},
)

# Use MCP Manager
manager = MCPManager()
manager.add_server(server)

# Get CLI args to pass to client
cli_args = manager.get_cli_args()

# Or add server directly to Claude Code
await manager.add_server_to_claude(server)

# List configured servers
servers = await manager.list_configured_servers()
```

**Transport Types:**
- `MCPTransport.STDIO` - Local command execution
- `MCPTransport.HTTP` - HTTP-based server
- `MCPTransport.SSE` - Server-Sent Events

---

## Cost Tracking

Monitor token usage and estimated costs.

```python
from claude_code_provider import CostTracker

tracker = CostTracker()

# Record a request
cost = tracker.record_request(
    model="sonnet",
    input_tokens=1000,
    output_tokens=500,
)
print(f"Request cost: ${cost.total_cost:.4f}")

# Get summary
summary = tracker.get_summary()
print(f"Total spent: ${summary.total_cost:.4f}")
print(f"Total requests: {summary.total_requests}")
print(f"By model: {summary.by_model}")

# Set budget with alert
def budget_alert(summary):
    print(f"Over budget! Total: ${summary.total_cost:.2f}")

tracker.set_budget(max_cost=10.0, alert_callback=budget_alert)

# Check budget status
if tracker.is_over_budget():
    print("Budget exceeded!")
print(f"Remaining: ${tracker.get_remaining_budget():.2f}")
```

### Token Breakdown

Claude Code CLI reports input tokens in three categories for cache-aware billing:

```python
from claude_code_provider._cli_executor import CLIResult

# After a CLI execution, access detailed token breakdown:
result: CLIResult = ...  # From executor.execute()

# Total input tokens (sum of all three)
print(f"Total input: {result.input_tokens}")

# Individual components
print(f"Raw input (new tokens): {result.raw_input_tokens}")
print(f"Cache creation: {result.cache_creation_tokens}")
print(f"Cache read: {result.cache_read_tokens}")
print(f"Output: {result.output_tokens}")

# Or get all at once
breakdown = result.token_breakdown
# {
#   "input_tokens": 18305,       # Total
#   "raw_input_tokens": 5,       # New tokens
#   "cache_creation_tokens": 4000,
#   "cache_read_tokens": 14300,
#   "output_tokens": 500,
# }
```

This breakdown is also preserved in `ChatResponse.usage_details.additional_counts` for MAF integration.

---

## Multi-Model Routing

Route tasks to optimal models based on complexity, cost, or custom rules.

```python
from claude_code_provider import (
    ModelRouter, ComplexityRouter, CostOptimizedRouter,
    TaskTypeRouter, CustomRouter, RoutingContext
)

# Complexity-based routing (default)
router = ModelRouter()
router.set_strategy(ComplexityRouter())

model = router.route("Simple question?")  # -> haiku
model = router.route("Analyze this complex algorithm step by step")  # -> sonnet/opus

# Cost-optimized routing
router.set_strategy(CostOptimizedRouter(budget_remaining=5.0))

# Task-type routing
router.set_strategy(TaskTypeRouter())
model = router.route("Summarize this text")  # -> haiku
model = router.route("Design a new architecture")  # -> opus

# Custom routing
def my_router(context: RoutingContext) -> str:
    if "urgent" in context.prompt.lower():
        return "opus"
    return "sonnet"

router.set_strategy(CustomRouter(my_router))
```

---

## Logging & Debugging

Structured logging for troubleshooting.

```python
from claude_code_provider import setup_logging, DebugLogger

# Quick setup
logger = setup_logging(level="DEBUG", json_output=False)

# Or detailed setup
logger = DebugLogger.setup(
    level="DEBUG",
    json_output=True,  # Structured JSON logs
    include_time=True,
)

# Log with context
logger.info("Request started", model="sonnet", tokens=500)
logger.debug_cli_call(["claude", "-p", "hello"], {"timeout": 30})
logger.debug_response({"result": "Hello!", "usage": {"input_tokens": 10}})

# Capture logs for testing
logger.start_capture()
# ... operations ...
entries = logger.stop_capture()
```

---

## Session Management

Track, export, and cleanup conversation sessions.

```python
from claude_code_provider import SessionManager

manager = SessionManager()

# Track a session
info = manager.track_session("session-123", model="sonnet")
print(f"Session: {info.session_id}, Messages: {info.message_count}")

# List sessions
sessions = manager.list_sessions(sort_by="last_used")
for s in sessions:
    print(f"{s.session_id}: {s.message_count} messages")

# Get recent sessions
recent = manager.get_recent_sessions(limit=5)

# Get statistics
stats = manager.get_stats()
print(f"Total sessions: {stats['total_sessions']}")
print(f"Models used: {stats['models_used']}")

# Export a session
export = await manager.export_session("session-123")
export.save("session_backup.json")

# Cleanup old sessions
deleted = await manager.cleanup_old_sessions(days=30)
print(f"Cleaned up {len(deleted)} old sessions")

# Delete specific session
manager.delete_session("session-123")
```

---

## Batch Processing

Process multiple prompts efficiently with concurrency control.

```python
from claude_code_provider import ClaudeCodeClient, BatchProcessor

client = ClaudeCodeClient(model="haiku")
processor = BatchProcessor(client, default_concurrency=3)

# Process multiple prompts
prompts = [
    "Summarize quantum computing",
    "Explain machine learning",
    "What is blockchain?",
]

result = await processor.process_batch(
    prompts=prompts,
    system_prompt="Be concise.",
    concurrency=5,
    progress_callback=lambda done, total: print(f"{done}/{total}"),
)

print(f"Success rate: {result.success_rate:.1%}")
for item in result.items:
    if item.status == "completed":
        print(f"{item.id}: {item.result[:50]}...")

# Process with template
files = ["file1.py", "file2.py", "file3.py"]
result = await processor.process_with_transform(
    items=files,
    prompt_template="Analyze the code in {filename}",
    transform=lambda f: {"filename": f},
)

# Map-reduce pattern
summaries = ["Summary of chapter 1", "Summary of chapter 2", "Summary of chapter 3"]
map_result, final = await processor.map_reduce(
    prompts=[f"Expand on: {s}" for s in summaries],
    reduce_prompt="Combine these expansions into a cohesive narrative:\n{results}",
)
print(f"Final result: {final}")
```

---

## MAF-Standard Methods

ClaudeAgent provides full compatibility with MAF's ChatAgent interface:

```python
# Run a conversation
response = await agent.run("Hello, how are you?")

# Stream responses
async for update in agent.run_stream("Tell me a story"):
    if update.text:
        print(update.text, end="")

# Agent properties
agent.name           # Agent name
agent.instructions   # System instructions
agent.display_name   # Display name for UI

# Serialization
data = agent.to_dict()
json_str = agent.to_json()

# Convert to tool
tool = agent.as_tool(name="helper_tool")

# Expose as MCP server
server = agent.as_mcp_server(server_name="MyAgent")

# Threading
thread = agent.get_new_thread()
response = await agent.run("Hello", thread=thread)
```

---

## CLI Limitations

Some MAF parameters may not be fully supported by Claude Code CLI:

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

---

## Testing

```bash
# Run all tests
python -m pytest tests/test_claude_code_provider.py -v

# Run only unit tests (no CLI required)
python -m pytest tests/test_claude_code_provider.py -v -k "not Integration"
```

---

## API Reference

### Core Classes

```python
from claude_code_provider import (
    # Core
    ClaudeCodeClient,      # Main client
    ClaudeCodeSettings,    # Configuration
    ClaudeAgent,           # Enhanced agent wrapper

    # MCP
    MCPServer,             # MCP server config
    MCPTransport,          # Transport types
    MCPManager,            # MCP management

    # Cost Tracking
    CostTracker,           # Cost monitoring
    RequestCost,           # Single request cost
    CostSummary,           # Aggregated costs

    # Model Routing
    ModelRouter,           # Main router
    ComplexityRouter,      # Complexity-based
    CostOptimizedRouter,   # Cost-optimized
    TaskTypeRouter,        # Task-type based
    CustomRouter,          # Custom function

    # Logging
    DebugLogger,           # Enhanced logger
    setup_logging,         # Quick setup

    # Sessions
    SessionManager,        # Session tracking
    SessionInfo,           # Session metadata
    SessionExport,         # Exported session

    # Batch
    BatchProcessor,        # Batch operations
    BatchResult,           # Batch results
    BatchItem,             # Single item
    BatchStatus,           # Item status

    # Resilience
    RetryConfig,           # Retry settings
    CircuitBreaker,        # Circuit breaker
)
```

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

---

## License

Proprietary - All Rights Reserved. See LICENSE file.
