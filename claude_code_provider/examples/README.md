# Claude Code Provider Examples

This folder contains examples demonstrating the features of `claude_code_provider`,
organized from simple to advanced usage patterns.

## Running Examples

Each example can be run as a Python module:

```bash
# From the package root
python -m claude_code_provider.examples.01_hello_world

# Or from anywhere after pip install
python -m claude_code_provider.examples.03_streaming
```

## Examples Overview

### Getting Started (Difficulty: Easy)

| Example | Description | Key Concepts |
|---------|-------------|--------------|
| `01_hello_world.py` | The simplest possible usage | ClaudeCodeClient, create_agent, run() |
| `02_model_selection.py` | Using different models | haiku/sonnet/opus, ClaudeModel enum |
| `03_streaming.py` | Real-time streaming responses | run_stream(), chunk processing |
| `04_conversation.py` | Multi-turn conversations | Session continuity, context retention |

### Observability (Difficulty: Medium)

| Example | Description | Key Concepts |
|---------|-------------|--------------|
| `05_logging_debug.py` | Debug logging and structured logs | setup_logging, DebugLogger, log capture |
| `06_cost_tracking.py` | Monitor token usage and costs | CostTracker, budgets, summaries |
| `07_session_management.py` | Session persistence | SessionManager, track/list/clear sessions |

### Smart Model Selection (Difficulty: Medium)

| Example | Description | Key Concepts |
|---------|-------------|--------------|
| `08_model_routing_basic.py` | Simple model routing | ModelRouter, SimpleRouter |
| `09_model_routing_advanced.py` | Advanced routing strategies | ComplexityRouter, TaskTypeRouter |

### Multi-Agent Patterns (Difficulty: Medium-Hard)

| Example | Description | Key Concepts |
|---------|-------------|--------------|
| `10_two_agent_handoff.py` | Basic handoff pattern | Two agents, manual handoff |
| `11_batch_processing.py` | Concurrent batch processing | BatchProcessor, map-reduce pattern |
| `12_pipeline_sequential.py` | Sequential agent pipeline | SequentialOrchestrator, multi-stage |
| `13_parallel_agents.py` | Parallel agent execution | ConcurrentOrchestrator, fan-out/fan-in |

### Advanced (Difficulty: Hard)

| Example | Description | Key Concepts |
|---------|-------------|--------------|
| `14_retry_resilience.py` | Retry logic and circuit breaker | RetryConfig, CircuitBreaker |
| `15_advanced_pipeline.py` | Production-grade pipeline | Pool, streaming, progress reporter, multi-model |
| `16_codebase_reviewer.py` | Full codebase review system | 3So 3i teams, verification team, AgentLogger |
| `17_opus_codebase_reviewer.py` | Premium Opus version of codebase reviewer | Opus models, higher quality analysis |

## Learning Path

**Recommended order for new users:**

1. Start with `01_hello_world.py` - understand the basic pattern
2. Try `03_streaming.py` - see real-time responses
3. Explore `06_cost_tracking.py` - understand token usage
4. Learn `10_two_agent_handoff.py` - basic multi-agent pattern
5. Progress to `13_parallel_agents.py` - concurrent execution
6. Study `15_advanced_pipeline.py` - production patterns
7. Master `16_codebase_reviewer.py` - full multi-agent architecture

## Prerequisites

- Python 3.10+
- Claude Code CLI installed and authenticated (`claude` command available)
- Package installed: `pip install claude-code-provider`

## Common Issues

### "claude" command not found
Install Claude Code CLI: https://github.com/anthropics/claude-code

### Rate limiting errors
The examples use connection pools and retries to handle rate limits.
See `15_advanced_pipeline.py` for production-grade rate limit handling.

### Import errors
Make sure you're running from the package root or have installed via pip.
