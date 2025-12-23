#!/usr/bin/env python3
# Copyright (c) 2025. Claude Code Provider for Microsoft Agent Framework.

"""Comprehensive test suite for Claude Code Provider.

Run with:
    python -m pytest tests/test_claude_code_provider.py -v

Or run directly:
    python tests/test_claude_code_provider.py
"""

import asyncio
import json
import sys
from pathlib import Path

# Add parent directory to allow importing the package
sys.path.insert(0, str(Path(__file__).parent.parent))


# =============================================================================
# UNIT TESTS (No CLI calls)
# =============================================================================

class TestSettings:
    """Test ClaudeCodeSettings."""

    def test_default_settings(self):
        """Test default settings initialization."""
        from claude_code_provider._settings import ClaudeCodeSettings

        settings = ClaudeCodeSettings()
        assert settings.cli_path == "claude"
        assert settings.model is None
        assert settings.default_max_turns is None

    def test_custom_settings(self):
        """Test custom settings."""
        from claude_code_provider._settings import ClaudeCodeSettings

        settings = ClaudeCodeSettings(
            model="opus",
            default_max_turns=10,
            tools=["Bash", "Read"],
        )
        assert settings.model == "opus"
        assert settings.default_max_turns == 10
        assert settings.tools == ["Bash", "Read"]

    def test_to_cli_args(self):
        """Test CLI args generation."""
        from claude_code_provider._settings import ClaudeCodeSettings

        settings = ClaudeCodeSettings(
            model="sonnet",
            default_max_turns=5,
            tools=["Bash", "Read"],
        )
        args = settings.to_cli_args()

        assert "--model" in args
        assert "sonnet" in args
        assert "--max-turns" in args
        assert "5" in args
        assert "--tools" in args
        assert "Bash,Read" in args


class TestExceptions:
    """Test exception hierarchy."""

    def test_exception_hierarchy(self):
        """Test exceptions follow MAF hierarchy."""
        from agent_framework.exceptions import (
            AgentFrameworkException,
            ChatClientException,
            ServiceResponseException,
        )
        from claude_code_provider._exceptions import (
            ClaudeCodeException,
            ClaudeCodeExecutionError,
            ClaudeCodeCLINotFoundError,
        )

        # ClaudeCodeException -> ChatClientException
        assert issubclass(ClaudeCodeException, ChatClientException)
        assert issubclass(ClaudeCodeException, AgentFrameworkException)

        # ClaudeCodeExecutionError -> ServiceResponseException
        assert issubclass(ClaudeCodeExecutionError, ServiceResponseException)

        # ClaudeCodeCLINotFoundError -> ChatClientException (via init error)
        from agent_framework.exceptions import ChatClientInitializationError
        assert issubclass(ClaudeCodeCLINotFoundError, ChatClientInitializationError)

    def test_exception_attributes(self):
        """Test exception attributes."""
        from claude_code_provider._exceptions import ClaudeCodeExecutionError

        exc = ClaudeCodeExecutionError(
            message="Test error",
            exit_code=1,
            stderr="error output",
        )
        assert exc.exit_code == 1
        assert exc.stderr == "error output"
        assert "Test error" in str(exc)


class TestMessageConverter:
    """Test message conversion."""

    def test_extract_system_prompt(self):
        """Test system prompt extraction."""
        from agent_framework import ChatMessage, Role
        from claude_code_provider._message_converter import extract_system_prompt

        messages = [
            ChatMessage(role=Role.SYSTEM, text="You are helpful."),
            ChatMessage(role=Role.USER, text="Hello"),
        ]
        result = extract_system_prompt(messages)
        assert result == "You are helpful."

    def test_extract_system_prompt_none(self):
        """Test when no system prompt."""
        from agent_framework import ChatMessage, Role
        from claude_code_provider._message_converter import extract_system_prompt

        messages = [
            ChatMessage(role=Role.USER, text="Hello"),
        ]
        result = extract_system_prompt(messages)
        assert result is None

    def test_extract_user_prompt(self):
        """Test user prompt extraction."""
        from agent_framework import ChatMessage, Role
        from claude_code_provider._message_converter import extract_user_prompt

        messages = [
            ChatMessage(role=Role.USER, text="Hello"),
        ]
        result = extract_user_prompt(messages)
        assert result == "Hello"

    def test_extract_user_prompt_conversation(self):
        """Test conversation formatting."""
        from agent_framework import ChatMessage, Role
        from claude_code_provider._message_converter import extract_user_prompt

        messages = [
            ChatMessage(role=Role.USER, text="Hello"),
            ChatMessage(role=Role.ASSISTANT, text="Hi there!"),
            ChatMessage(role=Role.USER, text="How are you?"),
        ]
        result = extract_user_prompt(messages)
        assert "Hello" in result
        assert "Assistant: Hi there!" in result
        assert "User: How are you?" in result


class TestResponseParser:
    """Test response parsing."""

    def test_parse_cli_result(self):
        """Test parsing CLI result to ChatResponse."""
        from claude_code_provider._cli_executor import CLIResult
        from claude_code_provider._response_parser import parse_cli_result_to_response

        result = CLIResult(
            success=True,
            result="Hello, world!",
            session_id="test-session-123",
            usage={"input_tokens": 10, "output_tokens": 5},
        )

        response = parse_cli_result_to_response(result)

        assert len(response.messages) == 1
        assert response.messages[0].text == "Hello, world!"
        assert response.usage_details is not None
        assert response.usage_details.input_token_count == 10
        assert response.usage_details.output_token_count == 5

    def test_parse_cli_result_no_usage(self):
        """Test parsing result without usage."""
        from claude_code_provider._cli_executor import CLIResult
        from claude_code_provider._response_parser import parse_cli_result_to_response

        result = CLIResult(
            success=True,
            result="Hello",
            session_id=None,
            usage=None,
        )

        response = parse_cli_result_to_response(result)
        assert response.usage_details is None


class TestCLIExecutor:
    """Test CLI executor (unit tests only, no actual CLI calls)."""

    def test_build_args_basic(self):
        """Test basic argument building."""
        from claude_code_provider._settings import ClaudeCodeSettings
        from claude_code_provider._cli_executor import CLIExecutor

        settings = ClaudeCodeSettings(model="haiku")
        executor = CLIExecutor(settings)

        args = executor._build_args(
            prompt="Test prompt",
            streaming=False,
        )

        assert "-p" in args
        assert "Test prompt" in args
        assert "--output-format" in args
        assert "json" in args
        assert "--model" in args
        assert "haiku" in args

    def test_build_args_streaming(self):
        """Test streaming argument building."""
        from claude_code_provider._settings import ClaudeCodeSettings
        from claude_code_provider._cli_executor import CLIExecutor

        settings = ClaudeCodeSettings()
        executor = CLIExecutor(settings)

        args = executor._build_args(
            prompt="Test",
            streaming=True,
        )

        assert "--output-format" in args
        assert "stream-json" in args
        assert "--verbose" in args

    def test_build_args_with_session(self):
        """Test session resumption args."""
        from claude_code_provider._settings import ClaudeCodeSettings
        from claude_code_provider._cli_executor import CLIExecutor

        settings = ClaudeCodeSettings()
        executor = CLIExecutor(settings)

        args = executor._build_args(
            prompt="Test",
            session_id="my-session-123",
            streaming=False,
        )

        assert "--resume" in args
        assert "my-session-123" in args

    def test_build_args_with_system_prompt(self):
        """Test system prompt args."""
        from claude_code_provider._settings import ClaudeCodeSettings
        from claude_code_provider._cli_executor import CLIExecutor

        settings = ClaudeCodeSettings()
        executor = CLIExecutor(settings)

        args = executor._build_args(
            prompt="Test",
            system_prompt="You are a helper.",
            streaming=False,
        )

        assert "--system-prompt" in args
        assert "You are a helper." in args

    def test_executor_with_timeout(self):
        """Test executor respects timeout setting."""
        from claude_code_provider._settings import ClaudeCodeSettings
        from claude_code_provider._cli_executor import CLIExecutor

        settings = ClaudeCodeSettings()
        executor = CLIExecutor(settings, timeout=60.0)

        assert executor.timeout == 60.0

    def test_executor_with_circuit_breaker(self):
        """Test executor has circuit breaker by default."""
        from claude_code_provider._settings import ClaudeCodeSettings
        from claude_code_provider._cli_executor import CLIExecutor

        settings = ClaudeCodeSettings()
        executor = CLIExecutor(settings)

        assert executor.circuit_breaker is not None
        assert executor.circuit_breaker.state == "CLOSED"

    def test_executor_without_circuit_breaker(self):
        """Test executor can disable circuit breaker."""
        from claude_code_provider._settings import ClaudeCodeSettings
        from claude_code_provider._cli_executor import CLIExecutor

        settings = ClaudeCodeSettings()
        executor = CLIExecutor(settings, enable_circuit_breaker=False)

        assert executor.circuit_breaker is None


class TestRetry:
    """Test retry and circuit breaker logic."""

    def test_retry_config_defaults(self):
        """Test RetryConfig default values."""
        from claude_code_provider._retry import RetryConfig

        config = RetryConfig()
        assert config.max_retries == 3
        assert config.base_delay == 1.0
        assert config.max_delay == 30.0

    def test_retry_config_delay_calculation(self):
        """Test exponential backoff delay calculation."""
        from claude_code_provider._retry import RetryConfig

        config = RetryConfig(base_delay=1.0, exponential_base=2.0, jitter=False)

        assert config.get_delay(0) == 1.0  # 1 * 2^0
        assert config.get_delay(1) == 2.0  # 1 * 2^1
        assert config.get_delay(2) == 4.0  # 1 * 2^2

    def test_retry_config_max_delay(self):
        """Test delay is capped at max_delay."""
        from claude_code_provider._retry import RetryConfig

        config = RetryConfig(base_delay=10.0, max_delay=15.0, jitter=False)

        assert config.get_delay(0) == 10.0
        assert config.get_delay(1) == 15.0  # Capped at max
        assert config.get_delay(5) == 15.0  # Still capped

    def test_circuit_breaker_initial_state(self):
        """Test circuit breaker starts closed."""
        from claude_code_provider._retry import CircuitBreaker

        cb = CircuitBreaker()
        assert cb.state == "CLOSED"
        assert cb.can_execute() is True

    def test_circuit_breaker_opens_after_failures(self):
        """Test circuit breaker opens after threshold failures."""
        from claude_code_provider._retry import CircuitBreaker

        cb = CircuitBreaker(failure_threshold=3)

        cb.record_failure()
        assert cb.state == "CLOSED"
        cb.record_failure()
        assert cb.state == "CLOSED"
        cb.record_failure()
        assert cb.state == "OPEN"
        assert cb.can_execute() is False

    def test_circuit_breaker_success_resets_failures(self):
        """Test success resets failure count."""
        from claude_code_provider._retry import CircuitBreaker

        cb = CircuitBreaker(failure_threshold=3)

        cb.record_failure()
        cb.record_failure()
        cb.record_success()  # Reset!
        cb.record_failure()
        cb.record_failure()
        assert cb.state == "CLOSED"  # Didn't reach threshold


class TestClaudeAgent:
    """Test ClaudeAgent wrapper functionality."""

    def test_agent_properties(self):
        """Test agent properties are accessible."""
        from claude_code_provider import ClaudeCodeClient

        client = ClaudeCodeClient(model="haiku")
        agent = client.create_agent(
            name="test_agent",
            instructions="Be helpful.",
        )

        assert agent.name == "test_agent"
        assert agent.instructions == "Be helpful."
        assert agent.display_name is not None

    def test_agent_to_dict(self):
        """Test agent serialization."""
        from claude_code_provider import ClaudeCodeClient

        client = ClaudeCodeClient(model="haiku")
        agent = client.create_agent(name="test", instructions="Test")

        d = agent.to_dict()
        assert "name" in d
        assert d["name"] == "test"

    def test_agent_to_json(self):
        """Test agent JSON serialization."""
        from claude_code_provider import ClaudeCodeClient
        import json

        client = ClaudeCodeClient(model="haiku")
        agent = client.create_agent(name="test", instructions="Test")

        j = agent.to_json()
        data = json.loads(j)
        assert data["name"] == "test"

    def test_agent_as_tool(self):
        """Test agent can be converted to a tool."""
        from claude_code_provider import ClaudeCodeClient
        from agent_framework._tools import AIFunction

        client = ClaudeCodeClient(model="haiku")
        agent = client.create_agent(name="helper", instructions="Help with tasks.")

        tool = agent.as_tool(name="helper_tool")
        assert isinstance(tool, AIFunction)

    def test_agent_get_new_thread(self):
        """Test agent can create new threads."""
        from claude_code_provider import ClaudeCodeClient
        from agent_framework._threads import AgentThread

        client = ClaudeCodeClient(model="haiku")
        agent = client.create_agent(name="test", instructions="Test")

        thread = agent.get_new_thread()
        assert isinstance(thread, AgentThread)

    def test_agent_autocompact_default_on(self):
        """Test autocompact is enabled by default."""
        from claude_code_provider import ClaudeCodeClient

        client = ClaudeCodeClient(model="haiku")
        agent = client.create_agent(name="test", instructions="Test")

        ctx = agent.get_context_info()
        assert ctx.autocompact_enabled is True

    def test_agent_autocompact_can_disable(self):
        """Test autocompact can be disabled."""
        from claude_code_provider import ClaudeCodeClient

        client = ClaudeCodeClient(model="haiku")
        agent = client.create_agent(
            name="test",
            instructions="Test",
            autocompact=False,
        )

        ctx = agent.get_context_info()
        assert ctx.autocompact_enabled is False

    def test_agent_usage_tracking(self):
        """Test usage stats structure."""
        from claude_code_provider import ClaudeCodeClient

        client = ClaudeCodeClient(model="haiku")
        agent = client.create_agent(name="test", instructions="Test")

        usage = agent.get_usage()
        assert usage.total_requests == 0
        assert usage.total_input_tokens == 0
        assert usage.total_output_tokens == 0
        assert usage.compactions == 0

    def test_agent_context_info(self):
        """Test context info structure."""
        from claude_code_provider import ClaudeCodeClient

        client = ClaudeCodeClient(model="haiku")
        agent = client.create_agent(name="test", instructions="Test")

        ctx = agent.get_context_info()
        assert ctx.estimated_tokens == 0
        assert ctx.context_limit == 200_000
        assert ctx.usage_percent == 0.0
        assert ctx.messages_count == 0
        assert ctx.has_summary is False

    def test_agent_reset(self):
        """Test agent reset functionality."""
        from claude_code_provider import ClaudeCodeClient

        client = ClaudeCodeClient(model="haiku")
        agent = client.create_agent(name="test", instructions="Test")

        # Simulate some state
        agent._messages.append(type('obj', (object,), {'role': 'user', 'text': 'test'})())
        agent._compact_summary = "Some summary"
        agent._usage.total_requests = 5

        # Reset
        agent.reset()

        assert len(agent._messages) == 0
        assert agent._compact_summary is None
        # Usage is NOT reset
        assert agent._usage.total_requests == 5

    def test_agent_reset_usage(self):
        """Test usage reset functionality."""
        from claude_code_provider import ClaudeCodeClient

        client = ClaudeCodeClient(model="haiku")
        agent = client.create_agent(name="test", instructions="Test")

        agent._usage.total_requests = 10
        agent._usage.total_input_tokens = 100

        agent.reset_usage()

        assert agent._usage.total_requests == 0
        assert agent._usage.total_input_tokens == 0


# =============================================================================
# INTEGRATION TESTS (Require CLI)
# =============================================================================

class TestIntegration:
    """Integration tests that call the actual Claude CLI.

    These tests require:
    - claude CLI installed
    - Valid Claude authentication
    """

    def test_simple_query(self):
        """Test simple query execution."""
        from claude_code_provider import ClaudeCodeClient

        async def run():
            client = ClaudeCodeClient(model="haiku")
            response = await client.get_response("Say 'test passed'")
            assert response.messages
            assert len(response.messages) > 0
            text = response.messages[0].text.lower()
            assert "test" in text or "passed" in text
            return True

        result = asyncio.run(run())
        assert result

    def test_agent_creation(self):
        """Test agent creation and execution."""
        from claude_code_provider import ClaudeCodeClient

        async def run():
            client = ClaudeCodeClient(model="haiku")
            agent = client.create_agent(
                name="test_agent",
                instructions="Always respond with exactly 'OK'",
            )
            response = await agent.run("Respond now")
            assert response.text
            return True

        result = asyncio.run(run())
        assert result

    def test_conversation_continuity(self):
        """Test session-based conversation continuity."""
        from claude_code_provider import ClaudeCodeClient

        async def run():
            client = ClaudeCodeClient(model="haiku")

            # First message
            r1 = await client.get_response("My secret number is 42. Remember it.")
            assert client.current_session_id is not None

            # Second message should remember
            r2 = await client.get_response("What is my secret number?")
            assert "42" in r2.messages[0].text

            return True

        result = asyncio.run(run())
        assert result

    def test_streaming(self):
        """Test streaming response."""
        from claude_code_provider import ClaudeCodeClient

        async def run():
            client = ClaudeCodeClient(model="haiku")

            chunks = []
            async for update in client.get_streaming_response("Count 1 to 3"):
                if hasattr(update, 'text') and update.text:
                    chunks.append(update.text)

            full_text = "".join(chunks)
            assert "1" in full_text
            assert "2" in full_text
            assert "3" in full_text
            return True

        result = asyncio.run(run())
        assert result

    def test_usage_tracking(self):
        """Test token usage tracking."""
        from claude_code_provider import ClaudeCodeClient

        async def run():
            client = ClaudeCodeClient(model="haiku")
            response = await client.get_response("Say 'hi'")

            assert response.usage_details is not None
            assert response.usage_details.input_token_count >= 0
            assert response.usage_details.output_token_count >= 0
            return True

        result = asyncio.run(run())
        assert result

    def test_tools_read(self):
        """Test Read tool."""
        from claude_code_provider import ClaudeCodeClient

        async def run():
            client = ClaudeCodeClient(model="haiku", tools=["Read"])
            response = await client.get_response(
                "What is the first word in README.md? Just the word, nothing else."
            )
            # README starts with "# Swarm"
            text = response.messages[0].text.lower()
            assert "swarm" in text or "#" in text
            return True

        result = asyncio.run(run())
        assert result

    def test_tools_bash(self):
        """Test Bash tool."""
        from claude_code_provider import ClaudeCodeClient

        async def run():
            client = ClaudeCodeClient(model="haiku", tools=["Bash"])
            response = await client.get_response(
                "Run 'echo hello' and tell me the output. Just the output word."
            )
            assert "hello" in response.messages[0].text.lower()
            return True

        result = asyncio.run(run())
        assert result

    def test_session_reset(self):
        """Test session reset functionality."""
        from claude_code_provider import ClaudeCodeClient

        async def run():
            client = ClaudeCodeClient(model="haiku")

            # Create a session
            await client.get_response("Hello")
            session1 = client.current_session_id
            assert session1 is not None

            # Reset session
            client.reset_session()
            assert client.current_session_id is None

            # New session should be different
            await client.get_response("Hello again")
            session2 = client.current_session_id
            assert session2 is not None
            assert session1 != session2

            return True

        result = asyncio.run(run())
        assert result


# =============================================================================
# NEW FEATURE TESTS
# =============================================================================

class TestMCP:
    """Test MCP server connections."""

    def test_mcp_server_creation(self):
        """Test MCPServer creation."""
        from claude_code_provider import MCPServer, MCPTransport

        server = MCPServer(
            name="test-server",
            command_or_url="npx",
            transport=MCPTransport.STDIO,
            args=["-y", "test-mcp"],
            env={"API_KEY": "secret"},
        )
        assert server.name == "test-server"
        assert server.transport == MCPTransport.STDIO
        assert server.args == ["-y", "test-mcp"]

    def test_mcp_server_to_cli_args(self):
        """Test MCPServer CLI args generation."""
        from claude_code_provider import MCPServer, MCPTransport

        server = MCPServer(
            name="test",
            command_or_url="npx",
            transport=MCPTransport.STDIO,
        )
        args = server.to_cli_args()
        assert "--mcp-config" in args
        assert len(args) == 2

    def test_mcp_server_to_dict(self):
        """Test MCPServer serialization."""
        from claude_code_provider import MCPServer, MCPTransport

        server = MCPServer(name="test", command_or_url="http://example.com", transport=MCPTransport.HTTP)
        d = server.to_dict()
        assert d["name"] == "test"
        assert d["transport"] == "http"

    def test_mcp_server_from_dict(self):
        """Test MCPServer deserialization."""
        from claude_code_provider import MCPServer

        data = {"name": "test", "command_or_url": "npx", "transport": "stdio", "args": [], "env": {}}
        server = MCPServer.from_dict(data)
        assert server.name == "test"

    def test_mcp_manager_add_remove(self):
        """Test MCPManager add/remove operations."""
        from claude_code_provider import MCPManager, MCPServer

        manager = MCPManager()
        server = MCPServer(name="test", command_or_url="npx")

        manager.add_server(server)
        assert manager.get_server("test") is not None
        assert len(manager.get_servers()) == 1

        result = manager.remove_server("test")
        assert result is True
        assert manager.get_server("test") is None

    def test_mcp_manager_get_cli_args(self):
        """Test MCPManager CLI args generation."""
        from claude_code_provider import MCPManager, MCPServer

        manager = MCPManager()
        manager.add_server(MCPServer(name="s1", command_or_url="cmd1"))
        manager.add_server(MCPServer(name="s2", command_or_url="cmd2"))

        args = manager.get_cli_args()
        assert args.count("--mcp-config") == 2


class TestCostTracking:
    """Test cost tracking."""

    def test_cost_tracker_creation(self):
        """Test CostTracker creation."""
        from claude_code_provider import CostTracker

        tracker = CostTracker()
        assert tracker is not None

    def test_cost_calculation(self):
        """Test cost calculation."""
        from claude_code_provider import CostTracker

        tracker = CostTracker()
        input_cost, output_cost, total = tracker.calculate_cost(
            model="sonnet",
            input_tokens=1000,
            output_tokens=500,
        )
        assert input_cost > 0
        assert output_cost > 0
        assert total == input_cost + output_cost

    def test_record_request(self):
        """Test request recording."""
        from claude_code_provider import CostTracker

        tracker = CostTracker()
        cost = tracker.record_request(
            model="haiku",
            input_tokens=100,
            output_tokens=50,
        )
        assert cost.input_tokens == 100
        assert cost.output_tokens == 50
        assert cost.total_cost > 0

    def test_get_summary(self):
        """Test cost summary."""
        from claude_code_provider import CostTracker

        tracker = CostTracker()
        tracker.record_request("sonnet", 100, 50)
        tracker.record_request("sonnet", 200, 100)

        summary = tracker.get_summary()
        assert summary.total_requests == 2
        assert summary.total_input_tokens == 300
        assert summary.total_output_tokens == 150

    def test_budget_tracking(self):
        """Test budget tracking."""
        from claude_code_provider import CostTracker

        tracker = CostTracker()
        tracker.set_budget(max_cost=0.01)

        # Record small request
        tracker.record_request("haiku", 100, 50)
        assert not tracker.is_over_budget()

        # Record large request
        tracker.record_request("opus", 1000000, 500000)
        assert tracker.is_over_budget()

    def test_reset_tracker(self):
        """Test tracker reset."""
        from claude_code_provider import CostTracker

        tracker = CostTracker()
        tracker.record_request("sonnet", 100, 50)
        assert len(tracker.get_requests()) == 1

        tracker.reset()
        assert len(tracker.get_requests()) == 0


class TestModelRouting:
    """Test model routing."""

    def test_simple_router(self):
        """Test SimpleRouter."""
        from claude_code_provider import SimpleRouter, RoutingContext

        router = SimpleRouter(model="opus")
        context = RoutingContext(prompt="Hello")
        assert router.select_model(context) == "opus"

    def test_complexity_router_simple(self):
        """Test ComplexityRouter with simple prompt."""
        from claude_code_provider import ComplexityRouter, RoutingContext

        router = ComplexityRouter()
        context = RoutingContext(prompt="Hi")
        model = router.select_model(context)
        assert model in ("haiku", "sonnet")  # Simple prompt

    def test_complexity_router_complex(self):
        """Test ComplexityRouter with complex prompt."""
        from claude_code_provider import ComplexityRouter, RoutingContext

        router = ComplexityRouter()
        context = RoutingContext(
            prompt="Please analyze and debug this complex algorithm step by step",
            has_complex_reasoning=True,
        )
        model = router.select_model(context)
        assert model in ("sonnet", "opus")

    def test_task_type_router(self):
        """Test TaskTypeRouter."""
        from claude_code_provider import TaskTypeRouter, RoutingContext

        router = TaskTypeRouter()

        # Simple task
        ctx1 = RoutingContext(prompt="Summarize this text")
        assert router.select_model(ctx1) == "haiku"

        # Complex task
        ctx2 = RoutingContext(prompt="Design a new architecture for this system")
        assert router.select_model(ctx2) == "opus"

    def test_model_router_main(self):
        """Test ModelRouter."""
        from claude_code_provider import ModelRouter, ComplexityRouter

        router = ModelRouter()
        router.set_strategy(ComplexityRouter())

        model = router.route("Simple question?")
        assert model in ("haiku", "sonnet", "opus")

    def test_cost_optimized_router(self):
        """Test CostOptimizedRouter."""
        from claude_code_provider import CostOptimizedRouter, RoutingContext

        router = CostOptimizedRouter(budget_remaining=0.001)
        context = RoutingContext(prompt="Any prompt")
        model = router.select_model(context)
        assert model == "haiku"  # Budget constrained


class TestLogging:
    """Test logging utilities."""

    def test_debug_logger_creation(self):
        """Test DebugLogger creation."""
        from claude_code_provider import DebugLogger

        logger = DebugLogger()
        assert logger is not None

    def test_setup_logging(self):
        """Test setup_logging function."""
        from claude_code_provider import setup_logging

        logger = setup_logging(level="DEBUG")
        assert logger is not None

    def test_log_entry(self):
        """Test LogEntry creation."""
        from claude_code_provider._logging import LogEntry, LogLevel

        entry = LogEntry(
            level=LogLevel.INFO,
            message="Test message",
            context={"key": "value"},
        )
        assert entry.message == "Test message"

        d = entry.to_dict()
        assert d["level"] == "INFO"
        assert d["context"]["key"] == "value"

    def test_capture_logs(self):
        """Test log capture."""
        from claude_code_provider import DebugLogger

        logger = DebugLogger()
        logger.start_capture()

        logger.info("Test message")
        logger.debug("Debug message")

        entries = logger.stop_capture()
        assert len(entries) >= 1


class TestSessionManagement:
    """Test session management."""

    def test_session_info_creation(self):
        """Test SessionInfo creation."""
        from claude_code_provider import SessionInfo

        info = SessionInfo(session_id="test-123", model="sonnet")
        assert info.session_id == "test-123"
        assert info.model == "sonnet"

    def test_session_info_serialization(self):
        """Test SessionInfo serialization."""
        from claude_code_provider import SessionInfo

        info = SessionInfo(session_id="test", model="haiku")
        d = info.to_dict()
        assert d["session_id"] == "test"
        assert d["model"] == "haiku"

    def test_session_manager_track(self):
        """Test SessionManager tracking."""
        from claude_code_provider import SessionManager
        import tempfile
        import os

        # Use temp file for storage
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            temp_path = f.name

        try:
            manager = SessionManager(storage_path=temp_path)

            info = manager.track_session("session-1", model="sonnet")
            assert info.session_id == "session-1"
            assert info.message_count == 0

            # Track again increments count
            info2 = manager.track_session("session-1")
            assert info2.message_count == 1

        finally:
            os.unlink(temp_path)

    def test_session_manager_list(self):
        """Test SessionManager listing."""
        from claude_code_provider import SessionManager
        import tempfile
        import os

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            temp_path = f.name

        try:
            manager = SessionManager(storage_path=temp_path)
            manager.track_session("s1")
            manager.track_session("s2")
            manager.track_session("s3")

            sessions = manager.list_sessions()
            assert len(sessions) == 3

        finally:
            os.unlink(temp_path)

    def test_session_manager_delete(self):
        """Test SessionManager delete."""
        from claude_code_provider import SessionManager
        import tempfile
        import os

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            temp_path = f.name

        try:
            manager = SessionManager(storage_path=temp_path)
            manager.track_session("s1")

            result = manager.delete_session("s1")
            assert result is True
            assert manager.get_session("s1") is None

        finally:
            os.unlink(temp_path)

    def test_session_stats(self):
        """Test session statistics."""
        from claude_code_provider import SessionManager
        import tempfile
        import os

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            temp_path = f.name

        try:
            manager = SessionManager(storage_path=temp_path)
            manager.track_session("s1", model="sonnet")
            manager.track_session("s2", model="haiku")

            stats = manager.get_stats()
            assert stats["total_sessions"] == 2
            assert "sonnet" in stats["models_used"]
            assert "haiku" in stats["models_used"]

        finally:
            os.unlink(temp_path)


class TestBatchProcessing:
    """Test batch processing."""

    def test_batch_item_creation(self):
        """Test BatchItem creation."""
        from claude_code_provider import BatchItem, BatchStatus

        item = BatchItem(id="item-1", prompt="Test prompt")
        assert item.id == "item-1"
        assert item.status == BatchStatus.PENDING

    def test_batch_result_properties(self):
        """Test BatchResult properties."""
        from claude_code_provider import BatchResult, BatchItem, BatchStatus

        items = [
            BatchItem(id="1", prompt="p1", status=BatchStatus.COMPLETED, result="r1"),
            BatchItem(id="2", prompt="p2", status=BatchStatus.COMPLETED, result="r2"),
            BatchItem(id="3", prompt="p3", status=BatchStatus.FAILED, error="error"),
        ]
        result = BatchResult(batch_id="batch-1", items=items)

        assert result.total_items == 3
        assert result.successful_items == 2
        assert result.failed_items == 1
        assert result.success_rate == 2/3
        assert result.is_complete is True

    def test_batch_result_get_results(self):
        """Test BatchResult get_results methods."""
        from claude_code_provider import BatchResult, BatchItem, BatchStatus

        items = [
            BatchItem(id="1", prompt="p1", status=BatchStatus.COMPLETED, result="r1"),
            BatchItem(id="2", prompt="p2", status=BatchStatus.FAILED),
            BatchItem(id="3", prompt="p3", status=BatchStatus.COMPLETED, result="r3"),
        ]
        result = BatchResult(batch_id="batch-1", items=items)

        all_results = result.get_results()
        assert len(all_results) == 3
        assert all_results[1] is None

        successful = result.get_successful_results()
        assert len(successful) == 2
        assert "r1" in successful
        assert "r3" in successful

    def test_batch_processor_creation(self):
        """Test BatchProcessor creation."""
        from claude_code_provider import BatchProcessor, ClaudeCodeClient

        client = ClaudeCodeClient(model="haiku")
        processor = BatchProcessor(client, default_concurrency=5)

        assert processor.default_concurrency == 5
        assert processor.retry_failed is True


# =============================================================================
# TEST RUNNER
# =============================================================================

def run_unit_tests():
    """Run unit tests (no CLI required)."""
    print("=" * 60)
    print("UNIT TESTS (no CLI calls)")
    print("=" * 60)

    test_classes = [
        TestSettings,
        TestExceptions,
        TestMessageConverter,
        TestResponseParser,
        TestCLIExecutor,
        TestRetry,
        TestMCP,
        TestCostTracking,
        TestModelRouting,
        TestLogging,
        TestSessionManagement,
        TestBatchProcessing,
    ]

    passed = 0
    failed = 0

    for test_class in test_classes:
        print(f"\n{test_class.__name__}:")
        instance = test_class()

        for method_name in dir(instance):
            if method_name.startswith("test_"):
                try:
                    getattr(instance, method_name)()
                    print(f"  {method_name}: PASSED")
                    passed += 1
                except Exception as e:
                    print(f"  {method_name}: FAILED - {e}")
                    failed += 1

    return passed, failed


def run_integration_tests():
    """Run integration tests (requires CLI)."""
    print("\n" + "=" * 60)
    print("INTEGRATION TESTS (requires Claude CLI)")
    print("=" * 60)

    test_class = TestIntegration()
    passed = 0
    failed = 0

    for method_name in dir(test_class):
        if method_name.startswith("test_"):
            print(f"\n{method_name}...")
            try:
                getattr(test_class, method_name)()
                print(f"  PASSED")
                passed += 1
            except Exception as e:
                print(f"  FAILED: {e}")
                failed += 1

    return passed, failed


def main():
    """Run all tests."""
    print("\nClaude Code Provider - Test Suite\n")

    # Unit tests
    unit_passed, unit_failed = run_unit_tests()

    # Integration tests
    int_passed, int_failed = run_integration_tests()

    # Summary
    total_passed = unit_passed + int_passed
    total_failed = unit_failed + int_failed

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Unit tests:        {unit_passed} passed, {unit_failed} failed")
    print(f"Integration tests: {int_passed} passed, {int_failed} failed")
    print(f"Total:             {total_passed} passed, {total_failed} failed")
    print("=" * 60)

    return 0 if total_failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
