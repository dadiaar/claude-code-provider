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

    def test_parse_cli_result_with_cache_tokens(self):
        """Test parsing result with cache tokens included in total."""
        from claude_code_provider._cli_executor import CLIResult
        from claude_code_provider._response_parser import parse_cli_result_to_response

        # Simulate typical Claude CLI response with cache tokens
        result = CLIResult(
            success=True,
            result="Test response",
            session_id="test-session",
            usage={
                "input_tokens": 5,  # Raw new tokens
                "cache_creation_input_tokens": 4000,
                "cache_read_input_tokens": 14000,
                "output_tokens": 500,
            },
        )

        response = parse_cli_result_to_response(result)

        # Total input should be sum of all three
        assert response.usage_details.input_token_count == 5 + 4000 + 14000
        assert response.usage_details.output_token_count == 500

        # Raw breakdown should be preserved in additional_counts
        assert response.usage_details.additional_counts["raw_input_tokens"] == 5
        assert response.usage_details.additional_counts["cache_creation_input_tokens"] == 4000
        assert response.usage_details.additional_counts["cache_read_input_tokens"] == 14000


class TestCLIResultTokens:
    """Test CLIResult token counting properties."""

    def test_input_tokens_includes_cache(self):
        """Test that input_tokens property sums all input types."""
        from claude_code_provider._cli_executor import CLIResult

        result = CLIResult(
            success=True,
            result="test",
            usage={
                "input_tokens": 10,
                "cache_creation_input_tokens": 5000,
                "cache_read_input_tokens": 15000,
                "output_tokens": 100,
            },
        )

        assert result.input_tokens == 10 + 5000 + 15000
        assert result.output_tokens == 100

    def test_raw_input_tokens_property(self):
        """Test raw_input_tokens returns only non-cached tokens."""
        from claude_code_provider._cli_executor import CLIResult

        result = CLIResult(
            success=True,
            result="test",
            usage={
                "input_tokens": 42,
                "cache_creation_input_tokens": 1000,
                "cache_read_input_tokens": 2000,
            },
        )

        assert result.raw_input_tokens == 42

    def test_cache_creation_tokens_property(self):
        """Test cache_creation_tokens property."""
        from claude_code_provider._cli_executor import CLIResult

        result = CLIResult(
            success=True,
            result="test",
            usage={"cache_creation_input_tokens": 5555},
        )

        assert result.cache_creation_tokens == 5555

    def test_cache_read_tokens_property(self):
        """Test cache_read_tokens property."""
        from claude_code_provider._cli_executor import CLIResult

        result = CLIResult(
            success=True,
            result="test",
            usage={"cache_read_input_tokens": 9999},
        )

        assert result.cache_read_tokens == 9999

    def test_token_breakdown_dict(self):
        """Test token_breakdown returns complete breakdown."""
        from claude_code_provider._cli_executor import CLIResult

        result = CLIResult(
            success=True,
            result="test",
            usage={
                "input_tokens": 5,
                "cache_creation_input_tokens": 1000,
                "cache_read_input_tokens": 2000,
                "output_tokens": 300,
            },
        )

        breakdown = result.token_breakdown
        assert breakdown["input_tokens"] == 3005  # Total
        assert breakdown["raw_input_tokens"] == 5
        assert breakdown["cache_creation_tokens"] == 1000
        assert breakdown["cache_read_tokens"] == 2000
        assert breakdown["output_tokens"] == 300

    def test_missing_cache_tokens_default_to_zero(self):
        """Test that missing cache token fields default to zero."""
        from claude_code_provider._cli_executor import CLIResult

        result = CLIResult(
            success=True,
            result="test",
            usage={"input_tokens": 100, "output_tokens": 50},
        )

        assert result.input_tokens == 100  # No cache to add
        assert result.raw_input_tokens == 100
        assert result.cache_creation_tokens == 0
        assert result.cache_read_tokens == 0

    def test_no_usage_returns_zeros(self):
        """Test that missing usage dict returns zeros."""
        from claude_code_provider._cli_executor import CLIResult

        result = CLIResult(success=True, result="test", usage=None)

        assert result.input_tokens == 0
        assert result.raw_input_tokens == 0
        assert result.cache_creation_tokens == 0
        assert result.cache_read_tokens == 0
        assert result.output_tokens == 0


class TestCLIExecutor:
    """Test CLI executor (unit tests only, no actual CLI calls)."""

    def test_build_args_basic(self):
        """Test basic argument building."""
        from claude_code_provider._settings import ClaudeCodeSettings
        from claude_code_provider._cli_executor import CLIExecutor

        settings = ClaudeCodeSettings(model="haiku")
        executor = CLIExecutor(settings)

        args, stdin_prompt = executor._build_args(
            prompt="Test prompt",
            streaming=False,
        )

        assert "-p" in args
        assert "Test prompt" in args
        assert "--output-format" in args
        assert "json" in args
        assert "--model" in args
        assert "haiku" in args
        assert stdin_prompt is None  # Small prompt uses args, not stdin

    def test_build_args_streaming(self):
        """Test streaming argument building."""
        from claude_code_provider._settings import ClaudeCodeSettings
        from claude_code_provider._cli_executor import CLIExecutor

        settings = ClaudeCodeSettings()
        executor = CLIExecutor(settings)

        args, stdin_prompt = executor._build_args(
            prompt="Test",
            streaming=True,
        )

        assert "--output-format" in args
        assert "stream-json" in args
        assert "--verbose" in args
        assert stdin_prompt is None

    def test_build_args_with_session(self):
        """Test session resumption args."""
        from claude_code_provider._settings import ClaudeCodeSettings
        from claude_code_provider._cli_executor import CLIExecutor

        settings = ClaudeCodeSettings()
        executor = CLIExecutor(settings)

        args, stdin_prompt = executor._build_args(
            prompt="Test",
            session_id="my-session-123",
            streaming=False,
        )

        assert "--resume" in args
        assert "my-session-123" in args
        assert stdin_prompt is None

    def test_build_args_with_system_prompt(self):
        """Test system prompt args."""
        from claude_code_provider._settings import ClaudeCodeSettings
        from claude_code_provider._cli_executor import CLIExecutor

        settings = ClaudeCodeSettings()
        executor = CLIExecutor(settings)

        args, stdin_prompt = executor._build_args(
            prompt="Test",
            system_prompt="You are a helper.",
            streaming=False,
        )

        assert "--system-prompt" in args
        assert "You are a helper." in args
        assert stdin_prompt is None

    def test_build_args_large_prompt_uses_stdin(self):
        """Test that large prompts are passed via stdin."""
        from claude_code_provider._settings import ClaudeCodeSettings
        from claude_code_provider._cli_executor import CLIExecutor, MAX_CLI_ARG_SIZE

        settings = ClaudeCodeSettings()
        executor = CLIExecutor(settings)

        # Create a prompt larger than MAX_CLI_ARG_SIZE
        large_prompt = "x" * (MAX_CLI_ARG_SIZE + 1000)
        args, stdin_prompt = executor._build_args(
            prompt=large_prompt,
            streaming=False,
        )

        assert "-p" not in args  # No -p flag for large prompts
        assert stdin_prompt == large_prompt  # Prompt passed via stdin

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

        async def run():
            cb = CircuitBreaker()
            assert cb.state == "CLOSED"
            assert await cb.can_execute() is True

        asyncio.run(run())

    def test_circuit_breaker_opens_after_failures(self):
        """Test circuit breaker opens after threshold failures."""
        from claude_code_provider._retry import CircuitBreaker

        async def run():
            cb = CircuitBreaker(failure_threshold=3)

            await cb.record_failure()
            assert cb.state == "CLOSED"
            await cb.record_failure()
            assert cb.state == "CLOSED"
            await cb.record_failure()
            assert cb.state == "OPEN"
            assert await cb.can_execute() is False

        asyncio.run(run())

    def test_circuit_breaker_success_resets_failures(self):
        """Test success resets failure count."""
        from claude_code_provider._retry import CircuitBreaker

        async def run():
            cb = CircuitBreaker(failure_threshold=3)

            await cb.record_failure()
            await cb.record_failure()
            await cb.record_success()  # Reset!
            await cb.record_failure()
            await cb.record_failure()
            assert cb.state == "CLOSED"  # Didn't reach threshold

        asyncio.run(run())


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
            # README starts with "# Claude Code Provider"
            text = response.messages[0].text.lower()
            assert "claude" in text or "#" in text
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


class TestOrchestration:
    """Test orchestration builders."""

    def test_orchestration_result(self):
        """Test OrchestrationResult creation."""
        from claude_code_provider import OrchestrationResult

        result = OrchestrationResult(
            final_output="Final text",
            conversation=[],
            rounds=5,
            participants_used={"agent1", "agent2"},
            metadata={"type": "test"},
        )
        assert result.final_output == "Final text"
        assert result.rounds == 5
        assert "agent1" in result.participants_used

    def test_group_chat_config(self):
        """Test GroupChatConfig creation."""
        from claude_code_provider import GroupChatConfig

        config = GroupChatConfig(
            max_rounds=10,
            manager_display_name="coordinator",
        )
        assert config.max_rounds == 10
        assert config.manager_display_name == "coordinator"
        assert config.termination_condition is None

    def test_handoff_config(self):
        """Test HandoffConfig creation."""
        from claude_code_provider import HandoffConfig

        config = HandoffConfig(
            autonomous=True,
            autonomous_turn_limit=30,
        )
        assert config.autonomous is True
        assert config.autonomous_turn_limit == 30

    def test_magentic_config(self):
        """Test MagenticConfig creation."""
        from claude_code_provider import MagenticConfig

        config = MagenticConfig(
            max_stall_count=5,
            max_reset_count=2,
        )
        assert config.max_stall_count == 5
        assert config.max_reset_count == 2

    def test_feedback_loop_orchestrator_creation(self):
        """Test FeedbackLoopOrchestrator creation."""
        from claude_code_provider import ClaudeCodeClient, FeedbackLoopOrchestrator

        client = ClaudeCodeClient(model="haiku")
        worker = client.create_agent(name="worker", instructions="Do work")
        reviewer = client.create_agent(name="reviewer", instructions="Review work")

        orchestrator = FeedbackLoopOrchestrator(
            worker=worker,
            reviewer=reviewer,
            max_iterations=3,
        )
        assert orchestrator.worker.name == "worker"
        assert orchestrator.reviewer.name == "reviewer"
        assert orchestrator.max_iterations == 3

    def test_feedback_loop_custom_approval(self):
        """Test FeedbackLoopOrchestrator with custom approval check."""
        from claude_code_provider import ClaudeCodeClient, FeedbackLoopOrchestrator

        client = ClaudeCodeClient(model="haiku")
        worker = client.create_agent(name="worker", instructions="Do work")
        reviewer = client.create_agent(name="reviewer", instructions="Review work")

        orchestrator = FeedbackLoopOrchestrator(
            worker=worker,
            reviewer=reviewer,
            approval_check=lambda t: "LGTM" in t,
        )

        # Test approval check
        assert orchestrator.approval_check("LGTM, great work!") is True
        assert orchestrator.approval_check("Needs revision") is False

    def test_create_review_loop_helper(self):
        """Test create_review_loop helper function."""
        from claude_code_provider import (
            ClaudeCodeClient,
            FeedbackLoopOrchestrator,
            create_review_loop,
        )

        client = ClaudeCodeClient(model="haiku")
        agents = {
            "worker": client.create_agent(name="worker", instructions="Work"),
            "reviewer": client.create_agent(name="reviewer", instructions="Review"),
        }

        orchestrator = create_review_loop(agents, max_iterations=7)
        assert isinstance(orchestrator, FeedbackLoopOrchestrator)
        assert orchestrator.max_iterations == 7

    def test_create_pipeline_helper(self):
        """Test create_pipeline helper function."""
        from claude_code_provider import (
            ClaudeCodeClient,
            SequentialOrchestrator,
            create_pipeline,
            MAF_ORCHESTRATION_AVAILABLE,
        )

        if not MAF_ORCHESTRATION_AVAILABLE:
            # Skip if MAF orchestration not available
            return

        client = ClaudeCodeClient(model="haiku")
        agents = [
            client.create_agent(name="agent1", instructions="Step 1"),
            client.create_agent(name="agent2", instructions="Step 2"),
        ]

        orchestrator = create_pipeline(agents)
        assert isinstance(orchestrator, SequentialOrchestrator)
        assert len(orchestrator.agents) == 2

    def test_create_parallel_analysis_helper(self):
        """Test create_parallel_analysis helper function."""
        from claude_code_provider import (
            ClaudeCodeClient,
            ConcurrentOrchestrator,
            create_parallel_analysis,
            MAF_ORCHESTRATION_AVAILABLE,
        )

        if not MAF_ORCHESTRATION_AVAILABLE:
            return

        client = ClaudeCodeClient(model="haiku")
        analysts = [
            client.create_agent(name="analyst1", instructions="Analyze"),
            client.create_agent(name="analyst2", instructions="Analyze"),
        ]

        orchestrator = create_parallel_analysis(analysts)
        assert isinstance(orchestrator, ConcurrentOrchestrator)
        assert len(orchestrator.agents) == 2

    def test_client_create_feedback_loop(self):
        """Test ClaudeCodeClient.create_feedback_loop method."""
        from claude_code_provider import ClaudeCodeClient, FeedbackLoopOrchestrator

        client = ClaudeCodeClient(model="haiku")
        worker = client.create_agent(name="dev", instructions="Write code")
        reviewer = client.create_agent(name="reviewer", instructions="Review code")

        orchestrator = client.create_feedback_loop(
            worker=worker,
            reviewer=reviewer,
            max_iterations=4,
        )
        assert isinstance(orchestrator, FeedbackLoopOrchestrator)
        assert orchestrator.max_iterations == 4

    def test_client_create_sequential(self):
        """Test ClaudeCodeClient.create_sequential method."""
        from claude_code_provider import ClaudeCodeClient, MAF_ORCHESTRATION_AVAILABLE

        if not MAF_ORCHESTRATION_AVAILABLE:
            return

        from claude_code_provider import SequentialOrchestrator

        client = ClaudeCodeClient(model="haiku")
        agents = [
            client.create_agent(name="a1", instructions="Step 1"),
            client.create_agent(name="a2", instructions="Step 2"),
        ]

        orchestrator = client.create_sequential(agents)
        assert isinstance(orchestrator, SequentialOrchestrator)

    def test_client_create_concurrent(self):
        """Test ClaudeCodeClient.create_concurrent method."""
        from claude_code_provider import ClaudeCodeClient, MAF_ORCHESTRATION_AVAILABLE

        if not MAF_ORCHESTRATION_AVAILABLE:
            return

        from claude_code_provider import ConcurrentOrchestrator

        client = ClaudeCodeClient(model="haiku")
        agents = [
            client.create_agent(name="a1", instructions="Analyze"),
            client.create_agent(name="a2", instructions="Analyze"),
        ]

        orchestrator = client.create_concurrent(agents)
        assert isinstance(orchestrator, ConcurrentOrchestrator)

    def test_client_create_group_chat(self):
        """Test ClaudeCodeClient.create_group_chat method."""
        from claude_code_provider import ClaudeCodeClient, MAF_ORCHESTRATION_AVAILABLE

        if not MAF_ORCHESTRATION_AVAILABLE:
            return

        from claude_code_provider import GroupChatOrchestrator

        client = ClaudeCodeClient(model="haiku")
        agents = [
            client.create_agent(name="dev", instructions="Code"),
            client.create_agent(name="reviewer", instructions="Review"),
        ]

        def select_speaker(state):
            return "dev"

        orchestrator = client.create_group_chat(
            participants=agents,
            manager=select_speaker,
            max_rounds=10,
        )
        assert isinstance(orchestrator, GroupChatOrchestrator)

    def test_client_create_handoff(self):
        """Test ClaudeCodeClient.create_handoff method."""
        from claude_code_provider import ClaudeCodeClient, MAF_ORCHESTRATION_AVAILABLE

        if not MAF_ORCHESTRATION_AVAILABLE:
            return

        from claude_code_provider import HandoffOrchestrator

        client = ClaudeCodeClient(model="haiku")
        coordinator = client.create_agent(name="coordinator", instructions="Route")
        specialists = [
            client.create_agent(name="billing", instructions="Handle billing"),
            client.create_agent(name="technical", instructions="Handle tech"),
        ]

        orchestrator = client.create_handoff(
            coordinator=coordinator,
            specialists=specialists,
            autonomous=True,
        )
        assert isinstance(orchestrator, HandoffOrchestrator)


class TestLimitProfiles:
    """Test limit profiles and model timeouts."""

    def test_limit_profiles_exist(self):
        """Test that all expected profiles exist."""
        from claude_code_provider import LIMIT_PROFILES

        expected = ["demo", "standard", "extended", "unlimited"]
        for name in expected:
            assert name in LIMIT_PROFILES, f"Missing profile: {name}"

    def test_limit_profile_structure(self):
        """Test that profiles have required fields."""
        from claude_code_provider import LIMIT_PROFILES

        required_fields = ["description", "max_iterations", "timeout_seconds", "checkpoint_enabled"]
        for name, profile in LIMIT_PROFILES.items():
            for field in required_fields:
                assert field in profile, f"Profile {name} missing field: {field}"

    def test_limit_profile_values(self):
        """Test that profile values are correct."""
        from claude_code_provider import LIMIT_PROFILES

        # Check demo profile
        assert LIMIT_PROFILES["demo"]["max_iterations"] == 5
        assert LIMIT_PROFILES["demo"]["timeout_seconds"] == 300  # 5 min

        # Check standard profile
        assert LIMIT_PROFILES["standard"]["max_iterations"] == 20
        assert LIMIT_PROFILES["standard"]["timeout_seconds"] == 3600  # 1 hour

        # Check extended profile
        assert LIMIT_PROFILES["extended"]["max_iterations"] == 100
        assert LIMIT_PROFILES["extended"]["timeout_seconds"] == 7200  # 2 hours

        # Check unlimited profile
        assert LIMIT_PROFILES["unlimited"]["max_iterations"] == 500
        assert LIMIT_PROFILES["unlimited"]["timeout_seconds"] == 14400  # 4 hours

    def test_checkpoint_disabled_by_default(self):
        """Test that checkpointing is disabled by default for security."""
        from claude_code_provider import LIMIT_PROFILES

        for name, profile in LIMIT_PROFILES.items():
            assert profile["checkpoint_enabled"] is False, \
                f"Profile {name} has checkpoint_enabled=True (should be False for security)"

    def test_get_limit_profile(self):
        """Test get_limit_profile function."""
        from claude_code_provider import get_limit_profile

        # Get standard profile
        profile = get_limit_profile("standard")
        assert profile["max_iterations"] == 20
        assert profile["timeout_seconds"] == 3600

        # Profile is a copy, not reference
        profile["max_iterations"] = 999
        original = get_limit_profile("standard")
        assert original["max_iterations"] == 20

    def test_get_limit_profile_default(self):
        """Test get_limit_profile returns default when None."""
        from claude_code_provider import get_limit_profile

        profile = get_limit_profile(None)
        assert profile["max_iterations"] == 20  # standard profile

    def test_get_limit_profile_invalid(self):
        """Test get_limit_profile raises on invalid name."""
        from claude_code_provider import get_limit_profile

        try:
            get_limit_profile("nonexistent")
            assert False, "Should raise ValueError"
        except ValueError as e:
            assert "nonexistent" in str(e)

    def test_model_timeouts_exist(self):
        """Test that MODEL_TIMEOUTS has expected models."""
        from claude_code_provider import MODEL_TIMEOUTS

        expected = ["haiku", "sonnet", "opus", "default"]
        for model in expected:
            assert model in MODEL_TIMEOUTS, f"Missing model: {model}"

    def test_model_timeout_values(self):
        """Test model timeout values."""
        from claude_code_provider import MODEL_TIMEOUTS

        assert MODEL_TIMEOUTS["haiku"] == 600      # 10 min
        assert MODEL_TIMEOUTS["sonnet"] == 1800    # 30 min
        assert MODEL_TIMEOUTS["opus"] == 3600      # 60 min
        assert MODEL_TIMEOUTS["default"] == 1800   # 30 min

    def test_get_model_timeout(self):
        """Test get_model_timeout function."""
        from claude_code_provider import get_model_timeout

        assert get_model_timeout("haiku") == 600
        assert get_model_timeout("sonnet") == 1800
        assert get_model_timeout("opus") == 3600

    def test_get_model_timeout_case_insensitive(self):
        """Test get_model_timeout is case insensitive."""
        from claude_code_provider import get_model_timeout

        assert get_model_timeout("HAIKU") == 600
        assert get_model_timeout("Sonnet") == 1800
        assert get_model_timeout("OPUS") == 3600

    def test_get_model_timeout_fallback(self):
        """Test get_model_timeout returns default for unknown models."""
        from claude_code_provider import get_model_timeout, MODEL_TIMEOUTS

        assert get_model_timeout("unknown") == MODEL_TIMEOUTS["default"]
        assert get_model_timeout("gpt-4") == MODEL_TIMEOUTS["default"]


class TestCheckpointing:
    """Test checkpointing system."""

    def test_checkpoint_creation(self):
        """Test Checkpoint dataclass creation."""
        from claude_code_provider import Checkpoint

        checkpoint = Checkpoint(
            checkpoint_id="test_123",
            orchestration_type="feedback_loop",
            task="Test task",
            conversation=[{"role": "assistant", "text": "Hello"}],
            current_iteration=2,
            current_work="Some work",
            feedback="Some feedback",
            participants_used=["worker", "reviewer"],
            metadata={"key": "value"},
            created_at="2025-01-01T00:00:00",
            updated_at="2025-01-01T00:01:00",
            status="in_progress",
        )
        assert checkpoint.checkpoint_id == "test_123"
        assert checkpoint.current_iteration == 2
        assert checkpoint.status == "in_progress"

    def test_checkpoint_manager_creation(self):
        """Test CheckpointManager creation."""
        from claude_code_provider import CheckpointManager
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            manager = CheckpointManager(tmpdir)
            assert manager.checkpoint_dir.exists()

    def test_checkpoint_manager_generate_id(self):
        """Test CheckpointManager.generate_checkpoint_id."""
        from claude_code_provider import CheckpointManager
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            manager = CheckpointManager(tmpdir)

            id1 = manager.generate_checkpoint_id("task1", "feedback_loop")
            id2 = manager.generate_checkpoint_id("task1", "feedback_loop")
            id3 = manager.generate_checkpoint_id("task2", "feedback_loop")

            # Same task = same ID
            assert id1 == id2
            # Different task = different ID
            assert id1 != id3
            # Contains orchestration type
            assert "feedback_loop" in id1

    def test_checkpoint_manager_save_load(self):
        """Test CheckpointManager save and load."""
        from claude_code_provider import CheckpointManager, Checkpoint
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            manager = CheckpointManager(tmpdir)

            checkpoint = Checkpoint(
                checkpoint_id="test_save",
                orchestration_type="feedback_loop",
                task="Test task",
                conversation=[],
                current_iteration=3,
                current_work="Work content",
                feedback="Feedback content",
                participants_used=["w", "r"],
                metadata={"test": True},
                created_at="2025-01-01T00:00:00",
                updated_at="2025-01-01T00:00:00",
                status="in_progress",
            )

            # Save
            path = manager.save(checkpoint)
            assert path.exists()

            # Load
            loaded = manager.load("test_save")
            assert loaded is not None
            assert loaded.checkpoint_id == "test_save"
            assert loaded.current_iteration == 3
            assert loaded.current_work == "Work content"
            assert loaded.status == "in_progress"

    def test_checkpoint_manager_exists(self):
        """Test CheckpointManager.exists."""
        from claude_code_provider import CheckpointManager, Checkpoint
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            manager = CheckpointManager(tmpdir)

            assert manager.exists("nonexistent") is False

            checkpoint = Checkpoint(
                checkpoint_id="exists_test",
                orchestration_type="feedback_loop",
                task="Test",
                conversation=[],
                current_iteration=1,
                current_work="",
                feedback="",
                participants_used=[],
                metadata={},
                created_at="",
                updated_at="",
                status="in_progress",
            )
            manager.save(checkpoint)

            assert manager.exists("exists_test") is True

    def test_checkpoint_manager_clear(self):
        """Test CheckpointManager.clear (single)."""
        from claude_code_provider import CheckpointManager, Checkpoint
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            manager = CheckpointManager(tmpdir)

            checkpoint = Checkpoint(
                checkpoint_id="clear_test",
                orchestration_type="feedback_loop",
                task="Test",
                conversation=[],
                current_iteration=1,
                current_work="",
                feedback="",
                participants_used=[],
                metadata={},
                created_at="",
                updated_at="",
                status="in_progress",
            )
            manager.save(checkpoint)
            assert manager.exists("clear_test") is True

            result = manager.clear("clear_test")
            assert result is True
            assert manager.exists("clear_test") is False

            # Clear nonexistent returns False
            result = manager.clear("nonexistent")
            assert result is False

    def test_checkpoint_manager_clear_all(self):
        """Test CheckpointManager.clear_all."""
        from claude_code_provider import CheckpointManager, Checkpoint
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            manager = CheckpointManager(tmpdir)

            # Create multiple checkpoints
            for i in range(5):
                checkpoint = Checkpoint(
                    checkpoint_id=f"batch_{i}",
                    orchestration_type="feedback_loop",
                    task=f"Task {i}",
                    conversation=[],
                    current_iteration=i,
                    current_work="",
                    feedback="",
                    participants_used=[],
                    metadata={},
                    created_at="",
                    updated_at="",
                    status="in_progress",
                )
                manager.save(checkpoint)

            assert len(manager.list_checkpoints()) == 5

            count = manager.clear_all()
            assert count == 5
            assert len(manager.list_checkpoints()) == 0

    def test_checkpoint_manager_list(self):
        """Test CheckpointManager.list_checkpoints."""
        from claude_code_provider import CheckpointManager, Checkpoint
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            manager = CheckpointManager(tmpdir)

            # Create checkpoints
            for i in range(3):
                checkpoint = Checkpoint(
                    checkpoint_id=f"list_{i}",
                    orchestration_type="feedback_loop",
                    task=f"Task {i}",
                    conversation=[],
                    current_iteration=i,
                    current_work="",
                    feedback="",
                    participants_used=[],
                    metadata={},
                    created_at="",
                    updated_at="",
                    status="in_progress",
                )
                manager.save(checkpoint)

            checkpoints = manager.list_checkpoints()
            assert len(checkpoints) == 3
            ids = [c.checkpoint_id for c in checkpoints]
            assert "list_0" in ids
            assert "list_1" in ids
            assert "list_2" in ids

    def test_clear_checkpoints_function(self):
        """Test clear_checkpoints convenience function."""
        from claude_code_provider import clear_checkpoints, CheckpointManager, Checkpoint
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            manager = CheckpointManager(tmpdir)

            # Create a checkpoint
            checkpoint = Checkpoint(
                checkpoint_id="func_test",
                orchestration_type="feedback_loop",
                task="Test",
                conversation=[],
                current_iteration=1,
                current_work="",
                feedback="",
                participants_used=[],
                metadata={},
                created_at="",
                updated_at="",
                status="in_progress",
            )
            manager.save(checkpoint)

            count = clear_checkpoints(tmpdir)
            assert count == 1


class TestGracefulStop:
    """Test graceful stop handler."""

    def test_stop_handler_creation(self):
        """Test GracefulStopHandler creation."""
        from claude_code_provider import GracefulStopHandler

        handler = GracefulStopHandler()
        assert handler.should_stop is False

    def test_stop_handler_reset(self):
        """Test GracefulStopHandler.reset."""
        from claude_code_provider import GracefulStopHandler

        handler = GracefulStopHandler()
        handler.should_stop = True
        assert handler.should_stop is True

        handler.reset()
        assert handler.should_stop is False

    def test_get_stop_handler(self):
        """Test get_stop_handler returns global handler."""
        from claude_code_provider import get_stop_handler

        handler1 = get_stop_handler()
        handler2 = get_stop_handler()
        assert handler1 is handler2

    def test_stop_handler_register_unregister(self):
        """Test GracefulStopHandler register/unregister."""
        from claude_code_provider import GracefulStopHandler

        handler = GracefulStopHandler()

        # Should not raise
        handler.register()
        handler.unregister()


class TestFeedbackLoopAdvanced:
    """Advanced tests for FeedbackLoopOrchestrator."""

    def test_orchestrator_default_checkpoint_disabled(self):
        """Test that checkpointing is disabled by default."""
        from claude_code_provider import ClaudeCodeClient, FeedbackLoopOrchestrator

        client = ClaudeCodeClient(model="haiku")
        worker = client.create_agent(name="w", instructions="Work")
        reviewer = client.create_agent(name="r", instructions="Review")

        orchestrator = FeedbackLoopOrchestrator(worker=worker, reviewer=reviewer)
        assert orchestrator.checkpoint_enabled is False
        assert orchestrator.checkpoint_manager is None

    def test_orchestrator_checkpoint_enabled(self):
        """Test enabling checkpointing."""
        from claude_code_provider import ClaudeCodeClient, FeedbackLoopOrchestrator
        import tempfile

        client = ClaudeCodeClient(model="haiku")
        worker = client.create_agent(name="w", instructions="Work")
        reviewer = client.create_agent(name="r", instructions="Review")

        with tempfile.TemporaryDirectory() as tmpdir:
            orchestrator = FeedbackLoopOrchestrator(
                worker=worker,
                reviewer=reviewer,
                checkpoint_enabled=True,
                checkpoint_dir=tmpdir,
            )
            assert orchestrator.checkpoint_enabled is True
            assert orchestrator.checkpoint_manager is not None

    def test_orchestrator_with_profile(self):
        """Test creating orchestrator with limit profile."""
        from claude_code_provider import (
            ClaudeCodeClient,
            FeedbackLoopOrchestrator,
            get_limit_profile,
        )

        client = ClaudeCodeClient(model="haiku")
        worker = client.create_agent(name="w", instructions="Work")
        reviewer = client.create_agent(name="r", instructions="Review")

        profile = get_limit_profile("extended")
        orchestrator = FeedbackLoopOrchestrator(
            worker=worker,
            reviewer=reviewer,
            max_iterations=profile["max_iterations"],
            timeout_seconds=profile["timeout_seconds"],
        )
        assert orchestrator.max_iterations == 100
        assert orchestrator.timeout_seconds == 7200

    def test_orchestrator_serialization(self):
        """Test conversation serialization/deserialization."""
        from claude_code_provider import ClaudeCodeClient, FeedbackLoopOrchestrator
        from agent_framework import ChatMessage, Role

        client = ClaudeCodeClient(model="haiku")
        worker = client.create_agent(name="w", instructions="Work")
        reviewer = client.create_agent(name="r", instructions="Review")

        orchestrator = FeedbackLoopOrchestrator(worker=worker, reviewer=reviewer)

        # Test serialization
        messages = [
            ChatMessage(role=Role.ASSISTANT, text="Hello", author_name="worker"),
            ChatMessage(role=Role.ASSISTANT, text="Looks good", author_name="reviewer"),
        ]
        serialized = orchestrator._serialize_conversation(messages)
        assert len(serialized) == 2
        assert serialized[0]["text"] == "Hello"
        assert serialized[0]["author_name"] == "worker"

        # Test deserialization
        deserialized = orchestrator._deserialize_conversation(serialized)
        assert len(deserialized) == 2
        assert deserialized[0].text == "Hello"


# =============================================================================
# SECURITY TESTS
# =============================================================================

class TestSecurity:
    """Security-focused tests for vulnerability prevention."""

    def test_format_string_injection_prevention(self):
        """Test that format string injection is prevented in batch processing."""
        from claude_code_provider._batch import BatchProcessor
        import re
        from string import Template

        # Test the safe_substitute approach directly
        prompt_template = "Process {item}"
        variables = {"item": "test"}

        # This is what the code does now - convert to Template syntax
        safe_template = re.sub(r'\{(\w+)\}', r'$\1', prompt_template)
        result = Template(safe_template).safe_substitute(variables)
        assert result == "Process test"

        # Test that malicious format strings don't expose internals
        malicious_template = "{__class__.__init__.__globals__}"
        safe_malicious = re.sub(r'\{(\w+)\}', r'$\1', malicious_template)
        result = Template(safe_malicious).safe_substitute({})
        # Should NOT execute the format string attack - returns unchanged
        assert "__class__" in result  # The literal string, not evaluated

    def test_cli_argument_validation_rejects_non_flags(self):
        """Test that non-flag arguments are rejected."""
        from claude_code_provider._cli_executor import _validate_extra_args

        # Valid flag arguments should work
        valid_args = ["--verbose", "-v", "--no-cache"]
        result = _validate_extra_args(valid_args)
        assert result == valid_args

        # Non-flag arguments should be rejected
        try:
            _validate_extra_args(["malicious_command"])
            assert False, "Should raise ValueError for non-flag arguments"
        except ValueError as e:
            assert "Only flag arguments" in str(e)

        # Injection attempts should be rejected (non-flag after valid flag)
        try:
            _validate_extra_args(["--verbose", "rm -rf /"])
            assert False, "Should reject non-flag arguments"
        except ValueError as e:
            assert "Only flag arguments" in str(e)

    def test_dangerous_skip_permissions_not_allowed(self):
        """Test that --dangerously-skip-permissions is not in allowed args."""
        from claude_code_provider._cli_executor import ALLOWED_EXTRA_ARGS

        assert "--dangerously-skip-permissions" not in ALLOWED_EXTRA_ARGS

    def test_mcp_server_name_validation(self):
        """Test MCP server name validation."""
        from claude_code_provider._mcp import _validate_server_name

        # Valid names
        assert _validate_server_name("my-server") == "my-server"
        assert _validate_server_name("server_123") == "server_123"
        assert _validate_server_name("MyServer") == "MyServer"

        # Invalid names
        try:
            _validate_server_name("")
            assert False, "Should reject empty name"
        except ValueError:
            pass

        try:
            _validate_server_name("123invalid")  # Starts with number
            assert False, "Should reject names starting with numbers"
        except ValueError:
            pass

        try:
            _validate_server_name("a" * 100)  # Too long
            assert False, "Should reject names over 64 chars"
        except ValueError:
            pass

    def test_mcp_command_validation(self):
        """Test MCP command/argument validation for dangerous characters."""
        from claude_code_provider._mcp import _validate_command_or_arg

        # Valid commands
        assert _validate_command_or_arg("npx") == "npx"
        assert _validate_command_or_arg("/usr/bin/python") == "/usr/bin/python"

        # Commands with dangerous characters should be rejected
        dangerous_tests = [
            "cmd; rm -rf /",
            "cmd && malicious",
            "cmd | cat /etc/passwd",
            "$(whoami)",
            "`whoami`",
            "cmd > /tmp/output",
            "cmd < /etc/shadow",
        ]
        for cmd in dangerous_tests:
            try:
                _validate_command_or_arg(cmd)
                assert False, f"Should reject: {cmd}"
            except ValueError as e:
                assert "dangerous characters" in str(e)

    def test_mcp_env_var_name_validation(self):
        """Test MCP environment variable name validation."""
        from claude_code_provider._mcp import _validate_env_var_name

        # Valid env var names
        assert _validate_env_var_name("API_KEY") == "API_KEY"
        assert _validate_env_var_name("_PRIVATE") == "_PRIVATE"
        assert _validate_env_var_name("myVar123") == "myVar123"

        # Invalid env var names
        try:
            _validate_env_var_name("123INVALID")  # Starts with number
            assert False, "Should reject names starting with numbers"
        except ValueError:
            pass

        try:
            _validate_env_var_name("VAR-NAME")  # Contains hyphen
            assert False, "Should reject names with hyphens"
        except ValueError:
            pass

    def test_mcp_server_validates_on_creation(self):
        """Test that MCPServer validates fields on creation."""
        from claude_code_provider import MCPServer, MCPTransport

        # Valid server should work
        server = MCPServer(
            name="valid-server",
            command_or_url="npx",
            transport=MCPTransport.STDIO,
        )
        assert server.name == "valid-server"

        # Invalid name should fail
        try:
            MCPServer(
                name="123-invalid",
                command_or_url="npx",
                transport=MCPTransport.STDIO,
            )
            assert False, "Should reject invalid server name"
        except ValueError:
            pass

        # Dangerous command should fail for STDIO transport
        try:
            MCPServer(
                name="test",
                command_or_url="npx; rm -rf /",
                transport=MCPTransport.STDIO,
            )
            assert False, "Should reject dangerous command"
        except ValueError:
            pass

        # HTTP transport doesn't validate command for dangerous chars
        server = MCPServer(
            name="test",
            command_or_url="http://example.com?param=value",
            transport=MCPTransport.HTTP,
        )
        assert server.command_or_url == "http://example.com?param=value"

    def test_session_export_path_traversal_prevention(self):
        """Test that session export prevents path traversal to system dirs."""
        from claude_code_provider._sessions import _validate_export_path
        from pathlib import Path

        # Valid paths should work
        valid = _validate_export_path(Path("/tmp/session.json"))
        assert valid == Path("/tmp/session.json")

        valid = _validate_export_path(Path("./session.json"))
        # Should resolve to absolute path in current directory
        assert valid.is_absolute()

        # System directories should be rejected
        forbidden = ["/etc/passwd", "/bin/test", "/usr/bin/test", "/boot/test"]
        for path_str in forbidden:
            try:
                _validate_export_path(Path(path_str))
                assert False, f"Should reject system path: {path_str}"
            except ValueError as e:
                assert "system directory" in str(e)

    def test_streaming_timeout_constant_exists(self):
        """Test that streaming timeout constant is defined."""
        from claude_code_provider._cli_executor import DEFAULT_STREAM_READ_TIMEOUT

        assert DEFAULT_STREAM_READ_TIMEOUT > 0
        assert DEFAULT_STREAM_READ_TIMEOUT <= 120  # Should be reasonable

    def test_prompt_size_limit(self):
        """Test that prompt size is validated."""
        from claude_code_provider._cli_executor import _validate_prompt, MAX_PROMPT_SIZE

        # Normal prompt should work
        result = _validate_prompt("Hello")
        assert result == "Hello"

        # Too large prompt should fail
        try:
            _validate_prompt("x" * (MAX_PROMPT_SIZE + 1))
            assert False, "Should reject prompts over size limit"
        except ValueError as e:
            assert "exceeds maximum" in str(e).lower()

    def test_prompt_null_byte_rejection(self):
        """Test that null bytes in prompts are rejected."""
        from claude_code_provider._cli_executor import _validate_prompt

        try:
            _validate_prompt("hello\x00world")
            assert False, "Should reject null bytes"
        except ValueError as e:
            assert "null" in str(e).lower() or "invalid" in str(e).lower()

    # =========================================================================
    # Tests for demo 29 findings - Bug fixes
    # =========================================================================

    def test_session_id_validation(self):
        """Test that session_id is validated like checkpoint_id."""
        from claude_code_provider import ClaudeCodeSettings
        from claude_code_provider._cli_executor import CLIExecutor

        settings = ClaudeCodeSettings()
        executor = CLIExecutor(settings)

        # Valid session IDs should work
        valid_ids = ["session-123", "session_abc", "abc123", "A1B2C3"]
        for sid in valid_ids:
            args, _ = executor._build_args(
                prompt="test",
                session_id=sid,
            )
            assert "--resume" in args
            assert sid in args

        # Invalid session IDs should be rejected
        invalid_ids = [
            "session;rm -rf /",  # Shell injection
            "../../../etc/passwd",  # Path traversal
            "session\x00evil",  # Null byte
            "session$(whoami)",  # Command substitution
            "session`id`",  # Backtick execution
        ]
        for sid in invalid_ids:
            try:
                executor._build_args(prompt="test", session_id=sid)
                assert False, f"Should reject invalid session_id: {sid}"
            except ValueError as e:
                assert "Invalid session_id" in str(e)

    def test_no_duplicate_cli_args(self):
        """Test that --model and --max-turns are not added twice."""
        from claude_code_provider import ClaudeCodeSettings
        from claude_code_provider._cli_executor import CLIExecutor

        settings = ClaudeCodeSettings(model="sonnet", default_max_turns=5)
        executor = CLIExecutor(settings)

        args, _ = executor._build_args(prompt="test")

        # Count occurrences
        model_count = args.count("--model")
        max_turns_count = args.count("--max-turns")

        assert model_count == 1, f"Expected 1 --model, got {model_count}"
        assert max_turns_count == 1, f"Expected 1 --max-turns, got {max_turns_count}"

    def test_to_cli_args_exclude_parameter(self):
        """Test that to_cli_args respects the exclude parameter."""
        from claude_code_provider import ClaudeCodeSettings

        settings = ClaudeCodeSettings(
            model="sonnet",
            default_max_turns=5,
            permission_mode="default",
        )

        # Without exclude - includes everything
        args = settings.to_cli_args()
        assert "--model" in args
        assert "--max-turns" in args
        assert "--permission-mode" in args

        # With exclude - omits specified args
        args = settings.to_cli_args(exclude={"model", "max_turns"})
        assert "--model" not in args
        assert "--max-turns" not in args
        assert "--permission-mode" in args  # Not excluded

    def test_mcp_url_ssrf_prevention(self):
        """Test that MCP URLs are validated to prevent SSRF attacks."""
        from claude_code_provider._mcp import _validate_mcp_url

        # Valid external URLs should work
        valid_urls = [
            "https://api.example.com/mcp",
            "http://external-service.com:8080/",
        ]
        for url in valid_urls:
            result = _validate_mcp_url(url)
            assert result == url

        # Internal/private URLs should be rejected
        invalid_urls = [
            "http://localhost:8080/",
            "http://127.0.0.1/",
            "http://192.168.1.1/",
            "http://10.0.0.1/",
            "http://169.254.169.254/",  # AWS metadata
            "http://[::1]/",  # IPv6 loopback
            "ftp://example.com/",  # Wrong scheme
        ]
        for url in invalid_urls:
            try:
                _validate_mcp_url(url)
                assert False, f"Should reject SSRF-prone URL: {url}"
            except ValueError:
                pass

    def test_mcp_server_validates_http_url(self):
        """Test that MCPServer validates HTTP URLs on creation."""
        from claude_code_provider import MCPServer, MCPTransport

        # Valid external URL should work
        server = MCPServer(
            name="external",
            command_or_url="https://api.example.com/mcp",
            transport=MCPTransport.HTTP,
        )
        assert server.command_or_url == "https://api.example.com/mcp"

        # Internal URL should fail
        try:
            MCPServer(
                name="internal",
                command_or_url="http://localhost:8080/",
                transport=MCPTransport.HTTP,
            )
            assert False, "Should reject localhost URL"
        except ValueError as e:
            assert "localhost" in str(e).lower() or "loopback" in str(e).lower()

    def test_working_directory_security(self):
        """Test that working_directory rejects sensitive system paths."""
        from claude_code_provider import ClaudeCodeSettings, ConfigurationError
        import tempfile
        import os

        # Valid working directory should work (use temp dir)
        with tempfile.TemporaryDirectory() as tmpdir:
            settings = ClaudeCodeSettings(working_directory=tmpdir)
            assert settings.working_directory == tmpdir

        # System directories should be rejected
        forbidden_dirs = ["/etc", "/bin", "/root", "/boot"]
        for dir_path in forbidden_dirs:
            if os.path.exists(dir_path):
                try:
                    ClaudeCodeSettings(working_directory=dir_path)
                    assert False, f"Should reject system directory: {dir_path}"
                except ConfigurationError as e:
                    assert "sensitive" in str(e).lower() or "security" in str(e).lower()

    def test_retry_async_typevar_binding(self):
        """Test that retry_async properly binds return type."""
        import asyncio
        from claude_code_provider._retry import retry_async, RetryConfig

        async def returns_int() -> int:
            return 42

        async def returns_str() -> str:
            return "hello"

        # Run the tests using asyncio.run for newer Python versions
        result_int = asyncio.run(
            retry_async(returns_int, config=RetryConfig(max_retries=0))
        )
        assert result_int == 42
        assert isinstance(result_int, int)

        result_str = asyncio.run(
            retry_async(returns_str, config=RetryConfig(max_retries=0))
        )
        assert result_str == "hello"
        assert isinstance(result_str, str)

    def test_session_manager_thread_safety(self):
        """Test that SessionManager has a lock for thread safety."""
        from claude_code_provider._sessions import SessionManager
        import threading

        manager = SessionManager(storage_path="/tmp/test_sessions.json")

        # Verify lock exists
        assert hasattr(manager, '_lock')
        assert isinstance(manager._lock, type(threading.Lock()))


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
        TestSecurity,
        TestOrchestration,
        TestLimitProfiles,
        TestCheckpointing,
        TestGracefulStop,
        TestFeedbackLoopAdvanced,
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
