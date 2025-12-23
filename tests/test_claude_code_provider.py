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
