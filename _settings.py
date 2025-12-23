# Copyright (c) 2025. Claude Code Provider for Microsoft Agent Framework.

"""Settings for Claude Code CLI provider."""

import os
import shutil
from dataclasses import dataclass, field
from typing import Literal


@dataclass
class ClaudeCodeSettings:
    """Settings for Claude Code CLI client.

    Attributes:
        cli_path: Path to the claude CLI executable. Defaults to 'claude' (uses PATH).
        model: Default model to use. Options: 'sonnet', 'opus', 'haiku', or full model name.
        default_max_turns: Default maximum agentic turns. None means no limit.
        permission_mode: Permission mode for tool execution.
        working_directory: Working directory for CLI execution. Defaults to current directory.
        tools: List of tools to enable. None means use defaults.
        allowed_tools: Tools that run without prompts.
        disallowed_tools: Tools that are blocked.

    Example:
        settings = ClaudeCodeSettings(
            model="sonnet",
            default_max_turns=10,
        )
    """

    cli_path: str = "claude"
    model: str | None = None  # None means use CLI default
    default_max_turns: int | None = None
    permission_mode: Literal["default", "bypassPermissions", "plan"] | None = None
    working_directory: str | None = None
    tools: list[str] | None = None
    allowed_tools: list[str] | None = None
    disallowed_tools: list[str] | None = None

    # Environment variable overrides
    env_prefix: str = field(default="CLAUDE_CODE_", repr=False)

    def __post_init__(self) -> None:
        """Load settings from environment variables if not explicitly set."""
        # CLI path from environment
        if self.cli_path == "claude":
            env_path = os.environ.get(f"{self.env_prefix}CLI_PATH")
            if env_path:
                self.cli_path = env_path

        # Model from environment
        if self.model is None:
            self.model = os.environ.get(f"{self.env_prefix}MODEL")

        # Max turns from environment
        if self.default_max_turns is None:
            env_turns = os.environ.get(f"{self.env_prefix}MAX_TURNS")
            if env_turns:
                self.default_max_turns = int(env_turns)

    def validate(self) -> None:
        """Validate that the CLI is available.

        Raises:
            FileNotFoundError: If the claude CLI is not found.
        """
        if not shutil.which(self.cli_path):
            raise FileNotFoundError(
                f"Claude CLI not found at '{self.cli_path}'. "
                "Please install Claude Code or set CLAUDE_CODE_CLI_PATH."
            )

    def to_cli_args(self) -> list[str]:
        """Convert settings to CLI arguments.

        Returns:
            List of CLI arguments based on current settings.
        """
        args: list[str] = []

        if self.model:
            args.extend(["--model", self.model])

        if self.default_max_turns is not None:
            args.extend(["--max-turns", str(self.default_max_turns)])

        if self.permission_mode:
            args.extend(["--permission-mode", self.permission_mode])

        if self.tools is not None:
            args.extend(["--tools", ",".join(self.tools)])

        if self.allowed_tools:
            args.extend(["--allowedTools", ",".join(self.allowed_tools)])

        if self.disallowed_tools:
            args.extend(["--disallowedTools", ",".join(self.disallowed_tools)])

        return args
