# Copyright (c) 2025. Claude Code Provider for Microsoft Agent Framework.

"""Settings for Claude Code CLI provider with validation."""

import os
import platform
import re
import shutil
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Literal

from ._exceptions import ClaudeCodeException


class ClaudeModel(str, Enum):
    """Supported Claude models with type-safe selection.

    Example:
        from claude_code_provider import ClaudeCodeClient, ClaudeModel

        # Type-safe (IDE autocomplete, catches typos)
        client = ClaudeCodeClient(model=ClaudeModel.SONNET)

        # String still works (backward compatible)
        client = ClaudeCodeClient(model="sonnet")
    """
    # Aliases (recommended)
    SONNET = "sonnet"
    OPUS = "opus"
    HAIKU = "haiku"
    # Full model names
    SONNET_3_5 = "claude-3-5-sonnet-20241022"
    HAIKU_3_5 = "claude-3-5-haiku-20241022"
    OPUS_3 = "claude-3-opus-20240229"
    SONNET_4 = "claude-sonnet-4-20250514"
    OPUS_4 = "claude-opus-4-20250514"

    def __str__(self) -> str:
        return self.value


# Valid model names and aliases
VALID_MODELS = frozenset({
    # Aliases
    "sonnet", "opus", "haiku",
    # Full model names (common ones)
    "claude-3-5-sonnet-20241022",
    "claude-3-5-haiku-20241022",
    "claude-3-opus-20240229",
    "claude-sonnet-4-20250514",
    "claude-opus-4-20250514",
})

# Valid permission modes
VALID_PERMISSION_MODES = frozenset({
    "default",
    "bypassPermissions",
    "plan",
})

# Directories that should not be used as working directories for security
# Unix/Linux/macOS forbidden directories
_FORBIDDEN_WORKING_DIRS_UNIX = frozenset({
    "/", "/etc", "/bin", "/sbin", "/usr/bin", "/usr/sbin",
    "/boot", "/dev", "/proc", "/sys", "/var/run",
    "/lib", "/lib64", "/usr/lib", "/usr/lib64",
    "/root", "/var/log", "/var/spool",
})

# Windows forbidden directories (fix for #32)
_FORBIDDEN_WORKING_DIRS_WINDOWS = frozenset({
    "C:\\", "C:\\Windows", "C:\\Windows\\System32",
    "C:\\Windows\\SysWOW64", "C:\\Program Files",
    "C:\\Program Files (x86)", "C:\\Users",
})

# Combined forbidden directories based on platform
_FORBIDDEN_WORKING_DIRS = (
    _FORBIDDEN_WORKING_DIRS_WINDOWS
    if platform.system() == "Windows"
    else _FORBIDDEN_WORKING_DIRS_UNIX
)

# Pattern for safe environment variable values
# Allows alphanumeric, path separators, common safe chars
_SAFE_ENV_VALUE_PATTERN = re.compile(r'^[a-zA-Z0-9_./:@=+,\-\\\s]*$')


class ConfigurationError(ClaudeCodeException):
    """Raised when configuration validation fails.

    Inherits from ClaudeCodeException to be part of the unified exception
    hierarchy (fix for #13).

    Attributes:
        field: The field that failed validation.
        message: Description of the validation error.
    """

    def __init__(self, field: str, message: str) -> None:
        self.field = field
        self.message = message
        super().__init__(f"Configuration error in '{field}': {message}")


@dataclass
class ClaudeCodeSettings:
    """Settings for Claude Code CLI client.

    All settings are validated at initialization when validate_on_init=True (default).

    Attributes:
        cli_path: Path to the claude CLI executable. Defaults to 'claude' (uses PATH).
        model: Default model to use. Options: 'sonnet', 'opus', 'haiku', or full model name.
        default_max_turns: Default maximum agentic turns. None means no limit.
        permission_mode: Permission mode for tool execution.
        working_directory: Working directory for CLI execution. Defaults to current directory.
        tools: List of tools to enable. None means use defaults.
        allowed_tools: Tools that run without prompts.
        disallowed_tools: Tools that are blocked.
        validate_on_init: Whether to validate all settings on initialization.

    Example:
        ```python
        # Basic usage (validates automatically)
        settings = ClaudeCodeSettings(
            model="sonnet",
            default_max_turns=10,
        )

        # Skip validation (e.g., for testing)
        settings = ClaudeCodeSettings(
            model="sonnet",
            validate_on_init=False,
        )

        # Validate manually later
        settings.validate()
        ```

    Raises:
        ConfigurationError: If validation fails and validate_on_init=True.
    """

    cli_path: str = "claude"
    model: str | None = None  # None means use CLI default
    default_max_turns: int | None = None
    permission_mode: Literal["default", "bypassPermissions", "plan"] | None = None
    working_directory: str | None = None
    tools: list[str] | None = None
    allowed_tools: list[str] | None = None
    disallowed_tools: list[str] | None = None

    # Validation options
    validate_on_init: bool = field(default=True, repr=False)

    # Environment variable overrides
    env_prefix: str = field(default="CLAUDE_CODE_", repr=False)

    def _validate_env_value(self, field: str, env_var: str, value: str) -> str:
        """Validate an environment variable value for injection attacks.

        Fix for #1: Environment variable values are validated before use
        to prevent shell injection via malicious env vars.

        Args:
            field: The setting field name (for error messages).
            env_var: The environment variable name.
            value: The value to validate.

        Returns:
            The validated value.

        Raises:
            ConfigurationError: If value contains potentially dangerous characters.
        """
        if not _SAFE_ENV_VALUE_PATTERN.match(value):
            raise ConfigurationError(
                field,
                f"Environment variable {env_var} contains potentially dangerous "
                f"characters. Value: '{value}'. Only alphanumeric characters, "
                "path separators, and common safe characters are allowed."
            )
        return value

    def __post_init__(self) -> None:
        """Load settings from environment variables and optionally validate.

        Note on TOCTOU (fix for #17): Working directory validation happens at
        initialization time. The directory could theoretically be modified
        between validation and CLI execution. Full prevention would require
        chroot or namespaces which is beyond scope. Users should ensure their
        working directories are stable.
        """
        # CLI path from environment (with injection validation)
        if self.cli_path == "claude":
            env_var = f"{self.env_prefix}CLI_PATH"
            env_path = os.environ.get(env_var)
            if env_path:
                self.cli_path = self._validate_env_value("cli_path", env_var, env_path)

        # Model from environment (with injection validation)
        if self.model is None:
            env_var = f"{self.env_prefix}MODEL"
            env_model = os.environ.get(env_var)
            if env_model:
                self.model = self._validate_env_value("model", env_var, env_model)

        # Max turns from environment
        if self.default_max_turns is None:
            env_var = f"{self.env_prefix}MAX_TURNS"
            env_turns = os.environ.get(env_var)
            if env_turns:
                # Validate format before parsing
                self._validate_env_value("default_max_turns", env_var, env_turns)
                try:
                    self.default_max_turns = int(env_turns)
                except ValueError:
                    raise ConfigurationError(
                        "default_max_turns",
                        f"Environment variable {env_var} must be an integer, got '{env_turns}'"
                    )

        # Validate if requested
        if self.validate_on_init:
            self.validate()

    def validate(self) -> None:
        """Validate all settings.

        This performs comprehensive validation of all configuration values.
        Call this if you set validate_on_init=False and want to validate later.

        Raises:
            ConfigurationError: If any setting is invalid.
        """
        self._validate_cli_path()
        self._validate_model()
        self._validate_max_turns()
        self._validate_permission_mode()
        self._validate_working_directory()
        self._validate_tools()

    def _validate_cli_path(self) -> None:
        """Validate CLI path exists and is executable."""
        if not self.cli_path:
            raise ConfigurationError(
                "cli_path",
                "CLI path cannot be empty"
            )

        # Check if CLI exists
        cli_found = shutil.which(self.cli_path)
        if not cli_found:
            # Check if it's an absolute/relative path that exists
            path = Path(self.cli_path)
            if path.exists():
                if not path.is_file():
                    raise ConfigurationError(
                        "cli_path",
                        f"Path '{self.cli_path}' exists but is not a file"
                    )
                if not os.access(path, os.X_OK):
                    raise ConfigurationError(
                        "cli_path",
                        f"Path '{self.cli_path}' is not executable"
                    )
            else:
                raise ConfigurationError(
                    "cli_path",
                    f"Claude CLI not found at '{self.cli_path}'. "
                    "Please install Claude Code or set CLAUDE_CODE_CLI_PATH environment variable."
                )

    def _validate_model(self) -> None:
        """Validate model name if specified."""
        if self.model is None:
            return  # Use CLI default

        if not isinstance(self.model, str):
            raise ConfigurationError(
                "model",
                f"Model must be a string, got {type(self.model).__name__}"
            )

        if not self.model.strip():
            raise ConfigurationError(
                "model",
                "Model cannot be an empty string"
            )

        # Check against known models (warning for unknown, not error)
        # This allows for future models without breaking
        model_lower = self.model.lower()
        if model_lower not in VALID_MODELS and self.model not in VALID_MODELS:
            # Only warn via logging, don't fail - allows new models
            import logging
            logger = logging.getLogger("claude_code_provider")
            logger.warning(
                f"Unknown model '{self.model}'. Known models: {', '.join(sorted(VALID_MODELS))}. "
                "This may work if it's a valid Claude model name."
            )

    def _validate_max_turns(self) -> None:
        """Validate max turns is positive if specified."""
        if self.default_max_turns is None:
            return

        if not isinstance(self.default_max_turns, int):
            raise ConfigurationError(
                "default_max_turns",
                f"Max turns must be an integer, got {type(self.default_max_turns).__name__}"
            )

        if self.default_max_turns <= 0:
            raise ConfigurationError(
                "default_max_turns",
                f"Max turns must be positive, got {self.default_max_turns}"
            )

    def _validate_permission_mode(self) -> None:
        """Validate permission mode is valid."""
        if self.permission_mode is None:
            return

        if self.permission_mode not in VALID_PERMISSION_MODES:
            raise ConfigurationError(
                "permission_mode",
                f"Invalid permission mode '{self.permission_mode}'. "
                f"Valid modes: {', '.join(sorted(VALID_PERMISSION_MODES))}"
            )

    def _validate_working_directory(self) -> None:
        """Validate working directory exists, is a directory, and is safe."""
        if self.working_directory is None:
            return

        path = Path(self.working_directory)
        if not path.exists():
            raise ConfigurationError(
                "working_directory",
                f"Working directory '{self.working_directory}' does not exist"
            )

        if not path.is_dir():
            raise ConfigurationError(
                "working_directory",
                f"Working directory '{self.working_directory}' is not a directory"
            )

        # Security check: prevent using sensitive system directories
        resolved = path.resolve()
        resolved_str = str(resolved)

        # Check exact matches first
        if resolved_str in _FORBIDDEN_WORKING_DIRS:
            raise ConfigurationError(
                "working_directory",
                f"Working directory '{resolved_str}' is a sensitive system directory "
                "and cannot be used for security reasons."
            )

        # Check if path is under a forbidden directory
        for forbidden in _FORBIDDEN_WORKING_DIRS:
            if resolved_str.startswith(forbidden + "/"):
                raise ConfigurationError(
                    "working_directory",
                    f"Working directory '{resolved_str}' is under sensitive system path "
                    f"'{forbidden}' and cannot be used for security reasons."
                )

    def _validate_tools(self) -> None:
        """Validate tools lists are valid string lists."""
        for field_name in ("tools", "allowed_tools", "disallowed_tools"):
            value = getattr(self, field_name)
            if value is None:
                continue

            if not isinstance(value, list):
                raise ConfigurationError(
                    field_name,
                    f"Must be a list, got {type(value).__name__}"
                )

            for i, item in enumerate(value):
                if not isinstance(item, str):
                    raise ConfigurationError(
                        field_name,
                        f"Item {i} must be a string, got {type(item).__name__}"
                    )
                if not item.strip():
                    raise ConfigurationError(
                        field_name,
                        f"Item {i} cannot be empty"
                    )

    def to_cli_args(self, exclude: set[str] | None = None) -> list[str]:
        """Convert settings to CLI arguments.

        Args:
            exclude: Set of argument names to exclude (e.g., {"model", "max_turns"})
                    to avoid duplication when caller handles them separately.

        Returns:
            List of CLI arguments based on current settings.
        """
        args: list[str] = []
        exclude = exclude or set()

        if self.model and "model" not in exclude:
            args.extend(["--model", self.model])

        if self.default_max_turns is not None and "max_turns" not in exclude:
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
