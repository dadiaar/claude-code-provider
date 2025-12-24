# Copyright (c) 2025. Claude Code Provider for Microsoft Agent Framework.

r"""Structured logging and debugging utilities.

This module provides JSON-structured logging by default for production use.
Logs can be analyzed using various open-source tools.

Log Analysis Tools (Open Source):
---------------------------------
1. **jq** - Command-line JSON processor (https://jqlang.github.io/jq/)
   ```bash
   # Filter errors only
   cat app.log | jq 'select(.level == "ERROR")'

   # Extract messages
   cat app.log | jq '.message'

   # Filter by context field
   cat app.log | jq 'select(.context.model == "sonnet")'
   ```

2. **Loki + Grafana** - Log aggregation and visualization (https://grafana.com/oss/loki/)
   - Horizontally-scalable, multi-tenant log aggregation
   - Native Grafana integration for dashboards
   - LogQL query language

3. **OpenSearch** - Search and analytics (https://opensearch.org/)
   - Fork of Elasticsearch, fully open source (Apache 2.0)
   - Powerful full-text search and analytics
   - OpenSearch Dashboards for visualization

4. **Vector** - Log collection/transformation (https://vector.dev/)
   - High-performance log router
   - Parse, transform, and route logs
   - Integrates with many backends

5. **Fluentd/Fluent Bit** - Log collection (https://www.fluentd.org/)
   - Unified logging layer
   - 500+ plugins for inputs/outputs
   - CNCF graduated project

6. **GoAccess** - Real-time log analyzer (https://goaccess.io/)
   - Terminal-based real-time viewer
   - Generates HTML reports

7. **Logstash** - Log processing (https://www.elastic.co/logstash)
   - Part of ELK stack (Apache 2.0 licensed)
   - Powerful filtering and transformation

Quick Local Analysis:
--------------------
```bash
# Count logs by level
cat app.log | jq -s 'group_by(.level) | map({level: .[0].level, count: length})'

# Find slow operations (if duration tracked)
cat app.log | jq 'select(.context.duration_ms > 1000)'

# Get all unique error messages
cat app.log | jq -s '[.[] | select(.level == "ERROR") | .message] | unique'

# Pretty print with timestamps
cat app.log | jq -r '"\(.timestamp) [\(.level)] \(.message)"'
```

Environment Variables:
---------------------
- CLAUDE_CODE_LOG_LEVEL: Set log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
- CLAUDE_CODE_LOG_FORMAT: Set format (json, text) - default: json
- CLAUDE_CODE_LOG_FILE: Optional file path for log output

Example:
    ```python
    from claude_code_provider import configure_logging, get_logger

    # Auto-configure from environment (recommended for production)
    configure_logging()

    # Or configure explicitly
    configure_logging(level="DEBUG", format="text")  # For development

    # Get a logger and use it
    logger = get_logger(__name__)
    logger.info("Operation started", operation="create_agent", model="sonnet")
    ```
"""

import json
import logging
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, TextIO


# Environment variable names
ENV_LOG_LEVEL = "CLAUDE_CODE_LOG_LEVEL"
ENV_LOG_FORMAT = "CLAUDE_CODE_LOG_FORMAT"
ENV_LOG_FILE = "CLAUDE_CODE_LOG_FILE"

# Default configuration
DEFAULT_LOG_LEVEL = "INFO"
DEFAULT_LOG_FORMAT = "json"  # JSON is default for production


class LogLevel(str, Enum):
    """Log levels."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class LogFormat(str, Enum):
    """Log output formats."""
    JSON = "json"
    TEXT = "text"


@dataclass
class LogEntry:
    """A structured log entry.

    Attributes:
        level: Log level.
        message: Log message.
        timestamp: When the log was created.
        context: Additional context data.
        source: Source of the log (module/function).
    """
    level: LogLevel
    message: str
    timestamp: datetime = field(default_factory=datetime.now)
    context: dict[str, Any] = field(default_factory=dict)
    source: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "level": self.level.value,
            "message": self.message,
            "timestamp": self.timestamp.isoformat(),
            "context": self.context,
            "source": self.source,
        }

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict())


class StructuredFormatter(logging.Formatter):
    """Formatter that outputs structured JSON logs.

    Output format:
        {"level": "INFO", "message": "...", "timestamp": "...", "source": "...", "context": {...}}
    """

    def __init__(self, include_context: bool = True) -> None:
        super().__init__()
        self.include_context = include_context

    def format(self, record: logging.LogRecord) -> str:
        entry = {
            "level": record.levelname,
            "message": record.getMessage(),
            "timestamp": datetime.fromtimestamp(record.created).isoformat(),
            "source": f"{record.name}:{record.funcName}:{record.lineno}",
        }

        # Include exception info if present
        if record.exc_info:
            entry["exception"] = self.formatException(record.exc_info)

        # Include extra context
        if self.include_context:
            context = {}
            for key, value in record.__dict__.items():
                if key not in (
                    "name", "msg", "args", "created", "filename",
                    "funcName", "levelname", "levelno", "lineno",
                    "module", "msecs", "pathname", "process",
                    "processName", "relativeCreated", "stack_info",
                    "exc_info", "exc_text", "thread", "threadName",
                    "message", "taskName",
                ):
                    try:
                        json.dumps(value)  # Check if serializable
                        context[key] = value
                    except (TypeError, ValueError):
                        context[key] = str(value)

            if context:
                entry["context"] = context

        return json.dumps(entry, ensure_ascii=False)


class ColoredFormatter(logging.Formatter):
    """Formatter with colored output for terminals (development use)."""

    COLORS = {
        "DEBUG": "\033[36m",     # Cyan
        "INFO": "\033[32m",      # Green
        "WARNING": "\033[33m",   # Yellow
        "ERROR": "\033[31m",     # Red
        "CRITICAL": "\033[35m",  # Magenta
    }
    RESET = "\033[0m"

    def __init__(self, include_time: bool = True) -> None:
        super().__init__()
        self.include_time = include_time

    def format(self, record: logging.LogRecord) -> str:
        color = self.COLORS.get(record.levelname, "")
        reset = self.RESET

        message = record.getMessage()

        # Append context if present
        context_parts = []
        for key, value in record.__dict__.items():
            if key not in (
                "name", "msg", "args", "created", "filename",
                "funcName", "levelname", "levelno", "lineno",
                "module", "msecs", "pathname", "process",
                "processName", "relativeCreated", "stack_info",
                "exc_info", "exc_text", "thread", "threadName",
                "message", "taskName",
            ):
                context_parts.append(f"{key}={value}")

        if context_parts:
            message = f"{message} [{', '.join(context_parts)}]"

        if self.include_time:
            time_str = datetime.fromtimestamp(record.created).strftime("%H:%M:%S")
            result = f"{color}[{time_str}] {record.levelname:8}{reset} {message}"
        else:
            result = f"{color}{record.levelname:8}{reset} {message}"

        # Add exception info if present
        if record.exc_info:
            result += f"\n{self.formatException(record.exc_info)}"

        return result


def configure_logging(
    level: str | None = None,
    format: str | None = None,
    log_file: str | Path | None = None,
    stream: TextIO | None = None,
) -> logging.Logger:
    """Configure logging for Claude Code Provider.

    This function configures the root logger for the claude_code_provider package.
    By default, it uses JSON format for production environments.

    Configuration priority:
    1. Function arguments (if provided)
    2. Environment variables
    3. Defaults (INFO level, JSON format)

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
               Env: CLAUDE_CODE_LOG_LEVEL
        format: Output format ('json' or 'text').
                Env: CLAUDE_CODE_LOG_FORMAT
        log_file: Optional file path to write logs to.
                  Env: CLAUDE_CODE_LOG_FILE
        stream: Output stream for console logging (default: stderr).

    Returns:
        Configured logger instance.

    Example:
        ```python
        # Production (JSON to stderr)
        configure_logging()

        # Development (colored text)
        configure_logging(level="DEBUG", format="text")

        # Production with file output
        configure_logging(log_file="/var/log/claude-code.log")
        ```
    """
    # Resolve configuration with priority: args > env > defaults
    resolved_level = (
        level or
        os.environ.get(ENV_LOG_LEVEL) or
        DEFAULT_LOG_LEVEL
    ).upper()

    resolved_format = (
        format or
        os.environ.get(ENV_LOG_FORMAT) or
        DEFAULT_LOG_FORMAT
    ).lower()

    resolved_file = (
        log_file or
        os.environ.get(ENV_LOG_FILE)
    )

    # Get or create logger
    logger = logging.getLogger("claude_code_provider")
    logger.setLevel(getattr(logging, resolved_level))

    # Remove existing handlers
    logger.handlers.clear()

    # Create formatter based on format type
    if resolved_format == "json":
        formatter = StructuredFormatter()
    else:
        formatter = ColoredFormatter(include_time=True)

    # Add console handler
    console_handler = logging.StreamHandler(stream or sys.stderr)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # Add file handler if specified
    if resolved_file:
        file_path = Path(resolved_file)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(file_path)
        # Always use JSON for file output
        file_handler.setFormatter(StructuredFormatter())
        logger.addHandler(file_handler)

    # Prevent propagation to root logger
    logger.propagate = False

    return logger


class StructuredLogger:
    """Logger wrapper that supports structured context as keyword arguments.

    This wrapper allows passing context directly as keyword arguments instead
    of using the `extra` parameter.

    Example:
        ```python
        logger = get_logger(__name__)
        logger.info("Request completed", model="sonnet", duration_ms=150)
        ```
    """

    def __init__(self, logger: logging.Logger) -> None:
        self._logger = logger

    @property
    def name(self) -> str:
        return self._logger.name

    def _log(self, level: int, message: str, **context: Any) -> None:
        """Internal logging method with context support."""
        if context:
            self._logger.log(level, message, extra=context)
        else:
            self._logger.log(level, message)

    def debug(self, message: str, **context: Any) -> None:
        """Log debug message with optional context."""
        self._log(logging.DEBUG, message, **context)

    def info(self, message: str, **context: Any) -> None:
        """Log info message with optional context."""
        self._log(logging.INFO, message, **context)

    def warning(self, message: str, **context: Any) -> None:
        """Log warning message with optional context."""
        self._log(logging.WARNING, message, **context)

    def error(self, message: str, **context: Any) -> None:
        """Log error message with optional context."""
        self._log(logging.ERROR, message, **context)

    def critical(self, message: str, **context: Any) -> None:
        """Log critical message with optional context."""
        self._log(logging.CRITICAL, message, **context)

    def exception(self, message: str, **context: Any) -> None:
        """Log exception with traceback and optional context."""
        if context:
            self._logger.exception(message, extra=context)
        else:
            self._logger.exception(message)

    def setLevel(self, level: int | str) -> None:
        """Set logging level."""
        self._logger.setLevel(level)


def get_logger(name: str | None = None) -> StructuredLogger:
    """Get a structured logger instance for the given name.

    Returns a StructuredLogger wrapper that supports context as keyword arguments.

    Args:
        name: Logger name. If None, returns the root claude_code_provider logger.
              If provided, returns a child logger.

    Returns:
        StructuredLogger instance.

    Example:
        ```python
        logger = get_logger(__name__)
        logger.info("Operation completed", duration_ms=150, model="sonnet")
        ```
    """
    if name is None:
        base_logger = logging.getLogger("claude_code_provider")
    elif name.startswith("claude_code_provider"):
        base_logger = logging.getLogger(name)
    else:
        base_logger = logging.getLogger(f"claude_code_provider.{name}")

    return StructuredLogger(base_logger)


class DebugLogger:
    """Enhanced logger for debugging Claude Code Provider.

    Example:
        ```python
        # Setup logging (JSON is default)
        logger = DebugLogger.setup(level="DEBUG", json_output=True)

        # Log with context
        logger.info("Request started", model="sonnet", tokens=500)

        # Log CLI execution
        logger.debug_cli_call(["claude", "-p", "hello"], {"timeout": 30})

        # Log response
        logger.debug_response({"result": "Hello!", "usage": {...}})
        ```
    """

    def __init__(
        self,
        name: str = "claude_code_provider",
        level: LogLevel | str = LogLevel.INFO,
    ) -> None:
        """Initialize debug logger.

        Args:
            name: Logger name.
            level: Log level.
        """
        self._logger = logging.getLogger(name)
        self._set_level(level)
        self._entries: list[LogEntry] = []
        self._capture_entries = False

    def _set_level(self, level: LogLevel | str) -> None:
        """Set the log level."""
        if isinstance(level, LogLevel):
            level_str = level.value
        else:
            level_str = level.upper()

        self._logger.setLevel(getattr(logging, level_str))

    @classmethod
    def setup(
        cls,
        level: str = "INFO",
        json_output: bool = True,  # JSON is now default
        stream: TextIO | None = None,
        include_time: bool = True,
    ) -> "DebugLogger":
        """Setup logging for Claude Code Provider.

        Args:
            level: Log level (DEBUG, INFO, WARNING, ERROR).
            json_output: Whether to output structured JSON (default: True).
            stream: Output stream (default: stderr).
            include_time: Whether to include timestamps.

        Returns:
            Configured DebugLogger instance.
        """
        format_type = "json" if json_output else "text"
        configure_logging(level=level, format=format_type, stream=stream)
        return cls(level=level)

    def start_capture(self) -> None:
        """Start capturing log entries."""
        self._capture_entries = True
        self._entries = []

    def stop_capture(self) -> list[LogEntry]:
        """Stop capturing and return entries."""
        self._capture_entries = False
        entries = self._entries
        self._entries = []
        return entries

    def _log(
        self,
        level: LogLevel,
        message: str,
        **context: Any,
    ) -> None:
        """Internal logging method."""
        if self._capture_entries:
            self._entries.append(LogEntry(
                level=level,
                message=message,
                context=context,
                source=self._logger.name,
            ))

        log_func = getattr(self._logger, level.value.lower())
        if context:
            log_func(message, extra=context)
        else:
            log_func(message)

    def debug(self, message: str, **context: Any) -> None:
        """Log debug message."""
        self._log(LogLevel.DEBUG, message, **context)

    def info(self, message: str, **context: Any) -> None:
        """Log info message."""
        self._log(LogLevel.INFO, message, **context)

    def warning(self, message: str, **context: Any) -> None:
        """Log warning message."""
        self._log(LogLevel.WARNING, message, **context)

    def error(self, message: str, **context: Any) -> None:
        """Log error message."""
        self._log(LogLevel.ERROR, message, **context)

    def critical(self, message: str, **context: Any) -> None:
        """Log critical message."""
        self._log(LogLevel.CRITICAL, message, **context)

    def debug_cli_call(
        self,
        args: list[str],
        options: dict[str, Any] | None = None,
    ) -> None:
        """Log a CLI call for debugging.

        Args:
            args: CLI arguments.
            options: Additional options.
        """
        # Mask sensitive data in args
        masked_args = []
        for i, arg in enumerate(args):
            if i > 0 and args[i - 1] in ("--system-prompt", "-p"):
                # Truncate long prompts
                if len(arg) > 100:
                    masked_args.append(f"{arg[:100]}... ({len(arg)} chars)")
                else:
                    masked_args.append(arg)
            else:
                masked_args.append(arg)

        self.debug(
            "CLI call",
            command=" ".join(masked_args[:3]) + "...",
            full_args=masked_args,
            options=options,
        )

    def debug_response(
        self,
        response: dict[str, Any],
        include_content: bool = False,
    ) -> None:
        """Log a response for debugging.

        Args:
            response: Response data.
            include_content: Whether to include full content.
        """
        summary = {
            "success": response.get("success", not response.get("is_error")),
            "session_id": response.get("session_id"),
        }

        if "usage" in response:
            summary["usage"] = response["usage"]

        if include_content and "result" in response:
            result = response["result"]
            if len(result) > 500:
                summary["result_preview"] = result[:500] + "..."
            else:
                summary["result"] = result

        self.debug("CLI response", **summary)

    def debug_request(
        self,
        prompt: str,
        model: str | None = None,
        session_id: str | None = None,
    ) -> None:
        """Log a request for debugging.

        Args:
            prompt: The prompt.
            model: Model used.
            session_id: Session ID.
        """
        preview = prompt[:100] + "..." if len(prompt) > 100 else prompt
        self.debug(
            "Request",
            prompt_preview=preview,
            prompt_length=len(prompt),
            model=model,
            session_id=session_id,
        )


def setup_logging(
    level: str = "INFO",
    json_output: bool = True,  # JSON is now default
) -> DebugLogger:
    """Convenience function to setup logging.

    Args:
        level: Log level.
        json_output: Whether to use JSON output (default: True).

    Returns:
        Configured DebugLogger.
    """
    return DebugLogger.setup(level=level, json_output=json_output)
