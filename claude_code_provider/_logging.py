# Copyright (c) 2025. Claude Code Provider for Microsoft Agent Framework.

"""Structured logging and debugging utilities."""

import json
import logging
import sys
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, TextIO


class LogLevel(str, Enum):
    """Log levels."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


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
    """Formatter that outputs structured JSON logs."""

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

        return json.dumps(entry)


class ColoredFormatter(logging.Formatter):
    """Formatter with colored output for terminals."""

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

        if self.include_time:
            time_str = datetime.fromtimestamp(record.created).strftime("%H:%M:%S")
            return f"{color}[{time_str}] {record.levelname:8}{reset} {record.getMessage()}"
        else:
            return f"{color}{record.levelname:8}{reset} {record.getMessage()}"


class DebugLogger:
    """Enhanced logger for debugging Claude Code Provider.

    Example:
        ```python
        # Setup logging
        logger = DebugLogger.setup(level="DEBUG", json_output=False)

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
        json_output: bool = False,
        stream: TextIO | None = None,
        include_time: bool = True,
    ) -> "DebugLogger":
        """Setup logging for Claude Code Provider.

        Args:
            level: Log level (DEBUG, INFO, WARNING, ERROR).
            json_output: Whether to output structured JSON.
            stream: Output stream (default: stderr).
            include_time: Whether to include timestamps.

        Returns:
            Configured DebugLogger instance.
        """
        logger = logging.getLogger("claude_code_provider")
        logger.setLevel(getattr(logging, level.upper()))

        # Remove existing handlers
        logger.handlers.clear()

        # Create handler
        handler = logging.StreamHandler(stream or sys.stderr)

        # Set formatter
        if json_output:
            handler.setFormatter(StructuredFormatter())
        else:
            handler.setFormatter(ColoredFormatter(include_time=include_time))

        logger.addHandler(handler)

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
    json_output: bool = False,
) -> DebugLogger:
    """Convenience function to setup logging.

    Args:
        level: Log level.
        json_output: Whether to use JSON output.

    Returns:
        Configured DebugLogger.
    """
    return DebugLogger.setup(level=level, json_output=json_output)
