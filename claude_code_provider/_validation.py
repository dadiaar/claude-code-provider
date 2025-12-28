# Copyright (c) 2025. Claude Code Provider for Microsoft Agent Framework.

"""Input validation utilities for public APIs.

This module provides validation functions for all public API parameters,
ensuring clear error messages for invalid inputs.
"""

from typing import Any, TypeVar

from ._exceptions import ClaudeCodeException

T = TypeVar("T")


class ValidationError(ClaudeCodeException):
    """Raised when input validation fails.

    Attributes:
        parameter: The parameter that failed validation.
        message: Description of the validation error.
    """

    def __init__(self, parameter: str, message: str) -> None:
        self.parameter = parameter
        self.message = message
        super().__init__(f"Invalid '{parameter}': {message}")


def validate_string(
    value: Any,
    parameter: str,
    *,
    min_length: int = 1,
    max_length: int | None = None,
    allow_none: bool = False,
) -> str | None:
    """Validate a string parameter.

    Args:
        value: The value to validate.
        parameter: Parameter name for error messages.
        min_length: Minimum allowed length (default: 1).
        max_length: Maximum allowed length (default: None = no limit).
        allow_none: Whether None is a valid value.

    Returns:
        The validated string (or None if allowed).

    Raises:
        ValidationError: If validation fails.
    """
    if value is None:
        if allow_none:
            return None
        raise ValidationError(parameter, "cannot be None")

    if not isinstance(value, str):
        raise ValidationError(
            parameter,
            f"must be a string, got {type(value).__name__}"
        )

    if len(value) < min_length:
        if min_length == 1:
            raise ValidationError(parameter, "cannot be empty")
        raise ValidationError(
            parameter,
            f"must be at least {min_length} characters"
        )

    if max_length is not None and len(value) > max_length:
        raise ValidationError(
            parameter,
            f"cannot exceed {max_length} characters (got {len(value)})"
        )

    return value


def validate_positive_int(
    value: Any,
    parameter: str,
    *,
    allow_none: bool = False,
    min_value: int = 1,
    max_value: int | None = None,
) -> int | None:
    """Validate a positive integer parameter.

    Args:
        value: The value to validate.
        parameter: Parameter name for error messages.
        allow_none: Whether None is a valid value.
        min_value: Minimum allowed value (default: 1).
        max_value: Maximum allowed value (default: None = no limit).

    Returns:
        The validated integer (or None if allowed).

    Raises:
        ValidationError: If validation fails.
    """
    if value is None:
        if allow_none:
            return None
        raise ValidationError(parameter, "cannot be None")

    if not isinstance(value, int) or isinstance(value, bool):
        raise ValidationError(
            parameter,
            f"must be an integer, got {type(value).__name__}"
        )

    if value < min_value:
        raise ValidationError(
            parameter,
            f"must be at least {min_value}, got {value}"
        )

    if max_value is not None and value > max_value:
        raise ValidationError(
            parameter,
            f"cannot exceed {max_value}, got {value}"
        )

    return value


def validate_positive_float(
    value: Any,
    parameter: str,
    *,
    allow_none: bool = False,
    min_value: float = 0.0,
    max_value: float | None = None,
    allow_zero: bool = False,
) -> float | None:
    """Validate a positive float parameter.

    Args:
        value: The value to validate.
        parameter: Parameter name for error messages.
        allow_none: Whether None is a valid value.
        min_value: Minimum allowed value (default: 0.0).
        max_value: Maximum allowed value (default: None = no limit).
        allow_zero: Whether zero is allowed (default: False).

    Returns:
        The validated float (or None if allowed).

    Raises:
        ValidationError: If validation fails.
    """
    if value is None:
        if allow_none:
            return None
        raise ValidationError(parameter, "cannot be None")

    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValidationError(
            parameter,
            f"must be a number, got {type(value).__name__}"
        )

    value = float(value)

    if not allow_zero and value == 0:
        raise ValidationError(parameter, "cannot be zero")

    if value < min_value:
        raise ValidationError(
            parameter,
            f"must be at least {min_value}, got {value}"
        )

    if max_value is not None and value > max_value:
        raise ValidationError(
            parameter,
            f"cannot exceed {max_value}, got {value}"
        )

    return value


def validate_list(
    value: Any,
    parameter: str,
    *,
    item_type: type | None = None,
    min_length: int = 0,
    max_length: int | None = None,
    allow_none: bool = False,
) -> list | None:
    """Validate a list parameter.

    Args:
        value: The value to validate.
        parameter: Parameter name for error messages.
        item_type: Expected type of items (default: None = any type).
        min_length: Minimum list length (default: 0).
        max_length: Maximum list length (default: None = no limit).
        allow_none: Whether None is a valid value.

    Returns:
        The validated list (or None if allowed).

    Raises:
        ValidationError: If validation fails.
    """
    if value is None:
        if allow_none:
            return None
        raise ValidationError(parameter, "cannot be None")

    if not isinstance(value, list):
        raise ValidationError(
            parameter,
            f"must be a list, got {type(value).__name__}"
        )

    if len(value) < min_length:
        if min_length == 1:
            raise ValidationError(parameter, "cannot be empty")
        raise ValidationError(
            parameter,
            f"must have at least {min_length} items"
        )

    if max_length is not None and len(value) > max_length:
        raise ValidationError(
            parameter,
            f"cannot have more than {max_length} items (got {len(value)})"
        )

    if item_type is not None:
        for i, item in enumerate(value):
            if not isinstance(item, item_type):
                raise ValidationError(
                    parameter,
                    f"item {i} must be {item_type.__name__}, got {type(item).__name__}"
                )

    return value


def validate_callable(
    value: Any,
    parameter: str,
    *,
    allow_none: bool = False,
) -> Any:
    """Validate that a value is callable.

    Args:
        value: The value to validate.
        parameter: Parameter name for error messages.
        allow_none: Whether None is a valid value.

    Returns:
        The validated callable (or None if allowed).

    Raises:
        ValidationError: If validation fails.
    """
    if value is None:
        if allow_none:
            return None
        raise ValidationError(parameter, "cannot be None")

    if not callable(value):
        raise ValidationError(
            parameter,
            f"must be callable, got {type(value).__name__}"
        )

    return value


def validate_agent(
    value: Any,
    parameter: str,
    *,
    allow_none: bool = False,
) -> Any:
    """Validate that a value is a ClaudeAgent.

    Args:
        value: The value to validate.
        parameter: Parameter name for error messages.
        allow_none: Whether None is a valid value.

    Returns:
        The validated agent (or None if allowed).

    Raises:
        ValidationError: If validation fails.
    """
    if value is None:
        if allow_none:
            return None
        raise ValidationError(parameter, "cannot be None")

    # Check for ClaudeAgent by looking for expected attributes
    # This avoids circular imports
    if not hasattr(value, "run") or not hasattr(value, "name"):
        raise ValidationError(
            parameter,
            f"must be a ClaudeAgent, got {type(value).__name__}"
        )

    return value


def validate_agents_list(
    value: Any,
    parameter: str,
    *,
    min_length: int = 1,
) -> list:
    """Validate a list of agents.

    Args:
        value: The value to validate.
        parameter: Parameter name for error messages.
        min_length: Minimum number of agents (default: 1).

    Returns:
        The validated list of agents.

    Raises:
        ValidationError: If validation fails.
    """
    if not isinstance(value, list):
        raise ValidationError(
            parameter,
            f"must be a list, got {type(value).__name__}"
        )

    if len(value) < min_length:
        raise ValidationError(
            parameter,
            f"must have at least {min_length} agent(s)"
        )

    for i, agent in enumerate(value):
        if not hasattr(agent, "run") or not hasattr(agent, "name"):
            raise ValidationError(
                parameter,
                f"item {i} must be a ClaudeAgent, got {type(agent).__name__}"
            )

    return value
