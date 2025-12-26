# Copyright (c) 2025. Claude Code Provider for Microsoft Agent Framework.

"""MCP (Model Context Protocol) server management."""

import asyncio
import ipaddress
import json
import logging
import re
import unicodedata
from dataclasses import dataclass, field
from enum import Enum
from typing import Any
from urllib.parse import urlparse

logger = logging.getLogger("claude_code_provider")


# Validation patterns for MCP server configuration
_SERVER_NAME_PATTERN = re.compile(r'^[a-zA-Z][a-zA-Z0-9_-]*$')
_ENV_VAR_NAME_PATTERN = re.compile(r'^[A-Za-z_][A-Za-z0-9_]*$')
# Dangerous shell metacharacters that should not appear in commands/args
# Includes: shell operators, quotes, escapes, wildcards, whitespace controls, null byte
#
# DESIGN DECISION: The '/' character is intentionally NOT included.
# Rationale: Forward slashes are required for file paths (e.g., "npx /path/to/server").
# Security: This is safe because we use subprocess.exec() (not shell=True), so arguments
# are passed directly to the executable without shell interpretation.
# Reviewed: 2025-01 - Not a security vulnerability, intentional design choice.
_DANGEROUS_CHARS = set(';&|`$(){}[]<>!#\n\r\t\'"\\*?~\0')

# Hostnames that should be blocked for SSRF prevention
_BLOCKED_HOSTNAMES = frozenset({
    'localhost',
    'localhost.localdomain',
    'ip6-localhost',
    'ip6-loopback',
})


def _is_private_ip(ip_str: str) -> bool:
    """Check if an IP address is private/internal.

    Args:
        ip_str: IP address string.

    Returns:
        True if the IP is private/internal, False otherwise.
    """
    try:
        ip = ipaddress.ip_address(ip_str)
        return (
            ip.is_private
            or ip.is_loopback
            or ip.is_link_local
            or ip.is_reserved
            or ip.is_multicast
        )
    except ValueError:
        return False


def _validate_mcp_url(url: str) -> str:
    """Validate an MCP server URL for SSRF prevention.

    Args:
        url: URL to validate.

    Returns:
        The validated URL.

    Raises:
        ValueError: If URL is invalid or points to internal resources.
    """
    if not isinstance(url, str) or not url:
        raise ValueError("MCP URL must be a non-empty string")

    try:
        parsed = urlparse(url)
    except Exception as e:
        raise ValueError(f"Invalid URL format: {e}")

    # Check scheme
    if parsed.scheme not in ('http', 'https'):
        raise ValueError(
            f"MCP URL must use http or https scheme, got: {parsed.scheme}"
        )

    # Check for blocked hostnames
    hostname = parsed.hostname or ''
    hostname_lower = hostname.lower()

    if hostname_lower in _BLOCKED_HOSTNAMES:
        raise ValueError(
            f"MCP URL cannot point to localhost or loopback: {hostname}"
        )

    # Check for private/internal IP addresses
    if _is_private_ip(hostname):
        raise ValueError(
            f"MCP URL cannot point to private/internal IP address: {hostname}"
        )

    # Check for embedded credentials (potential security issue)
    if parsed.username or parsed.password:
        logger.warning(
            f"MCP URL contains embedded credentials for server. "
            "Consider using environment variables instead for security."
        )

    return url


def _validate_server_name(name: str) -> str:
    """Validate MCP server name.

    Args:
        name: Server name to validate.

    Returns:
        The validated name.

    Raises:
        ValueError: If name is invalid.
    """
    if not name or not isinstance(name, str):
        raise ValueError("Server name must be a non-empty string")
    if len(name) > 64:
        raise ValueError("Server name must be at most 64 characters")
    if not _SERVER_NAME_PATTERN.match(name):
        raise ValueError(
            f"Server name '{name}' is invalid. Must start with a letter and "
            "contain only alphanumeric characters, underscores, or hyphens."
        )
    return name


def _validate_env_var_name(name: str) -> str:
    """Validate environment variable name.

    Args:
        name: Environment variable name to validate.

    Returns:
        The validated name.

    Raises:
        ValueError: If name is invalid.
    """
    if not name or not isinstance(name, str):
        raise ValueError("Environment variable name must be a non-empty string")
    if not _ENV_VAR_NAME_PATTERN.match(name):
        raise ValueError(
            f"Environment variable name '{name}' is invalid. Must start with "
            "a letter or underscore and contain only alphanumeric characters or underscores."
        )
    return name


def _validate_command_or_arg(value: str, field_name: str = "value") -> str:
    """Validate a command or argument for dangerous characters.

    Performs Unicode NFKC normalization before validation to prevent
    homoglyph bypass attacks (e.g., fullwidth semicolon U+FF1B → ASCII ;).

    Args:
        value: The value to validate.
        field_name: Name of the field for error messages.

    Returns:
        The validated (and normalized) value.

    Raises:
        ValueError: If value contains dangerous characters.
    """
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be a string")

    # Normalize Unicode to NFKC to prevent homoglyph bypass attacks
    # E.g., fullwidth semicolon (U+FF1B) normalizes to ASCII semicolon (U+003B)
    normalized = unicodedata.normalize('NFKC', value)

    dangerous_found = _DANGEROUS_CHARS.intersection(normalized)
    if dangerous_found:
        # Show readable representation of dangerous chars found
        char_repr = ', '.join(repr(c) for c in dangerous_found)
        raise ValueError(
            f"{field_name} contains dangerous characters: {char_repr}. "
            "Shell metacharacters are not allowed for security reasons."
        )
    return normalized


class MCPTransport(str, Enum):
    """MCP transport types."""
    STDIO = "stdio"
    HTTP = "http"
    SSE = "sse"


@dataclass
class MCPServer:
    """Configuration for an MCP server.

    Attributes:
        name: Unique name for the server.
        command_or_url: Command (for stdio) or URL (for http/sse).
        transport: Transport type (stdio, http, sse).
        args: Additional arguments for stdio commands.
        env: Environment variables to set.

    Security Note:
        Environment variables passed via `env` may be visible in process
        listings (e.g., `ps aux`). For sensitive values like API keys,
        consider configuring MCP servers directly in Claude Code's settings
        file (`~/.claude/settings.json`) instead of programmatically.
    """
    name: str
    command_or_url: str
    transport: MCPTransport = MCPTransport.STDIO
    args: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate all fields after initialization."""
        # Validate server name
        _validate_server_name(self.name)

        # For stdio transport, validate command and args for dangerous characters
        if self.transport == MCPTransport.STDIO:
            _validate_command_or_arg(self.command_or_url, "command")
            for i, arg in enumerate(self.args):
                _validate_command_or_arg(arg, f"args[{i}]")
        else:
            # For HTTP/SSE transport, validate URL for SSRF prevention
            _validate_mcp_url(self.command_or_url)

        # Validate environment variable names
        for env_name in self.env.keys():
            _validate_env_var_name(env_name)

    def to_cli_args(self, include_env: bool = True) -> list[str]:
        """Convert to CLI arguments for --mcp-config.

        Args:
            include_env: If True, include env vars in config JSON.
                Set to False and use get_subprocess_env() for more secure
                handling of sensitive environment variables.

        Returns:
            CLI arguments for --mcp-config.
        """
        config = {
            "name": self.name,
            "transport": self.transport.value,
        }

        if self.transport == MCPTransport.STDIO:
            config["command"] = self.command_or_url
            if self.args:
                config["args"] = self.args
            if self.env and include_env:
                config["env"] = self.env
        else:
            config["url"] = self.command_or_url

        return ["--mcp-config", json.dumps(config)]

    def get_subprocess_env(self) -> dict[str, str]:
        """Get environment variables for subprocess execution.

        This is more secure than passing env vars through CLI arguments,
        as subprocess environment is not visible in process listings.

        Returns:
            Dictionary of environment variables to pass to subprocess.
        """
        import os
        # Start with current environment and add server-specific vars
        env = os.environ.copy()
        env.update(self.env)
        return env

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "command_or_url": self.command_or_url,
            "transport": self.transport.value,
            "args": self.args,
            "env": self.env,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MCPServer":
        """Create from dictionary."""
        return cls(
            name=data["name"],
            command_or_url=data["command_or_url"],
            transport=MCPTransport(data.get("transport", "stdio")),
            args=data.get("args", []),
            env=data.get("env", {}),
        )


@dataclass
class MCPServerInfo:
    """Information about a configured MCP server."""
    name: str
    transport: str
    status: str
    tools: list[str] = field(default_factory=list)
    resources: list[str] = field(default_factory=list)


class MCPManager:
    """Manager for MCP server connections.

    Example:
        ```python
        manager = MCPManager(cli_path="claude")

        # Add a server
        server = MCPServer(
            name="my-tool",
            command_or_url="npx",
            args=["-y", "my-mcp-server"],
            env={"API_KEY": "secret"},
        )
        await manager.add_server(server)

        # List servers
        servers = await manager.list_servers()

        # Get CLI args for a session
        args = manager.get_cli_args([server])
        ```
    """

    def __init__(self, cli_path: str = "claude") -> None:
        """Initialize MCP manager.

        Args:
            cli_path: Path to the claude CLI.
        """
        self.cli_path = cli_path
        self._servers: dict[str, MCPServer] = {}

    def add_server(self, server: MCPServer) -> None:
        """Add an MCP server configuration.

        Args:
            server: The MCP server configuration.
        """
        self._servers[server.name] = server
        logger.info(f"Added MCP server: {server.name}")

    def remove_server(self, name: str) -> bool:
        """Remove an MCP server configuration.

        Args:
            name: Name of the server to remove.

        Returns:
            True if removed, False if not found.
        """
        if name in self._servers:
            del self._servers[name]
            logger.info(f"Removed MCP server: {name}")
            return True
        return False

    def get_server(self, name: str) -> MCPServer | None:
        """Get an MCP server configuration by name.

        Args:
            name: Name of the server.

        Returns:
            The server configuration or None.
        """
        return self._servers.get(name)

    def get_servers(self) -> list[MCPServer]:
        """Get all configured MCP servers.

        Returns:
            List of MCP server configurations.
        """
        return list(self._servers.values())

    def get_cli_args(self, servers: list[MCPServer] | None = None) -> list[str]:
        """Get CLI arguments for MCP configuration.

        Args:
            servers: Specific servers to include. None means all.

        Returns:
            CLI arguments for --mcp-config.
        """
        target_servers = servers if servers is not None else self.get_servers()

        args = []
        for server in target_servers:
            args.extend(server.to_cli_args())

        return args

    async def list_configured_servers(self) -> list[MCPServerInfo]:
        """List MCP servers configured in Claude Code.

        Returns:
            List of configured server information.
        """
        try:
            process = await asyncio.create_subprocess_exec(
                self.cli_path, "mcp", "list",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, _ = await process.communicate()

            if process.returncode != 0:
                logger.warning("Failed to list MCP servers")
                return []

            # Parse output (format varies, handle gracefully)
            output = stdout.decode().strip()
            servers = []

            for line in output.split("\n"):
                if line and not line.startswith(("─", "│", "╭", "╯", "╮", "╰")):
                    parts = line.split()
                    if parts:
                        servers.append(MCPServerInfo(
                            name=parts[0],
                            transport=parts[1] if len(parts) > 1 else "unknown",
                            status=parts[2] if len(parts) > 2 else "unknown",
                        ))

            return servers

        except Exception as e:
            logger.error(f"Error listing MCP servers: {e}")
            return []

    async def add_server_to_claude(self, server: MCPServer) -> bool:
        """Add an MCP server to Claude Code configuration.

        Args:
            server: The server to add.

        Returns:
            True if successful.

        Security Note:
            Environment variables in server.env will be passed as CLI
            arguments and may be visible in process listings. For sensitive
            values, configure MCP servers directly in Claude Code's settings.
        """
        try:
            cmd = [
                self.cli_path, "mcp", "add",
                "--transport", server.transport.value,
                server.name, server.command_or_url,
            ]

            # Add env vars (warning: visible in process listings)
            for key, value in server.env.items():
                cmd.extend(["--env", f"{key}={value}"])

            # Add args
            if server.args:
                cmd.append("--")
                cmd.extend(server.args)

            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            _, stderr = await process.communicate()

            if process.returncode != 0:
                logger.error(f"Failed to add MCP server: {stderr.decode()}")
                return False

            self._servers[server.name] = server
            logger.info(f"Added MCP server to Claude: {server.name}")
            return True

        except Exception as e:
            logger.error(f"Error adding MCP server: {e}")
            return False

    async def remove_server_from_claude(self, name: str) -> bool:
        """Remove an MCP server from Claude Code configuration.

        Args:
            name: Name of the server to remove.

        Returns:
            True if successful.
        """
        try:
            process = await asyncio.create_subprocess_exec(
                self.cli_path, "mcp", "remove", name,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            _, stderr = await process.communicate()

            if process.returncode != 0:
                logger.error(f"Failed to remove MCP server: {stderr.decode()}")
                return False

            if name in self._servers:
                del self._servers[name]
            logger.info(f"Removed MCP server from Claude: {name}")
            return True

        except Exception as e:
            logger.error(f"Error removing MCP server: {e}")
            return False

    async def get_server_details(self, name: str) -> dict[str, Any] | None:
        """Get details about a specific MCP server.

        Args:
            name: Name of the server.

        Returns:
            Server details or None.
        """
        try:
            process = await asyncio.create_subprocess_exec(
                self.cli_path, "mcp", "get", name,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, _ = await process.communicate()

            if process.returncode != 0:
                return None

            # Try to parse as JSON
            try:
                return json.loads(stdout.decode())
            except json.JSONDecodeError:
                return {"raw_output": stdout.decode()}

        except Exception as e:
            logger.error(f"Error getting MCP server details: {e}")
            return None
