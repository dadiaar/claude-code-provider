# Copyright (c) 2025. Claude Code Provider for Microsoft Agent Framework.

"""MCP (Model Context Protocol) server management."""

import asyncio
import json
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

logger = logging.getLogger("claude_code_provider")


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
    """
    name: str
    command_or_url: str
    transport: MCPTransport = MCPTransport.STDIO
    args: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)

    def to_cli_args(self) -> list[str]:
        """Convert to CLI arguments for --mcp-config."""
        config = {
            "name": self.name,
            "transport": self.transport.value,
        }

        if self.transport == MCPTransport.STDIO:
            config["command"] = self.command_or_url
            if self.args:
                config["args"] = self.args
            if self.env:
                config["env"] = self.env
        else:
            config["url"] = self.command_or_url

        return ["--mcp-config", json.dumps(config)]

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
        """
        try:
            cmd = [
                self.cli_path, "mcp", "add",
                "--transport", server.transport.value,
                server.name, server.command_or_url,
            ]

            # Add env vars
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
