# Copyright (c) 2025. Claude Code Provider for Microsoft Agent Framework.

"""Session management for Claude Code conversations."""

import asyncio
import json
import logging
import os
import stat
import threading
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger("claude_code_provider")


# Directories that should not be written to for security
_FORBIDDEN_PATHS = frozenset({
    "/etc", "/bin", "/sbin", "/usr/bin", "/usr/sbin",
    "/boot", "/dev", "/proc", "/sys", "/var/run",
    "/lib", "/lib64", "/usr/lib", "/usr/lib64",
})


def _validate_export_path(path: Path) -> Path:
    """Validate that an export path is safe to write to.

    Args:
        path: Path to validate.

    Returns:
        Validated absolute path.

    Raises:
        ValueError: If path is invalid or unsafe.
    """
    # Resolve to absolute path to catch traversal attempts
    resolved = path.resolve()

    # Check against forbidden directories
    path_str = str(resolved)
    for forbidden in _FORBIDDEN_PATHS:
        if path_str == forbidden or path_str.startswith(forbidden + "/"):
            raise ValueError(
                f"Cannot write to system directory: {forbidden}"
            )

    # Ensure parent directory exists or can be created
    parent = resolved.parent
    if not parent.exists():
        # Check parent path too before creating
        parent_str = str(parent)
        for forbidden in _FORBIDDEN_PATHS:
            if parent_str == forbidden or parent_str.startswith(forbidden + "/"):
                raise ValueError(
                    f"Cannot create directory in system path: {forbidden}"
                )

    return resolved


@dataclass
class SessionInfo:
    """Information about a session.

    Attributes:
        session_id: Unique session identifier.
        created_at: When the session was created.
        last_used: When the session was last used.
        message_count: Number of messages in the session.
        model: Model used in the session.
        metadata: Additional metadata.
    """
    session_id: str
    created_at: datetime = field(default_factory=datetime.now)
    last_used: datetime = field(default_factory=datetime.now)
    message_count: int = 0
    model: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "session_id": self.session_id,
            "created_at": self.created_at.isoformat(),
            "last_used": self.last_used.isoformat(),
            "message_count": self.message_count,
            "model": self.model,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SessionInfo":
        """Create from dictionary."""
        return cls(
            session_id=data["session_id"],
            created_at=datetime.fromisoformat(data["created_at"]),
            last_used=datetime.fromisoformat(data["last_used"]),
            message_count=data.get("message_count", 0),
            model=data.get("model"),
            metadata=data.get("metadata", {}),
        )


@dataclass
class SessionExport:
    """Exported session data.

    Attributes:
        session_id: Session identifier.
        messages: Conversation messages.
        metadata: Session metadata.
        exported_at: When the export was created.
    """
    session_id: str
    messages: list[dict[str, Any]]
    metadata: dict[str, Any]
    exported_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "session_id": self.session_id,
            "messages": self.messages,
            "metadata": self.metadata,
            "exported_at": self.exported_at.isoformat(),
        }

    def to_json(self, indent: int = 2) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)

    def save(self, path: str | Path) -> None:
        """Save to file.

        Args:
            path: File path to save to.

        Raises:
            ValueError: If path is to a system directory.
        """
        path = Path(path)
        # Validate path to prevent traversal to sensitive directories
        validated_path = _validate_export_path(path)
        validated_path.write_text(self.to_json())
        logger.info(f"Session exported to {validated_path}")

    @classmethod
    def load(cls, path: str | Path) -> "SessionExport":
        """Load from file.

        Args:
            path: File path to load from.

        Returns:
            Loaded SessionExport.
        """
        path = Path(path)
        data = json.loads(path.read_text())
        return cls(
            session_id=data["session_id"],
            messages=data["messages"],
            metadata=data["metadata"],
            exported_at=datetime.fromisoformat(data["exported_at"]),
        )


class SessionManager:
    """Manager for Claude Code sessions.

    Example:
        ```python
        manager = SessionManager()

        # Track a session
        manager.track_session("session-123", model="sonnet")

        # List sessions
        sessions = manager.list_sessions()

        # Export a session
        export = await manager.export_session("session-123")
        export.save("session_backup.json")

        # Clean up old sessions
        await manager.cleanup_old_sessions(days=30)
        ```
    """

    def __init__(
        self,
        cli_path: str = "claude",
        storage_path: str | Path | None = None,
    ) -> None:
        """Initialize session manager.

        Args:
            cli_path: Path to the claude CLI.
            storage_path: Path for session metadata storage.
        """
        self.cli_path = cli_path
        self._sessions: dict[str, SessionInfo] = {}
        self._lock = threading.Lock()  # Protect concurrent session modifications

        # Default storage in home directory
        if storage_path is None:
            storage_path = Path.home() / ".claude-code-provider" / "sessions.json"
        self.storage_path = Path(storage_path)

        self._load_sessions()

    def _load_sessions(self) -> None:
        """Load sessions from storage."""
        if self.storage_path.exists():
            try:
                data = json.loads(self.storage_path.read_text())
                for session_data in data.get("sessions", []):
                    info = SessionInfo.from_dict(session_data)
                    self._sessions[info.session_id] = info
                logger.debug(f"Loaded {len(self._sessions)} sessions from storage")
            except Exception as e:
                logger.warning(f"Failed to load sessions: {e}")

    def _save_sessions(self) -> None:
        """Save sessions to storage with secure permissions.

        Uses atomic file creation with secure permissions to prevent TOCTOU
        race conditions (fixes #3). Raises exception on permission failure
        instead of silently continuing (fixes #6).
        """
        import tempfile

        try:
            # Create directory with secure permissions (owner only)
            parent_dir = self.storage_path.parent
            parent_dir.mkdir(parents=True, exist_ok=True)

            # Set directory permissions - log warning but don't fail
            # (directory may already have correct permissions)
            try:
                os.chmod(parent_dir, stat.S_IRWXU)  # 0o700
            except OSError as e:
                logger.debug(f"Could not set directory permissions: {e}")

            data = {
                "sessions": [s.to_dict() for s in self._sessions.values()],
                "updated_at": datetime.now().isoformat(),
            }
            json_content = json.dumps(data, indent=2)

            # Use atomic write with secure permissions from the start
            # This prevents TOCTOU race where file is briefly world-readable
            temp_fd = None
            temp_path = None
            try:
                # Create temp file with secure permissions (0o600) atomically
                temp_fd, temp_path = tempfile.mkstemp(
                    dir=str(parent_dir),
                    prefix='.sessions_',
                    suffix='.tmp'
                )
                # mkstemp creates with 0o600 by default, but ensure it
                os.fchmod(temp_fd, stat.S_IRUSR | stat.S_IWUSR)  # 0o600

                # Write content
                os.write(temp_fd, json_content.encode('utf-8'))
                os.fsync(temp_fd)  # Ensure data is on disk
                os.close(temp_fd)
                temp_fd = None

                # Verify permissions before rename
                file_stat = os.stat(temp_path)
                actual_mode = file_stat.st_mode & 0o777
                if actual_mode != 0o600:
                    raise PermissionError(
                        f"Session file has insecure permissions: {oct(actual_mode)} "
                        f"(expected 0o600). Cannot save session data securely."
                    )

                # Atomic rename to target
                os.replace(temp_path, str(self.storage_path))
                temp_path = None  # Successfully moved, don't cleanup

            finally:
                # Cleanup on failure
                if temp_fd is not None:
                    try:
                        os.close(temp_fd)
                    except OSError:
                        pass
                if temp_path is not None:
                    try:
                        os.unlink(temp_path)
                    except OSError:
                        pass

        except PermissionError:
            # Re-raise permission errors - these are security critical
            raise
        except Exception as e:
            logger.warning(f"Failed to save sessions: {e}")

    def track_session(
        self,
        session_id: str,
        model: str | None = None,
        **metadata: Any,
    ) -> SessionInfo:
        """Track a session.

        Args:
            session_id: Session identifier.
            model: Model used.
            **metadata: Additional metadata.

        Returns:
            SessionInfo for the tracked session.
        """
        with self._lock:
            if session_id in self._sessions:
                info = self._sessions[session_id]
                info.last_used = datetime.now()
                info.message_count += 1
                if model:
                    info.model = model
                info.metadata.update(metadata)
            else:
                info = SessionInfo(
                    session_id=session_id,
                    model=model,
                    metadata=metadata,
                )
                self._sessions[session_id] = info

            self._save_sessions()
            return info

    def get_session(self, session_id: str) -> SessionInfo | None:
        """Get session info.

        Args:
            session_id: Session identifier.

        Returns:
            SessionInfo or None if not found.
        """
        return self._sessions.get(session_id)

    def list_sessions(
        self,
        sort_by: str = "last_used",
        descending: bool = True,
    ) -> list[SessionInfo]:
        """List all tracked sessions.

        Args:
            sort_by: Field to sort by (last_used, created_at, message_count).
            descending: Sort in descending order.

        Returns:
            List of SessionInfo objects.
        """
        sessions = list(self._sessions.values())

        if sort_by == "last_used":
            sessions.sort(key=lambda s: s.last_used, reverse=descending)
        elif sort_by == "created_at":
            sessions.sort(key=lambda s: s.created_at, reverse=descending)
        elif sort_by == "message_count":
            sessions.sort(key=lambda s: s.message_count, reverse=descending)

        return sessions

    def delete_session(self, session_id: str) -> bool:
        """Delete a session from tracking.

        Args:
            session_id: Session identifier.

        Returns:
            True if deleted, False if not found.
        """
        if session_id in self._sessions:
            del self._sessions[session_id]
            self._save_sessions()
            logger.info(f"Deleted session: {session_id}")
            return True
        return False

    async def export_session(
        self,
        session_id: str,
        include_system: bool = True,
    ) -> SessionExport | None:
        """Export a session's conversation history.

        Note: This requires the session to still exist in Claude Code's storage.

        Args:
            session_id: Session identifier.
            include_system: Whether to include system messages.

        Returns:
            SessionExport or None if not found.
        """
        # Try to get session data from Claude Code
        # Note: Claude Code doesn't have a direct export command,
        # so we store messages as we see them

        info = self._sessions.get(session_id)
        if not info:
            return None

        # For now, we can only export metadata we've tracked
        # Full conversation export would require Claude Code CLI support
        return SessionExport(
            session_id=session_id,
            messages=[],  # Would need CLI support
            metadata={
                "model": info.model,
                "message_count": info.message_count,
                "created_at": info.created_at.isoformat(),
                "last_used": info.last_used.isoformat(),
                **info.metadata,
            },
        )

    async def cleanup_old_sessions(
        self,
        days: int = 30,
        dry_run: bool = False,
    ) -> list[str]:
        """Clean up sessions older than specified days.

        Args:
            days: Delete sessions older than this many days.
            dry_run: If True, only report what would be deleted.

        Returns:
            List of deleted (or would-be deleted) session IDs.
        """
        cutoff = datetime.now().timestamp() - (days * 24 * 60 * 60)
        to_delete = []

        for session_id, info in list(self._sessions.items()):
            if info.last_used.timestamp() < cutoff:
                to_delete.append(session_id)
                if not dry_run:
                    del self._sessions[session_id]

        if not dry_run and to_delete:
            self._save_sessions()
            logger.info(f"Cleaned up {len(to_delete)} old sessions")

        return to_delete

    def get_recent_sessions(self, limit: int = 10) -> list[SessionInfo]:
        """Get most recent sessions.

        Args:
            limit: Maximum number of sessions to return.

        Returns:
            List of recent SessionInfo objects.
        """
        return self.list_sessions(sort_by="last_used", descending=True)[:limit]

    def get_stats(self) -> dict[str, Any]:
        """Get session statistics.

        Returns:
            Dictionary with session statistics.
        """
        sessions = list(self._sessions.values())

        if not sessions:
            return {
                "total_sessions": 0,
                "total_messages": 0,
                "models_used": {},
            }

        models: dict[str, int] = {}
        total_messages = 0

        for session in sessions:
            total_messages += session.message_count
            if session.model:
                models[session.model] = models.get(session.model, 0) + 1

        return {
            "total_sessions": len(sessions),
            "total_messages": total_messages,
            "models_used": models,
            "oldest_session": min(s.created_at for s in sessions).isoformat(),
            "newest_session": max(s.created_at for s in sessions).isoformat(),
        }

    def clear_all(self) -> int:
        """Clear all tracked sessions.

        Returns:
            Number of sessions cleared.
        """
        count = len(self._sessions)
        self._sessions = {}
        self._save_sessions()
        logger.info(f"Cleared {count} sessions")
        return count
