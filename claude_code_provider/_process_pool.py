# Copyright (c) 2025. Claude Code Provider for Microsoft Agent Framework.

"""Claude Process Pool - Unified pool with lazy spawning and LRU eviction.

A pool of persistent Claude CLI processes that:
- Spawns processes lazily on-demand
- Supports mixed models in the same pool
- Uses LRU eviction when a different model is needed
- Properly cleans up on exit (atexit + signal handlers)

Example:
    ```python
    from claude_code_provider import ClaudeProcessPool

    pool = ClaudeProcessPool(max_size=4)

    async with pool.acquire_context("haiku") as proc:
        response = await proc.send_message("Hello!")
        print(response)

    # Or manual acquire/release:
    proc = await pool.acquire("sonnet")
    try:
        response = await proc.send_message("Hello!")
    finally:
        pool.release(proc)

    # Cleanup when done
    await pool.shutdown()
    ```
"""

import asyncio
import atexit
import json
import os
import signal
import sys
import time
from dataclasses import dataclass, field
from typing import Any


@dataclass
class PersistentProcess:
    """A persistent Claude CLI process using bidirectional stream-json.

    This class wraps an asyncio subprocess running `claude -p` in stream-json
    mode, allowing multiple messages to be sent without respawning.

    Attributes:
        proc: The underlying asyncio subprocess.
        model: The Claude model this process uses.
        process_id: Unique identifier for this process.
        last_used: Timestamp of last use (for LRU eviction).
        in_use: Whether this process is currently acquired.
    """

    proc: asyncio.subprocess.Process
    model: str
    process_id: int
    last_used: float = field(default_factory=time.time)
    in_use: bool = False
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    async def send_message(self, content: str, timeout: float = 120.0) -> str:
        """Send a message and wait for response.

        Args:
            content: The message content to send.
            timeout: Maximum time to wait for response in seconds.

        Returns:
            The response text from Claude.

        Raises:
            RuntimeError: If the process has terminated.
            TimeoutError: If no response within timeout.
        """
        async with self._lock:
            if self.proc.returncode is not None:
                raise RuntimeError(f"Process {self.process_id} has terminated")

            # Send message as JSON
            msg = {"type": "user", "message": {"role": "user", "content": content}}
            self.proc.stdin.write((json.dumps(msg) + "\n").encode())
            await self.proc.stdin.drain()

            # Read until we get a result
            response_text = ""
            try:
                while True:
                    line = await asyncio.wait_for(
                        self.proc.stdout.readline(),
                        timeout=timeout
                    )
                    if not line:
                        raise RuntimeError("Process closed unexpectedly")
                    try:
                        data = json.loads(line.decode())
                        if data.get("type") == "result":
                            response_text = data.get("result", "")
                            break
                    except json.JSONDecodeError:
                        continue
            except asyncio.TimeoutError:
                raise TimeoutError(f"Response timeout after {timeout}s")

            self.last_used = time.time()
            return response_text

    async def terminate(self) -> None:
        """Terminate the process gracefully."""
        if self.proc.returncode is None:
            self.proc.terminate()
            try:
                await asyncio.wait_for(self.proc.wait(), timeout=2.0)
            except asyncio.TimeoutError:
                self.proc.kill()
                await self.proc.wait()

    def is_alive(self) -> bool:
        """Check if process is still running.

        Returns:
            True if the process is running, False otherwise.
        """
        return self.proc.returncode is None


class ClaudeProcessPool:
    """Unified pool of Claude processes with lazy spawning and LRU eviction.

    This pool manages persistent Claude CLI processes to avoid the ~10s spawn
    overhead on each call. Once spawned, processes can be reused for ~2s latency.

    Features:
        - Lazy spawning: Processes are only created when needed
        - Mixed models: Different models can coexist in the same pool
        - LRU eviction: When full, oldest idle process is replaced
        - Clean shutdown: Proper cleanup via atexit and signal handlers

    Example:
        ```python
        pool = ClaudeProcessPool(max_size=4)

        # First call spawns process (~10s)
        async with pool.acquire_context("haiku") as proc:
            response = await proc.send_message("Hello!")

        # Second call reuses process (~2s)
        async with pool.acquire_context("haiku") as proc:
            response = await proc.send_message("World!")

        await pool.shutdown()
        ```
    """

    def __init__(self, max_size: int = 4) -> None:
        """Initialize the pool.

        Args:
            max_size: Maximum number of concurrent processes in the pool.
        """
        self.max_size = max_size
        self._processes: list[PersistentProcess] = []
        self._lock = asyncio.Lock()
        self._next_id = 0
        self._shutting_down = False

        # Register cleanup handlers, preserving original signal handlers
        atexit.register(self._sync_cleanup)
        self._original_sigterm = signal.signal(signal.SIGTERM, self._signal_handler)
        self._original_sigint = signal.signal(signal.SIGINT, self._signal_handler)

    def _signal_handler(self, signum: int, frame: Any) -> None:
        """Handle termination signals with proper chaining."""
        self._sync_cleanup()
        # Restore and call original handler if it exists
        original = self._original_sigterm if signum == signal.SIGTERM else self._original_sigint
        if original and callable(original) and original not in (signal.SIG_DFL, signal.SIG_IGN):
            original(signum, frame)
        sys.exit(0)

    def _sync_cleanup(self) -> None:
        """Synchronous cleanup for atexit/signals."""
        if self._shutting_down:
            return
        self._shutting_down = True

        for proc in self._processes:
            try:
                if proc.is_alive():
                    proc.proc.terminate()
            except Exception:
                pass

    async def _spawn_process(self, model: str) -> PersistentProcess:
        """Spawn a new persistent Claude process.

        Args:
            model: The Claude model to use (haiku, sonnet, opus).

        Returns:
            A new PersistentProcess ready for use.

        Raises:
            RuntimeError: If process fails to initialize.
        """
        process_id = self._next_id
        self._next_id += 1

        # Create environment without ANTHROPIC_MODEL so --model flag takes precedence
        env = os.environ.copy()
        env.pop("ANTHROPIC_MODEL", None)

        proc = await asyncio.create_subprocess_exec(
            "claude", "-p",
            "--input-format", "stream-json",
            "--output-format", "stream-json",
            "--verbose",
            "--model", model,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env,
            # Create new process group for clean termination
            start_new_session=True,
        )

        persistent = PersistentProcess(
            proc=proc,
            model=model,
            process_id=process_id,
        )

        # Wait for init by sending a ping message
        try:
            await persistent.send_message("Say OK", timeout=60.0)
        except Exception as e:
            await persistent.terminate()
            raise RuntimeError(f"Failed to initialize process: {e}")

        return persistent

    async def acquire(self, model: str) -> PersistentProcess:
        """Acquire a process for the specified model.

        Strategy:
        1. Find idle process with matching model -> return it
        2. Pool not full -> spawn new process with model
        3. Pool full, has idle process with different model -> kill oldest idle, spawn new
        4. Pool full, all busy -> wait for one to become free, then recycle if needed

        Args:
            model: The Claude model to use (haiku, sonnet, opus).

        Returns:
            A PersistentProcess ready for use.
        """
        async with self._lock:
            # Clean up dead processes first
            self._processes = [p for p in self._processes if p.is_alive()]

            # 1. Find idle process with matching model
            for proc in self._processes:
                if not proc.in_use and proc.model == model:
                    proc.in_use = True
                    proc.last_used = time.time()
                    return proc

            # 2. Pool not full -> spawn new
            if len(self._processes) < self.max_size:
                proc = await self._spawn_process(model)
                proc.in_use = True
                self._processes.append(proc)
                return proc

            # 3. Pool full, find oldest idle to kill and replace
            idle_procs = [p for p in self._processes if not p.in_use]
            if idle_procs:
                # Sort by last_used (oldest first)
                oldest = min(idle_procs, key=lambda p: p.last_used)
                await oldest.terminate()
                self._processes.remove(oldest)

                # Spawn new with requested model
                proc = await self._spawn_process(model)
                proc.in_use = True
                self._processes.append(proc)
                return proc

        # 4. All busy - wait outside the lock
        while True:
            await asyncio.sleep(0.1)
            async with self._lock:
                # Check for idle process with matching model
                for proc in self._processes:
                    if not proc.in_use and proc.model == model:
                        proc.in_use = True
                        proc.last_used = time.time()
                        return proc

                # Check for any idle process to recycle
                idle_procs = [p for p in self._processes if not p.in_use]
                if idle_procs:
                    oldest = min(idle_procs, key=lambda p: p.last_used)
                    if oldest.model == model:
                        oldest.in_use = True
                        oldest.last_used = time.time()
                        return oldest
                    else:
                        # Kill and replace
                        await oldest.terminate()
                        self._processes.remove(oldest)
                        proc = await self._spawn_process(model)
                        proc.in_use = True
                        self._processes.append(proc)
                        return proc

    def release(self, process: PersistentProcess) -> None:
        """Release a process back to the pool.

        Args:
            process: The process to release.
        """
        process.in_use = False
        process.last_used = time.time()

    async def shutdown(self) -> None:
        """Shutdown the pool and terminate all processes."""
        self._shutting_down = True
        async with self._lock:
            for proc in self._processes:
                await proc.terminate()
            self._processes.clear()

    def stats(self) -> dict[str, Any]:
        """Get pool statistics.

        Returns:
            Dictionary with pool stats including size, usage, and model breakdown.
        """
        return {
            "max_size": self.max_size,
            "current_size": len(self._processes),
            "in_use": sum(1 for p in self._processes if p.in_use),
            "idle": sum(1 for p in self._processes if not p.in_use),
            "by_model": {
                model: sum(1 for p in self._processes if p.model == model)
                for model in set(p.model for p in self._processes)
            }
        }

    # Context manager support
    class _AcquireContext:
        """Context manager for acquiring a process."""

        def __init__(self, pool: "ClaudeProcessPool", model: str) -> None:
            self.pool = pool
            self.model = model
            self.proc: PersistentProcess | None = None

        async def __aenter__(self) -> PersistentProcess:
            self.proc = await self.pool.acquire(self.model)
            return self.proc

        async def __aexit__(
            self,
            exc_type: type | None,
            exc_val: Exception | None,
            exc_tb: Any,
        ) -> None:
            if self.proc:
                self.pool.release(self.proc)

    def acquire_context(self, model: str) -> _AcquireContext:
        """Get a context manager for acquiring a process.

        This is the recommended way to use the pool as it ensures
        proper release even if an exception occurs.

        Args:
            model: The Claude model to use (haiku, sonnet, opus).

        Returns:
            An async context manager that yields a PersistentProcess.

        Example:
            ```python
            async with pool.acquire_context("haiku") as proc:
                response = await proc.send_message("Hello!")
            ```
        """
        return self._AcquireContext(self, model)
