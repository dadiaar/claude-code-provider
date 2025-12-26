# Copyright (c) 2025. Claude Code Provider for Microsoft Agent Framework.

"""Batch processing for multiple prompts."""

import asyncio
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from string import Template
from typing import TYPE_CHECKING, Any, Callable, Protocol, TypeVar, runtime_checkable

if TYPE_CHECKING:
    from ._chat_client import ClaudeCodeClient

logger = logging.getLogger("claude_code_provider")

T = TypeVar("T")


@runtime_checkable
class ChatClientProtocol(Protocol):
    """Protocol for chat clients used in batch processing."""

    async def get_response(
        self,
        prompt: str,
        *,
        system_prompt: str | None = None,
        model: str | None = None,
        **kwargs: Any,
    ) -> Any:
        """Get a response from the chat client."""
        ...


class BatchStatus(str, Enum):
    """Status of a batch job."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class BatchItem:
    """A single item in a batch.

    Attributes:
        id: Unique identifier for the item.
        prompt: The prompt to process.
        status: Current status.
        result: Result when completed.
        error: Error message if failed.
        started_at: When processing started.
        completed_at: When processing completed.
        metadata: Additional metadata.
    """
    id: str
    prompt: str
    status: BatchStatus = BatchStatus.PENDING
    result: str | None = None
    error: str | None = None
    started_at: datetime | None = None
    completed_at: datetime | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def duration_seconds(self) -> float | None:
        """Get processing duration in seconds."""
        if self.started_at and self.completed_at:
            return (self.completed_at - self.started_at).total_seconds()
        return None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "prompt": self.prompt,
            "status": self.status.value,
            "result": self.result,
            "error": self.error,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "duration_seconds": self.duration_seconds,
            "metadata": self.metadata,
        }


@dataclass
class BatchResult:
    """Result of a batch processing job.

    Attributes:
        batch_id: Unique batch identifier.
        items: List of processed items.
        started_at: When the batch started.
        completed_at: When the batch completed.
        total_items: Total number of items.
        successful_items: Number of successful items.
        failed_items: Number of failed items.
    """
    batch_id: str
    items: list[BatchItem]
    started_at: datetime = field(default_factory=datetime.now)
    completed_at: datetime | None = None

    @property
    def total_items(self) -> int:
        return len(self.items)

    @property
    def successful_items(self) -> int:
        return sum(1 for item in self.items if item.status == BatchStatus.COMPLETED)

    @property
    def failed_items(self) -> int:
        return sum(1 for item in self.items if item.status == BatchStatus.FAILED)

    @property
    def is_complete(self) -> bool:
        return all(
            item.status in (BatchStatus.COMPLETED, BatchStatus.FAILED, BatchStatus.CANCELLED)
            for item in self.items
        )

    @property
    def success_rate(self) -> float:
        if not self.items:
            return 0.0
        return self.successful_items / len(self.items)

    def get_results(self) -> list[str | None]:
        """Get all results in order."""
        return [item.result for item in self.items]

    def get_successful_results(self) -> list[str]:
        """Get only successful results."""
        return [
            item.result for item in self.items
            if item.status == BatchStatus.COMPLETED and item.result
        ]

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "batch_id": self.batch_id,
            "items": [item.to_dict() for item in self.items],
            "started_at": self.started_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "total_items": self.total_items,
            "successful_items": self.successful_items,
            "failed_items": self.failed_items,
            "success_rate": self.success_rate,
        }


class BatchProcessor:
    """Processor for batch operations.

    Example:
        ```python
        from claude_code_provider import ClaudeCodeClient

        client = ClaudeCodeClient(model="haiku")
        processor = BatchProcessor(client)

        # Process multiple prompts
        prompts = [
            "Summarize quantum computing",
            "Explain machine learning",
            "What is blockchain?",
        ]

        result = await processor.process_batch(prompts, concurrency=2)

        print(f"Success rate: {result.success_rate:.1%}")
        for item in result.items:
            print(f"{item.id}: {item.result[:50]}...")
        ```
    """

    def __init__(
        self,
        client: "ChatClientProtocol | ClaudeCodeClient",
        default_concurrency: int = 3,
        retry_failed: bool = True,
        max_retries: int = 2,
    ) -> None:
        """Initialize batch processor.

        Args:
            client: ClaudeCodeClient instance or any client implementing ChatClientProtocol.
            default_concurrency: Default number of concurrent requests.
            retry_failed: Whether to retry failed items.
            max_retries: Maximum retry attempts.
        """
        self.client: ChatClientProtocol = client
        self.default_concurrency = default_concurrency
        self.retry_failed = retry_failed
        self.max_retries = max_retries

        self._batch_counter = 0
        self._item_counter = 0

    def _generate_batch_id(self) -> str:
        """Generate a unique batch ID."""
        self._batch_counter += 1
        return f"batch-{self._batch_counter}-{datetime.now().strftime('%Y%m%d%H%M%S')}"

    def _generate_item_id(self) -> str:
        """Generate a unique item ID."""
        self._item_counter += 1
        return f"item-{self._item_counter}"

    async def _process_item(
        self,
        item: BatchItem,
        system_prompt: str | None = None,
        model: str | None = None,
        retries_left: int = 0,
    ) -> None:
        """Process a single batch item."""
        item.status = BatchStatus.RUNNING
        item.started_at = datetime.now()

        try:
            response = await self.client.get_response(
                item.prompt,
                system_prompt=system_prompt,
                model=model,
            )

            item.result = response.messages[0].text if response.messages else ""
            item.status = BatchStatus.COMPLETED

        except Exception as e:
            logger.error(f"Batch item {item.id} failed: {e}")

            if retries_left > 0:
                logger.info(f"Retrying batch item {item.id}, {retries_left} attempts left")
                await self._process_item(
                    item,
                    system_prompt=system_prompt,
                    model=model,
                    retries_left=retries_left - 1,
                )
            else:
                item.error = str(e)
                item.status = BatchStatus.FAILED

        finally:
            if item.status in (BatchStatus.COMPLETED, BatchStatus.FAILED):
                item.completed_at = datetime.now()

    async def process_batch(
        self,
        prompts: list[str],
        system_prompt: str | None = None,
        model: str | None = None,
        concurrency: int | None = None,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> BatchResult:
        """Process a batch of prompts.

        Args:
            prompts: List of prompts to process.
            system_prompt: Optional system prompt for all items.
            model: Optional model override.
            concurrency: Number of concurrent requests.
            progress_callback: Callback for progress updates (completed, total).

        Returns:
            BatchResult with all processed items.
        """
        batch_id = self._generate_batch_id()
        items = [
            BatchItem(id=self._generate_item_id(), prompt=prompt)
            for prompt in prompts
        ]

        result = BatchResult(batch_id=batch_id, items=items)

        effective_concurrency = concurrency or self.default_concurrency
        retries = self.max_retries if self.retry_failed else 0

        logger.info(
            f"Starting batch {batch_id}: {len(items)} items, "
            f"concurrency={effective_concurrency}"
        )

        # Process with semaphore for concurrency control
        semaphore = asyncio.Semaphore(effective_concurrency)
        completed = 0

        async def process_with_semaphore(item: BatchItem) -> None:
            nonlocal completed
            async with semaphore:
                await self._process_item(
                    item,
                    system_prompt=system_prompt,
                    model=model,
                    retries_left=retries,
                )
                # DESIGN DECISION: This increment is safe in asyncio.
                # Rationale: asyncio is single-threaded and only switches context at await points.
                # The `completed += 1` executes atomically between awaits (no thread races).
                # This is standard asyncio pattern - no lock needed for simple counter updates.
                # Reviewed: 2025-01 - Not a race condition, intentional design choice.
                completed += 1
                if progress_callback:
                    progress_callback(completed, len(items))

        # Process all items concurrently (up to semaphore limit)
        await asyncio.gather(*[process_with_semaphore(item) for item in items])

        result.completed_at = datetime.now()

        logger.info(
            f"Batch {batch_id} completed: "
            f"{result.successful_items}/{result.total_items} successful"
        )

        return result

    async def process_with_transform(
        self,
        items: list[T],
        prompt_template: str,
        transform: Callable[[T], dict[str, Any]],
        system_prompt: str | None = None,
        model: str | None = None,
        concurrency: int | None = None,
    ) -> BatchResult:
        """Process items by transforming them into prompts.

        Args:
            items: List of items to process.
            prompt_template: Template string with placeholders.
            transform: Function to convert item to template variables.
            system_prompt: Optional system prompt.
            model: Optional model override.
            concurrency: Number of concurrent requests.

        Returns:
            BatchResult with processed items.

        Example:
            ```python
            files = ["file1.py", "file2.py", "file3.py"]
            result = await processor.process_with_transform(
                items=files,
                prompt_template="Analyze the code in {filename}",
                transform=lambda f: {"filename": f},
            )
            ```
        """
        prompts = []
        for item in items:
            variables = transform(item)
            # Use safe_substitute to prevent format string injection attacks
            # Convert {var} syntax to $var for Template compatibility
            safe_template = re.sub(r'\{(\w+)\}', r'$\1', prompt_template)
            prompt = Template(safe_template).safe_substitute(variables)
            prompts.append(prompt)

        return await self.process_batch(
            prompts=prompts,
            system_prompt=system_prompt,
            model=model,
            concurrency=concurrency,
        )

    async def map_reduce(
        self,
        prompts: list[str],
        reduce_prompt: str,
        system_prompt: str | None = None,
        model: str | None = None,
        map_concurrency: int | None = None,
    ) -> tuple[BatchResult, str]:
        """Process prompts and reduce results.

        Args:
            prompts: List of prompts to process (map phase).
            reduce_prompt: Prompt template for reduction.
                Use {results} placeholder for combined results.
            system_prompt: Optional system prompt.
            model: Optional model override.
            map_concurrency: Concurrency for map phase.

        Returns:
            Tuple of (map_result, reduced_result).

        Example:
            ```python
            prompts = [
                "Summarize chapter 1",
                "Summarize chapter 2",
                "Summarize chapter 3",
            ]
            map_result, final = await processor.map_reduce(
                prompts=prompts,
                reduce_prompt="Combine these summaries:\\n{results}",
            )
            print(f"Final summary: {final}")
            ```
        """
        # Map phase
        map_result = await self.process_batch(
            prompts=prompts,
            system_prompt=system_prompt,
            model=model,
            concurrency=map_concurrency,
        )

        # Combine results
        combined = "\n\n".join(
            f"Result {i+1}:\n{r}"
            for i, r in enumerate(map_result.get_successful_results())
        )

        # Reduce phase - use safe template substitution to prevent injection
        safe_reduce_template = re.sub(r'\{(\w+)\}', r'$\1', reduce_prompt)
        final_prompt = Template(safe_reduce_template).safe_substitute(results=combined)
        response = await self.client.get_response(
            final_prompt,
            system_prompt=system_prompt,
            model=model,
        )

        reduced_result = response.messages[0].text if response.messages else ""

        return map_result, reduced_result
