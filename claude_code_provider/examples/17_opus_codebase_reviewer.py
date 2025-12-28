#!/usr/bin/env python3
# Copyright (c) 2025. Claude Code Provider for Microsoft Agent Framework.

"""Example 17: OPUS Codebase Reviewer - Full Premium Model Orchestra

This is the premium version of example 16, upgrading all models:
    - Sonnet → Opus (5x more capable, 5x more expensive)
    - Haiku → Sonnet (3x more capable, 3x more expensive)

Model Configuration:
    - 5 Domain Teams: 3 OPUS agents × 3 iterations each (was: 3 Sonnet)
    - Evidence validation: 5 SONNET validators (was: 5 Haiku)
    - Per-domain synthesis: 5 OPUS synthesizers (was: 5 Sonnet)
    - Verification Team: 3 OPUS teams × 3 iterations each (was: 3 Sonnet)
    - Cross-domain merge: 1 OPUS (unchanged)
    - Final polish: 1 OPUS (was: 1 Sonnet)
    - Progress reporting: 1 SONNET (was: 1 Haiku)

Purpose:
    Compare accuracy vs example 16 to determine:
    1. Does Opus produce fewer false positives than Sonnet debate?
    2. Does the multi-agent pattern add value, or does Opus self-correct?
    3. Is the 5x cost justified by accuracy gains?

Expected Differences:
    - Smarter defenders = fewer inflated findings
    - Better code reading = more accurate verification
    - Higher quality synthesis = tighter final report
    - Possibly SHORTER output (Opus is more discerning)

Architecture:
    +-----------------------------------------------------------------------+
    |  PHASE 1: DOMAIN TEAMS (5 x 3Op 3i)                                   |
    |  Security, API, Performance, Testing, Architecture                    |
    |  -> Each team: 3 OPUS agents debate for 3 iterations                 |
    |  -> Connection pool limits concurrency                                |
    |  -> Sonnet reports on each team completion                           |
    +-----------------------------------------------------------------------+
    |  PHASE 1.5: EVIDENCE VALIDATION (5 x Sonnet)                         |
    |  Verify file:line references are real                                 |
    +-----------------------------------------------------------------------+
    |  PHASE 2: PER-DOMAIN SYNTHESIS (5 x Opus)                            |
    |  Create focused summary for each domain                               |
    +-----------------------------------------------------------------------+
    |  PHASE 3: VERIFICATION TEAM (3 x 3Op 3i)                             |
    |  Threat Model, Code Verifier, Severity Calibrator                    |
    +-----------------------------------------------------------------------+
    |  PHASE 4: CROSS-DOMAIN MERGE (1 x Opus)                              |
    |  Merge, deduplicate, prioritize all findings                         |
    +-----------------------------------------------------------------------+
    |  PHASE 5: FINAL POLISH (1 x Opus)                                    |
    |  QA review and formatting                                             |
    +-----------------------------------------------------------------------+

Run:
    # From ~/swarm (outside the reviewed repo)
    python3 17_opus_codebase_reviewer.py

    # Review a specific codebase
    python3 17_opus_codebase_reviewer.py --codebase /path/to/code

    # Resume from last checkpoint
    python3 17_opus_codebase_reviewer.py --resume

Output:
    Results are saved to ./results/17_opus_codebase_review/
    - logs/all_calls.jsonl (real-time call log)
    - checkpoints/*.json (phase checkpoints)
    - reports/FINAL_REPORT.md (main output)
    - reports/analysis.md (agent behavior analysis)

Cost Estimate:
    ~5x more expensive than example 16 (~$25-75 vs ~$5-15)
"""

import asyncio
import json
import subprocess
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from functools import wraps
from pathlib import Path
from typing import Any, Callable

from claude_code_provider import ClaudeCodeClient, CostTracker


# =============================================================================
# AGENT LOGGER - Comprehensive logging for multi-agent systems
# =============================================================================

@dataclass
class AgentCall:
    """Record of a single agent invocation."""
    call_id: int
    timestamp: str
    phase: str
    agent_name: str
    model: str
    prompt: str
    response: str
    duration_seconds: float
    input_tokens: int = 0
    output_tokens: int = 0
    cost: float = 0.0
    error: str | None = None
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class PhaseCheckpoint:
    """Checkpoint saved after each phase."""
    phase_name: str
    timestamp: str
    duration_seconds: float
    agent_calls: int
    total_tokens: int
    total_cost: float
    findings: dict[str, Any]


class AgentLogger:
    """Comprehensive logger for multi-agent orchestration."""

    def __init__(
        self,
        demo_name: str,
        output_dir: str | Path | None = None,
        verbose: bool = True,
    ):
        self.demo_name = demo_name
        self.verbose = verbose
        self.start_time = datetime.now()

        # Set up output directories
        base_dir = Path(output_dir) if output_dir else Path.cwd() / "results"
        base_dir.mkdir(exist_ok=True)

        self.run_dir = base_dir / demo_name
        self.run_dir.mkdir(exist_ok=True)

        self.logs_dir = self.run_dir / "logs"
        self.logs_dir.mkdir(exist_ok=True)
        self.checkpoints_dir = self.run_dir / "checkpoints"
        self.checkpoints_dir.mkdir(exist_ok=True)
        self.reports_dir = self.run_dir / "reports"
        self.reports_dir.mkdir(exist_ok=True)

        self.output_dir = self.run_dir

        self.calls: list[AgentCall] = []
        self.checkpoints: list[PhaseCheckpoint] = []
        self.current_phase: str | None = None
        self.phase_start_time: float | None = None
        self.call_counter = 0

        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_cost = 0.0
        self.config: dict[str, Any] = {}

    def set_config(self, **kwargs) -> None:
        """Set configuration metadata."""
        self.config.update(kwargs)
        if self.verbose:
            print(f"[CONFIG] {kwargs}")

    def start_phase(self, phase_name: str) -> None:
        """Start a new phase."""
        self.current_phase = phase_name
        self.phase_start_time = time.time()
        if self.verbose:
            print(f"\n{'='*70}")
            print(f"[PHASE START] {phase_name.upper()}")
            print(f"{'='*70}")

    def end_phase(
        self,
        phase_name: str,
        findings: dict[str, Any] | None = None,
    ) -> None:
        """End a phase and save checkpoint."""
        duration = time.time() - self.phase_start_time if self.phase_start_time else 0.0

        phase_calls = [c for c in self.calls if c.phase == phase_name]
        phase_tokens = sum(c.input_tokens + c.output_tokens for c in phase_calls)
        phase_cost = sum(c.cost for c in phase_calls)

        checkpoint = PhaseCheckpoint(
            phase_name=phase_name,
            timestamp=datetime.now().isoformat(),
            duration_seconds=duration,
            agent_calls=len(phase_calls),
            total_tokens=phase_tokens,
            total_cost=phase_cost,
            findings=findings or {},
        )
        self.checkpoints.append(checkpoint)
        self._save_checkpoint(checkpoint, findings)

        if self.verbose:
            print(f"\n[PHASE END] {phase_name.upper()}")
            print(f"  Duration: {duration:.1f}s")
            print(f"  Agent Calls: {len(phase_calls)}")
            print(f"  Tokens: {phase_tokens:,}")
            print(f"  Cost: ${phase_cost:.4f}")

        self.current_phase = None
        self.phase_start_time = None

    def log_call(
        self,
        phase: str,
        agent_name: str,
        model: str,
        prompt: str,
        response: str,
        duration: float,
        input_tokens: int = 0,
        output_tokens: int = 0,
        cost: float = 0.0,
        error: str | None = None,
        **metadata,
    ) -> AgentCall:
        """Log a single agent call."""
        self.call_counter += 1

        call = AgentCall(
            call_id=self.call_counter,
            timestamp=datetime.now().isoformat(),
            phase=phase,
            agent_name=agent_name,
            model=model,
            prompt=prompt,
            response=response,
            duration_seconds=duration,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost=cost,
            error=error,
            metadata=metadata,
        )
        self.calls.append(call)

        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens
        self.total_cost += cost

        if self.verbose:
            status = "ERROR" if error else "OK"
            print(f"  [{agent_name}] {status} ({duration:.1f}s, {input_tokens+output_tokens} tokens)")

        return call

    def flush_call_to_disk(self, call: AgentCall | None = None) -> Path:
        """Immediately flush a call to JSONL file (crash-safe)."""
        if call is None:
            if not self.calls:
                raise ValueError("No calls to flush")
            call = self.calls[-1]

        jsonl_path = self.logs_dir / "all_calls.jsonl"
        with open(jsonl_path, 'a') as f:
            f.write(json.dumps(call.to_dict(), default=str) + "\n")
        return jsonl_path

    def _save_checkpoint(
        self,
        checkpoint: PhaseCheckpoint,
        findings: dict[str, Any] | None,
    ) -> Path:
        """Save a phase checkpoint to disk."""
        meta_path = self.checkpoints_dir / f"{checkpoint.phase_name}_meta.json"
        with open(meta_path, 'w') as f:
            json.dump(asdict(checkpoint), f, indent=2, default=str)

        if findings:
            findings_path = self.checkpoints_dir / f"{checkpoint.phase_name}_findings.json"
            with open(findings_path, 'w') as f:
                json.dump(findings, f, indent=2, default=str)

        phase_calls = [c.to_dict() for c in self.calls if c.phase == checkpoint.phase_name]
        calls_path = self.checkpoints_dir / f"{checkpoint.phase_name}_calls.json"
        with open(calls_path, 'w') as f:
            json.dump(phase_calls, f, indent=2, default=str)

        return self.checkpoints_dir

    def generate_report(self) -> str:
        """Generate a comprehensive markdown report."""
        lines = []
        total_duration = (datetime.now() - self.start_time).total_seconds()

        lines.append(f"# {self.demo_name} - Codebase Review Report")
        lines.append(f"\n**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"**Total Duration**: {total_duration:.1f}s ({total_duration/60:.1f} min)")
        lines.append(f"**Total Agent Calls**: {len(self.calls)}")
        lines.append(f"**Total Tokens**: {self.total_input_tokens + self.total_output_tokens:,}")
        lines.append(f"**Total Cost**: ${self.total_cost:.4f}")
        lines.append("")

        if self.config:
            lines.append("## Configuration")
            lines.append("")
            for key, value in self.config.items():
                lines.append(f"- **{key}**: {value}")
            lines.append("")

        lines.append("## Phase Summaries")
        lines.append("")
        lines.append("| Phase | Calls | Duration | Tokens | Cost |")
        lines.append("|-------|-------|----------|--------|------|")
        for cp in self.checkpoints:
            lines.append(
                f"| {cp.phase_name} | {cp.agent_calls} | "
                f"{cp.duration_seconds:.1f}s | {cp.total_tokens:,} | ${cp.total_cost:.4f} |"
            )
        lines.append("")

        lines.append("## Agent Call Log")
        lines.append("")

        current_phase = None
        for call in self.calls:
            if call.phase != current_phase:
                current_phase = call.phase
                lines.append(f"### Phase: {current_phase}")
                lines.append("")

            status = "ERROR" if call.error else "OK"
            lines.append(f"#### Call #{call.call_id}: {call.agent_name} ({call.model}) - {status}")
            lines.append(f"*Duration: {call.duration_seconds:.1f}s, Tokens: {call.input_tokens + call.output_tokens}*")
            lines.append("")

            lines.append("**Prompt:**")
            lines.append("```")
            prompt_preview = call.prompt[:1000] + "..." if len(call.prompt) > 1000 else call.prompt
            lines.append(prompt_preview)
            lines.append("```")
            lines.append("")

            lines.append("**Response:**")
            lines.append("```")
            response_preview = call.response[:2000] + "..." if len(call.response) > 2000 else call.response
            lines.append(response_preview)
            lines.append("```")
            lines.append("")

        return "\n".join(lines)

    def generate_analysis(self) -> str:
        """Generate an analysis of agent behavior patterns."""
        lines = []
        lines.append("# Agent Behavior Analysis")
        lines.append("")

        lines.append("## Response Time Analysis")
        lines.append("")
        agent_times: dict[str, list[float]] = {}
        for call in self.calls:
            if call.agent_name not in agent_times:
                agent_times[call.agent_name] = []
            agent_times[call.agent_name].append(call.duration_seconds)

        lines.append("| Agent | Calls | Avg Time | Min | Max |")
        lines.append("|-------|-------|----------|-----|-----|")
        for agent, times in sorted(agent_times.items()):
            avg = sum(times) / len(times)
            lines.append(f"| {agent} | {len(times)} | {avg:.1f}s | {min(times):.1f}s | {max(times):.1f}s |")
        lines.append("")

        lines.append("## Token Usage by Agent")
        lines.append("")
        agent_tokens: dict[str, dict] = {}
        for call in self.calls:
            if call.agent_name not in agent_tokens:
                agent_tokens[call.agent_name] = {"input": 0, "output": 0, "cost": 0}
            agent_tokens[call.agent_name]["input"] += call.input_tokens
            agent_tokens[call.agent_name]["output"] += call.output_tokens
            agent_tokens[call.agent_name]["cost"] += call.cost

        lines.append("| Agent | Input Tokens | Output Tokens | Cost |")
        lines.append("|-------|--------------|---------------|------|")
        for agent, stats in sorted(agent_tokens.items()):
            lines.append(f"| {agent} | {stats['input']:,} | {stats['output']:,} | ${stats['cost']:.4f} |")
        lines.append("")

        errors = [c for c in self.calls if c.error]
        if errors:
            lines.append("## Errors")
            lines.append("")
            for call in errors:
                lines.append(f"- **{call.agent_name}** (Call #{call.call_id}): {call.error}")
            lines.append("")

        return "\n".join(lines)

    def save(self) -> tuple[Path, Path]:
        """Save all logs and generate reports."""
        end_time = datetime.now()

        report_path = self.reports_dir / "FINAL_REPORT.md"
        analysis_path = self.reports_dir / "analysis.md"
        data_path = self.reports_dir / "summary.json"
        metadata_path = self.run_dir / "metadata.json"

        report_path.write_text(self.generate_report())
        analysis_path.write_text(self.generate_analysis())

        summary_data = {
            "demo_name": self.demo_name,
            "start_time": self.start_time.isoformat(),
            "end_time": end_time.isoformat(),
            "duration_seconds": (end_time - self.start_time).total_seconds(),
            "config": self.config,
            "summary": {
                "total_calls": len(self.calls),
                "total_input_tokens": self.total_input_tokens,
                "total_output_tokens": self.total_output_tokens,
                "total_cost": self.total_cost,
            },
            "checkpoints": [asdict(cp) for cp in self.checkpoints],
        }
        with open(data_path, 'w') as f:
            json.dump(summary_data, f, indent=2, default=str)

        metadata = {
            "demo_name": self.demo_name,
            "start_time": self.start_time.isoformat(),
            "end_time": end_time.isoformat(),
            "duration_seconds": (end_time - self.start_time).total_seconds(),
            "total_calls": len(self.calls),
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_cost": self.total_cost,
            "config": self.config,
        }
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)

        full_data_path = self.logs_dir / "all_calls.json"
        with open(full_data_path, 'w') as f:
            json.dump([c.to_dict() for c in self.calls], f, indent=2, default=str)

        if self.verbose:
            print(f"\n[SAVED] Report: {report_path}")
            print(f"[SAVED] Analysis: {analysis_path}")
            print(f"[SAVED] Summary: {data_path}")

        return report_path, analysis_path

    def progress(self, message: str) -> None:
        """Log a progress message."""
        if self.verbose:
            timestamp = datetime.now().strftime("%H:%M:%S")
            print(f"[{timestamp}] {message}")


# =============================================================================
# CONNECTION POOL - Limits concurrent API calls
# =============================================================================

class ConnectionPool:
    """Manages concurrent connections to Claude API with rate limiting."""

    def __init__(self, max_concurrent: int = 6):
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._queue_count = 0
        self._active_count = 0
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        """Acquire a connection slot."""
        async with self._lock:
            self._queue_count += 1
        await self._semaphore.acquire()
        async with self._lock:
            self._queue_count -= 1
            self._active_count += 1

    async def release(self) -> None:
        """Release a connection slot."""
        async with self._lock:
            self._active_count -= 1
        self._semaphore.release()

    @property
    async def status(self) -> dict:
        """Get current pool status."""
        async with self._lock:
            return {"active": self._active_count, "queued": self._queue_count}


# Global instances
_connection_pool = ConnectionPool(max_concurrent=6)
_cost_tracker: CostTracker | None = None


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class DomainFindings:
    """Findings from a single domain team."""
    domain: str
    initial_findings: str = ""
    debate_rounds: list[dict[str, str]] = field(default_factory=list)
    final_consensus: str = ""
    validation_report: str = ""
    synthesis: str = ""


@dataclass
class ReviewState:
    """Complete state of the review - can be checkpointed and restored."""
    domain_findings: dict[str, DomainFindings] = field(default_factory=dict)
    verification_findings: dict[str, str] = field(default_factory=dict)
    cross_domain_synthesis: str = ""
    final_report: str = ""


# =============================================================================
# PROGRESS REPORTER - Haiku reports on team completion
# =============================================================================

class ProgressReporter:
    """Reports progress using Haiku via claude -p (no follow-up questions)."""

    def __init__(self, logger: AgentLogger):
        self.logger = logger
        self._report_count = 0
        self._start_time = time.time()

    async def report_team_completion(
        self,
        team_name: str,
        phase: str,
        summary: str,
        teams_completed: int,
        teams_total: int,
    ) -> None:
        """Report on team completion using Haiku via claude -p (print mode)."""
        self._report_count += 1
        elapsed = time.time() - self._start_time

        prompt = f"""You are a concise progress reporter. Create a brief status update (2-3 sentences).

Team: {team_name}
Phase: {phase}
Completed: {teams_completed}/{teams_total}
Elapsed: {elapsed/60:.1f} minutes
Summary: {summary[:200]}...

Report what just completed, overall progress, and time elapsed. Be factual. No follow-up questions."""

        try:
            result = subprocess.run(
                ["claude", "-p", prompt, "--model", "sonnet"],  # UPGRADED: was haiku
                capture_output=True,
                text=True,
                timeout=60,
            )
            status = result.stdout.strip() if result.returncode == 0 else f"Progress: {teams_completed}/{teams_total} complete"

            print(f"\n{'='*60}")
            print(f"PROGRESS REPORT #{self._report_count}")
            print(f"{'='*60}")
            print(status)
            print(f"{'='*60}\n")
        except Exception:
            print(f"\n[PROGRESS] {team_name} complete ({teams_completed}/{teams_total})")


# =============================================================================
# STREAMING RESULT
# =============================================================================

@dataclass
class StreamingResult:
    """Result object for streaming responses."""
    text: str
    input_tokens: int = 0
    output_tokens: int = 0


# =============================================================================
# LOGGED AGENT WITH POOL AND IMMEDIATE FLUSH
# =============================================================================

class LoggedAgent:
    """Wrapper that logs all agent interactions with pool, retry, and streaming.

    Note: USE_STREAMING is disabled for codebase review because the per-chunk
    streaming timeout (60s) is too short for complex multi-file analysis.
    The non-streaming mode uses the full client timeout (1200s for Sonnet).
    """

    MAX_RETRIES = 3
    BASE_DELAY = 30
    MAX_DELAY = 120
    USE_STREAMING = False  # Disabled for complex codebase review tasks

    def __init__(
        self,
        agent,
        logger: AgentLogger,
        agent_name: str,
        model: str,
        use_pool: bool = True,
    ):
        self.agent = agent
        self.logger = logger
        self.agent_name = agent_name
        self.model = model
        self.use_pool = use_pool

    async def run(self, prompt: str) -> Any:
        """Run agent with pool, retry logic, and immediate flush."""
        phase = self.logger.current_phase or "unknown"
        start = time.time()

        if self.use_pool:
            await _connection_pool.acquire()

        try:
            return await self._run_with_retry(prompt, phase, start)
        finally:
            if self.use_pool:
                await _connection_pool.release()

    async def _run_streaming(self, prompt: str) -> StreamingResult:
        """Run agent using streaming mode for faster hung detection."""
        chunks = []
        async for chunk in self.agent.run_stream(prompt):
            if hasattr(chunk, 'text') and chunk.text:
                chunks.append(chunk.text)
        return StreamingResult(text="".join(chunks))

    async def _run_with_retry(self, prompt: str, phase: str, start: float) -> Any:
        """Execute with retry logic."""
        last_exception = None

        for attempt in range(self.MAX_RETRIES + 1):
            try:
                if attempt > 0:
                    delay = min(self.BASE_DELAY * (2 ** (attempt - 1)), self.MAX_DELAY)
                    print(f"  [{self.agent_name}] Retry {attempt}/{self.MAX_RETRIES} after {delay}s...")
                    await asyncio.sleep(delay)

                if self.USE_STREAMING:
                    result = await self._run_streaming(prompt)
                else:
                    result = await self.agent.run(prompt)
                response_text = result.text if hasattr(result, 'text') else str(result)

                input_tokens, output_tokens = self._extract_usage(result)

                cost = 0.0
                if _cost_tracker and (input_tokens > 0 or output_tokens > 0):
                    req_cost = _cost_tracker.record_request(
                        self.model, input_tokens, output_tokens
                    )
                    cost = req_cost.total_cost

                duration = time.time() - start
                self._log_and_flush(
                    phase=phase,
                    prompt=prompt,
                    response=response_text,
                    duration=duration,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    cost=cost,
                    error=None,
                )
                return result

            except Exception as e:
                last_exception = e
                error = str(e)

                is_retryable = any(p in error.lower() for p in [
                    "exit code: 1", "rate limit", "overloaded",
                    "timeout", "timed out", "connection", "temporarily unavailable",
                ])

                if not is_retryable or attempt >= self.MAX_RETRIES:
                    duration = time.time() - start
                    self._log_and_flush(
                        phase=phase,
                        prompt=prompt,
                        response="",
                        duration=duration,
                        input_tokens=0,
                        output_tokens=0,
                        cost=0.0,
                        error=f"FAILED after {attempt + 1} attempts: {error}",
                    )
                    raise

                print(f"  [{self.agent_name}] Error (attempt {attempt + 1}): {error[:100]}...")

        raise last_exception

    def _extract_usage(self, result) -> tuple[int, int]:
        """Extract token usage from result.

        MAF AgentRunResult has usage_details with input_token_count/output_token_count.
        """
        input_tokens = 0
        output_tokens = 0

        # MAF AgentRunResult structure
        if hasattr(result, 'usage_details') and result.usage_details:
            ud = result.usage_details
            input_tokens = getattr(ud, 'input_token_count', 0) or 0
            output_tokens = getattr(ud, 'output_token_count', 0) or 0

        # Fallback: direct attributes
        if input_tokens == 0 and hasattr(result, 'input_tokens'):
            input_tokens = result.input_tokens or 0
        if output_tokens == 0 and hasattr(result, 'output_tokens'):
            output_tokens = result.output_tokens or 0

        return input_tokens, output_tokens

    def _log_and_flush(
        self,
        phase: str,
        prompt: str,
        response: str,
        duration: float,
        input_tokens: int,
        output_tokens: int,
        cost: float,
        error: str | None,
    ) -> None:
        """Log call and immediately flush to disk."""
        self.logger.log_call(
            phase=phase,
            agent_name=self.agent_name,
            model=self.model,
            prompt=prompt,
            response=response,
            duration=duration,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost=cost,
            error=error,
        )
        self.logger.flush_call_to_disk()


# =============================================================================
# 3So 3i TEAM - 3 Sonnet agents with 3 debate iterations
# =============================================================================

class SonnetTeam:
    """A team of 3 Sonnet agents that debate for 3 iterations."""

    def __init__(
        self,
        name: str,
        sonnet_client: ClaudeCodeClient,
        logger: AgentLogger,
        context: dict[str, str],
        instructions: str,
        iterations: int = 3,
    ):
        self.name = name
        self.sonnet = sonnet_client
        self.logger = logger
        self.context = context
        self.instructions = instructions
        self.iterations = iterations
        self._create_agents()

    def _create_agents(self) -> None:
        """Create 3 Sonnet agents with different perspectives.

        The defender role is critical for preventing severity inflation.
        They argue AGAINST findings to ensure only real issues are reported.
        """
        perspectives = [
            ("analyst", """You are a methodical ANALYST who focuses on evidence and facts.
When reporting issues, always consider:
- WHO controls the input? (user config vs untrusted external input)
- Is this a REAL attack vector or theoretical?
- Check for design decision comments that explain the code behavior."""),
            ("defender", """You are a DEFENDER who argues AGAINST findings.
Your job is to PREVENT FALSE POSITIVES and SEVERITY INFLATION.

For each claimed issue, ask:
1. Is this a real vulnerability or a theoretical concern?
2. Who controls the input? If the user controls it, it's NOT an attack vector.
3. Are there design decision comments explaining why the code is this way?
4. Would fixing this actually improve security, or just add complexity?
5. Is the severity inflated? CRITICAL means RCE/data breach, not theoretical issues.

Push back hard on inflated claims. A report with fewer TRUE issues is better
than one with many FALSE POSITIVES."""),
            ("synthesizer", """You are a SYNTHESIZER who integrates viewpoints and finds consensus.
When the analyst and defender disagree:
- VERIFY the claim by reading the actual code
- Check for design decision comments
- Consider the ACTUAL threat model, not theoretical attacks
- Err on the side of the DEFENDER if evidence is weak
- Only mark as CRITICAL if there's clear, verified evidence of exploitability"""),
        ]

        self.agents = []
        for role, perspective in perspectives:
            agent = self.sonnet.create_agent(
                name=f"{self.name}_{role}",
                instructions=f"""{perspective}

{self.instructions}

{self.context['base_instructions']}""",
            )
            self.agents.append(
                LoggedAgent(agent, self.logger, f"{self.name}_{role}", "opus")  # UPGRADED
            )

    async def run(self, initial_prompt: str) -> tuple[str, list[dict]]:
        """Run 3 iterations of debate among 3 agents."""
        debate_rounds = []
        current_context = ""

        # Iteration 1: Initial analysis
        self.logger.progress(f"[{self.name}] Iteration 1/3 - Initial analysis...")
        round1 = {}
        for i, agent in enumerate(self.agents):
            response = await agent.run(initial_prompt)
            round1[f"agent_{i+1}"] = response.text if hasattr(response, 'text') else str(response)
        debate_rounds.append(round1)

        current_context = "\n\n---\n\n".join(
            f"Agent {i+1}:\n{text}" for i, text in enumerate(round1.values())
        )

        # Iteration 2: Critique and refinement
        self.logger.progress(f"[{self.name}] Iteration 2/3 - Critique and refinement...")
        round2 = {}
        for i, agent in enumerate(self.agents):
            critique_prompt = f"""Review and critique these analyses from your colleagues:

{current_context}

Original question: {initial_prompt}

Provide your refined analysis, addressing any gaps or disagreements."""

            response = await agent.run(critique_prompt)
            round2[f"agent_{i+1}"] = response.text if hasattr(response, 'text') else str(response)
        debate_rounds.append(round2)

        current_context = "\n\n---\n\n".join(
            f"Agent {i+1} (refined):\n{text}" for i, text in enumerate(round2.values())
        )

        # Iteration 3: Final consensus
        self.logger.progress(f"[{self.name}] Iteration 3/3 - Building consensus...")
        round3 = {}

        consensus_prompt = f"""Create a FINAL CONSENSUS from all analyses:

Previous analyses:
{current_context}

Original question: {initial_prompt}

Synthesize all viewpoints into a single, comprehensive response.
Include all important findings, prioritized by severity.
Resolve any disagreements with clear reasoning."""

        response = await self.agents[2].run(consensus_prompt)  # synthesizer
        final_consensus = response.text if hasattr(response, 'text') else str(response)
        round3["consensus"] = final_consensus
        debate_rounds.append(round3)

        self.logger.progress(f"[{self.name}] Complete")
        return final_consensus, debate_rounds


# =============================================================================
# DOMAIN TEAM (using 3So 3i)
# =============================================================================

class DomainTeam:
    """A review team for a specific domain using 3So 3i pattern."""

    def __init__(
        self,
        domain: str,
        focus_areas: list[str],
        sonnet: ClaudeCodeClient,
        logger: AgentLogger,
        context: dict[str, str],
    ):
        self.domain = domain
        self.focus_areas = focus_areas
        self.sonnet = sonnet
        self.logger = logger
        self.context = context
        self.findings = DomainFindings(domain=domain)

        focus_list = "\n".join(f"- {area}" for area in self.focus_areas)
        instructions = f"""You are a SENIOR {domain.upper()} EXPERT conducting a code review.

YOUR FOCUS AREAS:
{focus_list}

THREAT MODEL ANALYSIS (REQUIRED for security issues):
Before marking any issue as CRITICAL or HIGH, you MUST analyze:
1. WHO controls the input? (User config = NOT an attack vector)
2. What's the ACTUAL attack scenario? (Be specific, not theoretical)
3. Are there DESIGN DECISION comments explaining the code? (Check for "# DESIGN" or similar)
4. Is this EXPLOITABLE in practice, or just theoretically concerning?

SEVERITY DEFINITIONS (be strict):
- CRITICAL: Proven RCE, data breach, or privilege escalation with clear exploit path
- HIGH: Verified vulnerability with realistic attack scenario
- MEDIUM: Real issue but requires specific conditions or has limited impact
- LOW: Code quality issue, not a security vulnerability
- FALSE POSITIVE: Theoretical issue that's not actually exploitable

For EACH finding, include:
- Confidence: VERIFIED (read the code) / LIKELY (probable) / THEORETICAL (unverified)
- Threat Model: Who is the attacker? What do they control?

OUTPUT FORMAT:
## {domain.title()} Analysis

### Critical Issues (CRITICAL) - Only verified, exploitable vulnerabilities
### High Priority (HIGH) - Verified issues with clear attack scenarios
### Medium Priority (MEDIUM) - Real issues, limited impact
### Low Priority (LOW) - Code quality, not security
### False Positives Cleared - Issues that looked bad but aren't real
### Patterns & Insights

Be thorough but ACCURATE. Include file:line references and code evidence.
A shorter report with TRUE findings is better than a longer one with FALSE POSITIVES."""

        self.team = SonnetTeam(
            name=f"{domain}_team",
            sonnet_client=sonnet,
            logger=logger,
            context=context,
            instructions=instructions,
        )

    async def run_review(self) -> DomainFindings:
        """Run the full team review."""
        self.logger.progress(f"[{self.domain.upper()}] Starting 3So 3i analysis...")

        initial_prompt = f"""Conduct a comprehensive {self.domain} review of:
{self.context['codebase_path']}

Files to review:
{self.context['file_list']}

Read ALL files. Be exhaustive."""

        final_consensus, debate_rounds = await self.team.run(initial_prompt)

        self.findings.final_consensus = final_consensus
        self.findings.debate_rounds = debate_rounds

        return self.findings


# =============================================================================
# EVIDENCE VALIDATOR (uses Haiku)
# =============================================================================

class EvidenceValidator:
    """Validates findings against actual code and challenges severity claims."""

    def __init__(
        self,
        haiku: ClaudeCodeClient,
        logger: AgentLogger,
        context: dict[str, str],
    ):
        self.haiku = haiku
        self.logger = logger
        self.context = context

    async def validate(self, domain: str, findings: str) -> str:
        """Validate a domain's findings and challenge severity."""
        self.logger.progress(f"[VALIDATOR] Checking {domain}...")

        validator_agent = self.haiku.create_agent(
            name=f"{domain}_validator",
            instructions=f"""You are a SKEPTICAL VALIDATOR. Your job is to VERIFY findings and CHALLENGE severity claims.

{self.context['base_instructions']}

For each finding:
1. READ the actual code at the file:line reference
2. Check for DESIGN DECISION comments (# DESIGN, # NOTE, # Security, etc.)
3. Analyze the THREAT MODEL - who controls the input?
4. CHALLENGE the severity if it seems inflated

Verification Status:
- VERIFIED: Code exists and issue is real as described
- DOWNGRADE: Issue exists but severity is inflated (explain why)
- INVALID: Claim doesn't match the actual code
- FALSE_POSITIVE: Theoretical issue that's not exploitable (e.g., user-controlled config)
- UNVERIFIABLE: Can't find the referenced code

IMPORTANT: If you find a design decision comment that explains the behavior,
the finding should be marked FALSE_POSITIVE unless there's a clear flaw in the reasoning.

Be thorough but fast.""",
        )
        validator = LoggedAgent(validator_agent, self.logger, f"{domain}_validator", "sonnet")  # UPGRADED

        response = await validator.run(
            f"""Validate and challenge these {domain} findings:

{findings}

For EACH finding:
1. Read the actual code
2. Look for design decision comments
3. Analyze the threat model
4. Verify or challenge the severity

Output format for each:
- Finding: [description]
- File: [path:line]
- Claimed Severity: [CRITICAL/HIGH/MEDIUM/LOW]
- Actual Code: [what you found]
- Design Comments: [any explanatory comments found]
- Threat Model: [who controls the input?]
- Status: [VERIFIED/DOWNGRADE/INVALID/FALSE_POSITIVE/UNVERIFIABLE]
- Recommended Severity: [if different from claimed]
- Reasoning: [why]"""
        )
        return response.text if hasattr(response, 'text') else str(response)


# =============================================================================
# DOMAIN SYNTHESIZER (uses 1 Sonnet)
# =============================================================================

class DomainSynthesizer:
    """Creates per-domain synthesis."""

    def __init__(self, sonnet: ClaudeCodeClient, logger: AgentLogger):
        self.sonnet = sonnet
        self.logger = logger

    async def synthesize(self, domain: str, findings: DomainFindings) -> str:
        """Create a focused synthesis for one domain."""
        self.logger.progress(f"[SYNTH] Summarizing {domain}...")

        synth_agent = self.sonnet.create_agent(
            name=f"{domain}_synthesizer",
            instructions=f"""Create a focused summary of {domain} findings.

OUTPUT:
## {domain.title()} Summary

### Key Issues (prioritized)
### Validated vs Invalidated
### Action Items

Keep it concise. This will be merged with other domains later.""",
        )
        synthesizer = LoggedAgent(synth_agent, self.logger, f"{domain}_synth", "opus")  # UPGRADED

        debate_summary = "\n\n".join(
            f"Round {i+1}: {list(r.values())[0][:500]}..."
            for i, r in enumerate(findings.debate_rounds)
        ) if findings.debate_rounds else ""

        response = await synthesizer.run(
            f"""Synthesize these {domain} findings:

TEAM CONSENSUS:
{findings.final_consensus}

VALIDATION REPORT:
{findings.validation_report}

DEBATE HIGHLIGHTS:
{debate_summary}

Create a focused, prioritized summary."""
        )
        return response.text if hasattr(response, 'text') else str(response)


# =============================================================================
# VERIFICATION TEAM (challenges findings, replaces old "Red Team")
# =============================================================================

class VerificationTeam:
    """Verification team that challenges and validates findings.

    Unlike the old "Red Team" that tried to find MORE issues,
    this team's job is to VERIFY claims and REDUCE false positives.
    """

    def __init__(
        self,
        sonnet: ClaudeCodeClient,
        logger: AgentLogger,
        context: dict[str, str],
    ):
        self.sonnet = sonnet
        self.logger = logger
        self.context = context

    async def verify(self, domain_syntheses: dict[str, str]) -> dict[str, str]:
        """Verify findings from domain teams."""
        self.logger.progress("[VERIFICATION TEAM] Starting 3 verification cells...")

        syntheses_text = "\n\n---\n\n".join(
            f"## {domain.upper()}:\n{synthesis}"
            for domain, synthesis in domain_syntheses.items()
        )

        cells = [
            ("threat_model_analyst", """You are a THREAT MODEL ANALYST.
Your job is to CHALLENGE the threat models in the findings:

For each CRITICAL or HIGH finding, verify:
1. WHO is the attacker? (External hacker? Malicious insider? User with config access?)
2. WHAT do they control? (Network input? Config files? Environment?)
3. Is this REALISTIC? Can this actually happen in production?

If the "attacker" is just a user configuring their own system, it's NOT a vulnerability.
If the finding assumes the attacker has already compromised the system, it's circular reasoning.

Mark findings as:
- VALID_THREAT: Realistic attack scenario with external attacker
- INSIDER_ONLY: Requires malicious insider with significant access
- USER_CONFIG: User controls their own config, not a vulnerability
- CIRCULAR: Assumes attacker already has access they'd need the vuln to get"""),

            ("code_verifier", """You are a CODE VERIFIER.
Your job is to READ THE ACTUAL CODE and verify claims:

For each finding:
1. Go to the EXACT file:line cited
2. Read the surrounding context (20 lines before and after)
3. Look for DESIGN DECISION comments (# DESIGN, # NOTE, # Security, # Rationale)
4. Check if the code actually does what the finding claims

Mark findings as:
- CODE_MATCHES: The code does exactly what the finding claims
- CODE_DIFFERS: The code is different from what's claimed
- HAS_MITIGATION: There's mitigation code the finding missed
- DESIGN_EXPLAINED: There's a design comment explaining the behavior
- CANT_FIND: Can't locate the referenced code"""),

            ("severity_calibrator", """You are a SEVERITY CALIBRATOR.
Your job is to ensure severity ratings are ACCURATE, not inflated:

CRITICAL means: Proven, exploitable RCE/data breach/privilege escalation
- Attacker can execute arbitrary code
- Attacker can read/write data they shouldn't
- Attacker can gain admin access
- AND there's a clear, realistic exploit path

HIGH means: Verified vulnerability with realistic attack scenario
- Not just theoretical
- Attacker doesn't need pre-existing access
- Impact is significant

MEDIUM means: Real issue with limited impact or complex requirements
- Requires specific conditions
- Limited blast radius
- Defense in depth applies

For each CRITICAL/HIGH finding, ask:
- Would this pass a security audit as written?
- Is there a CVE for similar issues in other projects?
- Or is this severity inflation to seem thorough?

Recommend DOWNGRADES where appropriate."""),
        ]

        teams = []
        for name, instructions in cells:
            team = SonnetTeam(
                name=f"verify_{name}",
                sonnet_client=self.sonnet,
                logger=self.logger,
                context=self.context,
                instructions=instructions,
            )
            teams.append((name, team))

        async def run_cell(name: str, team: SonnetTeam) -> tuple[str, str]:
            prompt = f"""Review these findings and CHALLENGE them:

{syntheses_text}

READ THE CODE at {self.context['codebase_path']} to verify or refute each claim.

Your goal is to REDUCE FALSE POSITIVES, not find more issues.
A shorter, accurate report is better than a longer one with inflated findings."""
            consensus, _ = await team.run(prompt)
            return name, consensus

        results = await asyncio.gather(*[run_cell(n, t) for n, t in teams])

        self.logger.progress("[VERIFICATION TEAM] Complete")
        return {name: consensus for name, consensus in results}


# =============================================================================
# CROSS-DOMAIN MERGER (1 Opus)
# =============================================================================

class CrossDomainMerger:
    """Merges per-domain syntheses into unified report using 1 Opus."""

    def __init__(self, opus: ClaudeCodeClient, logger: AgentLogger):
        self.opus = opus
        self.logger = logger

    async def merge(
        self,
        domain_syntheses: dict[str, str],
        verification_findings: dict[str, str],
    ) -> str:
        """Merge all findings into unified report."""
        self.logger.progress("[MERGE] Creating unified synthesis (1 Opus)...")

        merger_agent = self.opus.create_agent(
            name="cross_domain_merger",
            instructions="""You are a CHIEF ARCHITECT creating a unified code review.

Your job:
1. MERGE findings from all domains
2. DEDUPLICATE overlapping issues
3. Apply VERIFICATION TEAM feedback to adjust/remove false positives
4. PRIORITIZE by business impact (CRITICAL > HIGH > MEDIUM > LOW)
5. Only include VERIFIED findings in CRITICAL/HIGH sections
6. CREATE actionable recommendations

IMPORTANT:
- If the verification team marked something as FALSE_POSITIVE, USER_CONFIG, or
  CIRCULAR, it should NOT appear in the final report as a vulnerability.
- If severity was DOWNGRADED by the verification team, use the lower severity.
- A shorter report with TRUE findings is better than a longer one with false positives.

Output a comprehensive but ACCURATE report.
Do NOT include findings that were refuted by the verification team.""",
        )
        merger = LoggedAgent(merger_agent, self.logger, "merger", "opus")

        all_syntheses = "\n\n---\n\n".join(
            f"## {domain.upper()} SYNTHESIS:\n{synthesis}"
            for domain, synthesis in domain_syntheses.items()
        )

        verification_text = "\n\n---\n\n".join(
            f"## {verify_type.replace('_', ' ').title()} VERIFICATION:\n{findings}"
            for verify_type, findings in verification_findings.items()
        )

        response = await merger.run(
            f"""Merge these domain syntheses, applying verification team feedback:

DOMAIN SYNTHESES (initial findings):
{all_syntheses}

VERIFICATION TEAM FEEDBACK (use this to filter/adjust findings):
{verification_text}

Instructions:
1. For each finding in the syntheses, check what the verification team said
2. REMOVE findings marked as FALSE_POSITIVE, USER_CONFIG, or CIRCULAR
3. DOWNGRADE severity where the verification team recommended
4. Keep only VERIFIED findings in CRITICAL/HIGH sections
5. Create a unified, ACCURATE, prioritized report"""
        )
        return response.text if hasattr(response, 'text') else str(response)


# =============================================================================
# FINAL POLISHER (1 Sonnet)
# =============================================================================

class FinalPolisher:
    """QA review, verification, and final formatting."""

    def __init__(self, sonnet: ClaudeCodeClient, logger: AgentLogger, context: dict[str, str]):
        self.sonnet = sonnet
        self.logger = logger
        self.context = context

    async def polish(self, merged_report: str) -> str:
        """QA review, final verification, and format final report."""
        self.logger.progress("[POLISH] Final QA, verification, and formatting (1 Sonnet)...")

        qa_agent = self.sonnet.create_agent(
            name="qa_reviewer",
            instructions=f"""You are the FINAL QA REVIEWER. Your job is to ensure the report is ACCURATE.

{self.context['base_instructions']}

CRITICAL VERIFICATION (do this FIRST):
For each CRITICAL or HIGH finding in the report:
1. READ the actual code at the file:line reference
2. Verify the claim matches reality
3. If you find a design decision comment that explains the behavior, REMOVE the finding
4. If the threat model is wrong (user controls input), REMOVE or DOWNGRADE the finding

CONFIDENCE SCORING:
Add a confidence score to each finding:
- ✅ VERIFIED: You read the code and confirmed the issue
- ⚠️ LIKELY: Probable issue but couldn't fully verify
- ❓ UNVERIFIED: Claim couldn't be verified against code

FINAL CHECKS:
1. Remove any findings that are FALSE POSITIVES
2. Ensure severity ratings are accurate (not inflated)
3. Add executive summary with accurate issue counts
4. Format consistently with severity badges

Remember: A report with FEWER TRUE findings is better than one with FALSE POSITIVES.
If you find inflated claims during verification, REMOVE THEM.""",
        )
        qa = LoggedAgent(qa_agent, self.logger, "qa", "opus")  # UPGRADED

        response = await qa.run(
            f"""Review, verify, and polish this report:

{merged_report}

IMPORTANT STEPS:
1. For each CRITICAL/HIGH finding, READ the actual code to verify
2. Remove any false positives you discover
3. Add confidence scores (✅ VERIFIED / ⚠️ LIKELY / ❓ UNVERIFIED)
4. Add executive summary with ACCURATE counts
5. Ensure consistent formatting

The goal is an ACCURATE report, not a long one."""
        )
        return response.text if hasattr(response, 'text') else str(response)


# =============================================================================
# STATE CHECKPOINT HELPERS
# =============================================================================

def save_state_checkpoint(logger: AgentLogger, state: ReviewState, phase: str) -> None:
    """Save current state for crash recovery."""
    state_file = logger.checkpoints_dir / "current_state.json"

    state_data = {
        "phase": phase,
        "timestamp": datetime.now().isoformat(),
        "domain_findings": {
            d: {
                "domain": f.domain,
                "final_consensus": f.final_consensus,
                "validation_report": f.validation_report,
                "synthesis": f.synthesis,
            }
            for d, f in state.domain_findings.items()
        },
        "verification_findings": state.verification_findings,
        "cross_domain_synthesis": state.cross_domain_synthesis,
        "final_report": state.final_report,
    }

    with open(state_file, 'w') as f:
        json.dump(state_data, f, indent=2, default=str)
        f.flush()


def load_state_checkpoint(logger: AgentLogger) -> tuple[str, ReviewState] | None:
    """Load state from checkpoint for resume."""
    state_file = logger.checkpoints_dir / "current_state.json"

    if not state_file.exists():
        return None

    with open(state_file) as f:
        data = json.load(f)

    state = ReviewState()
    for domain, findings_data in data.get("domain_findings", {}).items():
        state.domain_findings[domain] = DomainFindings(
            domain=findings_data["domain"],
            final_consensus=findings_data.get("final_consensus", ""),
            validation_report=findings_data.get("validation_report", ""),
            synthesis=findings_data.get("synthesis", ""),
        )
    state.verification_findings = data.get("verification_findings", {})
    state.cross_domain_synthesis = data.get("cross_domain_synthesis", "")
    state.final_report = data.get("final_report", "")

    return data.get("phase", ""), state


# =============================================================================
# MAIN
# =============================================================================

async def main():
    """Run the codebase reviewer."""
    import argparse

    parser = argparse.ArgumentParser(description="Production Codebase Reviewer")
    parser.add_argument(
        "--codebase",
        default=str(Path.home() / "claude-code-provider" / "claude_code_provider"),
        help="Path to codebase to review"
    )
    parser.add_argument(
        "--ext",
        default=None,
        help="File extensions to review (e.g., 'js,ts' or 'py'). Auto-detects if not specified."
    )
    parser.add_argument(
        "--resume",
        nargs="?",
        const="auto",
        default=None,
        help="Resume from last checkpoint"
    )
    args = parser.parse_args()

    # Initialize clients - UPGRADED for Example 17:
    # - What was "sonnet" now uses Opus
    # - What was "haiku" now uses Sonnet
    # - Opus stays Opus (merger)
    opus = ClaudeCodeClient(model="opus", timeout=1800.0)
    sonnet = ClaudeCodeClient(model="opus", timeout=1800.0)  # UPGRADED: Sonnet → Opus
    haiku = ClaudeCodeClient(model="sonnet", timeout=1200.0)  # UPGRADED: Haiku → Sonnet
    logger = AgentLogger("17_opus_codebase_review", verbose=True)

    # Initialize global cost tracker
    global _cost_tracker
    _cost_tracker = CostTracker()

    # Initialize progress reporter
    progress_reporter = ProgressReporter(logger)

    print("=" * 70)
    print("OPUS CODEBASE REVIEWER (Example 17 - Full Premium)")
    print("=" * 70)
    print("""
MODEL UPGRADES (vs Example 16):
- Sonnet → Opus (5x capability, 5x cost)
- Haiku → Sonnet (3x capability, 3x cost)

Architecture:
- Domain Teams: 5 cells, each 3Op 3i (analyst + DEFENDER + synthesizer) [was: Sonnet]
- Validation: 5x Sonnet (checks design decision comments) [was: Haiku]
- Synthesis: 5x Opus [was: Sonnet]
- Verification Team: 3 cells × 3Op 3i each [was: Sonnet]
- Merge: 1 Opus (filters out false positives) [unchanged]
- Polish: 1 Opus (final verification with confidence scores) [was: Sonnet]

Anti-Inflation Measures:
- DEFENDER role in each team argues AGAINST findings
- Validation checks for design decision comments
- Verification team challenges threat models and severities
- Merger filters claims marked as false positives
- Final QA verifies CRITICAL/HIGH by reading code
- Confidence scores: VERIFIED / LIKELY / UNVERIFIED

Features:
- Connection pool: Semaphore(6) limits concurrent API calls
- Progress reporter: Sonnet reports on team completion [was: Haiku]
- Immediate log flush: Crash-safe logging
- State checkpoints: Resume capability

Estimated cost: ~$25-75 (5x more than Example 16)
""")

    # Target codebase
    codebase_path = args.codebase
    codebase_dir = Path(codebase_path)

    # Language extension mapping
    LANG_EXTENSIONS = {
        "python": ["py"],
        "javascript": ["js", "mjs", "cjs"],
        "typescript": ["ts", "tsx"],
        "rust": ["rs"],
        "go": ["go"],
        "java": ["java"],
    }

    # Determine extensions
    if args.ext:
        extensions = [e.strip().lstrip('.') for e in args.ext.split(',')]
    else:
        ext_counts = {}
        for ext_list in LANG_EXTENSIONS.values():
            for ext in ext_list:
                count = len(list(codebase_dir.glob(f"**/*.{ext}")))
                if count > 0:
                    ext_counts[ext] = count

        extensions = list(ext_counts.keys()) if ext_counts else ["py"]
        if ext_counts:
            print(f"Auto-detected: {ext_counts}")

    # Gather files
    code_files = []
    for ext in extensions:
        code_files.extend(codebase_dir.glob(f"*.{ext}"))
        code_files.extend(codebase_dir.glob(f"*/*.{ext}"))

    code_files = sorted(set(code_files))
    file_list = "\n".join(f"  - {f.relative_to(codebase_dir)}" for f in code_files)

    print(f"\nFiles to review: {len(code_files)}")
    for f in code_files[:10]:
        print(f"  - {f.relative_to(codebase_dir)}")
    if len(code_files) > 10:
        print(f"  ... and {len(code_files) - 10} more")
    print()

    # Context
    context = {
        "codebase_path": codebase_path,
        "file_list": file_list,
        "base_instructions": f"""
CODEBASE LOCATION: {codebase_path}

FILES:
{file_list}

INSTRUCTIONS:
- Use Read tool to examine files
- READ-ONLY - do not modify
- Include file:line references
- Rate severity: CRITICAL/HIGH/MEDIUM/LOW
""",
    }

    logger.set_config(
        codebase=codebase_path,
        files=len(code_files),
        extensions=extensions,
        architecture="3So 3i teams with connection pool",
        pool_size=6,
    )

    state = ReviewState()
    teams_completed = 0
    total_teams = 5 + 5 + 5 + 3 + 1 + 1  # domain + validation + synthesis + verification + merge + polish = 20
    resume_from_phase = None

    # Resume logic
    if args.resume:
        checkpoint_result = load_state_checkpoint(logger)
        if checkpoint_result:
            last_phase, loaded_state = checkpoint_result
            state = loaded_state

            phase_order = [
                "phase1_domain_teams",
                "phase1_5_validation",
                "phase2_domain_synthesis",
                "phase3_verification",
                "phase4_merge",
                "phase5_polish",
            ]

            if last_phase in phase_order:
                last_idx = phase_order.index(last_phase)
                if last_idx + 1 < len(phase_order):
                    resume_from_phase = phase_order[last_idx + 1]
                    teams_per_phase = [5, 5, 5, 3, 1, 1]
                    teams_completed = sum(teams_per_phase[:last_idx + 1])
                    print(f"\n[RESUME] Found checkpoint at {last_phase}")
                    print(f"         Resuming from {resume_from_phase}")
                    print(f"         Teams already completed: {teams_completed}/{total_teams}")
                else:
                    print(f"\n[RESUME] All phases already completed.")
                    return
            else:
                print(f"\n[RESUME] Unknown phase '{last_phase}'. Starting fresh.")
        else:
            print(f"\n[RESUME] No checkpoint found. Starting fresh.")

    def should_run_phase(phase_name: str) -> bool:
        if resume_from_phase is None:
            return True
        phase_order = [
            "phase1_domain_teams",
            "phase1_5_validation",
            "phase2_domain_synthesis",
            "phase3_verification",
            "phase4_merge",
            "phase5_polish",
        ]
        resume_idx = phase_order.index(resume_from_phase)
        current_idx = phase_order.index(phase_name)
        return current_idx >= resume_idx

    # =========================================================================
    # PHASE 1: Domain Teams (3So 3i each)
    # =========================================================================
    if should_run_phase("phase1_domain_teams"):
        logger.start_phase("phase1_domain_teams")

        teams_config = [
            ("security", ["Command injection", "Input validation", "Path traversal", "Secrets handling"]),
            ("api_design", ["API consistency", "Parameter validation", "Type hints", "Error messages"]),
            ("performance", ["Algorithm efficiency", "Memory management", "Async patterns", "Bottlenecks"]),
            ("testing", ["Test coverage", "Edge cases", "Error scenarios", "Mock usage"]),
            ("architecture", ["Module organization", "Separation of concerns", "Design patterns", "SOLID"]),
        ]

        teams = [
            DomainTeam(domain, focus, sonnet, logger, context)
            for domain, focus in teams_config
        ]

        async def run_domain_team(team: DomainTeam) -> DomainFindings:
            nonlocal teams_completed
            result = await team.run_review()
            teams_completed += 1

            await progress_reporter.report_team_completion(
                team_name=f"{team.domain} domain team",
                phase="Domain Analysis",
                summary=result.final_consensus[:200] if result.final_consensus else "Analysis complete",
                teams_completed=teams_completed,
                teams_total=total_teams,
            )
            return result

        team_results = await asyncio.gather(*[run_domain_team(t) for t in teams])

        for findings in team_results:
            state.domain_findings[findings.domain] = findings

        logger.end_phase("phase1_domain_teams", findings={
            d: f.final_consensus[:500] + "..." for d, f in state.domain_findings.items()
        })
        save_state_checkpoint(logger, state, "phase1_domain_teams")

    # =========================================================================
    # PHASE 1.5: Evidence Validation
    # =========================================================================
    if should_run_phase("phase1_5_validation"):
        logger.start_phase("phase1_5_validation")

        validator = EvidenceValidator(haiku, logger, context)

        async def run_validation(domain: str, findings: DomainFindings) -> tuple[str, str]:
            nonlocal teams_completed
            report = await validator.validate(domain, findings.final_consensus)
            teams_completed += 1
            return domain, report

        validation_results = await asyncio.gather(*[
            run_validation(domain, findings)
            for domain, findings in state.domain_findings.items()
        ])

        for domain, report in validation_results:
            state.domain_findings[domain].validation_report = report

        await progress_reporter.report_team_completion(
            team_name="All validation teams",
            phase="Validation",
            summary="All 5 domain findings validated",
            teams_completed=teams_completed,
            teams_total=total_teams,
        )

        logger.end_phase("phase1_5_validation", findings={
            d: f.validation_report[:300] + "..." for d, f in state.domain_findings.items()
        })
        save_state_checkpoint(logger, state, "phase1_5_validation")

    # =========================================================================
    # PHASE 2: Per-Domain Synthesis
    # =========================================================================
    if should_run_phase("phase2_domain_synthesis"):
        logger.start_phase("phase2_domain_synthesis")

        synthesizer = DomainSynthesizer(sonnet, logger)

        async def run_synthesis(domain: str, findings: DomainFindings) -> tuple[str, str]:
            nonlocal teams_completed
            synthesis = await synthesizer.synthesize(domain, findings)
            teams_completed += 1
            return domain, synthesis

        synth_results = await asyncio.gather(*[
            run_synthesis(domain, findings)
            for domain, findings in state.domain_findings.items()
        ])

        for domain, synthesis in synth_results:
            state.domain_findings[domain].synthesis = synthesis

        await progress_reporter.report_team_completion(
            team_name="All synthesis teams",
            phase="Synthesis",
            summary="All 5 domain syntheses complete",
            teams_completed=teams_completed,
            teams_total=total_teams,
        )

        logger.end_phase("phase2_domain_synthesis", findings={
            d: f.synthesis for d, f in state.domain_findings.items()
        })
        save_state_checkpoint(logger, state, "phase2_domain_synthesis")

    # Get domain_syntheses
    domain_syntheses = {d: f.synthesis for d, f in state.domain_findings.items()}

    # =========================================================================
    # PHASE 3: Verification Team (3 cells, each 3So 3i)
    # =========================================================================
    if should_run_phase("phase3_verification"):
        logger.start_phase("phase3_verification")

        verification_team = VerificationTeam(sonnet, logger, context)
        state.verification_findings = await verification_team.verify(domain_syntheses)
        teams_completed += 3

        await progress_reporter.report_team_completion(
            team_name="Verification Team (3 cells)",
            phase="Verification",
            summary="Claim verification and severity calibration complete",
            teams_completed=teams_completed,
            teams_total=total_teams,
        )

        logger.end_phase("phase3_verification", findings=state.verification_findings)
        save_state_checkpoint(logger, state, "phase3_verification")

    # =========================================================================
    # PHASE 4: Cross-Domain Merge (1 Opus)
    # =========================================================================
    if should_run_phase("phase4_merge"):
        logger.start_phase("phase4_merge")

        merger = CrossDomainMerger(opus, logger)
        state.cross_domain_synthesis = await merger.merge(
            domain_syntheses,
            state.verification_findings,
        )
        teams_completed += 1

        await progress_reporter.report_team_completion(
            team_name="Merge (Opus)",
            phase="Merge",
            summary="Cross-domain synthesis complete",
            teams_completed=teams_completed,
            teams_total=total_teams,
        )

        logger.end_phase("phase4_merge", findings={"merged": state.cross_domain_synthesis[:1000] + "..."})
        save_state_checkpoint(logger, state, "phase4_merge")

    # =========================================================================
    # PHASE 5: Final Polish (1 Sonnet)
    # =========================================================================
    if should_run_phase("phase5_polish"):
        logger.start_phase("phase5_polish")

        polisher = FinalPolisher(sonnet, logger, context)
        state.final_report = await polisher.polish(state.cross_domain_synthesis)
        teams_completed += 1

        logger.end_phase("phase5_polish", findings={"final": state.final_report[:1000] + "..."})
        save_state_checkpoint(logger, state, "phase5_polish")

    # =========================================================================
    # Output
    # =========================================================================
    print("\n" + "=" * 70)
    print("FINAL REPORT")
    print("=" * 70)
    print(state.final_report)

    # Save
    report_path, analysis_path = logger.save()

    final_path = logger.reports_dir / "CODEBASE_REVIEW.md"
    final_path.write_text(state.final_report)

    print("\n" + "=" * 70)
    print("OUTPUT FILES:")
    print(f"  Full Log: {report_path}")
    print(f"  Analysis: {analysis_path}")
    print(f"  Final Report: {final_path}")
    print("=" * 70)

    # Cost summary
    cost_summary = _cost_tracker.get_summary() if _cost_tracker else None

    print("\n" + "=" * 70)
    print("COST & TOKEN SUMMARY")
    print("=" * 70)

    if cost_summary:
        print(f"Total Requests: {cost_summary.total_requests}")
        print(f"Total Tokens: {cost_summary.total_input_tokens + cost_summary.total_output_tokens:,}")
        print(f"  - Input:  {cost_summary.total_input_tokens:,}")
        print(f"  - Output: {cost_summary.total_output_tokens:,}")
        print(f"Total Cost: ${cost_summary.total_cost:.4f}")

        if cost_summary.by_model:
            print("\nBy Model:")
            for model, stats in cost_summary.by_model.items():
                print(f"  {model}: {stats['requests']} calls, "
                      f"{stats['input_tokens'] + stats['output_tokens']:,} tokens, "
                      f"${stats['cost']:.4f}")
    else:
        print(f"Total Tokens: {logger.total_input_tokens + logger.total_output_tokens:,}")
        print(f"Total Cost: ${logger.total_cost:.4f}")

    print(f"\nTotal Agent Calls: {len(logger.calls)}")
    print(f"Progress Reports: {progress_reporter._report_count}")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
