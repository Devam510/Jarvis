"""
jarvis.utils.types — Core dataclasses used across all modules.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Optional, Union

import numpy as np


def _uid() -> str:
    return str(uuid.uuid4())


def _now() -> float:
    return time.time()


# ── Audio Events ─────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class WakeEvent:
    confidence: float = 1.0
    timestamp: float = field(default_factory=_now)
    correlation_id: str = field(default_factory=_uid)


@dataclass
class SpeechSegment:
    """Preprocessed speech ready for STT.

    `audio` is a float32 numpy array, peak-normalized, 16 kHz mono.
    Falls back to accepting raw bytes for backward compat.
    """

    audio: Union[np.ndarray, bytes]  # float32 numpy preferred, raw PCM int16 accepted
    sample_rate: int = 16000
    duration_ms: float = 0.0
    avg_speech_prob: float = 0.0
    peak: float = 0.0  # peak amplitude after normalization
    rms: float = 0.0  # RMS energy after normalization
    noise_floor: float = 0.0  # estimated noise floor
    correlation_id: str = field(default_factory=_uid)


# ── STT Events ───────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class TranscriptEvent:
    text: str
    confidence: float
    language: str = "en"
    timestamp: float = field(default_factory=_now)
    correlation_id: str = field(default_factory=_uid)


@dataclass(frozen=True)
class EmptyTranscript:
    timestamp: float = field(default_factory=_now)
    correlation_id: str = field(default_factory=_uid)


# ── Cognition Events ─────────────────────────────────────────────────────────


@dataclass
class PlannedAction:
    tool_name: str
    arguments: dict[str, Any]
    requires_confirmation: bool = False
    depends_on: Optional[int] = None  # index of prior action
    retry_count: int = 0
    rollback_command: Optional[dict[str, Any]] = None  # Phase 3: optional undo

    def to_dict(self) -> dict:
        d = {
            "tool": self.tool_name,
            "args": self.arguments,
            "requires_confirmation": self.requires_confirmation,
            "depends_on": self.depends_on,
        }
        if self.rollback_command:
            d["rollback_command"] = self.rollback_command
        return d


@dataclass
class Plan:
    thought: str
    confidence: float
    actions: list[PlannedAction]
    correlation_id: str = field(default_factory=_uid)

    @property
    def pending_steps(self) -> list[PlannedAction]:
        return list(self.actions)

    def next_step(self) -> PlannedAction:
        return self.actions.pop(0)

    @property
    def remaining_steps(self) -> list[PlannedAction]:
        return list(self.actions)

    def re_enqueue(self, step: PlannedAction):
        self.actions.insert(0, step)


@dataclass(frozen=True)
class ToolCallEvent:
    tool_name: str
    arguments: dict[str, Any]
    correlation_id: str = field(default_factory=_uid)
    source_transcript: str = ""


@dataclass(frozen=True)
class ValidatedToolCall:
    tool_name: str
    arguments: dict[str, Any]
    validated_at: float = field(default_factory=_now)
    correlation_id: str = field(default_factory=_uid)


# ── Risk Events ──────────────────────────────────────────────────────────────

from jarvis.utils.enums import RiskTier, ConfirmStatus


@dataclass(frozen=True)
class HeuristicResult:
    score: float
    tier: RiskTier
    triggers: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class LLMRiskResult:
    risk_score: float
    reasoning: str
    recommended_tier: int


@dataclass(frozen=True)
class RiskClassification:
    tool_call: ValidatedToolCall
    tier: RiskTier
    heuristic_score: float
    llm_risk_score: float = 0.0
    reasoning: str = ""
    correlation_id: str = field(default_factory=_uid)


@dataclass(frozen=True)
class ConfirmResult:
    status: ConfirmStatus
    action_summary: str = ""
    timestamp: float = field(default_factory=_now)


# ── Execution Events ─────────────────────────────────────────────────────────


@dataclass
class StepResult:
    status: str  # "success", "failed", "timeout"
    output: Any = None
    error: Optional[str] = None
    duration_ms: float = 0.0

    def to_dict(self) -> dict:
        return {
            "status": self.status,
            "output": str(self.output)[:500] if self.output else None,
            "error": self.error,
        }


@dataclass
class GoalResult:
    goal: str = ""
    status: str = "completed"  # completed | partial | aborted | timeout | deadlock
    steps_completed: int = 0
    steps_failed: int = 0
    total_iterations: int = 0
    results: list[StepResult] = field(default_factory=list)
    # Phase 3 extensions
    correlation_id: str = field(default_factory=_uid)  # [S12]
    duration_ms: float = 0.0
    confidence: float = 0.0
    rollback_log: list[str] = field(default_factory=list)  # [S7]
    telemetry: Optional["AutonomyTelemetry"] = None  # [S17]


# ── Phase 3: Autonomy Types ──────────────────────────────────────────────────


@dataclass
class GoalSnapshot:
    """[S13] Frozen initial context captured before autonomous execution.
    Rollback uses this snapshot, not mutated runtime state."""

    goal: str
    correlation_id: str
    captured_at: float = field(default_factory=_now)
    initial_context: str = ""
    working_directory: str = ""


@dataclass
class ResourceBudget:
    """[S16] Per-goal resource limits. Hard-stop when any counter hits 0."""

    max_tool_calls: int = 30
    max_tokens: int = 10000
    max_execution_time: float = 300.0
    # Mutable counters
    tool_calls_used: int = 0
    tokens_used: int = 0
    start_time: float = field(default_factory=_now)

    @property
    def tool_calls_remaining(self) -> int:
        return self.max_tool_calls - self.tool_calls_used

    @property
    def tokens_remaining(self) -> int:
        return self.max_tokens - self.tokens_used

    @property
    def time_remaining(self) -> float:
        return max(0.0, self.max_execution_time - (time.time() - self.start_time))

    @property
    def exhausted(self) -> bool:
        return (
            self.tool_calls_remaining <= 0
            or self.tokens_remaining <= 0
            or self.time_remaining <= 0
        )

    def consume_tool_call(self):
        self.tool_calls_used += 1

    def consume_tokens(self, n: int):
        self.tokens_used += n


@dataclass
class AutonomyTelemetry:
    """[S17] Metrics recorded per autonomous goal for future tuning."""

    iteration_count: int = 0
    confidences: list[float] = field(default_factory=list)
    step_latencies_ms: list[float] = field(default_factory=list)
    rollback_count: int = 0
    total_tool_calls: int = 0
    total_tokens: int = 0
    deadlock_detected: bool = False
    aborted: bool = False

    @property
    def avg_step_latency_ms(self) -> float:
        return (
            sum(self.step_latencies_ms) / len(self.step_latencies_ms)
            if self.step_latencies_ms
            else 0.0
        )


@dataclass
class ExecutionTransaction:
    """Tracks executed steps and their rollback commands for atomic reversal."""

    correlation_id: str = field(default_factory=_uid)
    completed_steps: list[tuple[PlannedAction, StepResult]] = field(
        default_factory=list
    )

    def record(self, action: PlannedAction, result: StepResult):
        self.completed_steps.append((action, result))

    @property
    def rollback_actions(self) -> list[dict[str, Any]]:
        """Return rollback commands in reverse order (LIFO)."""
        cmds = []
        for action, _ in reversed(self.completed_steps):
            if action.rollback_command:
                cmds.append(action.rollback_command)
        return cmds


@dataclass
class LoopDetectionState:
    """[S3] Sliding window of canonical plan hashes for deadlock detection."""

    window_size: int = 3
    plan_hashes: list[str] = field(default_factory=list)

    def push(self, plan_hash: str) -> bool:
        """Push a new hash. Returns True if deadlock detected."""
        self.plan_hashes.append(plan_hash)
        # Keep only the last `window_size` hashes
        if len(self.plan_hashes) > self.window_size * 2:
            self.plan_hashes = self.plan_hashes[-self.window_size * 2 :]
        # Deadlock = last N hashes are all identical
        if len(self.plan_hashes) >= self.window_size:
            recent = self.plan_hashes[-self.window_size :]
            if len(set(recent)) == 1:
                return True
        return False


@dataclass
class SandboxResult:
    status: str
    stdout: str = ""
    stderr: str = ""
    exit_code: int = 0


# ── Memory Types ─────────────────────────────────────────────────────────────

from jarvis.utils.enums import MemoryType


@dataclass
class MemoryEntry:
    content: str
    entry_type: MemoryType = MemoryType.CONVERSATION
    importance: float = 0.5
    id: str = field(default_factory=_uid)
    embedding: list[float] = field(default_factory=list)
    access_count: int = 0
    last_accessed: float = field(default_factory=_now)
    created_at: float = field(default_factory=_now)
    source_correlation_id: str = ""
    metadata: dict = field(default_factory=dict)


# ── Phase 4: Perception Types ────────────────────────────────────────────


@dataclass(frozen=True)  # [S6] immutable once emitted on EventBus
class ScreenContext:
    """Live snapshot of what's on screen. Frozen to prevent mutation after emit."""

    active_app: str = ""
    window_title: str = ""
    visible_text: str = ""
    detected_errors: tuple[str, ...] = ()  # tuple for frozen compat
    timestamp: float = field(default_factory=_now)
    screenshot_path: str = ""
    change_magnitude: float = 0.0  # 0-1, how much the screen changed
    is_blacked_out: bool = False  # [S2] True when privacy mode active


@dataclass(frozen=True)  # [S8] informational only — never auto-executed
class Suggestion:
    """A proactive suggestion from the perception engine."""

    trigger: str  # what triggered it (e.g. "error_detected")
    message: str  # what to say to the user
    confidence: float  # 0-1
    action_hint: str = ""  # informational only — never auto-executed
    timestamp: float = field(default_factory=_now)
    id: str = field(default_factory=_uid)


@dataclass
class BehaviorPattern:
    """A learned user behavior pattern for intent prediction."""

    context_hash: str  # hash of (time_bucket, active_app)
    action: str  # what the user typically does
    count: int = 1
    last_seen: float = field(default_factory=_now)
    metadata: dict = field(default_factory=dict)
    id: str = field(default_factory=_uid)


# ── Vision Types ─────────────────────────────────────────────────────────


@dataclass(frozen=True)
class VisionResult:
    description: str
    screenshot_path: str
    timestamp: float = field(default_factory=_now)


# ── Health Types ─────────────────────────────────────────────────────────────


@dataclass
class HealthCheck:
    ok: bool
    latency_ms: float = 0.0
    error: str = ""


@dataclass
class HealthReport:
    ollama: HealthCheck = field(default_factory=lambda: HealthCheck(ok=False))
    chromadb: HealthCheck = field(default_factory=lambda: HealthCheck(ok=False))
    audio: HealthCheck = field(default_factory=lambda: HealthCheck(ok=False))
    docker: HealthCheck = field(default_factory=lambda: HealthCheck(ok=False))
    disk: HealthCheck = field(default_factory=lambda: HealthCheck(ok=False))
    overall: str = "unknown"


# ── Observability Types ──────────────────────────────────────────────────────


@dataclass
class TraceStage:
    stage_name: str
    started_at: float = field(default_factory=_now)
    completed_at: float = 0.0
    input_summary: str = ""
    output_summary: str = ""
    status: str = "in_progress"
    metadata: dict = field(default_factory=dict)


@dataclass
class ExecutionTrace:
    correlation_id: str = field(default_factory=_uid)
    stages: list[TraceStage] = field(default_factory=list)


# ── Critique / Reflection Types ──────────────────────────────────────────────


@dataclass
class CritiqueResult:
    approved: bool = True
    issues: list[str] = field(default_factory=list)
    revised_actions: Optional[list[dict]] = None


@dataclass
class Reflection:
    should_re_plan: bool = False
    reasoning: str = ""
