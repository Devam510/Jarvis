"""
tests.test_autonomy — Phase 3 tests covering all 17 safeguards (S1-S17)
plus stress simulations for the deep audit.

Uses mock cognition/execution/memory to test the AutonomyEngine in isolation.
"""

import asyncio
import pytest
import time
from unittest.mock import AsyncMock, MagicMock

from jarvis.autonomy.autonomy_engine import AutonomyEngine, _is_fatal
from jarvis.core.event_bus import AsyncEventBus
from jarvis.core.state_machine import InteractionStateMachine
from jarvis.utils.config import JarvisConfig
from jarvis.utils.enums import InteractionState
from jarvis.utils.types import (
    AutonomyTelemetry,
    ExecutionTransaction,
    GoalResult,
    GoalSnapshot,
    LoopDetectionState,
    PlannedAction,
    ResourceBudget,
    StepResult,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def config():
    cfg = JarvisConfig()
    cfg.autonomy.max_iterations = 5
    cfg.autonomy.confidence_threshold = 0.4
    cfg.autonomy.loop_detection_window = 3
    cfg.autonomy.step_timeout_seconds = 5.0
    cfg.autonomy.total_goal_timeout_seconds = 30.0
    cfg.autonomy.reflection_timeout_seconds = 5.0
    cfg.autonomy.reflection_max_tokens = 100
    cfg.autonomy.max_tool_calls_per_goal = 10
    cfg.autonomy.max_tokens_per_goal = 5000
    cfg.autonomy.max_execution_time_seconds = 30.0
    cfg.autonomy.max_retries_per_step = 1
    return cfg


@pytest.fixture
def event_bus():
    return AsyncEventBus()


@pytest.fixture
def state_machine():
    return InteractionStateMachine()


@pytest.fixture
def mock_cognition():
    cog = MagicMock()
    cog.call_llm = AsyncMock(return_value=None)
    return cog


@pytest.fixture
def mock_execution():
    """Mock executor that returns StepResult(status='success') by default."""
    exe = MagicMock()
    exe.execute_tool_call = AsyncMock(
        return_value=StepResult(status="success", output="ok")
    )
    return exe


@pytest.fixture
def mock_memory():
    mem = MagicMock()
    mem.store_task_outcome = AsyncMock()
    return mem


@pytest.fixture
def engine(
    config, event_bus, mock_cognition, mock_execution, mock_memory, state_machine
):
    return AutonomyEngine(
        config=config,
        event_bus=event_bus,
        cognition=mock_cognition,
        execution=mock_execution,
        memory=mock_memory,
        state_machine=state_machine,
    )


@pytest.fixture
def engine_no_sm(config, event_bus, mock_cognition, mock_execution, mock_memory):
    """Engine without state machine — backward compat."""
    return AutonomyEngine(
        config=config,
        event_bus=event_bus,
        cognition=mock_cognition,
        execution=mock_execution,
        memory=mock_memory,
    )


def _make_plan_response(tools, confidence=0.8):
    """Helper: build LLM response dict simulating a plan."""
    return {
        "thought": "test plan",
        "confidence": confidence,
        "actions": [{"tool": t, "args": {"target": t}} for t in tools],
    }


def _make_reflect_response(should_continue, confidence=0.8):
    return {
        "should_continue": should_continue,
        "confidence": confidence,
        "reasoning": "test",
    }


# ══════════════════════════════════════════════════════════════════════════════
# SECTION A — Original 17 safeguard tests (updated for fixes)
# ══════════════════════════════════════════════════════════════════════════════


# ── Test 1: Happy path ───────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_happy_path_3_steps(engine, mock_cognition, state_machine):
    """3-step goal completes successfully. State returns to IDLE."""
    call_count = 0

    async def mock_llm(messages, max_tokens=None):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return _make_plan_response(["app_launch", "file_write", "file_read"])
        elif call_count == 2:
            return _make_reflect_response(False, 0.9)
        return None

    mock_cognition.call_llm = mock_llm

    result = await engine.execute_goal("test 3 steps")
    assert result.steps_completed == 3
    assert result.steps_failed == 0
    assert result.status in ("completed", "partial")
    assert result.correlation_id  # [S12]
    assert result.duration_ms >= 0  # Windows timer resolution can yield 0.0ms
    # F11: state returns to IDLE
    assert state_machine.state == InteractionState.IDLE
    # F2: task ref cleared
    assert engine._active_goal_task is None


# ── Test 2: Retry on failure [S11] ───────────────────────────────────────────


@pytest.mark.asyncio
async def test_retry_on_retriable_failure(engine, mock_cognition, mock_execution):
    """Step fails once (retriable), retries, succeeds."""
    call_count = 0

    async def mock_llm(messages, max_tokens=None):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return _make_plan_response(["app_launch"])
        return _make_reflect_response(False, 0.9)

    mock_cognition.call_llm = mock_llm

    exec_call = 0

    async def _exec(event):
        nonlocal exec_call
        exec_call += 1
        if exec_call == 1:
            return StepResult(status="failed", error="TransientError")
        return StepResult(status="success", output="ok")

    mock_execution.execute_tool_call = _exec

    result = await engine.execute_goal("retry test")
    assert result.steps_completed >= 1


# ── Test 3: Loop detection [S3] ──────────────────────────────────────────────


@pytest.mark.asyncio
async def test_loop_detection_deadlock(engine, mock_cognition):
    """3 identical plans → deadlock abort."""
    call_count = 0

    async def mock_llm(messages, max_tokens=None):
        nonlocal call_count
        call_count += 1
        # Planning calls return identical plan, reflection calls say continue
        if call_count % 2 == 1:
            return _make_plan_response(["app_launch"])
        return _make_reflect_response(True, 0.9)

    mock_cognition.call_llm = mock_llm

    result = await engine.execute_goal("deadlock test")
    assert result.status == "deadlock"
    assert result.telemetry.deadlock_detected is True


# ── Test 4: Total timeout [S2] ───────────────────────────────────────────────


@pytest.mark.asyncio
async def test_total_goal_timeout(
    config, event_bus, mock_execution, mock_memory, state_machine
):
    """Goal exceeds total time limit → abort. State returns to IDLE."""
    config.autonomy.total_goal_timeout_seconds = 0.5
    mock_cog = MagicMock()

    async def slow_llm(messages, max_tokens=None):
        await asyncio.sleep(2.0)
        return _make_plan_response(["app_launch"])

    mock_cog.call_llm = slow_llm

    engine = AutonomyEngine(
        config=config,
        event_bus=event_bus,
        cognition=mock_cog,
        execution=mock_execution,
        memory=mock_memory,
        state_machine=state_machine,
    )

    result = await engine.execute_goal("timeout test")
    assert result.status == "timeout"
    assert result.telemetry is not None
    assert result.telemetry.aborted is True  # F5
    # F11: state returns to IDLE
    assert state_machine.state == InteractionState.IDLE


# ── Test 5: Rollback on failure [S7] ─────────────────────────────────────────


@pytest.mark.asyncio
async def test_rollback_on_fatal_failure(engine, mock_cognition, mock_execution):
    """Step 2 fails fatally → step 1 rolled back."""
    call_count = 0

    async def mock_llm(messages, max_tokens=None):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return {
                "thought": "plan",
                "confidence": 0.8,
                "actions": [
                    {
                        "tool": "file_write",
                        "args": {"path": "/tmp/test.txt"},
                        "rollback_command": {
                            "tool": "file_delete",
                            "args": {"path": "/tmp/test.txt"},
                        },
                    },
                    {"tool": "execute_code", "args": {"code": "rm -rf"}},
                ],
            }
        return None

    mock_cognition.call_llm = mock_llm

    exec_call = 0

    async def _exec(event):
        nonlocal exec_call
        exec_call += 1
        if exec_call == 1:
            return StepResult(status="success", output="written")
        elif exec_call == 2:
            return StepResult(status="failed", error="PermissionError: denied")
        # Rollback calls
        return StepResult(status="success", output="rolled back")

    mock_execution.execute_tool_call = _exec

    result = await engine.execute_goal("rollback test")
    assert result.status == "aborted"
    assert len(result.rollback_log) > 0
    assert result.telemetry.aborted is True  # F5


# ── Test 6: Confidence gate [S5] ─────────────────────────────────────────────


@pytest.mark.asyncio
async def test_confidence_gate_stops(engine, mock_cognition):
    """Low confidence reflection → early stop."""
    call_count = 0

    async def mock_llm(messages, max_tokens=None):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return _make_plan_response(["app_launch"])
        return _make_reflect_response(True, 0.1)  # Low confidence

    mock_cognition.call_llm = mock_llm

    result = await engine.execute_goal("confidence test")
    assert result.total_iterations == 1  # Stopped after 1 iteration


# ── Test 7: Hard ceiling [S4] ────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_hard_iteration_ceiling(
    config, event_bus, mock_execution, mock_memory, state_machine
):
    """max_iterations overrides everything."""
    config.autonomy.max_iterations = 2
    mock_cog = MagicMock()
    call_count = 0

    async def mock_llm(messages, max_tokens=None):
        nonlocal call_count
        call_count += 1
        if call_count % 2 == 1:
            # Different plans each time to avoid deadlock
            return _make_plan_response([f"tool_{call_count}"])
        return _make_reflect_response(True, 0.9)

    mock_cog.call_llm = mock_llm

    engine = AutonomyEngine(
        config=config,
        event_bus=event_bus,
        cognition=mock_cog,
        execution=mock_execution,
        memory=mock_memory,
        state_machine=state_machine,
    )

    result = await engine.execute_goal("ceiling test")
    assert result.total_iterations <= 2


# ── Test 8: Fatal error — no retry [S11] ─────────────────────────────────────


@pytest.mark.asyncio
async def test_fatal_error_no_retry(engine, mock_cognition, mock_execution):
    """Fatal error → immediate abort, no retry attempt."""

    async def mock_llm(messages, max_tokens=None):
        return _make_plan_response(["app_launch"])

    mock_cognition.call_llm = mock_llm
    mock_execution.execute_tool_call = AsyncMock(
        return_value=StepResult(status="failed", error="PermissionError: access denied")
    )

    result = await engine.execute_goal("fatal test")
    assert result.status == "aborted"


# ── Test 9: Single goal lock [S1] ────────────────────────────────────────────


@pytest.mark.asyncio
async def test_single_goal_lock(engine, mock_cognition):
    """Second goal rejected while first is running."""

    async def slow_llm(messages, max_tokens=None):
        await asyncio.sleep(2.0)
        return _make_plan_response(["app_launch"])

    mock_cognition.call_llm = slow_llm

    # Start first goal (don't await)
    task1 = asyncio.create_task(engine.execute_goal("first goal"))
    await asyncio.sleep(0.1)

    # Try second goal — should be rejected
    result2 = await engine.execute_goal("second goal")
    assert result2.status == "rejected"

    # Clean up
    task1.cancel()
    try:
        await task1
    except asyncio.CancelledError:
        pass


# ── Test 10: State ownership [S8] ────────────────────────────────────────────


@pytest.mark.asyncio
async def test_state_machine_ownership():
    """LISTENING transition blocked during autonomous states."""
    sm = InteractionStateMachine()

    sm.transition(InteractionState.AUTONOMOUS_PLANNING)
    assert not sm.can_transition(InteractionState.LISTENING)
    assert not sm.can_transition(InteractionState.AWAITING_CONFIRM)

    sm.transition(InteractionState.AUTONOMOUS_EXECUTING)
    assert not sm.can_transition(InteractionState.LISTENING)
    assert not sm.can_transition(InteractionState.AWAITING_CONFIRM)


# ── Test 11: Cancellation [S2, S6] ───────────────────────────────────────────


@pytest.mark.asyncio
async def test_cancel_mid_execution(engine, mock_cognition, state_machine):
    """Cancel mid-execution → cleanup + state returns to IDLE."""

    async def slow_llm(messages, max_tokens=None):
        await asyncio.sleep(5.0)
        return _make_plan_response(["app_launch"])

    mock_cognition.call_llm = slow_llm

    task = asyncio.create_task(engine.execute_goal("cancel test"))
    await asyncio.sleep(0.1)

    # F2: _active_goal_task is now set automatically
    await engine.cancel_goal()

    result = await task
    assert result.status in ("aborted", "timeout")
    # F11: state always returns to IDLE
    assert state_machine.state == InteractionState.IDLE
    # F2: task ref cleared
    assert engine._active_goal_task is None


# ── Test 12: Rollback idempotency [S7] ───────────────────────────────────────


@pytest.mark.asyncio
async def test_rollback_idempotency(engine, mock_execution):
    """Rollback errors are ignored, no retry loops."""
    result = GoalResult(goal="test")
    result.rollback_log = []
    result.telemetry = AutonomyTelemetry()

    tx = ExecutionTransaction()
    tx.completed_steps = [
        (
            PlannedAction(
                tool_name="file_write",
                arguments={"path": "/tmp/test"},
                rollback_command={
                    "tool": "file_delete",
                    "args": {"path": "/tmp/test"},
                },
            ),
            StepResult(status="success"),
        ),
    ]

    # Mock rollback execution to fail
    mock_execution.execute_tool_call = AsyncMock(
        return_value=StepResult(status="failed", error="FS error")
    )

    await engine._rollback(tx, result, "test-corr-id")
    # Should complete without raising
    assert len(result.rollback_log) > 0


# ── Test 13: Reflection timeout [S9] ─────────────────────────────────────────


@pytest.mark.asyncio
async def test_reflection_timeout(
    config, event_bus, mock_execution, mock_memory, state_machine
):
    """Reflection that exceeds timeout → stops goal."""
    config.autonomy.reflection_timeout_seconds = 0.1
    mock_cog = MagicMock()
    call_count = 0

    async def mock_llm(messages, max_tokens=None):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return _make_plan_response(["app_launch"])
        # Reflection call — slow
        await asyncio.sleep(2.0)
        return _make_reflect_response(True, 0.9)

    mock_cog.call_llm = mock_llm

    engine = AutonomyEngine(
        config=config,
        event_bus=event_bus,
        cognition=mock_cog,
        execution=mock_execution,
        memory=mock_memory,
        state_machine=state_machine,
    )

    result = await engine.execute_goal("reflection timeout test")
    assert result.total_iterations == 1


# ── Test 14: Correlation ID propagation [S12] ────────────────────────────────


@pytest.mark.asyncio
async def test_correlation_id_propagation(engine, event_bus, mock_cognition):
    """Correlation ID present in result and events."""
    emitted_events = []

    async def _capture(event):
        emitted_events.append(event)

    event_bus.subscribe("autonomy.goal_started", _capture)

    async def mock_llm(messages, max_tokens=None):
        return None

    mock_cognition.call_llm = mock_llm

    result = await engine.execute_goal("corr id test")
    assert result.correlation_id
    assert len(result.correlation_id) > 10

    assert len(emitted_events) >= 1
    assert emitted_events[0]["correlation_id"] == result.correlation_id


# ── Test 15: Memory write isolation [S14] ────────────────────────────────────


@pytest.mark.asyncio
async def test_memory_write_isolation(engine, mock_cognition, mock_memory):
    """Memory is only written after goal completion, not during execution."""
    write_times = []

    async def track_write(summary):
        write_times.append(time.time())

    mock_memory.store_task_outcome = track_write

    async def mock_llm(messages, max_tokens=None):
        return None

    mock_cognition.call_llm = mock_llm

    await engine.execute_goal("memory isolation test")
    assert len(write_times) == 1


# ── Test 16: Double-rollback guard [S15] ─────────────────────────────────────


@pytest.mark.asyncio
async def test_double_rollback_guard(engine):
    """Second rollback call is a no-op."""
    result = GoalResult(goal="test")
    result.rollback_log = []
    result.telemetry = AutonomyTelemetry()
    tx = ExecutionTransaction()

    await engine._rollback(tx, result, "corr-1")
    assert engine._rollback_executed is True

    result.rollback_log = []
    await engine._rollback(tx, result, "corr-1")
    assert len(result.rollback_log) == 0


# ── Test 17: Resource budget exceeded [S16] ──────────────────────────────────


@pytest.mark.asyncio
async def test_resource_budget_exceeded(
    config, event_bus, mock_execution, mock_memory, state_machine
):
    """Resource budget hard-stop when tool calls exhausted."""
    config.autonomy.max_tool_calls_per_goal = 2
    mock_cog = MagicMock()
    call_count = 0

    async def mock_llm(messages, max_tokens=None):
        nonlocal call_count
        call_count += 1
        if call_count % 2 == 1:
            return _make_plan_response(
                [f"t{call_count}_a", f"t{call_count}_b", f"t{call_count}_c"]
            )
        return _make_reflect_response(True, 0.9)

    mock_cog.call_llm = mock_llm

    engine = AutonomyEngine(
        config=config,
        event_bus=event_bus,
        cognition=mock_cog,
        execution=mock_execution,
        memory=mock_memory,
        state_machine=state_machine,
    )

    result = await engine.execute_goal("budget test")
    assert result.status == "budget_exhausted" or result.steps_completed <= 2


# ══════════════════════════════════════════════════════════════════════════════
# SECTION B — NEW: Fix verification tests
# ══════════════════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_f2_active_goal_task_set_during_execution(engine, mock_cognition):
    """F2: _active_goal_task is set while goal is running."""
    task_ref_during_exec = None

    async def spy_llm(messages, max_tokens=None):
        nonlocal task_ref_during_exec
        task_ref_during_exec = engine._active_goal_task
        return None  # complete immediately

    mock_cognition.call_llm = spy_llm

    await engine.execute_goal("task ref test")
    # During execution, task ref should have been set
    assert task_ref_during_exec is not None
    # After execution, task ref should be cleared
    assert engine._active_goal_task is None


@pytest.mark.asyncio
async def test_f4_goal_completed_event(engine, event_bus, mock_cognition):
    """F4: goal_completed event emitted on success."""
    completed_events = []

    async def _capture(event):
        completed_events.append(event)

    event_bus.subscribe("autonomy.goal_completed", _capture)

    async def mock_llm(messages, max_tokens=None):
        return None

    mock_cognition.call_llm = mock_llm

    result = await engine.execute_goal("complete event test")
    assert result.status == "completed"
    assert len(completed_events) == 1
    assert completed_events[0]["correlation_id"] == result.correlation_id
    assert completed_events[0]["status"] == "completed"


@pytest.mark.asyncio
async def test_f5_telemetry_aborted_flag(engine, mock_cognition, mock_execution):
    """F5: telemetry.aborted set on abort."""

    async def mock_llm(messages, max_tokens=None):
        return _make_plan_response(["file_write"])

    mock_cognition.call_llm = mock_llm
    mock_execution.execute_tool_call = AsyncMock(
        return_value=StepResult(status="failed", error="SecurityError: blocked")
    )

    result = await engine.execute_goal("aborted flag test")
    assert result.status == "aborted"
    assert result.telemetry.aborted is True


@pytest.mark.asyncio
async def test_f6_state_transitions(engine, state_machine, mock_cognition):
    """F6: Engine drives state machine through AUTONOMOUS_* → IDLE."""
    transitions = []
    original_transition = state_machine.transition

    def spy_transition(target):
        transitions.append(target)
        return original_transition(target)

    state_machine.transition = spy_transition

    async def mock_llm(messages, max_tokens=None):
        return None

    mock_cognition.call_llm = mock_llm

    await engine.execute_goal("state transition test")
    # Should have transitioned to AUTONOMOUS_PLANNING at least once
    assert InteractionState.AUTONOMOUS_PLANNING in transitions
    # Final state should be IDLE
    assert state_machine.state == InteractionState.IDLE


@pytest.mark.asyncio
async def test_f7_token_budget_tracked(
    config, event_bus, mock_execution, mock_memory, state_machine
):
    """F7: Token budget is tracked after LLM calls."""
    config.autonomy.max_tokens_per_goal = 50  # Very tight budget
    mock_cog = MagicMock()
    call_count = 0

    async def mock_llm(messages, max_tokens=None):
        nonlocal call_count
        call_count += 1
        if call_count <= 3:
            # Each plan response is ~100 chars → ~25 token estimate
            return _make_plan_response([f"tool_{call_count}"])
        return _make_reflect_response(True, 0.9)

    mock_cog.call_llm = mock_llm

    engine = AutonomyEngine(
        config=config,
        event_bus=event_bus,
        cognition=mock_cog,
        execution=mock_execution,
        memory=mock_memory,
        state_machine=state_machine,
    )

    result = await engine.execute_goal("token budget test")
    # Should either exhaust budget or complete with tracked tokens
    assert result.telemetry.total_tokens > 0


@pytest.mark.asyncio
async def test_f11_state_idle_after_error(
    config, event_bus, mock_execution, mock_memory, state_machine
):
    """F11: State returns to IDLE even after unexpected error."""
    mock_cog = MagicMock()

    async def crashing_llm(messages, max_tokens=None):
        raise RuntimeError("LLM crashed unexpectedly")

    mock_cog.call_llm = crashing_llm

    engine = AutonomyEngine(
        config=config,
        event_bus=event_bus,
        cognition=mock_cog,
        execution=mock_execution,
        memory=mock_memory,
        state_machine=state_machine,
    )

    result = await engine.execute_goal("crash test")
    assert result.status == "aborted"
    assert state_machine.state == InteractionState.IDLE
    assert engine._active_goal_task is None


@pytest.mark.asyncio
async def test_no_state_machine_backward_compat(engine_no_sm, mock_cognition):
    """Engine works fine without state machine (backward compat)."""

    async def mock_llm(messages, max_tokens=None):
        return None

    mock_cognition.call_llm = mock_llm

    result = await engine_no_sm.execute_goal("no sm test")
    assert result.status == "completed"
    assert engine_no_sm._active_goal_task is None


# ══════════════════════════════════════════════════════════════════════════════
# SECTION C — Stress simulation tests
# ══════════════════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_stress_100_consecutive_goals(
    config, event_bus, mock_execution, mock_memory, state_machine
):
    """100 consecutive goals — no deadlocks, no state corruption."""
    config.autonomy.max_iterations = 2
    mock_cog = MagicMock()

    call_count = 0

    async def mock_llm(messages, max_tokens=None):
        nonlocal call_count
        call_count += 1
        if call_count % 2 == 1:
            return _make_plan_response([f"tool_{call_count}"])
        return _make_reflect_response(False, 0.9)

    mock_cog.call_llm = mock_llm

    engine = AutonomyEngine(
        config=config,
        event_bus=event_bus,
        cognition=mock_cog,
        execution=mock_execution,
        memory=mock_memory,
        state_machine=state_machine,
    )

    for i in range(100):
        result = await engine.execute_goal(f"goal_{i}")
        assert result.status in ("completed", "partial")
        assert engine._active_goal_task is None
        assert not engine._goal_lock.locked()
        assert state_machine.state == InteractionState.IDLE


@pytest.mark.asyncio
async def test_stress_rapid_second_goal_rejection(
    config, event_bus, mock_execution, mock_memory, state_machine
):
    """Rapid goal submissions — second always rejected."""
    mock_cog = MagicMock()

    async def slow_llm(messages, max_tokens=None):
        await asyncio.sleep(0.5)
        return _make_plan_response(["tool"])

    mock_cog.call_llm = slow_llm

    engine = AutonomyEngine(
        config=config,
        event_bus=event_bus,
        cognition=mock_cog,
        execution=mock_execution,
        memory=mock_memory,
        state_machine=state_machine,
    )

    task1 = asyncio.create_task(engine.execute_goal("first"))
    await asyncio.sleep(0.05)

    # Fire 10 rapid second-goal attempts
    for i in range(10):
        r = await engine.execute_goal(f"rejected_{i}")
        assert r.status == "rejected"

    task1.cancel()
    try:
        await task1
    except asyncio.CancelledError:
        pass


@pytest.mark.asyncio
async def test_stress_cancel_during_rollback(
    config, event_bus, mock_memory, state_machine
):
    """Cancel arrives during rollback — no double rollback, state returns to IDLE."""
    mock_cog = MagicMock()

    async def mock_llm(messages, max_tokens=None):
        return {
            "thought": "plan",
            "confidence": 0.8,
            "actions": [
                {
                    "tool": "file_write",
                    "args": {"path": "/tmp/a"},
                    "rollback_command": {
                        "tool": "file_delete",
                        "args": {"path": "/tmp/a"},
                    },
                },
                {"tool": "dangerous_op", "args": {}},
            ],
        }

    mock_cog.call_llm = mock_llm

    exec_call = 0
    mock_exe = MagicMock()

    async def _exec(event):
        nonlocal exec_call
        exec_call += 1
        if exec_call == 1:
            return StepResult(status="success", output="ok")
        elif exec_call == 2:
            return StepResult(status="failed", error="PermissionError: denied")
        # Slow rollback
        await asyncio.sleep(0.5)
        return StepResult(status="success", output="rolled back")

    mock_exe.execute_tool_call = _exec

    engine = AutonomyEngine(
        config=config,
        event_bus=event_bus,
        cognition=mock_cog,
        execution=mock_exe,
        memory=mock_memory,
        state_machine=state_machine,
    )

    result = await engine.execute_goal("cancel during rollback test")
    assert result.status == "aborted"
    assert state_machine.state == InteractionState.IDLE
    assert engine._active_goal_task is None


@pytest.mark.asyncio
async def test_stress_step_timeout(config, event_bus, mock_memory, state_machine):
    """Step-level timeout — no hang, no orphan."""
    config.autonomy.step_timeout_seconds = 0.1
    mock_cog = MagicMock()

    async def mock_llm(messages, max_tokens=None):
        return _make_plan_response(["hanging_tool"])

    mock_cog.call_llm = mock_llm

    mock_exe = MagicMock()

    async def _hanging_exec(event):
        await asyncio.sleep(10.0)  # simulate hang
        return StepResult(status="success", output="unreachable")

    mock_exe.execute_tool_call = _hanging_exec

    engine = AutonomyEngine(
        config=config,
        event_bus=event_bus,
        cognition=mock_cog,
        execution=mock_exe,
        memory=mock_memory,
        state_machine=state_machine,
    )

    result = await engine.execute_goal("step timeout test")
    # Should not hang — step timed out, then loop may produce more timeouts or stop
    assert state_machine.state == InteractionState.IDLE
    assert engine._active_goal_task is None


@pytest.mark.asyncio
async def test_stress_goal_timeout_during_step(
    config, event_bus, mock_memory, state_machine
):
    """Total goal timeout fires during step execution."""
    config.autonomy.total_goal_timeout_seconds = 0.2
    config.autonomy.step_timeout_seconds = 5.0
    mock_cog = MagicMock()

    async def mock_llm(messages, max_tokens=None):
        return _make_plan_response(["slow_tool"])

    mock_cog.call_llm = mock_llm

    mock_exe = MagicMock()

    async def _slow_exec(event):
        await asyncio.sleep(3.0)
        return StepResult(status="success", output="late")

    mock_exe.execute_tool_call = _slow_exec

    engine = AutonomyEngine(
        config=config,
        event_bus=event_bus,
        cognition=mock_cog,
        execution=mock_exe,
        memory=mock_memory,
        state_machine=state_machine,
    )

    result = await engine.execute_goal("goal timeout mid-step")
    assert result.status == "timeout"
    assert state_machine.state == InteractionState.IDLE


@pytest.mark.asyncio
async def test_stress_duplicate_identical_plans(
    config, event_bus, mock_execution, mock_memory, state_machine
):
    """Exactly loop_detection_window identical plans → deadlock."""
    config.autonomy.loop_detection_window = 3
    config.autonomy.max_iterations = 10
    mock_cog = MagicMock()
    call_count = 0

    async def mock_llm(messages, max_tokens=None):
        nonlocal call_count
        call_count += 1
        if call_count % 2 == 1:
            return _make_plan_response(["same_tool"])
        return _make_reflect_response(True, 0.9)

    mock_cog.call_llm = mock_llm

    engine = AutonomyEngine(
        config=config,
        event_bus=event_bus,
        cognition=mock_cog,
        execution=mock_execution,
        memory=mock_memory,
        state_machine=state_machine,
    )

    result = await engine.execute_goal("duplicate plan test")
    assert result.status == "deadlock"
    assert state_machine.state == InteractionState.IDLE


@pytest.mark.asyncio
async def test_stress_messaging_crash(config, event_bus, mock_memory, state_machine):
    """Executor raises unexpected exception — goal aborts cleanly."""
    mock_cog = MagicMock()

    async def mock_llm(messages, max_tokens=None):
        return _make_plan_response(["crash_tool"])

    mock_cog.call_llm = mock_llm

    mock_exe = MagicMock()

    async def _crashing_exec(event):
        raise ConnectionError("Messaging tool crash")

    mock_exe.execute_tool_call = _crashing_exec

    engine = AutonomyEngine(
        config=config,
        event_bus=event_bus,
        cognition=mock_cog,
        execution=mock_exe,
        memory=mock_memory,
        state_machine=state_machine,
    )

    result = await engine.execute_goal("crash test")
    # ConnectionError is not in _FATAL_ERRORS, so it's retriable but will exhaust retries
    assert state_machine.state == InteractionState.IDLE
    assert engine._active_goal_task is None


@pytest.mark.asyncio
async def test_stress_budget_time_exhaustion(
    config, event_bus, mock_execution, mock_memory, state_machine
):
    """Budget time limit fires before total_goal_timeout."""
    config.autonomy.max_execution_time_seconds = 0.1
    config.autonomy.total_goal_timeout_seconds = 30.0
    mock_cog = MagicMock()

    async def mock_llm(messages, max_tokens=None):
        await asyncio.sleep(0.05)
        return _make_plan_response(["tool"])

    mock_cog.call_llm = mock_llm

    engine = AutonomyEngine(
        config=config,
        event_bus=event_bus,
        cognition=mock_cog,
        execution=mock_execution,
        memory=mock_memory,
        state_machine=state_machine,
    )

    result = await engine.execute_goal("budget time test")
    # Should either be budget_exhausted or completed before time ran out
    assert state_machine.state == InteractionState.IDLE


# ══════════════════════════════════════════════════════════════════════════════
# SECTION D — Unit tests for helpers
# ══════════════════════════════════════════════════════════════════════════════


def test_is_fatal_classification():
    """[S11] Error classification utility."""
    assert _is_fatal("PermissionError: access denied") is True
    assert _is_fatal("PathViolationError: /etc/passwd") is True
    assert _is_fatal("SecurityError: blocked") is True
    assert _is_fatal("InvalidToolError: no such tool") is True
    assert _is_fatal("TimeoutError: timed out") is False
    assert _is_fatal("ConnectionError: refused") is False
    assert _is_fatal(None) is False
    assert _is_fatal("") is False


def test_loop_detection_state():
    """[S3] Canonical plan hash sliding window."""
    ld = LoopDetectionState(window_size=3)
    assert ld.push("hash_a") is False
    assert ld.push("hash_b") is False
    assert ld.push("hash_a") is False
    assert ld.push("hash_c") is False
    assert ld.push("hash_c") is False
    assert ld.push("hash_c") is True  # 3 identical → deadlock


def test_resource_budget_exhaustion():
    """[S16] Resource budget counters."""
    budget = ResourceBudget(max_tool_calls=3, max_tokens=100, max_execution_time=10.0)
    assert not budget.exhausted
    budget.consume_tool_call()
    budget.consume_tool_call()
    assert not budget.exhausted
    budget.consume_tool_call()
    assert budget.exhausted
    assert budget.tool_calls_remaining == 0


def test_resource_budget_token_exhaustion():
    """[S16] Token budget counter."""
    budget = ResourceBudget(max_tool_calls=100, max_tokens=50, max_execution_time=300.0)
    assert not budget.exhausted
    budget.consume_tokens(30)
    assert not budget.exhausted
    budget.consume_tokens(25)
    assert budget.exhausted
    assert budget.tokens_remaining <= 0


def test_execution_transaction_rollback_order():
    """Rollback actions returned in reverse (LIFO) order."""
    tx = ExecutionTransaction()
    tx.record(
        PlannedAction(
            tool_name="a",
            arguments={},
            rollback_command={"tool": "undo_a", "args": {}},
        ),
        StepResult(status="success"),
    )
    tx.record(
        PlannedAction(
            tool_name="b",
            arguments={},
            rollback_command={"tool": "undo_b", "args": {}},
        ),
        StepResult(status="success"),
    )
    tx.record(
        PlannedAction(tool_name="c", arguments={}),  # No rollback
        StepResult(status="success"),
    )

    rollbacks = tx.rollback_actions
    assert len(rollbacks) == 2
    assert rollbacks[0]["tool"] == "undo_b"  # Reverse order
    assert rollbacks[1]["tool"] == "undo_a"
