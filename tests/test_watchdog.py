"""
Phase 5: Watchdog tests — loop detection, execution timeout, stuck-state reset.
"""

import asyncio
import time

import pytest

from jarvis.core.event_bus import AsyncEventBus
from jarvis.core.state_machine import InteractionStateMachine
from jarvis.core.watchdog import Watchdog
from jarvis.utils.config import WatchdogConfig
from jarvis.utils.enums import InteractionState


# ── Helpers ──────────────────────────────────────────────────────────────


@pytest.fixture(params=["default"])
def watchdog_config(request):
    """Fast-ticking config for tests."""
    return WatchdogConfig(
        enabled=True,
        loop_detect_window=12,
        loop_cycle_threshold=3,
        max_action_seconds=0.3,  # very short for tests
        max_state_seconds=0.3,  # very short for tests
        check_interval=0.05,  # 50ms ticks
    )


@pytest.fixture
def state_machine():
    return InteractionStateMachine()


@pytest.fixture
def event_bus(tmp_path):
    return AsyncEventBus(dlq_path=str(tmp_path / "dlq.json"))


@pytest.fixture
def watchdog(watchdog_config, event_bus, state_machine):
    return Watchdog(watchdog_config, event_bus, state_machine)


# ── Loop Detection ───────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_loop_detection_fires(watchdog, event_bus, state_machine):
    """Should detect a repeating cycle A→B→A→B→A→B and emit watchdog.loop_detected."""
    alerts = []

    async def on_loop(event):
        alerts.append(event)

    event_bus.subscribe("watchdog.loop_detected", on_loop)

    await watchdog.start()
    try:
        # Feed 3 repeats of [A, B] = 6 steps
        for _ in range(3):
            watchdog.record_plan_step("tool_a", "args_a")
            watchdog.record_plan_step("tool_b", "args_b")

        # Wait for watchdog to detect it
        await asyncio.sleep(0.2)
        assert len(alerts) >= 1
        assert (
            "loop" in alerts[0]["message"].lower() or "Infinite" in alerts[0]["message"]
        )
        assert state_machine.state == InteractionState.IDLE
    finally:
        await watchdog.stop()


@pytest.mark.asyncio
async def test_no_loop_with_varied_steps(watchdog, event_bus):
    """Should NOT fire loop detection for non-repeating sequences."""
    alerts = []

    async def on_loop(event):
        alerts.append(event)

    event_bus.subscribe("watchdog.loop_detected", on_loop)

    await watchdog.start()
    try:
        # Feed diverse steps
        for i in range(12):
            watchdog.record_plan_step(f"tool_{i}", f"args_{i}")

        await asyncio.sleep(0.2)
        assert len(alerts) == 0
    finally:
        await watchdog.stop()


# ── Execution Timeout ────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_execution_timeout(watchdog, event_bus, state_machine):
    """Should fire watchdog.timeout when action exceeds max_action_seconds."""
    alerts = []

    async def on_timeout(event):
        alerts.append(event)

    event_bus.subscribe("watchdog.timeout", on_timeout)

    # Move to EXECUTING state so stuck-state detector doesn't fire first
    state_machine.transition(InteractionState.REASONING)
    state_machine.transition(InteractionState.PLANNING)
    state_machine.transition(InteractionState.RISK_CLASSIFYING)
    state_machine.transition(InteractionState.EXECUTING)

    watchdog.action_started("slow_tool")

    await watchdog.start()
    try:
        await asyncio.sleep(0.5)  # > max_action_seconds (0.3s)
        assert len(alerts) >= 1
        assert "slow_tool" in alerts[0]["action"]
        assert state_machine.state == InteractionState.IDLE
    finally:
        await watchdog.stop()


@pytest.mark.asyncio
async def test_no_timeout_when_action_completes(watchdog, event_bus, state_machine):
    """Should NOT fire timeout if action completes in time."""
    alerts = []

    async def on_timeout(event):
        alerts.append(event)

    event_bus.subscribe("watchdog.timeout", on_timeout)

    watchdog.action_started("fast_tool")
    watchdog.action_completed()

    await watchdog.start()
    try:
        await asyncio.sleep(0.2)
        assert len(alerts) == 0
    finally:
        await watchdog.stop()


# ── Stuck-State Detection ────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_stuck_state_detection(watchdog, event_bus, state_machine):
    """Should fire watchdog.stuck_reset when stuck in non-IDLE state."""
    alerts = []

    async def on_stuck(event):
        alerts.append(event)

    event_bus.subscribe("watchdog.stuck_reset", on_stuck)

    # Move to REASONING and stay there
    state_machine.transition(InteractionState.REASONING)
    watchdog.record_transition()

    await watchdog.start()
    try:
        await asyncio.sleep(0.5)  # > max_state_seconds (0.3s)
        assert len(alerts) >= 1
        assert "REASONING" in alerts[0]["stuck_state"]
        assert state_machine.state == InteractionState.IDLE
    finally:
        await watchdog.stop()


@pytest.mark.asyncio
async def test_no_stuck_when_idle(watchdog, event_bus, state_machine):
    """Should NOT fire stuck detection when in IDLE state."""
    alerts = []

    async def on_stuck(event):
        alerts.append(event)

    event_bus.subscribe("watchdog.stuck_reset", on_stuck)

    # Stay in IDLE
    assert state_machine.state == InteractionState.IDLE

    await watchdog.start()
    try:
        await asyncio.sleep(0.2)
        assert len(alerts) == 0
    finally:
        await watchdog.stop()


# ── Lifecycle ────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_disabled_watchdog(event_bus, state_machine):
    """Disabled watchdog should not start a background task."""
    cfg = WatchdogConfig(enabled=False)
    wd = Watchdog(cfg, event_bus, state_machine)
    await wd.start()
    assert wd._task is None
    await wd.stop()


@pytest.mark.asyncio
async def test_clean_shutdown(watchdog):
    """Start and stop should not raise."""
    await watchdog.start()
    assert watchdog._running is True
    await watchdog.stop()
    assert watchdog._running is False
