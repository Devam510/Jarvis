"""
tests.test_suggestion_engine — Phase 4 tests for SuggestionEngine.

Tests cover safety requirements S5 and S8.
"""

import asyncio
import time

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from jarvis.core.event_bus import AsyncEventBus
from jarvis.perception.suggestion_engine import SuggestionEngine
from jarvis.utils.config import ProactiveConfig
from jarvis.utils.enums import InteractionState
from jarvis.utils.types import ScreenContext


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def config():
    return ProactiveConfig(
        enabled=True,
        confidence_threshold=0.8,
        cooldown_seconds=600.0,
        max_suggestions_per_hour=6,
        do_not_disturb=False,
    )


@pytest.fixture
def event_bus():
    return AsyncEventBus()


@pytest.fixture
def state_machine():
    from jarvis.core.state_machine import InteractionStateMachine

    return InteractionStateMachine()


@pytest.fixture
def engine(config, event_bus, state_machine):
    return SuggestionEngine(config, event_bus, state_machine=state_machine)


def _make_context(**overrides) -> ScreenContext:
    """Helper to create ScreenContext for tests."""
    defaults = dict(
        active_app="VS Code",
        window_title="main.py - Jarvis",
        visible_text="",
        detected_errors=(),
        timestamp=time.time(),
        screenshot_path="",
        change_magnitude=0.1,
        is_blacked_out=False,
    )
    defaults.update(overrides)
    return ScreenContext(**defaults)


# ══════════════════════════════════════════════════════════════════════════════
# Basic Trigger Tests
# ══════════════════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_error_trigger_fires(engine, event_bus):
    """Error in ScreenContext → suggestion emitted."""
    emitted = []

    async def _capture(event):
        emitted.append(event)

    event_bus.subscribe("suggestion.offered", _capture)

    ctx = _make_context(detected_errors=("ModuleNotFoundError: no module 'foo'",))
    await engine.on_context_updated(ctx)

    assert len(emitted) == 1
    assert "error" in emitted[0]["trigger"]
    assert emitted[0]["message"]


@pytest.mark.asyncio
async def test_no_trigger_on_clean_screen(engine, event_bus):
    """No errors, no stall → no suggestion."""
    emitted = []

    async def _capture(event):
        emitted.append(event)

    event_bus.subscribe("suggestion.offered", _capture)

    ctx = _make_context()
    await engine.on_context_updated(ctx)

    assert len(emitted) == 0


@pytest.mark.asyncio
async def test_build_failed_trigger(engine, event_bus):
    """'build failed' in visible text → suggestion."""
    emitted = []

    async def _capture(event):
        emitted.append(event)

    event_bus.subscribe("suggestion.offered", _capture)

    ctx = _make_context(visible_text="npm ERR! Build failed with exit code 1")
    await engine.on_context_updated(ctx)

    assert len(emitted) == 1
    assert emitted[0]["trigger"] == "build_failed"


# ══════════════════════════════════════════════════════════════════════════════
# Cooldown & Rate Limiting
# ══════════════════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_cooldown_prevents_spam(engine, event_bus):
    """Same trigger within cooldown period → suppressed."""
    emitted = []

    async def _capture(event):
        emitted.append(event)

    event_bus.subscribe("suggestion.offered", _capture)

    ctx = _make_context(detected_errors=("Error: something broke",))
    await engine.on_context_updated(ctx)  # First → fires
    await engine.on_context_updated(ctx)  # Second within cooldown → blocked

    assert len(emitted) == 1


@pytest.mark.asyncio
async def test_rate_limit_enforced(engine, event_bus):
    """More than max_suggestions_per_hour → excess suppressed."""
    emitted = []

    async def _capture(event):
        emitted.append(event)

    event_bus.subscribe("suggestion.offered", _capture)

    # Each needs its own trigger for cooldown, but should hit rate limit
    engine._config.cooldown_seconds = 0  # Disable cooldown for this test
    engine._config.max_suggestions_per_hour = 2

    for i in range(5):
        ctx = _make_context(
            detected_errors=(f"Error_{i}: unique error {i}",),
            change_magnitude=0.5,
        )
        await engine.on_context_updated(ctx)

    assert len(emitted) == 2  # Only 2 allowed per hour


# ══════════════════════════════════════════════════════════════════════════════
# S5: Escalation Guard
# ══════════════════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_s5_blocks_during_autonomous_planning(engine, event_bus, state_machine):
    """S5: No suggestions during AUTONOMOUS_PLANNING."""
    emitted = []

    async def _capture(event):
        emitted.append(event)

    event_bus.subscribe("suggestion.offered", _capture)

    state_machine.transition(InteractionState.AUTONOMOUS_PLANNING)
    ctx = _make_context(detected_errors=("Error: bad",))
    await engine.on_context_updated(ctx)

    assert len(emitted) == 0


@pytest.mark.asyncio
async def test_s5_blocks_during_autonomous_executing(engine, event_bus, state_machine):
    """S5: No suggestions during AUTONOMOUS_EXECUTING."""
    emitted = []

    async def _capture(event):
        emitted.append(event)

    event_bus.subscribe("suggestion.offered", _capture)

    state_machine.transition(InteractionState.AUTONOMOUS_PLANNING)
    state_machine.transition(InteractionState.AUTONOMOUS_EXECUTING)
    ctx = _make_context(detected_errors=("Error: bad",))
    await engine.on_context_updated(ctx)

    assert len(emitted) == 0


@pytest.mark.asyncio
async def test_s5_blocks_during_awaiting_confirm(engine, event_bus, state_machine):
    """S5: No suggestions during AWAITING_CONFIRM."""
    emitted = []

    async def _capture(event):
        emitted.append(event)

    event_bus.subscribe("suggestion.offered", _capture)

    # Directly set state (bypass transition validation for test)
    state_machine._state = InteractionState.AWAITING_CONFIRM
    ctx = _make_context(detected_errors=("Error: bad",))
    await engine.on_context_updated(ctx)

    assert len(emitted) == 0


@pytest.mark.asyncio
async def test_s5_dnd_flag(engine, event_bus):
    """S5: do_not_disturb=True → all suggestions blocked."""
    emitted = []

    async def _capture(event):
        emitted.append(event)

    event_bus.subscribe("suggestion.offered", _capture)

    engine._config.do_not_disturb = True
    ctx = _make_context(detected_errors=("Error: critical failure",))
    await engine.on_context_updated(ctx)

    assert len(emitted) == 0


# ══════════════════════════════════════════════════════════════════════════════
# S8: No Feedback Loop
# ══════════════════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_s8_suggestion_never_triggers_autonomy(engine, event_bus):
    """S8: Suggestion event contains no tool calls or autonomy commands."""
    emitted = []

    async def _capture(event):
        emitted.append(event)

    event_bus.subscribe("suggestion.offered", _capture)

    ctx = _make_context(detected_errors=("Error: something",))
    await engine.on_context_updated(ctx)

    assert len(emitted) == 1
    event = emitted[0]
    # Verify no auto-execution fields
    assert "tool_call" not in event
    assert "execute_goal" not in event
    assert "action" not in event  # Only action_hint (informational)
    assert "action_hint" in event  # Informational only


@pytest.mark.asyncio
async def test_s8_skips_blacked_out_context(engine, event_bus):
    """S8/S2: No suggestions on blacked-out context."""
    emitted = []

    async def _capture(event):
        emitted.append(event)

    event_bus.subscribe("suggestion.offered", _capture)

    ctx = _make_context(is_blacked_out=True, detected_errors=("Error: bad",))
    await engine.on_context_updated(ctx)

    assert len(emitted) == 0


@pytest.mark.asyncio
async def test_disabled_engine_emits_nothing(config, event_bus, state_machine):
    """Disabled engine emits no suggestions."""
    config.enabled = False
    engine = SuggestionEngine(config, event_bus, state_machine=state_machine)
    emitted = []

    async def _capture(event):
        emitted.append(event)

    event_bus.subscribe("suggestion.offered", _capture)

    ctx = _make_context(detected_errors=("Error: bad",))
    await engine.on_context_updated(ctx)

    assert len(emitted) == 0
