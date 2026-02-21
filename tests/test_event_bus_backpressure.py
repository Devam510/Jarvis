"""
Phase 5: Event Bus backpressure, priority dispatch, fire-and-forget, and DLQ tests.
"""

import asyncio
import json
import os
import tempfile

import pytest

from jarvis.core.event_bus import AsyncEventBus, DeadLetterQueue
from jarvis.utils.enums import EventPriority


# ── Helpers ──────────────────────────────────────────────────────────────


@pytest.fixture
def tmp_dlq_path(tmp_path):
    return str(tmp_path / "dlq.json")


@pytest.fixture
def bus(tmp_dlq_path):
    return AsyncEventBus(max_pending=5, dlq_path=tmp_dlq_path)


# ── Priority Resolution ──────────────────────────────────────────────────


def test_default_priority_critical():
    bus = AsyncEventBus()
    assert bus._get_priority("audio.wake_detected") == EventPriority.CRITICAL


def test_default_priority_background():
    bus = AsyncEventBus()
    assert bus._get_priority("state.changed") == EventPriority.BACKGROUND


def test_unknown_channel_defaults_to_normal():
    bus = AsyncEventBus()
    assert bus._get_priority("some.random.channel") == EventPriority.NORMAL


# ── Basic Emit & Handler ────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_basic_emit(bus):
    received = []

    async def handler(event):
        received.append(event)

    bus.subscribe("test.channel", handler)
    await bus.emit("test.channel", {"msg": "hello"})
    assert len(received) == 1
    assert received[0]["msg"] == "hello"


@pytest.mark.asyncio
async def test_multiple_handlers(bus):
    results = []

    async def h1(event):
        results.append("h1")

    async def h2(event):
        results.append("h2")

    bus.subscribe("test.channel", h1)
    bus.subscribe("test.channel", h2)
    await bus.emit("test.channel", None)
    assert "h1" in results
    assert "h2" in results


@pytest.mark.asyncio
async def test_unsubscribe(bus):
    called = []

    async def handler(event):
        called.append(True)

    bus.subscribe("test", handler)
    bus.unsubscribe("test", handler)
    await bus.emit("test", None)
    assert len(called) == 0


# ── Backpressure ─────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_p3_dropped_under_backpressure(tmp_dlq_path):
    """P3 events should be silently dropped when pending >= max_pending."""
    bus = AsyncEventBus(
        max_pending=0, dlq_path=tmp_dlq_path
    )  # max_pending=0 → always under pressure

    received = []

    async def handler(event):
        received.append(event)

    bus.subscribe("state.changed", handler)  # state.changed is P3/BACKGROUND

    await bus.emit("state.changed", {"msg": "dropped"})
    assert len(received) == 0
    assert bus.dropped_count == 1


@pytest.mark.asyncio
async def test_p0_never_dropped(tmp_dlq_path):
    """P0/CRITICAL events should always be dispatched regardless of backpressure."""
    bus = AsyncEventBus(max_pending=0, dlq_path=tmp_dlq_path)

    received = []

    async def handler(event):
        received.append(event)

    bus.subscribe("audio.wake_detected", handler)  # P0/CRITICAL

    await bus.emit("audio.wake_detected", {"wake": True})
    assert len(received) == 1


@pytest.mark.asyncio
async def test_p1_never_dropped(tmp_dlq_path):
    """P1/HIGH events should always be dispatched regardless of backpressure."""
    bus = AsyncEventBus(max_pending=0, dlq_path=tmp_dlq_path)

    received = []

    async def handler(event):
        received.append(event)

    bus.subscribe("risk.require_confirmation", handler)  # P1/HIGH

    await bus.emit("risk.require_confirmation", {"action": "delete"})
    assert len(received) == 1


@pytest.mark.asyncio
async def test_p2_dispatched_under_pressure(tmp_dlq_path):
    """P2/NORMAL events should still be dispatched (with warning) under pressure."""
    bus = AsyncEventBus(max_pending=0, dlq_path=tmp_dlq_path)

    received = []

    async def handler(event):
        received.append(event)

    bus.subscribe("execution.result", handler)  # P2/NORMAL

    await bus.emit("execution.result", {"result": "ok"})
    assert len(received) == 1  # dispatched (not dropped)


@pytest.mark.asyncio
async def test_explicit_priority_override(tmp_dlq_path):
    """Explicit priority override should take precedence over channel default."""
    bus = AsyncEventBus(max_pending=0, dlq_path=tmp_dlq_path)

    received = []

    async def handler(event):
        received.append(event)

    bus.subscribe("state.changed", handler)  # Default = P3 → would be dropped

    # Override to P0 → should go through
    await bus.emit("state.changed", {"msg": "forced"}, priority=EventPriority.CRITICAL)
    assert len(received) == 1


# ── Fire-and-Forget ──────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_fire_and_forget(bus):
    """Fire-and-forget should dispatch in background without awaiting."""
    received = []

    async def handler(event):
        await asyncio.sleep(0.01)
        received.append(event)

    bus.subscribe("test.channel", handler)
    await bus.emit("test.channel", {"msg": "bg"}, fire_and_forget=True)

    # Handler hasn't completed yet
    await asyncio.sleep(0.05)
    assert len(received) == 1  # Now it has


# ── Dead Letter Queue ────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_handler_error_goes_to_dlq(bus):
    """Failed handlers should be recorded in the DLQ."""

    async def bad_handler(event):
        raise ValueError("boom")

    bus.subscribe("test.fail", bad_handler)
    await bus.emit("test.fail", {"data": 42})

    assert len(bus.dlq) == 1
    assert bus.total_errors == 1
    entry = bus.dlq.entries[0]
    assert entry["channel"] == "test.fail"
    assert "boom" in entry["error"]


def test_dlq_persistence(tmp_path):
    """DLQ should persist entries to disk and reload them."""
    path = str(tmp_path / "dlq.json")

    dlq1 = DeadLetterQueue(path=path, max_entries=10)
    dlq1.record("test.ch", "error1")
    dlq1.record("test.ch", "error2")
    assert len(dlq1) == 2

    # Reload from disk
    dlq2 = DeadLetterQueue(path=path, max_entries=10)
    assert len(dlq2) == 2
    assert dlq2.entries[0]["error"] == "error1"


def test_dlq_max_entries(tmp_path):
    path = str(tmp_path / "dlq.json")
    dlq = DeadLetterQueue(path=path, max_entries=3)

    for i in range(5):
        dlq.record("ch", f"err{i}")

    assert len(dlq) == 3
    # Should have the latest entries only
    assert dlq.entries[0]["error"] == "err2"
    assert dlq.entries[2]["error"] == "err4"


# ── Stop Behavior ────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_stopped_bus_drops_all(bus):
    received = []

    async def handler(event):
        received.append(event)

    bus.subscribe("test", handler)
    bus.stop()
    await bus.emit("test", {"msg": "dropped"})
    assert len(received) == 0


# ── Stats ────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_stats(bus):
    async def handler(event):
        pass

    bus.subscribe("test", handler)
    await bus.emit("test", None)

    stats = bus.stats()
    assert stats["total_emitted"] == 1
    assert stats["pending"] == 0
    assert stats["stopped"] is False
