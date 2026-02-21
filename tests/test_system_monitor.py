"""
tests/test_system_monitor.py — Tests for SystemMonitor.

Covers:
  - Metric collection
  - Health summary generation
  - Start/stop lifecycle
  - Non-blocking behavior
  - StateStore push
"""

import asyncio
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from jarvis.execution.system_monitor import SystemMonitor


# ── Basic Collection ─────────────────────────────────────────────────────────


def test_collect_metrics_sync():
    """Metrics collection should return a dict with CPU/RAM/disk."""
    metrics = SystemMonitor._collect_metrics()
    assert isinstance(metrics, dict)
    assert "collected_at" in metrics
    # At least CPU and RAM should be collected (psutil likely installed)
    if "error" not in metrics:
        assert "cpu_percent" in metrics
        assert "ram_used_gb" in metrics
        assert "ram_total_gb" in metrics


# ── Health Summary ────────────────────────────────────────────────────────────


def test_health_summary_empty():
    """Before collection, summary should explain metrics aren't ready."""
    m = SystemMonitor()
    summary = m.get_health_summary()
    assert "not yet collected" in summary.lower()


def test_health_summary_with_data():
    """After manual collection, summary should contain CPU/RAM info."""
    m = SystemMonitor()
    m._snapshot = SystemMonitor._collect_metrics()
    summary = m.get_health_summary()
    if "error" not in m._snapshot:
        assert "CPU" in summary
        assert "RAM" in summary


# ── Lifecycle ─────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_start_stop():
    """Monitor should start and stop cleanly."""
    m = SystemMonitor(interval=5)
    await m.start()
    assert m._task is not None
    assert m._running is True

    await m.stop()
    assert m._task is None
    assert m._running is False


@pytest.mark.asyncio
async def test_double_start():
    """Starting twice should not create duplicate tasks."""
    m = SystemMonitor(interval=5)
    await m.start()
    task1 = m._task
    await m.start()  # second start should be no-op
    assert m._task is task1
    await m.stop()


# ── Collection Runs ───────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_collection_runs_after_start():
    """After starting, metrics should be collected within interval."""
    m = SystemMonitor(interval=5)  # Will use MIN_INTERVAL_S = 5
    await m.start()
    # Wait enough for one collection
    await asyncio.sleep(6)
    snapshot = m.get_snapshot()
    assert "collected_at" in snapshot
    await m.stop()


# ── Top Processes ─────────────────────────────────────────────────────────────


def test_top_processes():
    """get_top_processes should return a list."""
    m = SystemMonitor()
    m._snapshot = SystemMonitor._collect_metrics()
    procs = m.get_top_processes(3)
    assert isinstance(procs, list)
    assert len(procs) <= 3


# ── StateStore Push ───────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_state_store_push():
    """If a StateStore-like object is provided, patch() should be called."""

    class FakeStore:
        def __init__(self):
            self.patches = []

        async def patch(self, data):
            self.patches.append(data)

    store = FakeStore()
    m = SystemMonitor(state_store=store, interval=5)
    await m.start()
    await asyncio.sleep(6)
    await m.stop()

    assert len(store.patches) >= 1
    assert "system" in store.patches[0]
