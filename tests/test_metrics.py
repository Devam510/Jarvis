"""
Phase 5: Structured metrics tests — counters, gauges, histograms, snapshots, JSONL flush.
"""

import asyncio
import json
import os

import pytest

from jarvis.observability.metrics import MetricsRegistry, _metric_key


# ── Helpers ──────────────────────────────────────────────────────────────


@pytest.fixture
def registry(tmp_path):
    return MetricsRegistry(
        output_path=str(tmp_path / "metrics.jsonl"),
        flush_interval_seconds=60,
        max_histogram_size=100,
        enabled=True,
    )


@pytest.fixture
def registry_path(tmp_path):
    return str(tmp_path / "metrics.jsonl")


# ── Counter Tests ────────────────────────────────────────────────────────


def test_counter_increment(registry):
    registry.counter("test_counter")
    assert registry.get_counter("test_counter") == 1.0


def test_counter_increment_by_amount(registry):
    registry.counter("test_counter", 5)
    registry.counter("test_counter", 3)
    assert registry.get_counter("test_counter") == 8.0


def test_counter_with_labels(registry):
    registry.counter("tool_calls", labels={"tool": "file_read", "status": "ok"})
    registry.counter("tool_calls", labels={"tool": "file_read", "status": "ok"})
    registry.counter("tool_calls", labels={"tool": "file_write", "status": "error"})

    assert (
        registry.get_counter("tool_calls", {"tool": "file_read", "status": "ok"}) == 2.0
    )
    assert (
        registry.get_counter("tool_calls", {"tool": "file_write", "status": "error"})
        == 1.0
    )


def test_counter_missing_returns_zero(registry):
    assert registry.get_counter("nonexistent") == 0.0


# ── Gauge Tests ──────────────────────────────────────────────────────────


def test_gauge_set(registry):
    registry.gauge("cpu_percent", 45.2)
    assert registry.get_gauge("cpu_percent") == 45.2


def test_gauge_overwrite(registry):
    registry.gauge("cpu_percent", 45.2)
    registry.gauge("cpu_percent", 78.9)
    assert registry.get_gauge("cpu_percent") == 78.9


def test_gauge_missing_returns_zero(registry):
    assert registry.get_gauge("nonexistent") == 0.0


# ── Histogram Tests ──────────────────────────────────────────────────────


def test_histogram_observe(registry):
    for v in [10, 20, 30, 40, 50]:
        registry.histogram("stt_latency_ms", v)

    h = registry.get_histogram("stt_latency_ms")
    assert h is not None
    assert len(h.samples) == 5


def test_histogram_percentile_p50(registry):
    for v in range(1, 101):
        registry.histogram("latency_ms", float(v))

    h = registry.get_histogram("latency_ms")
    assert abs(h.percentile(50) - 50.5) < 1.0


def test_histogram_percentile_p99(registry):
    for v in range(1, 101):
        registry.histogram("latency_ms", float(v))

    h = registry.get_histogram("latency_ms")
    assert h.percentile(99) >= 99.0


def test_histogram_empty_percentile(registry):
    """Empty histogram should return 0 for any percentile."""
    registry.histogram("empty", 0)  # single value
    h = registry.get_histogram("empty")
    # With one sample, all percentiles = that value
    assert h.percentile(50) == 0.0


def test_histogram_max_size(tmp_path):
    reg = MetricsRegistry(
        output_path=str(tmp_path / "m.jsonl"),
        max_histogram_size=5,
    )
    for v in range(100):
        reg.histogram("test_h", float(v))

    h = reg.get_histogram("test_h")
    assert len(h.samples) == 5
    # Should keep last 5
    assert h.samples[0] == 95.0


# ── Snapshot ─────────────────────────────────────────────────────────────


def test_snapshot_structure(registry):
    registry.counter("c1")
    registry.gauge("g1", 42)
    registry.histogram("h1", 10)

    snap = registry.snapshot()
    assert "timestamp" in snap
    assert "c1" in snap["counters"]
    assert "g1" in snap["gauges"]
    assert "h1" in snap["histograms"]

    # Check counter snapshot
    assert snap["counters"]["c1"]["type"] == "counter"
    assert snap["counters"]["c1"]["value"] == 1.0

    # Check gauge snapshot
    assert snap["gauges"]["g1"]["value"] == 42

    # Check histogram snapshot
    assert snap["histograms"]["h1"]["count"] == 1


# ── JSONL Flush ──────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_flush_to_disk(tmp_path):
    path = str(tmp_path / "metrics.jsonl")
    reg = MetricsRegistry(
        output_path=path,
        flush_interval_seconds=60,
    )

    reg.counter("test_counter", 3)
    reg.gauge("test_gauge", 99.5)

    # Manual flush
    await reg._flush_to_disk()

    assert os.path.exists(path)
    with open(path, "r") as f:
        lines = f.readlines()
    assert len(lines) == 1

    data = json.loads(lines[0])
    assert "timestamp" in data
    assert "test_counter" in data["counters"]
    assert data["counters"]["test_counter"]["value"] == 3.0


@pytest.mark.asyncio
async def test_flush_loop_starts_and_stops(tmp_path):
    path = str(tmp_path / "metrics.jsonl")
    reg = MetricsRegistry(
        output_path=path,
        flush_interval_seconds=0.1,  # Flush every 100ms
    )
    reg.counter("c1")

    await reg.start_flush_loop()
    await asyncio.sleep(0.25)  # Should have flushed at least once
    await reg.stop()

    assert os.path.exists(path)
    with open(path, "r") as f:
        lines = f.readlines()
    assert len(lines) >= 1


# ── Disabled Mode ────────────────────────────────────────────────────────


def test_disabled_ignores_recordings(tmp_path):
    reg = MetricsRegistry(
        output_path=str(tmp_path / "m.jsonl"),
        enabled=False,
    )
    reg.counter("c1")
    reg.gauge("g1", 42)
    reg.histogram("h1", 10)

    assert reg.get_counter("c1") == 0.0
    assert reg.get_gauge("g1") == 0.0
    assert reg.get_histogram("h1") is None


# ── Metric Key Helper ────────────────────────────────────────────────────


def test_metric_key_no_labels():
    assert _metric_key("counter") == "counter"


def test_metric_key_with_labels():
    key = _metric_key("tool_calls", {"tool": "file_read", "status": "ok"})
    assert "status=ok" in key
    assert "tool=file_read" in key
