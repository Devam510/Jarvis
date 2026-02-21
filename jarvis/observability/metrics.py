"""
jarvis.observability.metrics — Structured metrics registry (Phase 5).

Provides three metric types:
- Counter:   monotonically increasing value (e.g. events emitted, tool calls)
- Gauge:     point-in-time value (e.g. pending count, CPU %)
- Histogram: latency distributions with P50/P95/P99 (e.g. STT latency)

Periodically flushes snapshots to JSONL for dashboarding.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


class _Counter:
    """Monotonic counter."""

    __slots__ = ("name", "value", "labels")

    def __init__(self, name: str, labels: Optional[dict] = None):
        self.name = name
        self.value: float = 0.0
        self.labels = labels or {}

    def inc(self, amount: float = 1.0):
        self.value += amount

    def snapshot(self) -> dict:
        return {
            "type": "counter",
            "name": self.name,
            "value": self.value,
            "labels": self.labels,
        }


class _Gauge:
    """Point-in-time value."""

    __slots__ = ("name", "value", "labels")

    def __init__(self, name: str, labels: Optional[dict] = None):
        self.name = name
        self.value: float = 0.0
        self.labels = labels or {}

    def set(self, value: float):
        self.value = value

    def snapshot(self) -> dict:
        return {
            "type": "gauge",
            "name": self.name,
            "value": self.value,
            "labels": self.labels,
        }


class _Histogram:
    """Latency distribution with percentile computation.

    [V2-06 FIX] Caches sorted array — invalidated on new observations.
    """

    __slots__ = ("name", "samples", "max_size", "labels", "_sorted_cache")

    def __init__(self, name: str, max_size: int = 1000, labels: Optional[dict] = None):
        self.name = name
        self.samples: list[float] = []
        self.max_size = max_size
        self.labels = labels or {}
        self._sorted_cache: Optional[list[float]] = None

    def observe(self, value: float):
        self.samples.append(value)
        self._sorted_cache = None  # invalidate cache
        if len(self.samples) > self.max_size:
            # Keep most recent samples
            self.samples = self.samples[-self.max_size :]

    def _get_sorted(self) -> list[float]:
        """Return sorted samples, using cache if available."""
        if self._sorted_cache is None:
            self._sorted_cache = sorted(self.samples)
        return self._sorted_cache

    def percentile(self, p: float) -> float:
        """Compute the p-th percentile (0-100)."""
        if not self.samples:
            return 0.0
        s = self._get_sorted()
        k = (len(s) - 1) * (p / 100.0)
        f = int(k)
        c = f + 1
        if c >= len(s):
            return s[f]
        return s[f] + (k - f) * (s[c] - s[f])

    def snapshot(self) -> dict:
        return {
            "type": "histogram",
            "name": self.name,
            "count": len(self.samples),
            "p50": round(self.percentile(50), 2),
            "p95": round(self.percentile(95), 2),
            "p99": round(self.percentile(99), 2),
            "min": round(min(self.samples), 2) if self.samples else 0.0,
            "max": round(max(self.samples), 2) if self.samples else 0.0,
            "labels": self.labels,
        }


# ── Label key helper ──────────────────────────────────────────────────────


def _metric_key(name: str, labels: Optional[dict] = None) -> str:
    """Create a unique key from name + sorted labels."""
    if not labels:
        return name
    parts = sorted(f"{k}={v}" for k, v in labels.items())
    return f"{name}{{{','.join(parts)}}}"


class MetricsRegistry:
    """Global in-memory metrics registry with periodic JSONL flushing.

    Thread-safe via single-threaded asyncio loop (no locks needed).
    """

    # [V2-05 FIX] Max JSONL file size before rotation (10 MB default)
    _MAX_FILE_SIZE_BYTES = 10 * 1024 * 1024

    def __init__(
        self,
        output_path: str = "logs/metrics.jsonl",
        flush_interval_seconds: int = 60,
        max_histogram_size: int = 1000,
        enabled: bool = True,
        max_file_size_mb: float = 10.0,
    ):
        self._output_path = str(Path(output_path).resolve())
        self._flush_interval = flush_interval_seconds
        self._max_histogram_size = max_histogram_size
        self._enabled = enabled
        self._MAX_FILE_SIZE_BYTES = int(max_file_size_mb * 1024 * 1024)

        self._counters: dict[str, _Counter] = {}
        self._gauges: dict[str, _Gauge] = {}
        self._histograms: dict[str, _Histogram] = {}

        self._flush_task: Optional[asyncio.Task] = None
        self._total_flushes: int = 0

        Path(self._output_path).parent.mkdir(parents=True, exist_ok=True)

    # ── Recording API ─────────────────────────────────────────────────────

    def counter(self, name: str, value: float = 1.0, labels: Optional[dict] = None):
        """Increment a named counter."""
        if not self._enabled:
            return
        key = _metric_key(name, labels)
        if key not in self._counters:
            self._counters[key] = _Counter(name, labels)
        self._counters[key].inc(value)

    def gauge(self, name: str, value: float, labels: Optional[dict] = None):
        """Set a named gauge to a specific value."""
        if not self._enabled:
            return
        key = _metric_key(name, labels)
        if key not in self._gauges:
            self._gauges[key] = _Gauge(name, labels)
        self._gauges[key].set(value)

    def histogram(self, name: str, value: float, labels: Optional[dict] = None):
        """Record a sample into a named histogram."""
        if not self._enabled:
            return
        key = _metric_key(name, labels)
        if key not in self._histograms:
            self._histograms[key] = _Histogram(name, self._max_histogram_size, labels)
        self._histograms[key].observe(value)

    # ── Snapshot ──────────────────────────────────────────────────────────

    def snapshot(self) -> dict[str, Any]:
        """Return a full metrics dump."""
        return {
            "timestamp": time.time(),
            "counters": {k: v.snapshot() for k, v in self._counters.items()},
            "gauges": {k: v.snapshot() for k, v in self._gauges.items()},
            "histograms": {k: v.snapshot() for k, v in self._histograms.items()},
        }

    def get_counter(self, name: str, labels: Optional[dict] = None) -> float:
        """Get current counter value (for testing / dashboards)."""
        key = _metric_key(name, labels)
        c = self._counters.get(key)
        return c.value if c else 0.0

    def get_gauge(self, name: str, labels: Optional[dict] = None) -> float:
        """Get current gauge value."""
        key = _metric_key(name, labels)
        g = self._gauges.get(key)
        return g.value if g else 0.0

    def get_histogram(
        self, name: str, labels: Optional[dict] = None
    ) -> Optional[_Histogram]:
        """Get histogram object (for percentile queries)."""
        key = _metric_key(name, labels)
        return self._histograms.get(key)

    # ── Flush loop ────────────────────────────────────────────────────────

    async def start_flush_loop(self):
        """Start the periodic JSONL flush."""
        if not self._enabled:
            return
        if self._flush_task is not None:
            return
        self._flush_task = asyncio.create_task(self._flush_loop())
        logger.info(
            "Metrics flush started (interval=%ds, output=%s)",
            self._flush_interval,
            self._output_path,
        )

    async def _flush_loop(self):
        """Background loop writing metric snapshots to JSONL."""
        try:
            while True:
                await asyncio.sleep(self._flush_interval)
                try:
                    await self._flush_to_disk()
                except Exception as e:
                    logger.error("Metrics flush error: %s", e)
        except asyncio.CancelledError:
            # Final flush on shutdown
            try:
                await self._flush_to_disk()
                logger.info("Metrics: final flush on shutdown complete")
            except Exception as e:
                logger.error("Metrics: final flush failed: %s", e)

    async def _flush_to_disk(self):
        """Serialize snapshot to JSONL file."""
        snap = self.snapshot()
        line = json.dumps(snap, default=str)
        await asyncio.get_event_loop().run_in_executor(None, self._append_line, line)
        self._total_flushes += 1
        logger.debug("Metrics flushed (flush #%d)", self._total_flushes)

    def _append_line(self, line: str):
        """Blocking append — always called inside run_in_executor.

        [V2-05 FIX] Rotates file when it exceeds max size.
        """
        try:
            if os.path.exists(self._output_path):
                size = os.path.getsize(self._output_path)
                if size > self._MAX_FILE_SIZE_BYTES:
                    bak = self._output_path + ".bak"
                    if os.path.exists(bak):
                        os.remove(bak)
                    os.rename(self._output_path, bak)
                    logger.info(
                        "Metrics file rotated (%.1f MB > %.1f MB limit)",
                        size / (1024 * 1024),
                        self._MAX_FILE_SIZE_BYTES / (1024 * 1024),
                    )
        except OSError as e:
            logger.warning("Metrics rotation failed: %s", e)

        with open(self._output_path, "a", encoding="utf-8") as f:
            f.write(line + "\n")

    async def stop(self):
        """Stop flush loop with final flush."""
        if self._flush_task and not self._flush_task.done():
            self._flush_task.cancel()
            try:
                await self._flush_task
            except asyncio.CancelledError:
                pass
            self._flush_task = None
        logger.info("Metrics stopped")
