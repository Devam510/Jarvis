"""
jarvis.execution.process_graph — System State Graph with anomaly detection.

Safety controls:
  - Collection frequency capped at MIN_INTERVAL_S
  - All psutil calls wrapped in per-call try/except
  - Anomalies emitted as events only — NEVER auto-acts or kills processes
  - Runs in executor thread (never blocks main async loop)
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Optional

logger = logging.getLogger(__name__)

MIN_INTERVAL_S = 5.0  # Hard minimum between collections


@dataclass
class ProcessInfo:
    """Information about a single OS process."""

    pid: int = 0
    name: str = ""
    cpu_percent: float = 0.0
    memory_mb: float = 0.0
    status: str = ""
    parent_pid: Optional[int] = None
    children_pids: list[int] = field(default_factory=list)


@dataclass
class Anomaly:
    """Detected system anomaly."""

    type: str = ""  # cpu_sustained | memory_high | disk_full
    metric: str = ""
    value: float = 0.0
    threshold: float = 0.0
    message: str = ""
    timestamp: float = 0.0


class ProcessGraph:
    """System State Graph — real-time PC health model.

    Invariants:
      - Never auto-acts on anomalies (emit-only)
      - Never kills or modifies processes
      - All psutil calls are individually exception-safe
      - Collection frequency enforced ≥ MIN_INTERVAL_S
    """

    def __init__(
        self,
        event_bus: Any = None,
        cpu_threshold: float = 80.0,
        mem_threshold: float = 85.0,
        disk_threshold: float = 90.0,
        sustained_seconds: float = 300.0,
        collection_interval: float = 10.0,
    ):
        self._event_bus = event_bus
        self._cpu_threshold = cpu_threshold
        self._mem_threshold = mem_threshold
        self._disk_threshold = disk_threshold
        self._sustained_secs = sustained_seconds
        self._interval = max(collection_interval, MIN_INTERVAL_S)

        # State
        self._process_map: dict[int, ProcessInfo] = {}
        self._last_collection: float = 0
        self._anomaly_start: dict[str, float] = {}  # type → first_seen
        self._task: Optional[asyncio.Task] = None
        self._running = False

        # Stats
        self.total_collections = 0
        self.total_anomalies_emitted = 0

    # ── Lifecycle ─────────────────────────────────────────────────────────

    async def start(self):
        """Start periodic collection loop."""
        if self._running:
            return
        self._running = True
        self._task = asyncio.create_task(self._collection_loop())
        logger.info("ProcessGraph started (interval=%.0fs)", self._interval)

    async def stop(self):
        """Stop the collection loop."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

    async def _collection_loop(self):
        """Periodic metric collection with rate enforcement."""
        while self._running:
            now = time.time()
            elapsed = now - self._last_collection

            if elapsed < MIN_INTERVAL_S:
                await asyncio.sleep(MIN_INTERVAL_S - elapsed)

            try:
                loop = asyncio.get_event_loop()
                # [V2-02 FIX] Collect returns new map; assignment on event loop side
                new_map = await loop.run_in_executor(None, self._collect)
                if new_map is not None:
                    self._process_map = new_map
                self._last_collection = time.time()
                self.total_collections += 1

                # Check for anomalies
                anomalies = self._detect_anomalies()
                for a in anomalies:
                    await self._emit_anomaly(a)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.debug("ProcessGraph collection error: %s", e)

            await asyncio.sleep(self._interval)

    # ── Collection ────────────────────────────────────────────────────────

    def _collect(self) -> Optional[dict[int, "ProcessInfo"]]:
        """Collect process information via psutil. Every call is exception-safe.

        [V2-02 FIX] Returns the new map instead of mutating self._process_map.
        Assignment happens on the event loop side.
        """
        try:
            import psutil
        except ImportError:
            logger.debug("psutil not available for ProcessGraph")
            return None

        new_map: dict[int, ProcessInfo] = {}

        try:
            procs = list(
                psutil.process_iter(
                    ["pid", "name", "cpu_percent", "memory_info", "status", "ppid"]
                )
            )
        except Exception as e:
            logger.debug("process_iter failed: %s", e)
            return None

        for proc in procs:
            try:
                info = proc.info
                pid = info.get("pid", 0)
                mem_info = info.get("memory_info")
                mem_mb = (mem_info.rss / (1024 * 1024)) if mem_info else 0

                new_map[pid] = ProcessInfo(
                    pid=pid,
                    name=info.get("name", ""),
                    cpu_percent=info.get("cpu_percent", 0) or 0,
                    memory_mb=round(mem_mb, 1),
                    status=info.get("status", ""),
                    parent_pid=info.get("ppid"),
                )
            except Exception:
                continue  # Per-process exception safety

        # Build children map
        for pid, pinfo in new_map.items():
            if pinfo.parent_pid and pinfo.parent_pid in new_map:
                try:
                    new_map[pinfo.parent_pid].children_pids.append(pid)
                except Exception:
                    pass

        return new_map

    # ── Process Queries ───────────────────────────────────────────────────

    def build_graph(self) -> dict[int, ProcessInfo]:
        """Return the current process map."""
        return dict(self._process_map)

    def find_resource_hogs(
        self, cpu_threshold: float = 50.0, mem_mb_threshold: float = 2048.0
    ) -> list[ProcessInfo]:
        """Find processes exceeding resource thresholds."""
        hogs = []
        for pinfo in self._process_map.values():
            try:
                if (
                    pinfo.cpu_percent >= cpu_threshold
                    or pinfo.memory_mb >= mem_mb_threshold
                ):
                    hogs.append(pinfo)
            except Exception:
                continue
        return sorted(hogs, key=lambda p: p.cpu_percent + p.memory_mb, reverse=True)

    def get_natural_summary(self) -> str:
        """Generate human-readable health summary for LLM consumption."""
        try:
            import psutil
        except ImportError:
            return "System monitoring unavailable (psutil not installed)."

        parts = []

        # System-level metrics
        try:
            cpu = psutil.cpu_percent(interval=0)
            parts.append(f"CPU: {cpu}%")
        except Exception:
            parts.append("CPU: unknown")

        try:
            mem = psutil.virtual_memory()
            parts.append(
                f"RAM: {mem.percent}% used "
                f"({mem.used // (1024**3)}GB / {mem.total // (1024**3)}GB)"
            )
        except Exception:
            parts.append("RAM: unknown")

        try:
            disk = psutil.disk_usage("/")
            parts.append(
                f"Disk: {disk.percent}% used "
                f"({disk.used // (1024**3)}GB / {disk.total // (1024**3)}GB)"
            )
        except Exception:
            parts.append("Disk: unknown")

        # Top resource users
        hogs = self.find_resource_hogs(cpu_threshold=30, mem_mb_threshold=1024)
        if hogs:
            parts.append("\nTop resource consumers:")
            for p in hogs[:5]:
                parts.append(
                    f"  - {p.name} (PID {p.pid}): "
                    f"CPU {p.cpu_percent:.0f}%, RAM {p.memory_mb:.0f}MB"
                )

        # Anomalies
        anomalies = self._detect_anomalies()
        if anomalies:
            parts.append("\n⚠️ Active anomalies:")
            for a in anomalies:
                parts.append(f"  - {a.message}")

        return "\n".join(parts)

    # ── Anomaly Detection ─────────────────────────────────────────────────

    def _detect_anomalies(self) -> list[Anomaly]:
        """Detect system anomalies. Returns list — NEVER auto-acts."""
        anomalies = []
        now = time.time()

        try:
            import psutil
        except ImportError:
            return []

        # CPU check
        try:
            cpu = psutil.cpu_percent(interval=0)
            if cpu >= self._cpu_threshold:
                key = "cpu_sustained"
                if key not in self._anomaly_start:
                    self._anomaly_start[key] = now
                elif now - self._anomaly_start[key] >= self._sustained_secs:
                    anomalies.append(
                        Anomaly(
                            type="cpu_sustained",
                            metric="cpu_percent",
                            value=cpu,
                            threshold=self._cpu_threshold,
                            message=f"CPU at {cpu}% for >{self._sustained_secs/60:.0f} min",
                            timestamp=now,
                        )
                    )
            else:
                self._anomaly_start.pop("cpu_sustained", None)
        except Exception:
            pass

        # Memory check
        try:
            mem = psutil.virtual_memory()
            if mem.percent >= self._mem_threshold:
                key = "memory_high"
                if key not in self._anomaly_start:
                    self._anomaly_start[key] = now
                elif now - self._anomaly_start[key] >= self._sustained_secs:
                    anomalies.append(
                        Anomaly(
                            type="memory_high",
                            metric="memory_percent",
                            value=mem.percent,
                            threshold=self._mem_threshold,
                            message=f"RAM at {mem.percent}% for >{self._sustained_secs/60:.0f} min",
                            timestamp=now,
                        )
                    )
            else:
                self._anomaly_start.pop("memory_high", None)
        except Exception:
            pass

        # Disk check
        try:
            disk = psutil.disk_usage("/")
            if disk.percent >= self._disk_threshold:
                anomalies.append(
                    Anomaly(
                        type="disk_full",
                        metric="disk_percent",
                        value=disk.percent,
                        threshold=self._disk_threshold,
                        message=f"Disk at {disk.percent}% — critically full",
                        timestamp=now,
                    )
                )
        except Exception:
            pass

        return anomalies

    async def _emit_anomaly(self, anomaly: Anomaly):
        """Emit anomaly as event — never auto-acts."""
        self.total_anomalies_emitted += 1
        logger.warning(
            "System anomaly: %s (%s=%.1f > %.1f)",
            anomaly.type,
            anomaly.metric,
            anomaly.value,
            anomaly.threshold,
        )
        if self._event_bus:
            try:
                await self._event_bus.emit(
                    "system.anomaly_detected",
                    {
                        "type": anomaly.type,
                        "metric": anomaly.metric,
                        "value": anomaly.value,
                        "threshold": anomaly.threshold,
                        "message": anomaly.message,
                    },
                )
            except Exception:
                pass
