"""
jarvis.execution.system_monitor — Live OS metric collection daemon.

Guardrails:
  - Capped collection frequency (min 5s interval)
  - All psutil calls wrapped in try/except
  - Runs in executor thread (never blocks main async loop)
  - Pushes to StateStore via patch()
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

logger = logging.getLogger(__name__)

# Minimum collection interval (seconds) — prevents CPU overhead
MIN_INTERVAL_S = 5.0


class SystemMonitor:
    """Background daemon that collects OS-level metrics and pushes to StateStore.

    Usage:
        monitor = SystemMonitor(state_store=store, interval=10)
        await monitor.start()
        ...
        summary = monitor.get_health_summary()
        ...
        await monitor.stop()
    """

    def __init__(
        self,
        state_store: Any = None,
        interval: float = 10.0,
    ):
        self.state_store = state_store
        self.interval = max(interval, MIN_INTERVAL_S)
        self._task: asyncio.Task | None = None
        self._snapshot: dict[str, Any] = {}
        self._running = False

    # ── Lifecycle ─────────────────────────────────────────────────────────

    async def start(self):
        """Start the background collection loop."""
        if self._task is not None:
            return
        self._running = True
        self._task = asyncio.create_task(self._collection_loop())
        logger.info("SystemMonitor started (interval=%.0fs)", self.interval)

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
        logger.info("SystemMonitor stopped")

    # ── Collection Loop ───────────────────────────────────────────────────

    async def _collection_loop(self):
        """Periodic metric collection. Runs in background task."""
        while self._running:
            try:
                # Collect in thread pool — never block async loop
                snapshot = await asyncio.get_event_loop().run_in_executor(
                    None, self._collect_metrics
                )
                self._snapshot = snapshot

                # Push to StateStore if available
                if self.state_store is not None:
                    try:
                        await self.state_store.patch({"system": snapshot})
                    except Exception as e:
                        logger.debug("StateStore push failed: %s", e)

            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.warning("SystemMonitor collection error: %s", e)

            await asyncio.sleep(self.interval)

    # ── Metric Collection (runs in thread) ────────────────────────────────

    @staticmethod
    def _collect_metrics() -> dict[str, Any]:
        """Collect OS metrics via psutil. Every call is exception-safe."""
        metrics: dict[str, Any] = {"collected_at": time.time()}

        try:
            import psutil
        except ImportError:
            metrics["error"] = "psutil not installed"
            return metrics

        # CPU
        try:
            metrics["cpu_percent"] = psutil.cpu_percent(interval=0.5)
            metrics["cpu_count"] = psutil.cpu_count()
            metrics["cpu_freq_mhz"] = (
                round(psutil.cpu_freq().current) if psutil.cpu_freq() else None
            )
        except Exception as e:
            metrics["cpu_error"] = str(e)

        # Memory
        try:
            mem = psutil.virtual_memory()
            metrics["ram_used_gb"] = round(mem.used / (1024**3), 1)
            metrics["ram_total_gb"] = round(mem.total / (1024**3), 1)
            metrics["ram_percent"] = mem.percent
        except Exception as e:
            metrics["ram_error"] = str(e)

        # Disk
        try:
            disk = psutil.disk_usage("/")
            metrics["disk_free_gb"] = round(disk.free / (1024**3), 1)
            metrics["disk_total_gb"] = round(disk.total / (1024**3), 1)
            metrics["disk_percent"] = disk.percent
        except Exception as e:
            metrics["disk_error"] = str(e)

        # Network I/O
        try:
            net = psutil.net_io_counters()
            metrics["net_sent_mb"] = round(net.bytes_sent / (1024**2), 1)
            metrics["net_recv_mb"] = round(net.bytes_recv / (1024**2), 1)
        except Exception as e:
            metrics["net_error"] = str(e)

        # Top processes (by CPU)
        try:
            procs = []
            for p in psutil.process_iter(["pid", "name", "cpu_percent", "memory_info"]):
                try:
                    info = p.info
                    procs.append(
                        {
                            "pid": info["pid"],
                            "name": info["name"],
                            "cpu": info.get("cpu_percent", 0) or 0,
                            "ram_mb": round(
                                (
                                    info.get("memory_info")
                                    and info["memory_info"].rss
                                    or 0
                                )
                                / (1024**2),
                                1,
                            ),
                        }
                    )
                except (
                    psutil.NoSuchProcess,
                    psutil.AccessDenied,
                    psutil.ZombieProcess,
                ):
                    continue
            procs.sort(key=lambda x: x["cpu"], reverse=True)
            metrics["top_processes"] = procs[:10]
        except Exception as e:
            metrics["process_error"] = str(e)

        # GPU (optional — nvidia-smi)
        try:
            import subprocess

            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=utilization.gpu,memory.used,memory.total",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                parts = result.stdout.strip().split(", ")
                if len(parts) >= 3:
                    metrics["gpu_load_percent"] = int(parts[0])
                    metrics["gpu_memory_used_mb"] = int(parts[1])
                    metrics["gpu_memory_total_mb"] = int(parts[2])
        except Exception:
            pass  # GPU metrics are optional

        return metrics

    # ── Public API ────────────────────────────────────────────────────────

    def get_snapshot(self) -> dict[str, Any]:
        """Return the latest collected metrics (thread-safe read)."""
        return dict(self._snapshot)

    def get_top_processes(self, n: int = 5) -> list[dict]:
        """Return top N processes by CPU usage."""
        return self._snapshot.get("top_processes", [])[:n]

    def get_health_summary(self) -> str:
        """Return a human-readable health summary for the LLM to consume."""
        s = self._snapshot
        if not s:
            return "System metrics not yet collected."

        lines = ["**System Health Summary**"]

        # CPU
        cpu = s.get("cpu_percent")
        if cpu is not None:
            status = "🔴 HIGH" if cpu > 80 else "🟡 MODERATE" if cpu > 50 else "🟢 OK"
            lines.append(f"- CPU: {cpu}% {status}")

        # RAM
        ram_pct = s.get("ram_percent")
        ram_used = s.get("ram_used_gb")
        ram_total = s.get("ram_total_gb")
        if ram_pct is not None:
            status = (
                "🔴 HIGH"
                if ram_pct > 85
                else "🟡 MODERATE" if ram_pct > 60 else "🟢 OK"
            )
            lines.append(f"- RAM: {ram_used}/{ram_total} GB ({ram_pct}%) {status}")

        # Disk
        disk_pct = s.get("disk_percent")
        disk_free = s.get("disk_free_gb")
        if disk_pct is not None:
            status = "🔴 LOW" if disk_pct > 90 else "🟢 OK"
            lines.append(f"- Disk: {disk_free} GB free ({disk_pct}% used) {status}")

        # GPU
        gpu = s.get("gpu_load_percent")
        if gpu is not None:
            gpu_mem = s.get("gpu_memory_used_mb", "?")
            gpu_total = s.get("gpu_memory_total_mb", "?")
            lines.append(f"- GPU: {gpu}% load, {gpu_mem}/{gpu_total} MB VRAM")

        # Top processes
        top = s.get("top_processes", [])[:5]
        if top:
            lines.append("- Top processes:")
            for p in top:
                lines.append(
                    f"  - {p['name']} (PID {p['pid']}): "
                    f"CPU {p['cpu']}%, RAM {p['ram_mb']} MB"
                )

        return "\n".join(lines)
