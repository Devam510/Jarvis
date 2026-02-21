"""
jarvis.observability.self_healer — Local AI Supervisor / Self-Healing.

Monitors subsystem health (STT latency, LLM latency, TTS latency,
error rates) and automatically attempts recovery when degradation
is detected.

Safety:
  - Max 3 restart attempts per component per hour
  - All restart actions audit-logged
  - Never kills user processes — only restarts Jarvis subsystems
  - Emit-first: always emits degradation event before acting
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)

# Hard cap: max restarts per component per hour
_MAX_RESTARTS_PER_HOUR = 3


@dataclass
class HealthMetric:
    """Tracked health metric for a subsystem."""

    name: str
    current_value: float = 0.0
    threshold: float = 0.0
    unit: str = "ms"
    is_healthy: bool = True
    last_check: float = 0.0


@dataclass
class HealingAction:
    """Record of a self-healing action."""

    component: str
    action: str
    reason: str
    timestamp: float = 0.0
    success: bool = False
    attempt: int = 0


class SelfHealer:
    """Monitor subsystem health and auto-recover from degradation.

    Safety invariants:
      - Max _MAX_RESTARTS_PER_HOUR per component
      - All actions audit-logged via event bus
      - Only restarts Jarvis subsystems (STT, TTS, audio)
      - Emits event BEFORE attempting any healing action
    """

    def __init__(
        self,
        event_bus: Any = None,
        stt_latency_threshold: float = 2000.0,  # ms
        llm_latency_threshold: float = 5000.0,  # ms
        tts_latency_threshold: float = 3000.0,  # ms
        error_rate_threshold: float = 0.3,  # 30% error rate
        check_interval: float = 30.0,  # seconds
    ):
        self._event_bus = event_bus
        self._thresholds = {
            "stt": stt_latency_threshold,
            "llm": llm_latency_threshold,
            "tts": tts_latency_threshold,
            "error_rate": error_rate_threshold,
        }
        self._check_interval = max(check_interval, 10.0)

        # Restart tracking: component → list of timestamps
        self._restart_history: dict[str, list[float]] = defaultdict(list)

        # Registered restart handlers: component → async callable
        self._restart_handlers: dict[str, Callable] = {}

        # Current metrics
        self._metrics: dict[str, HealthMetric] = {}

        # Running state
        self._running = False
        self._task: Optional[asyncio.Task] = None

        # Stats
        self.total_checks = 0
        self.total_restarts = 0
        self.total_restart_failures = 0
        self._healing_log: list[HealingAction] = []

    # ── Registration ──────────────────────────────────────────────────────

    def register_handler(self, component: str, handler: Callable):
        """Register a restart handler for a component.

        Args:
            component: Component name (e.g., 'stt', 'tts', 'audio')
            handler: Async callable that restarts the component
        """
        self._restart_handlers[component] = handler
        logger.info("Self-healer: registered handler for '%s'", component)

    # ── Metric Reporting ──────────────────────────────────────────────────

    def report_metric(self, component: str, value: float, unit: str = "ms"):
        """Report a health metric for a component.

        Called by subsystems to report their current state.
        """
        threshold = self._thresholds.get(component, float("inf"))
        is_healthy = value < threshold

        self._metrics[component] = HealthMetric(
            name=component,
            current_value=value,
            threshold=threshold,
            unit=unit,
            is_healthy=is_healthy,
            last_check=time.time(),
        )

    def report_error(self, component: str):
        """Report an error occurrence for a component."""
        key = f"{component}_errors"
        metric = self._metrics.get(key)
        if metric:
            metric.current_value += 1
        else:
            self._metrics[key] = HealthMetric(
                name=key, current_value=1, unit="count", last_check=time.time()
            )

    # ── Health Check Loop ─────────────────────────────────────────────────

    async def start(self):
        """Start the self-healing monitor loop."""
        if self._running:
            return
        self._running = True
        self._task = asyncio.create_task(self._monitor_loop())
        logger.info("Self-healer started (interval=%.1fs)", self._check_interval)

    async def stop(self):
        """Stop the self-healing monitor."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

    async def _monitor_loop(self):
        """Periodic health check loop."""
        while self._running:
            try:
                await self._check_health()
                self.total_checks += 1
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Self-healer check failed: %s", e)

            await asyncio.sleep(self._check_interval)

    async def _check_health(self):
        """Check all metrics and trigger healing if needed."""
        for name, metric in list(self._metrics.items()):
            if metric.is_healthy:
                continue

            # Component is degraded — attempt healing
            component = name.replace("_errors", "")
            if component in self._restart_handlers:
                await self._attempt_healing(component, metric)

    # ── Healing Actions ───────────────────────────────────────────────────

    async def _attempt_healing(self, component: str, metric: HealthMetric):
        """Attempt to heal a degraded component.

        Safety: checks restart budget before acting.
        """
        # Check restart budget
        now = time.time()
        hour_ago = now - 3600

        # Prune old entries
        self._restart_history[component] = [
            t for t in self._restart_history[component] if t > hour_ago
        ]

        recent_restarts = len(self._restart_history[component])
        if recent_restarts >= _MAX_RESTARTS_PER_HOUR:
            logger.warning(
                "Self-healer: restart budget exhausted for '%s' "
                "(%d/%d in last hour)",
                component,
                recent_restarts,
                _MAX_RESTARTS_PER_HOUR,
            )
            return

        # Emit degradation event BEFORE acting
        await self._emit_event(
            "subsystem.degraded",
            {
                "component": component,
                "metric": metric.name,
                "value": metric.current_value,
                "threshold": metric.threshold,
                "unit": metric.unit,
                "action": "restart_attempt",
            },
        )

        # Attempt restart
        attempt = recent_restarts + 1
        action = HealingAction(
            component=component,
            action="restart",
            reason=f"{metric.name}={metric.current_value}{metric.unit} > {metric.threshold}{metric.unit}",
            timestamp=now,
            attempt=attempt,
        )

        try:
            handler = self._restart_handlers[component]
            await handler()
            action.success = True
            self.total_restarts += 1
            self._restart_history[component].append(now)
            logger.info(
                "Self-healer: restarted '%s' (attempt %d/%d)",
                component,
                attempt,
                _MAX_RESTARTS_PER_HOUR,
            )

            await self._emit_event(
                "subsystem.healed",
                {"component": component, "attempt": attempt},
            )

        except Exception as e:
            action.success = False
            self.total_restart_failures += 1
            self._restart_history[component].append(now)
            logger.error("Self-healer: restart of '%s' failed: %s", component, e)

        self._healing_log.append(action)
        # [V2-04 FIX] Cap healing log to prevent unbounded memory growth
        if len(self._healing_log) > 200:
            self._healing_log = self._healing_log[-100:]

    # ── Status ────────────────────────────────────────────────────────────

    def get_health_summary(self) -> dict:
        """Return current health status of all monitored components."""
        return {
            name: {
                "value": m.current_value,
                "threshold": m.threshold,
                "unit": m.unit,
                "healthy": m.is_healthy,
                "last_check": m.last_check,
            }
            for name, m in self._metrics.items()
        }

    def get_healing_log(self) -> list[dict]:
        """Return history of healing actions."""
        return [
            {
                "component": a.component,
                "action": a.action,
                "reason": a.reason,
                "timestamp": a.timestamp,
                "success": a.success,
                "attempt": a.attempt,
            }
            for a in self._healing_log[-50:]  # last 50 entries
        ]

    def restarts_remaining(self, component: str) -> int:
        """How many restarts are left in the current hour window."""
        hour_ago = time.time() - 3600
        recent = sum(1 for t in self._restart_history[component] if t > hour_ago)
        return max(0, _MAX_RESTARTS_PER_HOUR - recent)

    # ── Event Bus ─────────────────────────────────────────────────────────

    async def _emit_event(self, event_type: str, data: dict):
        if self._event_bus:
            try:
                await self._event_bus.emit(event_type, data)
            except Exception as e:
                logger.debug("Failed to emit %s: %s", event_type, e)
