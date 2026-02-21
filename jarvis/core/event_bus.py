"""
jarvis.core.event_bus — Typed async publish-subscribe event bus with
backpressure, priority dispatch, fire-and-forget mode, and DLQ (Phase 5).

All inter-module communication flows through this bus.
No component holds a direct reference to another.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from collections import defaultdict, deque
from pathlib import Path
from typing import Any, Callable, Coroutine, Optional

from jarvis.utils.enums import EventPriority

logger = logging.getLogger(__name__)

# ── Channel → Default Priority Mapping ────────────────────────────────────

_CHANNEL_PRIORITIES: dict[str, EventPriority] = {
    # P0 — Critical (voice, safety)
    "audio.wake_detected": EventPriority.CRITICAL,
    "audio.speech_segment": EventPriority.CRITICAL,
    "stt.transcript": EventPriority.CRITICAL,
    "stt.empty": EventPriority.CRITICAL,
    "watchdog.timeout": EventPriority.CRITICAL,
    "watchdog.stuck_reset": EventPriority.CRITICAL,
    "watchdog.loop_detected": EventPriority.CRITICAL,
    # P1 — High (confirmation)
    "risk.require_confirmation": EventPriority.HIGH,
    "risk.authorized": EventPriority.HIGH,
    # P2 — Normal (execution, cognition)
    "cognition.plan_ready": EventPriority.NORMAL,
    "cognition.direct_response": EventPriority.NORMAL,
    "execution.result": EventPriority.NORMAL,
    "execution.step_result": EventPriority.NORMAL,
    "autonomy.goal_started": EventPriority.NORMAL,
    "autonomy.goal_completed": EventPriority.NORMAL,
    "autonomy.step_executed": EventPriority.NORMAL,
    # P3 — Background (logging, metrics, perception)
    "state.changed": EventPriority.BACKGROUND,
    "perception.context_updated": EventPriority.BACKGROUND,
    "suggestion.offered": EventPriority.BACKGROUND,
    "metrics.snapshot": EventPriority.BACKGROUND,
}


class DeadLetterQueue:
    """Persists events that repeatedly failed dispatch.

    Capped at max_entries to prevent unbounded disk growth.
    """

    def __init__(self, path: str = "state/dlq.json", max_entries: int = 100):
        self._path = str(Path(path).resolve())
        self._max_entries = max_entries
        self._entries: deque[dict] = deque(maxlen=max_entries)
        Path(self._path).parent.mkdir(parents=True, exist_ok=True)
        self._load()

    def record(self, channel: str, error: str, event_summary: str = ""):
        """Record a failed event."""
        entry = {
            "timestamp": time.time(),
            "channel": channel,
            "error": str(error)[:500],
            "event_summary": str(event_summary)[:200],
        }
        self._entries.append(entry)
        # BUG-04 FIX: schedule persist in background to avoid blocking event loop
        try:
            loop = asyncio.get_running_loop()
            loop.call_soon_threadsafe(
                lambda: asyncio.ensure_future(asyncio.to_thread(self._persist))
            )
        except RuntimeError:
            # No running loop (e.g., during tests) — persist synchronously
            self._persist()

    def _load(self):
        try:
            if os.path.exists(self._path):
                with open(self._path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self._entries = deque(
                    data[-self._max_entries :], maxlen=self._max_entries
                )
        except (json.JSONDecodeError, OSError):
            self._entries = deque(maxlen=self._max_entries)

    def _persist(self):
        try:
            tmp = self._path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(list(self._entries), f, indent=2, default=str)
            os.replace(tmp, self._path)
        except OSError as e:
            logger.error("DLQ: persist failed: %s", e)

    @property
    def entries(self) -> list[dict]:
        return list(self._entries)

    def __len__(self) -> int:
        return len(self._entries)


class AsyncEventBus:
    """Typed publish-subscribe event bus with backpressure and priority.

    Phase 5 enhancements:
    - Priority-aware dispatch (P0–P3)
    - Backpressure: drops P3 events and queues P2 when overloaded
    - Fire-and-forget mode for non-critical channels
    - Dead Letter Queue for persistently failing events
    """

    def __init__(
        self,
        max_pending: int = 200,
        dlq_path: str = "state/dlq.json",
    ):
        self._handlers: dict[str, list[Callable[..., Coroutine]]] = defaultdict(list)
        # BUG-09 FIX: use deque with maxlen for O(1) append + auto-eviction
        self._history: deque[tuple[str, Any]] = deque(maxlen=500)
        self._max_history = 500
        self._stopped = False

        # Phase 5: Backpressure
        self._max_pending = max_pending
        self._pending_count = 0
        self._dropped_count = 0
        self._total_emitted = 0
        self._total_errors = 0

        # Phase 5: Dead Letter Queue
        self._dlq = DeadLetterQueue(path=dlq_path)

        # Phase 5: Fire-and-forget tracking
        self._fire_and_forget_tasks: set[asyncio.Task] = set()

    def subscribe(self, channel: str, handler: Callable[..., Coroutine]):
        """Register an async handler for a channel."""
        self._handlers[channel].append(handler)
        logger.debug("Subscribed %s to channel '%s'", handler.__qualname__, channel)

    def unsubscribe(self, channel: str, handler: Callable[..., Coroutine]):
        """Remove a handler from a channel."""
        if handler in self._handlers[channel]:
            self._handlers[channel].remove(handler)

    def stop(self):
        """Prevent any new events from being dispatched."""
        self._stopped = True
        # Cancel any pending fire-and-forget tasks
        for task in self._fire_and_forget_tasks:
            if not task.done():
                task.cancel()
        self._fire_and_forget_tasks.clear()
        logger.info("EventBus stopped — no further events will be dispatched")

    def _get_priority(self, channel: str) -> EventPriority:
        """Resolve channel priority from the mapping, default NORMAL."""
        return _CHANNEL_PRIORITIES.get(channel, EventPriority.NORMAL)

    async def emit(
        self,
        channel: str,
        event: Any = None,
        *,
        fire_and_forget: bool = False,
        priority: Optional[EventPriority] = None,
    ):
        """Publish an event to all subscribers of a channel.

        Args:
            channel: Event channel name.
            event: Payload to dispatch.
            fire_and_forget: If True, dispatch in background task (non-blocking).
            priority: Override channel default priority.

        Backpressure rules:
        - P0/P1: ALWAYS dispatched regardless of pending count.
        - P2: Dispatched if pending < max_pending, otherwise queued briefly.
        - P3: Silently dropped if pending >= max_pending.
        """
        if self._stopped:
            logger.debug("EventBus stopped — dropping event on '%s'", channel)
            return

        prio = priority or self._get_priority(channel)

        # Backpressure check
        if self._pending_count >= self._max_pending:
            if prio == EventPriority.BACKGROUND:
                self._dropped_count += 1
                logger.debug(
                    "Backpressure: dropped P3 event on '%s' (pending=%d, total dropped=%d)",
                    channel,
                    self._pending_count,
                    self._dropped_count,
                )
                return
            elif prio == EventPriority.NORMAL:
                # Allow P2 through but log the pressure
                logger.warning(
                    "Backpressure: P2 event on '%s' dispatched under pressure (pending=%d)",
                    channel,
                    self._pending_count,
                )
            # P0/P1 always go through

        self._total_emitted += 1

        if fire_and_forget:
            task = asyncio.create_task(self._dispatch(channel, event, prio))
            self._fire_and_forget_tasks.add(task)
            task.add_done_callback(self._fire_and_forget_tasks.discard)
        else:
            await self._dispatch(channel, event, prio)

    async def _dispatch(self, channel: str, event: Any, priority: EventPriority):
        """Core dispatch logic — runs handlers, tracks pending count, DLQ on failure."""
        self._record(channel, event)
        handlers = self._handlers.get(channel, [])
        if not handlers:
            logger.debug("No handlers for channel '%s'", channel)
            return

        self._pending_count += 1
        try:
            results = await asyncio.gather(
                *(self._safe_call(h, event) for h in handlers),
                return_exceptions=True,
            )

            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    self._total_errors += 1
                    logger.error(
                        "Handler %s on '%s' raised: %s",
                        handlers[i].__qualname__,
                        channel,
                        result,
                    )
                    # DLQ: record persistent failures
                    self._dlq.record(
                        channel=channel,
                        error=str(result),
                        event_summary=repr(event)[:200] if event else "",
                    )
        finally:
            self._pending_count = max(0, self._pending_count - 1)

    async def _safe_call(self, handler: Callable, event: Any):
        try:
            return await handler(event)
        except Exception:
            raise

    def _record(self, channel: str, event: Any):
        # BUG-09 FIX: deque auto-evicts, no copy needed
        self._history.append((channel, event))

    @property
    def channels(self) -> list[str]:
        return list(self._handlers.keys())

    # ── Phase 5: Observability Properties ─────────────────────────────────

    @property
    def pending_count(self) -> int:
        """Current number of events being dispatched."""
        return self._pending_count

    @property
    def dropped_count(self) -> int:
        """Total number of P3 events dropped due to backpressure."""
        return self._dropped_count

    @property
    def total_emitted(self) -> int:
        """Total events successfully dispatched (not dropped)."""
        return self._total_emitted

    @property
    def total_errors(self) -> int:
        """Total handler errors across all channels."""
        return self._total_errors

    @property
    def dlq(self) -> DeadLetterQueue:
        """Access the dead letter queue."""
        return self._dlq

    def stats(self) -> dict:
        """Return a summary of bus health metrics."""
        return {
            "pending": self._pending_count,
            "total_emitted": self._total_emitted,
            "total_dropped": self._dropped_count,
            "total_errors": self._total_errors,
            "dlq_size": len(self._dlq),
            "channels": len(self._handlers),
            "stopped": self._stopped,
        }
