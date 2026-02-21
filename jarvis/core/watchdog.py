"""
jarvis.core.watchdog — Deadlock, loop, and stuck-state detection (Phase 5).

Three watchdog subsystems running in a single background task:
1. Loop Detector:  hash recent plan steps, abort on 3× repeated cycle.
2. Execution Timeout: cancel actions running longer than max_action_seconds.
3. Stuck-State Detector: force-reset if non-IDLE for > max_state_seconds.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import threading
import time
from collections import deque
from typing import Optional

from jarvis.core.event_bus import AsyncEventBus
from jarvis.core.state_machine import InteractionStateMachine
from jarvis.utils.config import WatchdogConfig
from jarvis.utils.enums import InteractionState

logger = logging.getLogger(__name__)


class Watchdog:
    """Production watchdog for deadlocks, infinite loops, and stuck states.

    Runs a single asyncio background task that ticks at `check_interval`.
    """

    def __init__(
        self,
        config: WatchdogConfig,
        event_bus: AsyncEventBus,
        state_machine: InteractionStateMachine,
    ):
        self._config = config
        self._event_bus = event_bus
        self._state_machine = state_machine

        # Loop detection
        self._plan_step_hashes: deque[str] = deque(maxlen=config.loop_detect_window)

        # Execution timeout
        self._current_action_start: Optional[float] = None
        self._current_action_name: str = ""
        self._action_lock = threading.Lock()  # BUG-20 FIX

        # Stuck-state detection
        self._last_transition_time: float = time.time()
        self._last_known_state: InteractionState = InteractionState.IDLE

        # Lifecycle
        self._task: Optional[asyncio.Task] = None
        self._running = False

    # ── Lifecycle ─────────────────────────────────────────────────────────

    async def start(self):
        """Start the watchdog background loop."""
        if not self._config.enabled:
            logger.info("Watchdog disabled by config")
            return
        if self._running:
            return
        self._running = True
        self._task = asyncio.create_task(self._tick_loop())
        logger.info(
            "Watchdog started (interval=%.1fs, action_timeout=%.0fs, "
            "state_timeout=%.0fs, loop_threshold=%d)",
            self._config.check_interval,
            self._config.max_action_seconds,
            self._config.max_state_seconds,
            self._config.loop_cycle_threshold,
        )

    async def stop(self):
        """Stop the watchdog cleanly."""
        self._running = False
        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
        logger.info("Watchdog stopped")

    # ── External hooks (called by orchestrator / risk engine) ─────────────

    def record_plan_step(self, tool_name: str, args_summary: str = ""):
        """Record a plan step for loop detection.

        Called by orchestrator when a tool call is dispatched.
        """
        raw = f"{tool_name}:{args_summary}"
        h = hashlib.md5(raw.encode(), usedforsecurity=False).hexdigest()[:8]
        self._plan_step_hashes.append(h)

    def action_started(self, action_name: str):
        """Mark the start of an execution action for timeout tracking."""
        with self._action_lock:  # BUG-20 FIX: atomic start/name pair
            self._current_action_start = time.time()
            self._current_action_name = action_name

    def action_completed(self):
        """Mark the current action as completed."""
        with self._action_lock:  # BUG-20 FIX: atomic clear
            self._current_action_start = None
            self._current_action_name = ""

    def record_transition(self):
        """Notify the watchdog that a state transition just occurred."""
        self._last_transition_time = time.time()
        self._last_known_state = self._state_machine.state

    # ── Core tick loop ────────────────────────────────────────────────────

    async def _tick_loop(self):
        """Background loop that checks all watchdog conditions."""
        try:
            while self._running:
                await asyncio.sleep(self._config.check_interval)
                if not self._running:
                    break

                try:
                    await self._check_loop_detection()
                    await self._check_execution_timeout()
                    await self._check_stuck_state()
                except Exception as e:
                    logger.error("Watchdog tick error: %s", e)

        except asyncio.CancelledError:
            logger.debug("Watchdog tick loop cancelled")

    # ── 1. Loop Detection ─────────────────────────────────────────────────

    async def _check_loop_detection(self):
        """Detect repeated plan step cycles (e.g. [A,B,A,B,A,B])."""
        steps = list(self._plan_step_hashes)
        if len(steps) < 4:
            return

        # Try cycle lengths from 1 to len/threshold
        max_cycle_len = len(steps) // self._config.loop_cycle_threshold
        for cycle_len in range(1, max_cycle_len + 1):
            # Extract the candidate cycle from the tail
            tail = steps[-cycle_len * self._config.loop_cycle_threshold :]
            if len(tail) < cycle_len * self._config.loop_cycle_threshold:
                continue

            # Check if the tail is made of repeated cycles
            candidate = tail[:cycle_len]
            is_loop = True
            for i in range(1, self._config.loop_cycle_threshold):
                segment = tail[i * cycle_len : (i + 1) * cycle_len]
                if segment != candidate:
                    is_loop = False
                    break

            if is_loop:
                cycle_repr = " → ".join(candidate)
                logger.critical(
                    "WATCHDOG: Loop detected! Cycle [%s] repeated %d times",
                    cycle_repr,
                    self._config.loop_cycle_threshold,
                )
                self._plan_step_hashes.clear()

                await self._event_bus.emit(
                    "watchdog.loop_detected",
                    {
                        "cycle": candidate,
                        "repeats": self._config.loop_cycle_threshold,
                        "message": f"Infinite loop detected: {cycle_repr}",
                    },
                )

                # Force reset state machine
                self._state_machine.reset()
                self._last_transition_time = time.time()
                return  # Only fire once per detection

    # ── 2. Execution Timeout ──────────────────────────────────────────────

    async def _check_execution_timeout(self):
        """Cancel actions that exceed max_action_seconds."""
        if self._current_action_start is None:
            return

        elapsed = time.time() - self._current_action_start
        if elapsed > self._config.max_action_seconds:
            logger.critical(
                "WATCHDOG: Action '%s' timed out after %.0fs (limit: %.0fs)",
                self._current_action_name,
                elapsed,
                self._config.max_action_seconds,
            )

            await self._event_bus.emit(
                "watchdog.timeout",
                {
                    "action": self._current_action_name,
                    "elapsed_seconds": elapsed,
                    "limit_seconds": self._config.max_action_seconds,
                    "message": f"Action '{self._current_action_name}' timed out after {elapsed:.0f}s",
                },
            )

            # Clear the action tracking
            self._current_action_start = None
            self._current_action_name = ""

            # Force reset to IDLE
            self._state_machine.reset()
            self._last_transition_time = time.time()

    # ── 3. Stuck-State Detection ──────────────────────────────────────────

    async def _check_stuck_state(self):
        """Detect if the system is stuck in a non-IDLE state too long."""
        current = self._state_machine.state

        # BUG-19 FIX: only reset timer on ACTUAL state changes, not every IDLE tick
        # Check if state changed since last check
        if current != self._last_known_state:
            self._last_transition_time = time.time()
            self._last_known_state = current
            return

        # IDLE is normally safe — skip stuck detection for it
        if current == InteractionState.IDLE:
            return

        elapsed = time.time() - self._last_transition_time
        if elapsed > self._config.max_state_seconds:
            logger.critical(
                "WATCHDOG: Stuck in %s for %.0fs (limit: %.0fs) — forcing IDLE",
                current.name,
                elapsed,
                self._config.max_state_seconds,
            )

            await self._event_bus.emit(
                "watchdog.stuck_reset",
                {
                    "stuck_state": current.name,
                    "elapsed_seconds": elapsed,
                    "limit_seconds": self._config.max_state_seconds,
                    "message": f"Stuck in {current.name} for {elapsed:.0f}s — reset to IDLE",
                },
            )

            self._state_machine.reset()
            self._last_transition_time = time.time()
            self._last_known_state = InteractionState.IDLE
