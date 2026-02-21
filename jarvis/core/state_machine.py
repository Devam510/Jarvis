"""
jarvis.core.state_machine — Interaction state machine with guarded transitions.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Optional

from jarvis.utils.enums import InteractionState

logger = logging.getLogger(__name__)


class IllegalStateTransition(Exception):
    """Raised when an invalid state transition is attempted."""

    def __init__(self, current: InteractionState, target: InteractionState):
        self.current = current
        self.target = target
        super().__init__(f"Illegal transition: {current.name} → {target.name}")


# Valid transitions map: current_state → set of allowed next states
_TRANSITIONS: dict[InteractionState, set[InteractionState]] = {
    InteractionState.IDLE: {
        InteractionState.LISTENING,
        InteractionState.REASONING,  # text input skips audio
        InteractionState.RESPONDING,  # direct response from text input
        InteractionState.AUTONOMOUS_PLANNING,  # Phase 3: start autonomous goal
        InteractionState.ERROR,
    },
    InteractionState.LISTENING: {
        InteractionState.TRANSCRIBING,
        InteractionState.REASONING,  # text input while listening
        InteractionState.RESPONDING,  # error response while listening
        InteractionState.IDLE,  # timeout / cancel
        InteractionState.ERROR,
    },
    InteractionState.TRANSCRIBING: {
        InteractionState.REASONING,
        InteractionState.IDLE,  # empty transcript
        InteractionState.ERROR,
    },
    InteractionState.REASONING: {
        InteractionState.PLANNING,
        InteractionState.RESPONDING,  # direct response, no tools
        InteractionState.ERROR,
    },
    InteractionState.PLANNING: {
        InteractionState.RISK_CLASSIFYING,
        InteractionState.REASONING,  # self-critique loop
        InteractionState.ERROR,
    },
    InteractionState.RISK_CLASSIFYING: {
        InteractionState.AWAITING_CONFIRM,
        InteractionState.EXECUTING,  # Tier 1/2 auto-authorized
        InteractionState.ERROR,
    },
    InteractionState.AWAITING_CONFIRM: {
        InteractionState.EXECUTING,  # confirmed
        InteractionState.IDLE,  # denied / timeout
        InteractionState.ERROR,
    },
    InteractionState.EXECUTING: {
        InteractionState.RISK_CLASSIFYING,  # next step in plan
        InteractionState.RESPONDING,
        InteractionState.ERROR,
    },
    InteractionState.RESPONDING: {
        InteractionState.MEMORY_UPDATE,
        InteractionState.ERROR,
    },
    InteractionState.MEMORY_UPDATE: {
        InteractionState.IDLE,
    },
    InteractionState.ERROR: {
        InteractionState.IDLE,  # recovery always returns to IDLE
    },
    # Phase 3: Autonomous states — [S8] NO transitions to LISTENING or
    # AWAITING_CONFIRM from here. Autonomy owns the state machine.
    InteractionState.AUTONOMOUS_PLANNING: {
        InteractionState.AUTONOMOUS_EXECUTING,  # plan ready → execute
        InteractionState.RESPONDING,  # goal complete → speak result
        InteractionState.IDLE,  # abort / cancel
        InteractionState.ERROR,
    },
    InteractionState.AUTONOMOUS_EXECUTING: {
        InteractionState.AUTONOMOUS_PLANNING,  # re-plan after step
        InteractionState.RESPONDING,  # all steps done → speak result
        InteractionState.IDLE,  # abort / cancel
        InteractionState.ERROR,
    },
}


class InteractionStateMachine:
    """
    State machine governing the agent's interaction lifecycle.
    Enforces valid transitions; invalid ones raise IllegalStateTransition.
    """

    def __init__(self):
        self._state = InteractionState.IDLE
        self._last_transition_time = time.time()
        self._transition_log: list[tuple[float, InteractionState, InteractionState]] = (
            []
        )
        # BUG-03 FIX: serialize transitions against concurrent async handlers
        self._lock = threading.Lock()

    @property
    def state(self) -> InteractionState:
        return self._state

    @property
    def time_in_state(self) -> float:
        return time.time() - self._last_transition_time

    def transition(self, target: InteractionState):
        """Transition to a new state. Raises IllegalStateTransition if invalid."""
        with self._lock:  # BUG-03 FIX: atomic read-check-write
            allowed = _TRANSITIONS.get(self._state, set())
            if target not in allowed:
                raise IllegalStateTransition(self._state, target)

            old = self._state
            self._state = target
            self._last_transition_time = time.time()
            self._transition_log.append((self._last_transition_time, old, target))
            logger.info("State: %s → %s", old.name, target.name)

    def reset(self):
        """Force-reset to IDLE (used by watchdog)."""
        with self._lock:  # BUG-03 FIX: atomic reset
            old = self._state
            self._state = InteractionState.IDLE
            self._last_transition_time = time.time()
            logger.warning("State FORCE RESET: %s → IDLE", old.name)

    @property
    def history(self) -> list[tuple[float, InteractionState, InteractionState]]:
        return list(self._transition_log)

    def can_transition(self, target: InteractionState) -> bool:
        return target in _TRANSITIONS.get(self._state, set())
