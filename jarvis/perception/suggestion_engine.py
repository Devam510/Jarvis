"""
jarvis.perception.suggestion_engine — Proactive suggestion engine.

Safety guarantees:
  [S5] Escalation guard — blocks during AUTONOMOUS, AWAITING_CONFIRM, and DND
  [S8] No feedback loop — suggestions are informational only, never auto-executed
"""

from __future__ import annotations

import logging
import time
from typing import Optional

from jarvis.core.event_bus import AsyncEventBus
from jarvis.utils.config import ProactiveConfig
from jarvis.utils.enums import InteractionState
from jarvis.utils.types import ScreenContext, Suggestion

logger = logging.getLogger(__name__)

# States during which suggestions are BLOCKED [S5]
_BLOCKED_STATES = frozenset(
    {
        InteractionState.AUTONOMOUS_PLANNING,
        InteractionState.AUTONOMOUS_EXECUTING,
        InteractionState.AWAITING_CONFIRM,
    }
)


class SuggestionEngine:
    """
    Event-driven proactive suggestion engine.

    Subscribes to `perception.context_updated` events and evaluates
    heuristic triggers. Emits `suggestion.offered` events (TTS only).

    [S8] NEVER creates tool calls, autonomy goals, or auto-executes anything.
    """

    def __init__(
        self,
        config: ProactiveConfig,
        event_bus: AsyncEventBus,
        state_machine=None,
    ):
        self._config = config
        self._event_bus = event_bus
        self._state_machine = state_machine

        # Cooldown tracking: {trigger_type: last_fired_timestamp}
        self._cooldowns: dict[str, float] = {}

        # [S5] Hourly rate limiting
        self._suggestions_this_hour: int = 0
        self._hour_start: float = time.time()

        # Track last context for staleness detection
        self._last_change_time: float = time.time()
        self._stale_warned: bool = False

    async def on_context_updated(self, context: ScreenContext):
        """Handle perception context updates. Main entry point."""
        if not self._config.enabled:
            return

        # [S5] Check Do Not Disturb
        if self._config.do_not_disturb:
            return

        # [S5] Check state machine — block during autonomous/confirmation
        if self._state_machine and hasattr(self._state_machine, "state"):
            if self._state_machine.state in _BLOCKED_STATES:
                return

        # Skip blacked-out contexts
        if context.is_blacked_out:
            return

        # Track screen staleness
        if context.change_magnitude > 0.05:
            self._last_change_time = time.time()
            self._stale_warned = False

        # Evaluate triggers
        suggestions = self._evaluate_triggers(context)

        for suggestion in suggestions:
            if self._passes_cooldown(suggestion) and self._passes_rate_limit():
                await self._emit_suggestion(suggestion)

    def _evaluate_triggers(self, ctx: ScreenContext) -> list[Suggestion]:
        """Run all heuristic triggers against current context."""
        suggestions = []

        # Trigger 1: Error detected on screen
        if ctx.detected_errors:
            error_summary = ctx.detected_errors[0] if ctx.detected_errors else "error"
            suggestions.append(
                Suggestion(
                    trigger="error_detected",
                    message=f"I noticed an error on your screen: {error_summary}. "
                    "Would you like help debugging it?",
                    confidence=0.85,
                    action_hint="debug_assist",
                )
            )

        # Trigger 2: Screen stale for extended period (>10 min no change)
        if (
            time.time() - self._last_change_time > 600
            and not self._stale_warned
            and ctx.active_app  # Don't trigger on blank screens
        ):
            self._stale_warned = True
            suggestions.append(
                Suggestion(
                    trigger="screen_stale",
                    message="You've been on the same screen for a while. "
                    "Need any help or want to take a break?",
                    confidence=0.6,
                    action_hint="",
                )
            )

        # Trigger 3: "Failed" or crash keywords in visible text
        if ctx.visible_text:
            lower_text = ctx.visible_text.lower()
            if "build failed" in lower_text or "compilation error" in lower_text:
                suggestions.append(
                    Suggestion(
                        trigger="build_failed",
                        message="Looks like a build failed. Want me to check the error logs?",
                        confidence=0.9,
                        action_hint="log_analysis",
                    )
                )

        # Filter by confidence threshold
        return [
            s for s in suggestions if s.confidence >= self._config.confidence_threshold
        ]

    def _passes_cooldown(self, suggestion: Suggestion) -> bool:
        """Check if this trigger type is still in cooldown."""
        now = time.time()
        last_fired = self._cooldowns.get(suggestion.trigger, 0.0)
        return (now - last_fired) >= self._config.cooldown_seconds

    def _passes_rate_limit(self) -> bool:
        """Check hourly rate limit."""
        now = time.time()
        # Reset hourly counter
        if now - self._hour_start >= 3600:
            self._suggestions_this_hour = 0
            self._hour_start = now
        return self._suggestions_this_hour < self._config.max_suggestions_per_hour

    async def _emit_suggestion(self, suggestion: Suggestion):
        """
        Emit a suggestion event for TTS to speak.

        [S8] This ONLY emits an event. It NEVER calls execute_goal,
        execute_tool_call, or any autonomy method. Suggestions are
        purely informational.
        """
        self._cooldowns[suggestion.trigger] = time.time()
        self._suggestions_this_hour += 1

        logger.info(
            "Suggestion offered: trigger=%s confidence=%.2f message=%s",
            suggestion.trigger,
            suggestion.confidence,
            suggestion.message[:80],
        )

        # [S8] Emit event — TTS handler in orchestrator speaks this.
        # No tool calls, no autonomy goals.
        await self._event_bus.emit(
            "suggestion.offered",
            {
                "id": suggestion.id,
                "trigger": suggestion.trigger,
                "message": suggestion.message,
                "confidence": suggestion.confidence,
                "action_hint": suggestion.action_hint,
                "timestamp": suggestion.timestamp,
            },
        )
