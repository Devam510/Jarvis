"""
jarvis.risk.risk_engine — Hybrid risk classification + confirmation protocol.

Two-stage model:
  Stage 1: Static heuristic scoring based on tool name and arguments
  Stage 2: LLM-based assessment for ambiguous cases

Three tiers:
  TIER_1 (auto-execute): read-only operations
  TIER_2 (notify): write, non-destructive
  TIER_3 (confirm): destructive, irreversible, external
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Optional

from jarvis.core.event_bus import AsyncEventBus
from jarvis.utils.config import SafetyConfig
from jarvis.utils.enums import RiskTier, ConfirmStatus, InteractionState
from jarvis.utils.types import (
    Plan,
    PlannedAction,
    RiskClassification,
    StepResult,
    ConfirmResult,
)
from jarvis.observability.logger import RiskAuditLog

logger = logging.getLogger(__name__)


# ── Heuristic Rules ──────────────────────────────────────────────────────────

# Patterns that indicate risk keywords in arguments
DESTRUCTIVE_PATTERNS = re.compile(
    r"(rm\s+-rf|del\s+/s|format|rmdir|drop\s+table|truncate|shutdown|reboot)",
    re.IGNORECASE,
)
EXFILTRATION_PATTERNS = re.compile(
    r"(curl|wget|http[s]?://|ftp://|scp\s|rsync\s)", re.IGNORECASE
)
PATH_TRAVERSAL_PATTERN = re.compile(r"\.\.[/\\]")
SYSTEM_PATHS = re.compile(
    r"(C:\\Windows|C:\\Program Files|/etc/|/usr/|/bin/|/sbin/|System32)", re.IGNORECASE
)

TOOL_BASE_RISK: dict[str, float] = {
    "file_read": 0.0,
    "search_files": 0.0,
    "search_content": 0.0,
    "system_info": 0.0,
    "store_memory": 0.0,
    "set_reminder": 0.0,
    "get_time": 0.0,
    "system_query": 0.0,  # Phase 2: read-only system metrics
    "read_messages": 0.1,  # Phase 2: read-only messaging
    "app_launch": 0.1,
    "file_write": 0.3,
    "file_copy": 0.2,
    "file_move": 0.4,
    "send_message": 0.4,  # Phase 2: outbound messaging
    "run_script": 0.4,  # Phase 2: audited code execution
    "install_package": 0.4,  # Phase 2: pip install
    "file_delete": 0.7,
    "execute_code": 0.5,
    "browser_navigate": 0.4,
    "browser_interact": 0.5,
}


def compute_heuristic_score(tool_name: str, args: dict) -> tuple[float, list[str]]:
    """Compute a heuristic risk score [0.0, 1.0] and list of triggered rules."""
    score = TOOL_BASE_RISK.get(tool_name, 0.5)
    triggers = []

    # BUG-12 FIX: scan only string VALUES, not key names (avoids false positives
    # from arg keys like "format" or "path" matching DESTRUCTIVE_PATTERNS)
    arg_values = " ".join(
        str(v) for v in args.values() if isinstance(v, (str, int, float))
    )

    if DESTRUCTIVE_PATTERNS.search(arg_values):
        score = max(score, 0.8)
        triggers.append("destructive_command")

    if EXFILTRATION_PATTERNS.search(arg_values):
        score = max(score, 0.7)
        triggers.append("external_data_transfer")

    if PATH_TRAVERSAL_PATTERN.search(arg_values):
        score = max(score, 0.9)
        triggers.append("path_traversal")

    if SYSTEM_PATHS.search(arg_values):
        score = max(score, 0.8)
        triggers.append("system_path_access")

    # Recursive flag on destructive ops
    if args.get("recursive", False) and tool_name in ("file_delete", "file_move"):
        score = max(score, 0.7)
        triggers.append("recursive_operation")

    return score, triggers


def score_to_tier(score: float) -> RiskTier:
    if score <= 0.2:
        return RiskTier.TIER_1
    elif score <= 0.5:
        return RiskTier.TIER_2
    else:
        return RiskTier.TIER_3


# ── Confirmation Protocol ────────────────────────────────────────────────────

CONFIRM_PHRASES = {
    "yes",
    "yeah",
    "yep",
    "confirm",
    "do it",
    "go ahead",
    "proceed",
    "affirmative",
    "ok",
    "okay",
    "sure",
}
DENY_PHRASES = {
    "no",
    "nope",
    "cancel",
    "stop",
    "abort",
    "don't",
    "deny",
    "negative",
    "nah",
}


def classify_confirmation(text: str) -> ConfirmStatus:
    """Classify a spoken response as confirmation or denial."""
    lower = text.lower().strip()
    words = set(lower.split())

    # Check both single-word and multi-word phrase matches
    confirm_hit = bool(words & CONFIRM_PHRASES) or any(
        p in lower for p in CONFIRM_PHRASES if " " in p
    )
    deny_hit = bool(words & DENY_PHRASES) or any(
        p in lower for p in DENY_PHRASES if " " in p
    )

    if deny_hit and not confirm_hit:
        return ConfirmStatus.DENIED
    if confirm_hit and not deny_hit:
        return ConfirmStatus.CONFIRMED
    if confirm_hit and deny_hit:
        return ConfirmStatus.AMBIGUOUS
    return ConfirmStatus.AMBIGUOUS


@dataclass
class ConfirmationContext:
    """State object for a single pending confirmation request."""

    action_id: str  # uuid4 — unique per confirmation request
    action: PlannedAction
    description: str  # human-readable summary
    event: asyncio.Event = field(default_factory=asyncio.Event)
    result: Optional[bool] = None  # True=confirmed, False=denied, None=pending
    expired: bool = False  # set True after timeout — blocks late responses


class RiskEngine:
    """
    Risk classification and confirmation engine.
    Classifies each action in a plan, manages confirmation dialogues.
    """

    def __init__(
        self, config: SafetyConfig, event_bus: AsyncEventBus, tts=None, stt=None
    ):
        self.config = config
        self.event_bus = event_bus
        self.tts = tts
        self.stt = stt
        self.audit = RiskAuditLog()
        self._pending_confirm: Optional[ConfirmationContext] = None

    async def classify_plan(self, event: dict):
        """Classify all actions in a plan and authorize them sequentially."""
        plan: Plan = event.get("plan")
        response_text: str = event.get("response_text", "")
        correlation_id: str = event.get("correlation_id", "")

        if not plan:
            return

        logger.info("Classifying risk for %d-step plan", len(plan.actions))

        execution_results = []

        while plan.actions:
            action = plan.next_step()
            score, triggers = compute_heuristic_score(
                action.tool_name, action.arguments
            )
            tier = score_to_tier(score)

            # Force TIER_3 if the action itself says it requires confirmation
            if action.requires_confirmation:
                tier = RiskTier.TIER_3

            classification = RiskClassification(
                tool_call=None,
                tier=tier,
                heuristic_score=score,
                correlation_id=correlation_id,
            )

            logger.info(
                "Action %s: score=%.2f tier=%s triggers=%s",
                action.tool_name,
                score,
                tier.name,
                triggers,
            )

            # Audit log
            self.audit.record(
                correlation_id=correlation_id,
                tool_name=action.tool_name,
                arguments=action.arguments,
                heuristic_score=score,
                llm_risk_score=0.0,
                tier=tier.name,
                authorization="pending",
                reasoning=", ".join(triggers),
            )

            authorized = False
            if tier == RiskTier.TIER_1:
                authorized = True
                logger.info("TIER_1: auto-authorized %s", action.tool_name)

            elif tier == RiskTier.TIER_2:
                authorized = True
                logger.info(
                    "TIER_2: auto-authorized with notification %s", action.tool_name
                )

            elif tier == RiskTier.TIER_3:
                authorized = await self._confirm_action(action)
                if not authorized:
                    logger.info("TIER_3: user DENIED %s", action.tool_name)
                    if self.tts:
                        await self.tts.speak("Action cancelled.")
                    continue

            if authorized:
                # Subscribe to result BEFORE emitting authorized (prevents race condition)
                result_future: asyncio.Future = asyncio.get_event_loop().create_future()

                async def _capture(event, _future=result_future, _cid=correlation_id):
                    # BUG-01 FIX: only capture results for THIS action's correlation_id
                    if not _future.done() and event.get("correlation_id") == _cid:
                        _future.set_result(event)

                self.event_bus.subscribe("execution.step_result", _capture)

                await self.event_bus.emit(
                    "risk.authorized",
                    {
                        "action": action,
                        "tier": tier,
                        "correlation_id": correlation_id,
                    },
                )

                # Wait for execution result
                try:
                    result = await asyncio.wait_for(result_future, timeout=60)
                    if result:
                        execution_results.append(
                            {
                                "tool": action.tool_name,
                                "result": result.get("result"),
                            }
                        )
                except asyncio.TimeoutError:
                    logger.warning("Execution timeout for %s", action.tool_name)
                finally:
                    self.event_bus.unsubscribe("execution.step_result", _capture)

        # Build final spoken response from tool results
        final_response = self._build_response(response_text, execution_results)
        logger.info("Final response: %s", final_response[:200])

        # Emit execution.result for the orchestrator to speak
        await self.event_bus.emit(
            "execution.result",
            {
                "response": final_response,
                "correlation_id": correlation_id,
            },
        )

    def _build_response(self, llm_response: str, results: list) -> str:
        """Build a spoken response from LLM text and tool execution results."""
        if not results:
            return llm_response or "Done."

        parts = []
        for r in results:
            tool = r.get("tool", "")
            result = r.get("result")
            if result is None:
                continue

            # Extract the output from the StepResult
            output = (
                getattr(result, "output", None) if hasattr(result, "output") else None
            )
            status = (
                getattr(result, "status", "unknown")
                if hasattr(result, "status")
                else "unknown"
            )
            error = getattr(result, "error", None) if hasattr(result, "error") else None

            if status == "failed" and error:
                parts.append(f"Sorry, {tool} failed: {error}")
                continue

            if output is None:
                continue

            if tool == "get_time" and isinstance(output, dict):
                time_str = output.get("time", "")
                date_str = output.get("date", "")
                parts.append(f"The current time is {time_str} on {date_str}.")
            elif tool == "app_launch":
                parts.append(
                    f"Done! I've opened {output}."
                    if isinstance(output, str)
                    else "Application launched."
                )
            elif tool == "system_info" and isinstance(output, dict):
                parts.append(f"Here's your system info: {json.dumps(output)[:200]}")
            else:
                if isinstance(output, str):
                    parts.append(output[:300])
                elif isinstance(output, dict):
                    parts.append(json.dumps(output, default=str)[:300])

        if parts:
            return " ".join(parts)
        return llm_response or "Done."

    async def _confirm_action(self, action: PlannedAction) -> bool:
        """Run the confirmation protocol for a TIER_3 action.

        Emits a `risk.require_confirmation` event and waits for
        `receive_confirmation()` to be called by the Orchestrator.
        """
        summary = self._describe_action(action)
        action_id = str(uuid.uuid4())

        ctx = ConfirmationContext(
            action_id=action_id,
            action=action,
            description=summary,
        )
        self._pending_confirm = ctx

        logger.info("TIER_3 confirmation requested [%s]: %s", action_id[:8], summary)

        # Notify orchestrator (which will TTS the prompt and set state)
        await self.event_bus.emit(
            "risk.require_confirmation",
            {
                "action_id": action_id,
                "description": summary,
                "tool_name": action.tool_name,
            },
        )

        # Wait for the orchestrator to call receive_confirmation()
        try:
            await asyncio.wait_for(
                ctx.event.wait(),
                timeout=self.config.confirmation_timeout_seconds,
            )
            confirmed = ctx.result is True
            logger.info(
                "Confirmation [%s] result: %s",
                action_id[:8],
                "CONFIRMED" if confirmed else "DENIED",
            )
            return confirmed
        except asyncio.TimeoutError:
            ctx.expired = True
            self._pending_confirm = None
            logger.warning("Confirmation [%s] timed out", action_id[:8])
            if self.tts:
                await self.tts.speak("Confirmation timed out. Cancelling.")
            return False
        finally:
            # Always clean up after resolution
            if self._pending_confirm is ctx:
                self._pending_confirm = None

    def receive_confirmation(self, action_id: str, confirmed: bool) -> bool:
        """Accept a confirmation response from the Orchestrator.

        Returns True if the confirmation was applied, False if it was
        rejected (wrong ID, expired, or no pending request).
        """
        ctx = self._pending_confirm

        if ctx is None:
            logger.warning(
                "Received confirmation [%s] but no request is pending — ignoring",
                action_id[:8],
            )
            return False

        if ctx.expired:
            logger.warning(
                "Received confirmation [%s] but request already expired — ignoring",
                action_id[:8],
            )
            return False

        if ctx.action_id != action_id:
            logger.warning(
                "Confirmation ID mismatch: expected [%s], got [%s] — ignoring",
                ctx.action_id[:8],
                action_id[:8],
            )
            return False

        ctx.result = confirmed
        ctx.event.set()
        logger.info(
            "Confirmation [%s] delivered: %s",
            action_id[:8],
            "CONFIRMED" if confirmed else "DENIED",
        )
        return True

    async def _wait_for_result(
        self, correlation_id: str, timeout: float = 60
    ) -> Optional[dict]:
        """Wait for an execution result event."""
        result_future: asyncio.Future = asyncio.get_event_loop().create_future()

        async def _capture(event):
            if not result_future.done():
                result_future.set_result(event)

        self.event_bus.subscribe("execution.step_result", _capture)
        try:
            return await asyncio.wait_for(result_future, timeout=timeout)
        except asyncio.TimeoutError:
            logger.warning("Execution result timeout for %s", correlation_id)
            return None
        finally:
            self.event_bus.unsubscribe("execution.step_result", _capture)

    @staticmethod
    def _describe_action(action: PlannedAction) -> str:
        """Generate a human-readable description of an action."""
        args = action.arguments
        name = action.tool_name
        if name == "file_delete":
            path = args.get("path", "unknown")
            rec = "recursively " if args.get("recursive") else ""
            return f"{rec}delete {path}"
        if name == "file_write":
            return f"write to {args.get('path', 'a file')}"
        if name == "file_move":
            return f"move {args.get('source', '?')} to {args.get('destination', '?')}"
        if name == "execute_code":
            lang = args.get("language", "code")
            return f"execute {lang} code in a sandbox"
        if name == "browser_navigate":
            return f"navigate browser to {args.get('url', 'a URL')}"
        return f"execute {name}"
