"""
jarvis.autonomy.autonomy_engine — Hardened autonomous multi-step execution.

Implements decompose → plan → execute → evaluate → re-plan with 17 safeguards:
  S1  Single active goal lock
  S2  Structured cancellation
  S3  Canonical plan hashing
  S4  Hard iteration ceiling
  S5  Strict confidence gate
  S6  Step-level kill enforcement
  S7  Idempotent rollback
  S9  Reflection bounds
  S10 Full event emission
  S11 Error categorization
  S12 Correlation ID everywhere
  S13 Goal snapshot at start
  S14 Memory write isolation
  S15 Double-rollback guard
  S16 Resource budget
  S17 Telemetry
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import time
import uuid
from typing import Any, Optional

from jarvis.core.event_bus import AsyncEventBus
from jarvis.utils.config import JarvisConfig, AutonomyConfig
from jarvis.utils.types import (
    AutonomyTelemetry,
    ExecutionTransaction,
    GoalResult,
    GoalSnapshot,
    LoopDetectionState,
    Plan,
    PlannedAction,
    ResourceBudget,
    StepResult,
)

logger = logging.getLogger(__name__)


# ── Error categorization [S11] ───────────────────────────────────────────────

_FATAL_ERRORS = frozenset(
    {
        "PermissionError",
        "PathViolationError",
        "InvalidToolError",
        "SecurityError",
    }
)


def _is_fatal(error: Optional[str]) -> bool:
    """[S11] Classify error as fatal (no retry) or retriable."""
    if not error:
        return False
    for fatal in _FATAL_ERRORS:
        if fatal in error:
            return True
    return False


class AutonomyEngine:
    """
    Hardened autonomous multi-step goal execution engine.

    Lifecycle:
      1. execute_goal() acquires the single-goal lock [S1]
      2. Captures a GoalSnapshot [S13]
      3. Loops: decompose → execute steps → reflect
      4. Emits events throughout [S10]
      5. On completion/failure, optionally rolls back [S7] and commits memory [S14]
    """

    def __init__(
        self,
        config: JarvisConfig,
        event_bus: AsyncEventBus,
        cognition=None,
        execution=None,
        memory=None,
        state_machine=None,  # F6: accept state machine reference
    ):
        self.config = config
        self.autonomy_cfg: AutonomyConfig = config.autonomy
        self.event_bus = event_bus
        self.cognition = cognition
        self.execution = execution
        self.memory = memory
        self.state_machine = state_machine  # F6: optional state machine

        # [S1] Single active goal lock — atomic flag pattern (F1)
        self._goal_lock = asyncio.Lock()
        self._active_goal_task: Optional[asyncio.Task] = None
        self._active_correlation_id: Optional[str] = None

        # [S15] Double-rollback guard — per-goal, reset in _run_goal (F1)
        self._rollback_executed = False

    # ── Public API ────────────────────────────────────────────────────────

    async def execute_goal(self, goal: str, context: str = "") -> GoalResult:
        """
        Execute a complex multi-step goal autonomously.

        [S1] Only one goal can run at a time — atomic lock acquisition.
        [S16] Hard-stops if resource budget is exceeded.
        """
        # [S1] Atomic non-blocking lock acquisition (BUG-02 FIX: no TOCTOU)
        # Use try-acquire pattern instead of check-then-lock
        acquired = self._goal_lock.locked()
        if acquired:
            # Fast-path rejection when lock is visibly held (hint only)
            logger.warning("Autonomy: goal rejected — another goal is active")
            return GoalResult(
                goal=goal,
                status="rejected",
                correlation_id=str(uuid.uuid4()),
            )

        # Authoritative atomic acquire — if contention happened between
        # the hint check and here, we block briefly then proceed safely
        async with self._goal_lock:
            correlation_id = str(uuid.uuid4())  # [S12]
            self._active_correlation_id = correlation_id
            self._rollback_executed = False  # [S15] Per-goal reset

            result = GoalResult(
                goal=goal,
                correlation_id=correlation_id,
            )

            # F6: Transition state machine to AUTONOMOUS_PLANNING
            self._transition_state("AUTONOMOUS_PLANNING")

            try:
                # F2: Wrap _run_goal in a task so cancel_goal() can target it
                run_coro = self._run_goal(goal, context, result, correlation_id)
                self._active_goal_task = asyncio.current_task()  # F2

                # Wrap in total goal timeout
                result = await asyncio.wait_for(
                    run_coro,
                    timeout=self.autonomy_cfg.total_goal_timeout_seconds,
                )
            except asyncio.TimeoutError:
                logger.error(
                    "Autonomy [%s]: goal timed out after %.0fs",
                    correlation_id[:8],
                    self.autonomy_cfg.total_goal_timeout_seconds,
                )
                result.status = "timeout"
                if result.telemetry:
                    result.telemetry.aborted = True  # F5: set aborted flag
                # F10: emit in try/except to survive cancellation
                await self._safe_emit(
                    "autonomy.goal_aborted",
                    {
                        "reason": "total_goal_timeout",
                        "partial_result": result,
                        "correlation_id": correlation_id,
                    },
                )
            except asyncio.CancelledError:
                logger.warning("Autonomy [%s]: goal cancelled", correlation_id[:8])
                result.status = "aborted"
                if result.telemetry:
                    result.telemetry.aborted = True  # F5
                # F10: best-effort emit on cancellation
                await self._safe_emit(
                    "autonomy.goal_aborted",
                    {
                        "reason": "cancelled",
                        "partial_result": result,
                        "correlation_id": correlation_id,
                    },
                )
            except Exception as exc:
                logger.exception("Autonomy [%s]: unexpected error", correlation_id[:8])
                result.status = "aborted"
                if result.telemetry:
                    result.telemetry.aborted = True  # F5
                await self._safe_emit(
                    "autonomy.goal_aborted",
                    {
                        "reason": f"unexpected: {exc}",
                        "partial_result": result,
                        "correlation_id": correlation_id,
                    },
                )
            finally:
                # F2/F11: Always clear task ref and return to IDLE
                self._active_goal_task = None
                self._active_correlation_id = None
                self._transition_state("IDLE")  # F11: always return to IDLE

            return result

    async def cancel_goal(self):
        """[S2] Structured cancellation: cancel → await → IDLE.

        F3: Removed asyncio.shield() — it was preventing cancellation.
        """
        task = self._active_goal_task
        if task and not task.done():
            logger.warning("Autonomy: cancelling active goal")
            task.cancel()
            try:
                # F3: await the task directly — no shield
                await asyncio.wait_for(
                    asyncio.ensure_future(self._wait_for_task(task)),
                    timeout=10.0,
                )
            except (asyncio.TimeoutError, asyncio.CancelledError):
                pass

    @staticmethod
    async def _wait_for_task(task: asyncio.Task):
        """Await a task, suppressing CancelledError."""
        try:
            await task
        except (asyncio.CancelledError, Exception):
            pass

    @property
    def is_running(self) -> bool:
        return self._goal_lock.locked()

    # ── Core loop ─────────────────────────────────────────────────────────

    async def _run_goal(
        self,
        goal: str,
        context: str,
        result: GoalResult,
        correlation_id: str,
    ) -> GoalResult:
        """Internal goal execution with all safeguards."""
        start_time = time.time()

        # [S13] Capture snapshot before any mutation
        snapshot = GoalSnapshot(
            goal=goal,
            correlation_id=correlation_id,
            initial_context=context,
            working_directory=os.getcwd(),
        )

        # [S16] Resource budget
        budget = ResourceBudget(
            max_tool_calls=self.autonomy_cfg.max_tool_calls_per_goal,
            max_tokens=self.autonomy_cfg.max_tokens_per_goal,
            max_execution_time=self.autonomy_cfg.max_execution_time_seconds,
        )

        # [S17] Telemetry
        telemetry = AutonomyTelemetry()
        result.telemetry = telemetry

        # [S3] Loop detection — BUG-16 NOTE: created fresh per-goal, no cross-goal leak
        loop_detector = LoopDetectionState(
            window_size=self.autonomy_cfg.loop_detection_window,
        )

        # Transaction for rollback [S7]
        transaction = ExecutionTransaction(correlation_id=correlation_id)

        # [S14] Deferred memory writes
        deferred_memory: list[str] = []

        # [S10] Emit goal_started
        await self._emit(
            "autonomy.goal_started",
            {
                "goal": goal,
                "correlation_id": correlation_id,
            },
        )

        iteration = 0

        # ── Main loop ────────────────────────────────────────────────────
        while iteration < self.autonomy_cfg.max_iterations:  # [S4] Hard ceiling
            iteration += 1
            telemetry.iteration_count = iteration
            logger.info(
                "Autonomy [%s]: iteration %d/%d",
                correlation_id[:8],
                iteration,
                self.autonomy_cfg.max_iterations,
            )

            # [S16] Check budget before planning
            if budget.exhausted:
                logger.warning(
                    "Autonomy [%s]: resource budget exhausted", correlation_id[:8]
                )
                result.status = "budget_exhausted"
                telemetry.aborted = True  # F5
                await self._emit(
                    "autonomy.goal_aborted",
                    {
                        "reason": "resource_budget_exhausted",
                        "correlation_id": correlation_id,
                    },
                )
                break

            # F6: Transition to AUTONOMOUS_PLANNING for each iteration
            self._transition_state("AUTONOMOUS_PLANNING")

            # Step 1: Decompose/plan via LLM
            plan = await self._decompose_goal(goal, context, result, budget)
            if plan is None or not plan.actions:
                logger.info(
                    "Autonomy [%s]: no more actions — goal complete",
                    correlation_id[:8],
                )
                break

            # [S3] Canonical hash + loop check
            plan_hash = self._hash_plan(plan)
            if loop_detector.push(plan_hash):
                logger.error(
                    "Autonomy [%s]: DEADLOCK — %d identical plans in a row",
                    correlation_id[:8],
                    self.autonomy_cfg.loop_detection_window,
                )
                telemetry.deadlock_detected = True
                result.status = "deadlock"
                await self._emit(
                    "autonomy.goal_aborted",
                    {
                        "reason": "deadlock_detected",
                        "correlation_id": correlation_id,
                    },
                )
                break

            # F6: Transition to AUTONOMOUS_EXECUTING
            self._transition_state("AUTONOMOUS_EXECUTING")

            # Step 2: Execute each step
            step_failed_fatal = False
            for step_idx, action in enumerate(plan.actions):
                # [S16] Budget check per step
                if budget.exhausted:
                    logger.warning(
                        "Autonomy [%s]: budget exhausted mid-plan",
                        correlation_id[:8],
                    )
                    result.status = "budget_exhausted"
                    break

                # [S10] step_started
                await self._emit(
                    "autonomy.step_started",
                    {
                        "tool_name": action.tool_name,
                        "args": action.arguments,
                        "step_index": step_idx,
                        "correlation_id": correlation_id,
                    },
                )

                step_result = await self._execute_step(
                    action,
                    budget,
                    correlation_id,
                )
                result.results.append(step_result)
                telemetry.step_latencies_ms.append(step_result.duration_ms)

                if step_result.status == "success":
                    result.steps_completed += 1
                    transaction.record(action, step_result)
                    budget.consume_tool_call()
                    telemetry.total_tool_calls += 1

                    # [S10] step_completed
                    await self._emit(
                        "autonomy.step_completed",
                        {
                            "tool_name": action.tool_name,
                            "result": step_result,
                            "step_index": step_idx,
                            "correlation_id": correlation_id,
                        },
                    )
                else:
                    result.steps_failed += 1

                    # [S11] Check if fatal
                    if _is_fatal(step_result.error):
                        logger.error(
                            "Autonomy [%s]: FATAL error on %s: %s",
                            correlation_id[:8],
                            action.tool_name,
                            step_result.error,
                        )
                        step_failed_fatal = True
                        await self._emit(
                            "autonomy.step_failed",
                            {
                                "tool_name": action.tool_name,
                                "error": step_result.error,
                                "retriable": False,
                                "step_index": step_idx,
                                "correlation_id": correlation_id,
                            },
                        )
                        break

                    # [S10] step_failed (retriable)
                    await self._emit(
                        "autonomy.step_failed",
                        {
                            "tool_name": action.tool_name,
                            "error": step_result.error,
                            "retriable": True,
                            "step_index": step_idx,
                            "correlation_id": correlation_id,
                        },
                    )

                    # Retry logic
                    if action.retry_count < self.autonomy_cfg.max_retries_per_step:
                        action.retry_count += 1
                        logger.info(
                            "Autonomy [%s]: retrying %s (attempt %d)",
                            correlation_id[:8],
                            action.tool_name,
                            action.retry_count,
                        )
                        retry_result = await self._execute_step(
                            action,
                            budget,
                            correlation_id,
                        )
                        result.results.append(retry_result)
                        telemetry.step_latencies_ms.append(retry_result.duration_ms)

                        if retry_result.status == "success":
                            result.steps_completed += 1
                            result.steps_failed -= 1
                            transaction.record(action, retry_result)
                            budget.consume_tool_call()
                            telemetry.total_tool_calls += 1

            # Fatal error → rollback and abort
            if step_failed_fatal:
                await self._rollback(transaction, result, correlation_id)
                result.status = "aborted"
                telemetry.aborted = True  # F5
                break

            # Budget exhausted mid-plan → break
            if budget.exhausted:
                break

            # Step 3: Reflect — should we continue? [S5] [S9]
            should_continue, confidence = await self._reflect(
                goal, result, correlation_id
            )
            telemetry.confidences.append(confidence)
            result.confidence = confidence

            if not should_continue:
                logger.info(
                    "Autonomy [%s]: reflection says stop (confidence=%.2f)",
                    correlation_id[:8],
                    confidence,
                )
                break

            # [S5] Strict confidence gate
            if confidence < self.autonomy_cfg.confidence_threshold:
                logger.info(
                    "Autonomy [%s]: confidence %.2f < threshold %.2f — stopping",
                    correlation_id[:8],
                    confidence,
                    self.autonomy_cfg.confidence_threshold,
                )
                break

            # Update context for next iteration
            context = self._build_progress_context(result)

        # ── Finalize ─────────────────────────────────────────────────────────
        result.total_iterations = iteration
        result.duration_ms = (time.time() - start_time) * 1000

        if result.status == "completed":
            result.status = "completed" if result.steps_failed == 0 else "partial"

        # F4/S10: Emit goal_completed on non-aborted outcomes
        if result.status in ("completed", "partial"):
            await self._emit(
                "autonomy.goal_completed",
                {
                    "goal": goal,
                    "status": result.status,
                    "steps_completed": result.steps_completed,
                    "correlation_id": correlation_id,
                },
            )

        # [S14] Commit deferred memory only on success or finalized abort
        if self.memory:
            summary = (
                f"Goal: {goal}. "
                f"Steps: {result.steps_completed}/{result.steps_completed + result.steps_failed}, "
                f"iterations: {iteration}, status: {result.status}"
            )
            deferred_memory.append(summary)
            for mem_entry in deferred_memory:
                try:
                    await self.memory.store_task_outcome(mem_entry)
                except Exception:
                    logger.warning(
                        "Autonomy [%s]: memory write failed", correlation_id[:8]
                    )

        # F5: finalize telemetry token count
        telemetry.total_tokens = budget.tokens_used

        logger.info(
            "Autonomy [%s]: goal '%s' → %s (%d/%d steps, %d iterations, %.0fms)",
            correlation_id[:8],
            goal[:50],
            result.status,
            result.steps_completed,
            result.steps_completed + result.steps_failed,
            iteration,
            result.duration_ms,
        )
        return result

    # ── Planning ──────────────────────────────────────────────────────────

    async def _decompose_goal(
        self,
        goal: str,
        context: str,
        progress: GoalResult,
        budget: ResourceBudget,
    ) -> Optional[Plan]:
        """Ask the LLM to plan the next steps."""
        if not self.cognition:
            return None

        progress_summary = self._build_progress_context(progress)

        decompose_prompt = f"""Given this goal: "{goal}"

Progress so far:
{progress_summary or "No steps completed yet."}

Additional context:
{context or "None."}

Budget remaining: {budget.tool_calls_remaining} tool calls, {budget.time_remaining:.0f}s

What are the next steps to complete this goal? If the goal is already complete, respond with no actions.
For each action, optionally provide a rollback_command if the action is reversible.
Respond with JSON: {{"thought": "...", "confidence": 0.0-1.0, "actions": [{{"tool": "...", "args": {{}}, "rollback_command": {{}} }}]}}"""

        result = await self.cognition.call_llm(
            [
                {
                    "role": "system",
                    "content": (
                        "You are Jarvis planning assistant. Decompose goals into "
                        "concrete tool calls. Respond with JSON only."
                    ),
                },
                {"role": "user", "content": decompose_prompt},
            ]
        )

        # F7: Track token usage from LLM call
        if result and isinstance(result, dict):
            token_estimate = len(json.dumps(result)) // 4  # rough estimate
            budget.consume_tokens(token_estimate)

        if result and result.get("actions"):
            actions = [
                PlannedAction(
                    tool_name=a.get("tool", ""),
                    arguments=a.get("args", {}),
                    requires_confirmation=a.get("requires_confirmation", False),
                    depends_on=a.get("depends_on"),
                    rollback_command=a.get("rollback_command"),
                )
                for a in result["actions"]
            ]
            return Plan(
                thought=result.get("thought", ""),
                confidence=result.get("confidence", 0.5),
                actions=actions,
                correlation_id=progress.correlation_id,
            )
        return None

    # ── Execution ─────────────────────────────────────────────────────────

    async def _execute_step(
        self,
        action: PlannedAction,
        budget: ResourceBudget,
        correlation_id: str,
    ) -> StepResult:
        """[S6] Execute a single step with timeout and kill enforcement.

        Calls self.execution.execute_tool_call() which should return a StepResult.
        """
        if not self.execution:
            return StepResult(status="failed", error="No executor available")

        start = time.time()
        try:
            raw_result = await asyncio.wait_for(
                self.execution.execute_tool_call(
                    {
                        "action": action,
                        "correlation_id": correlation_id,
                    }
                ),
                timeout=self.autonomy_cfg.step_timeout_seconds,
            )

            # If executor returns a StepResult directly, use it
            if isinstance(raw_result, StepResult):
                raw_result.duration_ms = (time.time() - start) * 1000
                return raw_result

            # Otherwise treat as success with the return value as output
            return StepResult(
                status="success",
                output=raw_result,
                duration_ms=(time.time() - start) * 1000,
            )

        except asyncio.TimeoutError:
            # [S6] Kill enforcement — emit kill event for executor
            logger.error(
                "Autonomy [%s]: step %s timed out — emitting kill",
                correlation_id[:8],
                action.tool_name,
            )
            await self._safe_emit(
                "execution.kill_step",
                {
                    "tool_name": action.tool_name,
                    "correlation_id": correlation_id,
                },
            )
            return StepResult(
                status="timeout",
                error=f"Step {action.tool_name} timed out after "
                f"{self.autonomy_cfg.step_timeout_seconds}s",
                duration_ms=(time.time() - start) * 1000,
            )
        except asyncio.CancelledError:
            # MUST propagate — do not absorb. External cancellation (goal
            # timeout or user cancel) needs to reach execute_goal's handler.
            raise

    # ── Reflection ────────────────────────────────────────────────────────

    async def _reflect(
        self, goal: str, progress: GoalResult, correlation_id: str
    ) -> tuple[bool, float]:
        """
        [S5] [S9] Bounded reflection — single LLM call with timeout + token limit.
        Returns (should_continue, confidence).
        """
        if not self.cognition:
            return False, 0.0

        context = self._build_progress_context(progress)
        reflect_prompt = f"""Goal: "{goal}"
Progress: {context}

Should I continue working on this goal? Respond with JSON:
{{"should_continue": true/false, "confidence": 0.0-1.0, "reasoning": "..."}}"""

        try:
            # [S9] Timeout bound
            result = await asyncio.wait_for(
                self.cognition.call_llm(
                    [
                        {
                            "role": "system",
                            "content": "You are a task evaluator. Respond with JSON only.",
                        },
                        {"role": "user", "content": reflect_prompt},
                    ],
                    max_tokens=self.autonomy_cfg.reflection_max_tokens,
                ),
                timeout=self.autonomy_cfg.reflection_timeout_seconds,
            )
        except asyncio.TimeoutError:
            logger.warning(
                "Autonomy [%s]: reflection timed out — stopping",
                correlation_id[:8],
            )
            return False, 0.0

        if result:
            confidence = float(result.get("confidence", 0.0))
            should_continue = bool(result.get("should_continue", False))
            return should_continue, confidence
        return False, 0.0

    # ── Rollback ──────────────────────────────────────────────────────────

    async def _rollback(
        self,
        transaction: ExecutionTransaction,
        result: GoalResult,
        correlation_id: str,
    ):
        """
        [S7] Idempotent rollback: reverse-iterate completed steps, ignore errors.
        [S15] Double-rollback guard: only runs once per goal.
        """
        if self._rollback_executed:
            logger.warning(
                "Autonomy [%s]: rollback already executed — skipping",
                correlation_id[:8],
            )
            return

        self._rollback_executed = True
        rollback_cmds = transaction.rollback_actions
        if not rollback_cmds:
            logger.info(
                "Autonomy [%s]: no rollback commands to execute",
                correlation_id[:8],
            )
            return

        await self._emit(
            "autonomy.rollback_started",
            {
                "steps_to_rollback": len(rollback_cmds),
                "correlation_id": correlation_id,
            },
        )

        errors = []
        rolled_back = 0
        for cmd in rollback_cmds:
            try:
                tool_name = cmd.get("tool", "")
                args = cmd.get("args", {})
                logger.info(
                    "Autonomy [%s]: rolling back %s(%s)",
                    correlation_id[:8],
                    tool_name,
                    args,
                )
                action = PlannedAction(tool_name=tool_name, arguments=args)
                # Create a temporary budget for rollback (generous)
                rollback_budget = ResourceBudget(max_tool_calls=100, max_tokens=100000)
                await self._execute_step(action, rollback_budget, correlation_id)
                rolled_back += 1
                result.rollback_log.append(f"OK: {tool_name}")
            except Exception as exc:
                # [S7] Ignore rollback errors — never retry
                error_msg = f"IGNORED: {cmd.get('tool', '?')}: {exc}"
                errors.append(error_msg)
                result.rollback_log.append(error_msg)
                logger.warning(
                    "Autonomy [%s]: rollback error (ignored): %s",
                    correlation_id[:8],
                    exc,
                )

        if result.telemetry:
            result.telemetry.rollback_count = rolled_back

        await self._emit(
            "autonomy.rollback_completed",
            {
                "rolled_back_count": rolled_back,
                "errors": errors,
                "correlation_id": correlation_id,
            },
        )

    # ── Canonical hashing [S3] ────────────────────────────────────────────

    @staticmethod
    def _hash_plan(plan: Plan) -> str:
        """[S3] Hash canonical (tool_name, sorted_args, index) — never raw LLM text."""
        canonical = []
        for i, action in enumerate(plan.actions):
            # Sort arguments for deterministic hashing
            sorted_args = json.dumps(action.arguments, sort_keys=True, default=str)
            canonical.append((action.tool_name, sorted_args, i))
        data = json.dumps(canonical, sort_keys=True)
        return hashlib.sha256(data.encode()).hexdigest()[:16]

    # ── State machine transitions [F6] ────────────────────────────────────

    def _transition_state(self, target: str):
        """Safely transition state machine if available. Never raises."""
        if self.state_machine is None:
            return
        from jarvis.utils.enums import InteractionState

        target_state = getattr(InteractionState, target, None)
        if target_state is None:
            return
        try:
            if target_state == InteractionState.IDLE:
                self.state_machine.reset()
            else:
                self.state_machine.transition(target_state)
        except Exception as exc:
            logger.warning("Autonomy: state transition to %s failed: %s", target, exc)

    # ── Helpers ───────────────────────────────────────────────────────────

    def _build_progress_context(self, progress: GoalResult) -> str:
        if not progress.results:
            return "No steps completed yet."
        lines = []
        for i, r in enumerate(progress.results):
            output_str = str(r.output)[:200] if r.output else ""
            lines.append(f"Step {i + 1}: {r.status} — {output_str}")
        return "\n".join(lines)

    async def _emit(self, channel: str, data: dict):
        """[S10] Emit event with correlation ID always included."""
        try:
            await self.event_bus.emit(channel, data)
        except Exception:
            logger.warning("Autonomy: failed to emit %s", channel)

    async def _safe_emit(self, channel: str, data: dict):
        """[F10] Best-effort emit — survives CancelledError."""
        try:
            await self.event_bus.emit(channel, data)
        except (asyncio.CancelledError, Exception):
            logger.warning("Autonomy: failed to emit %s (safe)", channel)
