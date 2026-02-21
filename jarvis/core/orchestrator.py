"""
jarvis.core.orchestrator — Main agent orchestrator that wires everything together.
"""

from __future__ import annotations

import asyncio
import logging
import signal
import time
from typing import Optional

from jarvis.core.event_bus import AsyncEventBus
from jarvis.core.state_machine import InteractionStateMachine, IllegalStateTransition
from jarvis.core.state_store import StateStore
from jarvis.utils.config import JarvisConfig
from jarvis.utils.enums import InteractionState

logger = logging.getLogger(__name__)


class AgentOrchestrator:
    """
    Top-level orchestrator — wires all subsystem components together
    and drives the main event loop.
    """

    def __init__(self, config: JarvisConfig):
        self.config = config
        self.event_bus = AsyncEventBus()
        self.state_machine = InteractionStateMachine()
        self.state_store = StateStore(
            event_bus=self.event_bus,
            persist_path=config.state_store.persist_path,
            auto_persist_seconds=config.state_store.auto_persist_seconds,
            max_versions=config.state_store.max_versions,
        )
        self._running = False
        self._components: dict[str, object] = {}
        self._tasks: list[asyncio.Task] = []
        # Command queue — accepts transcripts while processing
        self._command_queue: asyncio.Queue = asyncio.Queue(maxsize=10)
        self._is_processing = False
        # Tracks the action_id of the current pending TIER_3 confirmation
        self._pending_action_id: Optional[str] = None

    def register_component(self, name: str, component: object):
        self._components[name] = component
        logger.info("Registered component: %s", name)

    async def start(self):
        """Initialize all components and start the main loop."""
        self.config.ensure_dirs()
        self._running = True
        logger.info("=== Jarvis Agent Starting ===")

        # Import and initialize components lazily
        await self._init_observability()
        await self._init_memory()
        await self._init_audio()
        await self._init_stt()
        await self._init_tts()
        await self._init_cognition()
        await self._init_risk()
        await self._init_execution()
        await self._init_autonomy()
        await self._init_perception()  # Phase 4
        await self._init_watchdog()  # Phase 5
        await self._init_metrics()  # Phase 5
        await self._init_guardrails()  # Phase 6
        await self._init_audit_ledger()  # Phase 6

        # Wire event handlers
        self._wire_events()

        # Load persisted state and start auto-persist
        await self.state_store.load()
        await self.state_store.start_auto_persist()
        self.register_component("state_store", self.state_store)

        # Start heartbeat
        self._tasks.append(asyncio.create_task(self._heartbeat_loop()))
        # Start command queue worker
        self._tasks.append(asyncio.create_task(self._queue_worker()))
        # Start text input listener (fallback when mic doesn't work)
        self._tasks.append(asyncio.create_task(self._text_input_loop()))

        logger.info("=== Jarvis Agent Ready — Listening for wake word ===")
        logger.info("💡 TIP: You can also type commands directly in this terminal")

        # Keep alive until stopped
        try:
            while self._running:
                await asyncio.sleep(1)
        except asyncio.CancelledError:
            pass
        finally:
            await self.stop()

    async def stop(self):
        """Gracefully shut down all components."""
        if not self._running:
            return
        self._running = False

        # Cancel pending confirmation so RiskEngine unblocks
        risk = self._components.get("risk")
        if risk and getattr(risk, "_pending_confirm", None) is not None:
            ctx = risk._pending_confirm
            if not ctx.event.is_set():
                ctx.expired = True
                ctx.result = False
                ctx.event.set()
        self._pending_action_id = None

        # Stop audio subsystem (pushes sentinel into frame queue)
        audio = self._components.get("audio")
        if audio and hasattr(audio, "stop"):
            audio.stop()

        # Stop Phase 5: Watchdog + Metrics (before perception)
        watchdog = self._components.get("watchdog")
        if watchdog and hasattr(watchdog, "stop"):
            await watchdog.stop()

        metrics = self._components.get("metrics")
        if metrics and hasattr(metrics, "stop"):
            await metrics.stop()

        # Stop Phase 4: Perception layer
        watcher = self._components.get("screen_watcher")
        if watcher and hasattr(watcher, "stop"):
            watcher.stop()

        behavior = self._components.get("behavioral_memory")
        if behavior and hasattr(behavior, "stop"):
            await behavior.stop()

        # Stop Phase 2 components (before StateStore flush)
        monitor = self._components.get("system_monitor")
        if monitor and hasattr(monitor, "stop"):
            await monitor.stop()

        messaging = self._components.get("messaging")
        if messaging and hasattr(messaging, "close"):
            await messaging.close()

        # BUG-21 FIX: Explicit stop for components that weren't previously cleaned up
        # Stop ProcessGraph (has its own running _task)
        process_graph = self._components.get("process_graph")
        if process_graph and hasattr(process_graph, "stop"):
            await process_graph.stop()

        # Cancel any running autonomous goal
        autonomy = self._components.get("autonomy")
        if autonomy and hasattr(autonomy, "cancel_goal"):
            try:
                await autonomy.cancel_goal()
            except Exception:
                pass

        # Stop SelfHealer (has monitoring tasks)
        self_healer = self._components.get("self_healer")
        if self_healer and hasattr(self_healer, "stop"):
            try:
                await self_healer.stop()
            except Exception:
                pass

        # Close executor browser + cancel reminder tasks (BUG-07/08)
        execution = self._components.get("execution")
        if execution:
            if hasattr(execution, "close_browser"):
                await execution.close_browser()
            if hasattr(execution, "_reminder_tasks"):
                for task in list(execution._reminder_tasks):
                    task.cancel()
                if execution._reminder_tasks:
                    await asyncio.gather(
                        *execution._reminder_tasks, return_exceptions=True
                    )
                execution._reminder_tasks.clear()

        # Flush and stop StateStore before EventBus shutdown
        await self.state_store.stop()

        # Stop EventBus — prevents new events from firing during teardown
        self.event_bus.stop()

        # Cancel all background tasks and wait for them
        # BUG-22 FIX: copy to list to avoid iteration over mutating set
        tasks_snapshot = list(self._tasks)
        for task in tasks_snapshot:
            task.cancel()
        if tasks_snapshot:
            await asyncio.gather(*tasks_snapshot, return_exceptions=True)
        self._tasks.clear()

        logger.info("=== Jarvis Agent Stopped ===")

    def _wire_events(self):
        """Connect event bus channels to component handlers."""
        bus = self.event_bus

        # Audio → Wake → start listening
        if "audio" in self._components:
            bus.subscribe("audio.wake_detected", self._on_wake_detected)

        # Speech segment → STT
        if "stt" in self._components:
            stt = self._components["stt"]
            bus.subscribe("audio.speech_segment", stt.transcribe)

        # Transcript → Cognition (goes through our handler for state management)
        if "cognition" in self._components:
            bus.subscribe("stt.transcript", self._on_transcript)

        # Tool calls → Risk classifier
        if "risk" in self._components:
            risk = self._components["risk"]
            bus.subscribe("cognition.plan_ready", risk.classify_plan)

        # Authorized → Executor
        if "execution" in self._components:
            execution = self._components["execution"]
            bus.subscribe("risk.authorized", execution.execute_tool_call)

        # Confirmation request → TTS prompt + state change
        bus.subscribe("risk.require_confirmation", self._on_require_confirmation)

        # Results → Response + Memory
        bus.subscribe("execution.result", self._on_execution_result)
        bus.subscribe("cognition.direct_response", self._on_direct_response)

        # Empty transcript → reset to IDLE
        bus.subscribe("stt.empty_transcript", self._on_empty_transcript)

        # Phase 3: Autonomy events → logging + TTS feedback
        bus.subscribe("autonomy.goal_started", self._on_autonomy_event)
        bus.subscribe("autonomy.step_started", self._on_autonomy_event)
        bus.subscribe("autonomy.step_completed", self._on_autonomy_event)
        bus.subscribe("autonomy.step_failed", self._on_autonomy_event)
        bus.subscribe("autonomy.rollback_started", self._on_autonomy_event)
        bus.subscribe("autonomy.rollback_completed", self._on_autonomy_event)
        bus.subscribe("autonomy.goal_aborted", self._on_autonomy_event)
        bus.subscribe("autonomy.goal_completed", self._on_autonomy_event)  # F4/F8

        # Phase 4: Perception → Suggestion + Behavioral Memory
        if "suggestion_engine" in self._components:
            bus.subscribe(
                "perception.context_updated",
                self._components["suggestion_engine"].on_context_updated,
            )
        if "behavioral_memory" in self._components:
            bus.subscribe(
                "perception.context_updated",
                self._components["behavioral_memory"].on_context_update,
            )
        # Phase 4: Suggestion → TTS (S8: speak only, never auto-execute)
        bus.subscribe("suggestion.offered", self._on_suggestion)

        # Phase 5: Watchdog alerts → TTS + logging
        bus.subscribe("watchdog.loop_detected", self._on_watchdog_alert)
        bus.subscribe("watchdog.timeout", self._on_watchdog_alert)
        bus.subscribe("watchdog.stuck_reset", self._on_watchdog_alert)

        # Phase 6: Audit ledger — record every tool execution
        if "audit_ledger" in self._components:
            bus.subscribe("execution.step_result", self._on_audit_record)

        logger.info("Event wiring complete")

    async def _on_autonomy_event(self, event):
        """Log autonomy lifecycle events and optionally speak status."""
        correlation_id = event.get("correlation_id", "?")[:8]
        if "goal" in str(event.get("reason", "")):
            logger.info("Autonomy [%s]: %s", correlation_id, event)
        else:
            logger.debug("Autonomy [%s]: event %s", correlation_id, event)

    async def _on_wake_detected(self, event):
        """Handle wake word detection."""
        if self._is_processing:
            logger.debug("Wake word ignored — processing in progress")
            return

        current = self.state_machine.state
        if current != InteractionState.IDLE:
            logger.debug("Wake word ignored (state=%s)", current.name)
            return

        try:
            self.state_machine.transition(InteractionState.LISTENING)
            logger.info("🎙️ Wake word detected — listening for speech")
        except IllegalStateTransition:
            pass

    async def _on_require_confirmation(self, event):
        """Handle TIER_3 confirmation request from RiskEngine."""
        action_id = event.get("action_id", "")
        description = event.get("description", "perform an action")
        self._pending_action_id = action_id

        prompt = (
            f"I'm about to {description}. This is a critical operation. Do you confirm?"
        )
        logger.info("⚠️  Confirmation required [%s]: %s", action_id[:8], description)

        if "tts" in self._components:
            await self._components["tts"].speak(prompt)

    async def _on_empty_transcript(self, event):
        """Reset state when STT produces empty/garbage output."""
        if self.state_machine.state in (
            InteractionState.LISTENING,
            InteractionState.TRANSCRIBING,
        ):
            self.state_machine.reset()

    async def _on_transcript(self, transcript):
        """Handle transcript events — filter for 'jarvis' keyword, then queue."""
        # [S8] Block all input while autonomy owns the state machine
        current = self.state_machine.state
        if current in (
            InteractionState.AUTONOMOUS_PLANNING,
            InteractionState.AUTONOMOUS_EXECUTING,
        ):
            logger.info(
                "Dropping transcript — autonomy active (state=%s)",
                current.name,
            )
            return

        # Only drop if queue already has a pending command (prevent buildup)
        if not self._command_queue.empty():
            logger.info(
                "Dropping audio transcript — queue not empty (queue=%d)",
                self._command_queue.qsize(),
            )
            return

        # STT-based keyword detection: check for "jarvis" and common mishearings
        text = transcript.text.lower().strip()

        # Fuzzy match: Whisper often mishears "jarvis" as these variants
        KEYWORD_VARIANTS = [
            "jarvis",
            "jarves",
            "jarvish",
            "jarvas",
            "jarvus",
            "arvish",
            "arvis",
            "jervis",
            "jarv ",
            "javis",
            "gervis",
            "jarovis",
            "darvish",
            "darvis",
            "jarwis",
            "jaris",
            "hey jarvis",
            "jarvis,",
            "jarvis.",
            "jarvis!",
        ]

        matched_keyword = None
        match_idx = -1
        for variant in KEYWORD_VARIANTS:
            idx = text.find(variant)
            if idx != -1:
                matched_keyword = variant
                match_idx = idx
                break

        if matched_keyword is None:
            logger.info("Discarding non-jarvis transcript: '%s'", transcript.text)
            return

        # Extract command after the matched keyword
        # e.g., "hey jarvis open chrome" → "open chrome"
        command_text = transcript.text[match_idx + len(matched_keyword) :].strip()
        # Clean leading punctuation/spaces
        command_text = command_text.lstrip(",.!? ")
        if not command_text:
            logger.info("Wake word '%s' heard but no command followed", matched_keyword)
            return

        # Create a new transcript with just the command
        from jarvis.utils.types import TranscriptEvent

        command_transcript = TranscriptEvent(
            text=command_text,
            confidence=transcript.confidence,
            language=transcript.language,
        )
        logger.info("🎯 Jarvis command detected: '%s'", command_text)

        try:
            self._command_queue.put_nowait(command_transcript)
            logger.info("Command queued (queue size: %d)", self._command_queue.qsize())
        except asyncio.QueueFull:
            logger.warning("Command queue full — dropping transcript")

    async def _queue_worker(self):
        """Background worker that processes commands from the queue sequentially."""
        while self._running:
            try:
                transcript = await self._command_queue.get()
                self._is_processing = True
                logger.info("Queue worker processing: '%s'", transcript.text)
                try:
                    # ── Confirmation routing ──────────────────────────
                    risk = self._components.get("risk")
                    if (
                        risk is not None
                        and getattr(risk, "_pending_confirm", None) is not None
                        and self._pending_action_id is not None
                    ):
                        from jarvis.risk.risk_engine import classify_confirmation
                        from jarvis.utils.enums import ConfirmStatus

                        status = classify_confirmation(transcript.text)
                        if status == ConfirmStatus.CONFIRMED:
                            risk.receive_confirmation(self._pending_action_id, True)
                            self._pending_action_id = None
                            logger.info(
                                "User confirmed via voice/text: '%s'", transcript.text
                            )
                        elif status == ConfirmStatus.DENIED:
                            risk.receive_confirmation(self._pending_action_id, False)
                            self._pending_action_id = None
                            logger.info(
                                "User denied via voice/text: '%s'", transcript.text
                            )
                        else:
                            # Ambiguous — keep _pending_action_id so user can retry
                            logger.info(
                                "Ambiguous confirmation input: '%s'", transcript.text
                            )
                            if "tts" in self._components:
                                await self._components["tts"].speak(
                                    "I didn't catch that. Please say yes or no."
                                )
                        # Skip normal processing — fall through to finally
                        continue

                    # ── Normal command processing ────────────────────
                    self.state_machine.reset()
                    cognition = self._components.get("cognition")
                    if cognition:
                        # Timeout: don't let a single command block forever
                        await asyncio.wait_for(
                            cognition.process_transcript(transcript),
                            timeout=45,
                        )
                except asyncio.TimeoutError:
                    logger.error("Command timed out (45s): '%s'", transcript.text)
                except Exception as e:
                    logger.exception("Error processing transcript: %s", e)
                finally:
                    self._is_processing = False
                    self.state_machine.reset()
                    self._command_queue.task_done()
                    logger.info(
                        "Queue worker ready (remaining=%d)",
                        self._command_queue.qsize(),
                    )
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Queue worker error: %s", e)
                self._is_processing = False

    async def _on_execution_result(self, event):
        """Handle tool execution results — speak the response."""
        try:
            if "tts" in self._components:
                tts = self._components["tts"]
                response_text = event.get("response", "Done.")
                await tts.speak(response_text)

            if "memory" in self._components:
                memory = self._components["memory"]
                try:
                    await memory.store_interaction(event)
                except Exception:
                    pass
        except Exception as e:
            logger.exception("Error in execution result handling: %s", e)

    async def _on_direct_response(self, event):
        """Handle direct LLM responses (no tool calls)."""
        try:
            if "tts" in self._components:
                tts = self._components["tts"]
                text = event.get("text", "")
                if text:
                    await tts.speak(text)
        except Exception as e:
            logger.exception("Error in direct response: %s", e)

    async def _heartbeat_loop(self):
        """Write heartbeat file for watchdog monitoring."""
        import os

        heartbeat_path = "state/heartbeat"
        os.makedirs(os.path.dirname(heartbeat_path), exist_ok=True)

        def _write_heartbeat():
            with open(heartbeat_path, "w") as f:
                f.write(str(time.time()))

        while self._running:
            try:
                await asyncio.get_event_loop().run_in_executor(None, _write_heartbeat)
            except OSError:
                pass
            await asyncio.sleep(5)

    async def _text_input_loop(self):
        """Read typed commands from stdin as a fallback for voice."""
        loop = asyncio.get_event_loop()

        def _read_line():
            try:
                return input("\n🎤 Jarvis> ").strip()
            except (EOFError, KeyboardInterrupt):
                return None

        while self._running:
            try:
                text = await loop.run_in_executor(None, _read_line)
                if text is None:
                    continue
                if text.lower() in ("exit", "quit", "stop"):
                    logger.info("Exit command received")
                    self._running = False
                    break
                if not text:
                    continue

                logger.info("Text input: '%s'", text)

                # Flush any pending audio transcripts from the queue
                drained = 0
                while not self._command_queue.empty():
                    try:
                        self._command_queue.get_nowait()
                        self._command_queue.task_done()
                        drained += 1
                    except asyncio.QueueEmpty:
                        break
                if drained:
                    logger.info("Drained %d stale audio commands from queue", drained)

                # Create a transcript event and push through the pipeline
                from jarvis.utils.types import TranscriptEvent

                transcript = TranscriptEvent(
                    text=text,
                    confidence=1.0,
                    language="en",
                )
                # Queue directly (bypasses _on_transcript's audio-busy guard)
                try:
                    self._command_queue.put_nowait(transcript)
                    logger.info(
                        "Text command queued (queue size: %d)",
                        self._command_queue.qsize(),
                    )
                except asyncio.QueueFull:
                    logger.warning("Command queue full — dropping text input")

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Text input error: %s", e)

    # ── Phase 4: Suggestion handler ──────────────────────────────────────

    async def _on_suggestion(self, event):
        """
        Handle proactive suggestions — TTS only.

        [S8] This handler ONLY speaks the suggestion via TTS.
        It NEVER creates tool calls, autonomy goals, or auto-executes anything.
        """
        message = event.get("message", "")
        if message and "tts" in self._components:
            logger.info("Speaking suggestion: %s", message[:80])
            await self._components["tts"].speak(message)

    # ── Phase 5: Watchdog alert handler ───────────────────────────────────

    async def _on_watchdog_alert(self, event):
        """
        Handle watchdog events — log critical + TTS announce.
        """
        message = event.get("message", "Watchdog alert")
        logger.critical("WATCHDOG ALERT: %s", message)

        # Record metric
        metrics = self._components.get("metrics")
        if metrics:
            metrics.counter("watchdog_alerts")

        # Speak the alert
        if "tts" in self._components:
            await self._components["tts"].speak(f"Warning: {message}")

    # ── Lazy component initialization ────────────────────────────────────

    async def _init_observability(self):
        from jarvis.observability.logger import setup_logging

        setup_logging(self.config.observability)
        logger.info("Observability initialized")

    async def _init_memory(self):
        try:
            from jarvis.memory.memory_manager import MemoryManager

            mgr = MemoryManager(self.config.memory, self.event_bus)
            await mgr.initialize()
            self.register_component("memory", mgr)
        except Exception as e:
            logger.warning("Memory init failed (non-fatal): %s", e)

    async def _init_audio(self):
        try:
            from jarvis.audio.audio_system import AudioSystem

            audio = AudioSystem(self.config.audio, self.event_bus)
            self.register_component("audio", audio)
            self._tasks.append(asyncio.create_task(audio.run()))
        except Exception as e:
            logger.error("Audio init failed: %s", e)

    async def _init_stt(self):
        try:
            from jarvis.stt.stt_engine import STTEngine

            stt = STTEngine(self.config.stt, self.event_bus)
            self.register_component("stt", stt)
        except Exception as e:
            logger.warning("STT init failed: %s", e)

    async def _init_tts(self):
        try:
            from jarvis.tts.tts_engine import TTSEngine

            tts = TTSEngine(self.config.tts)
            self.register_component("tts", tts)
        except Exception as e:
            logger.warning("TTS init failed: %s", e)

    async def _init_cognition(self):
        try:
            from jarvis.cognition.cognitive_core import CognitiveCore

            mem = self._components.get("memory")
            cog = CognitiveCore(self.config.llm, self.event_bus, memory=mem)
            self.register_component("cognition", cog)
        except Exception as e:
            logger.warning("Cognition init failed: %s", e)

    async def _init_risk(self):
        try:
            from jarvis.risk.risk_engine import RiskEngine

            tts = self._components.get("tts")
            stt = self._components.get("stt")
            risk = RiskEngine(self.config.safety, self.event_bus, tts=tts, stt=stt)
            self.register_component("risk", risk)
        except Exception as e:
            logger.warning("Risk engine init failed: %s", e)

    async def _init_execution(self):
        try:
            from jarvis.execution.executor import ExecutionDispatcher

            ex = ExecutionDispatcher(self.config.execution, self.event_bus)

            # ── Phase 2: Messaging automation ─────────────────────────
            try:
                from jarvis.execution.messaging import MessagingTool

                messaging = MessagingTool(
                    browser_user_data_dir=self.config.execution.browser_user_data_dir,
                )
                ex._messaging = messaging
                self.register_component("messaging", messaging)
                logger.info("MessagingTool initialized")
            except Exception as e:
                logger.warning("MessagingTool init skipped: %s", e)

            # ── Phase 2: Safe code execution ──────────────────────────
            try:
                from jarvis.execution.safe_executor import AuditedPythonRunner

                safe_exec = AuditedPythonRunner(
                    timeout_seconds=self.config.execution.sandbox_timeout_seconds,
                    memory_limit_mb=int(
                        self.config.execution.sandbox_memory_limit.rstrip("m")
                    ),
                )
                ex._safe_executor = safe_exec
                self.register_component("safe_executor", safe_exec)
                logger.info("AuditedPythonRunner initialized")
            except Exception as e:
                logger.warning("AuditedPythonRunner init skipped: %s", e)

            # ── Phase 2: System monitor ───────────────────────────────
            try:
                from jarvis.execution.system_monitor import SystemMonitor

                monitor = SystemMonitor(
                    state_store=self.state_store,
                    interval=10.0,
                )
                ex._system_monitor = monitor
                await monitor.start()
                self.register_component("system_monitor", monitor)
                logger.info("SystemMonitor started")
            except Exception as e:
                logger.warning("SystemMonitor init skipped: %s", e)

            self.register_component("execution", ex)
        except Exception as e:
            logger.warning("Execution init failed: %s", e)

    async def _init_autonomy(self):
        try:
            from jarvis.autonomy.autonomy_engine import AutonomyEngine

            ae = AutonomyEngine(
                self.config,
                self.event_bus,
                cognition=self._components.get("cognition"),
                execution=self._components.get("execution"),
                memory=self._components.get("memory"),
                state_machine=self.state_machine,  # F9: pass state machine
            )
            self.register_component("autonomy", ae)
        except Exception as e:
            logger.warning("Autonomy init failed: %s", e)

    async def _init_perception(self):
        """Phase 4: Initialize perception layer (ScreenWatcher + Suggestions + Behavior)."""
        if not self.config.perception.enabled:
            logger.info("Perception layer disabled (perception.enabled=False)")
            return

        try:
            from jarvis.perception.screen_watcher import ScreenWatcher

            watcher = ScreenWatcher(self.config.perception, self.event_bus)
            self.register_component("screen_watcher", watcher)
            self._tasks.append(asyncio.create_task(watcher.run()))
        except Exception as e:
            logger.warning("ScreenWatcher init failed (non-fatal): %s", e)

        try:
            from jarvis.perception.suggestion_engine import SuggestionEngine

            if self.config.proactive.enabled:
                suggestion = SuggestionEngine(
                    self.config.proactive,
                    self.event_bus,
                    state_machine=self.state_machine,
                )
                self.register_component("suggestion_engine", suggestion)
            else:
                logger.info("Suggestion engine disabled")
        except Exception as e:
            logger.warning("SuggestionEngine init failed (non-fatal): %s", e)

        try:
            from jarvis.perception.behavioral_memory import BehavioralMemory

            if self.config.behavioral_memory.enabled:
                behavior = BehavioralMemory(self.config.behavioral_memory)
                await behavior.initialize()
                self.register_component("behavioral_memory", behavior)
            else:
                logger.info("Behavioral memory disabled")
        except Exception as e:
            logger.warning("BehavioralMemory init failed (non-fatal): %s", e)

    async def _init_watchdog(self):
        """Phase 5: Initialize deadlock/loop/timeout watchdog."""
        if not self.config.watchdog.enabled:
            logger.info("Watchdog disabled by config")
            return

        try:
            from jarvis.core.watchdog import Watchdog

            wd = Watchdog(
                config=self.config.watchdog,
                event_bus=self.event_bus,
                state_machine=self.state_machine,
            )
            await wd.start()
            self.register_component("watchdog", wd)
        except Exception as e:
            logger.warning("Watchdog init failed (non-fatal): %s", e)

    async def _init_metrics(self):
        """Phase 5: Initialize structured metrics registry."""
        if not self.config.metrics.enabled:
            logger.info("Metrics disabled by config")
            return

        try:
            from jarvis.observability.metrics import MetricsRegistry

            reg = MetricsRegistry(
                output_path=self.config.metrics.output_path,
                flush_interval_seconds=self.config.metrics.flush_interval_seconds,
                max_histogram_size=self.config.metrics.max_histogram_size,
            )
            await reg.start_flush_loop()
            self.register_component("metrics", reg)
        except Exception as e:
            logger.warning("Metrics init failed (non-fatal): %s", e)

    async def _init_guardrails(self):
        """Phase 6: Initialize deterministic guardrail engine."""
        if not self.config.guardrail.enabled:
            logger.info("Guardrails disabled by config")
            return

        try:
            from jarvis.safety.guardrails import GuardrailEngine, GuardrailConfig

            gc = GuardrailConfig(
                enabled=True,
                max_file_deletes_per_min=self.config.guardrail.max_file_deletes_per_min,
                max_file_ops_per_min=self.config.guardrail.max_file_ops_per_min,
                max_code_exec_per_min=self.config.guardrail.max_code_exec_per_min,
                domain_whitelist=self.config.guardrail.domain_whitelist,
            )
            engine = GuardrailEngine(gc)
            # Inject into execution dispatcher
            execution = self._components.get("execution")
            if execution:
                execution._guardrail = engine
            self.register_component("guardrail", engine)
        except Exception as e:
            logger.warning("Guardrail init failed (non-fatal): %s", e)

    async def _init_audit_ledger(self):
        """Phase 6: Initialize cryptographic audit ledger."""
        if not self.config.audit.enabled:
            logger.info("Audit ledger disabled by config")
            return

        try:
            from jarvis.observability.audit_ledger import AuditLedger

            ledger = AuditLedger(
                ledger_path=self.config.audit.ledger_path,
                enabled=True,
            )
            self.register_component("audit_ledger", ledger)
        except Exception as e:
            logger.warning("Audit ledger init failed (non-fatal): %s", e)

    async def _on_audit_record(self, event: dict):
        """Phase 6: Record tool execution in the audit ledger."""
        ledger = self._components.get("audit_ledger")
        if not ledger:
            return
        try:
            action_name = event.get("action", "unknown")
            result = event.get("result")
            result_summary = ""
            risk_tier = "TIER_1"
            if result:
                result_summary = getattr(result, "status", str(result))
            # [V2-01] ledger.record() is now async (non-blocking I/O)
            await ledger.record(
                action="tool_executed",
                tool=action_name,
                args={"correlation_id": event.get("correlation_id", "")},
                result_summary=result_summary,
                risk_tier=risk_tier,
            )
        except Exception as e:
            logger.debug("Audit record failed: %s", e)
