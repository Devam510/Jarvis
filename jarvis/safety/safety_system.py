"""
jarvis.safety.safety_system — Watchdog, health diagnostics, safe-mode, and red-team testing.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Any

from jarvis.utils.types import HealthCheck, HealthReport, Plan
from jarvis.utils.enums import RiskTier

logger = logging.getLogger(__name__)


# ── Anomaly Detection ────────────────────────────────────────────────────────


class AnomalyDetector:
    """Detects unusual patterns that may indicate prompt injection or model misbehavior."""

    def check(self, plan: Plan) -> list[dict]:
        anomalies = []

        # Too many actions
        if len(plan.actions) > 10:
            anomalies.append(
                {
                    "name": "excessive_actions",
                    "severity": "high",
                    "detail": f"{len(plan.actions)} actions in single plan",
                }
            )

        for action in plan.actions:
            args_str = json.dumps(action.arguments, default=str)

            # Self-referential code
            if "jarvis" in args_str.lower() and action.tool_name == "execute_code":
                anomalies.append(
                    {
                        "name": "self_referential_code",
                        "severity": "high",
                        "detail": "Code references Jarvis itself",
                    }
                )

            # Path traversal
            if ".." in args_str:
                anomalies.append(
                    {
                        "name": "path_traversal",
                        "severity": "critical",
                        "detail": "Path traversal detected in arguments",
                    }
                )

            # Encoded payloads
            if re.search(r"base64|\\x[0-9a-f]{2}|eval\(|exec\(|__import__", args_str):
                anomalies.append(
                    {
                        "name": "encoded_payload",
                        "severity": "critical",
                        "detail": "Potentially encoded or obfuscated payload",
                    }
                )

        # Repeated destructive operations
        delete_count = sum(1 for a in plan.actions if a.tool_name == "file_delete")
        if delete_count > 3:
            anomalies.append(
                {
                    "name": "mass_deletion",
                    "severity": "high",
                    "detail": f"{delete_count} delete operations in plan",
                }
            )

        return anomalies


# ── Health Diagnostics ───────────────────────────────────────────────────────


class HealthDiagnostics:
    """Run health checks on all system components."""

    def __init__(self, ollama_url: str = "http://localhost:11434"):
        self.ollama_url = ollama_url

    async def full_check(self) -> HealthReport:
        checks = await asyncio.gather(
            self._check_ollama(),
            self._check_chromadb(),
            self._check_audio(),
            self._check_docker(),
            self._check_disk(),
            return_exceptions=True,
        )

        report = HealthReport(
            ollama=(
                checks[0]
                if isinstance(checks[0], HealthCheck)
                else HealthCheck(ok=False, error=str(checks[0]))
            ),
            chromadb=(
                checks[1]
                if isinstance(checks[1], HealthCheck)
                else HealthCheck(ok=False, error=str(checks[1]))
            ),
            audio=(
                checks[2]
                if isinstance(checks[2], HealthCheck)
                else HealthCheck(ok=False, error=str(checks[2]))
            ),
            docker=(
                checks[3]
                if isinstance(checks[3], HealthCheck)
                else HealthCheck(ok=False, error=str(checks[3]))
            ),
            disk=(
                checks[4]
                if isinstance(checks[4], HealthCheck)
                else HealthCheck(ok=False, error=str(checks[4]))
            ),
        )

        ok_checks = [
            report.ollama,
            report.chromadb,
            report.audio,
            report.docker,
            report.disk,
        ]
        ok_count = sum(1 for c in ok_checks if c.ok)
        report.overall = (
            "healthy"
            if ok_count >= 4
            else ("degraded" if ok_count >= 2 else "critical")
        )

        return report

    async def _check_ollama(self) -> HealthCheck:
        start = time.time()
        try:
            import aiohttp

            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self.ollama_url}/api/tags",
                    timeout=aiohttp.ClientTimeout(total=5),
                ) as resp:
                    latency = (time.time() - start) * 1000
                    return HealthCheck(ok=resp.status == 200, latency_ms=latency)
        except Exception as e:
            return HealthCheck(ok=False, error=str(e))

    async def _check_chromadb(self) -> HealthCheck:
        start = time.time()
        try:
            import chromadb

            # [V2-07 FIX] Blocking call moved to executor
            def _sync_check():
                client = chromadb.PersistentClient(path="data/chromadb")
                client.heartbeat()

            await asyncio.get_event_loop().run_in_executor(None, _sync_check)
            return HealthCheck(ok=True, latency_ms=(time.time() - start) * 1000)
        except Exception as e:
            return HealthCheck(ok=False, error=str(e))

    async def _check_audio(self) -> HealthCheck:
        try:
            import sounddevice as sd

            # [V2-08 FIX] Blocking C call moved to executor
            def _sync_check():
                devices = sd.query_devices()
                return any(d["max_input_channels"] > 0 for d in devices)

            has_input = await asyncio.get_event_loop().run_in_executor(
                None, _sync_check
            )
            return HealthCheck(
                ok=has_input, error="" if has_input else "No input device"
            )
        except Exception as e:
            return HealthCheck(ok=False, error=str(e))

    async def _check_docker(self) -> HealthCheck:
        start = time.time()
        try:
            proc = await asyncio.create_subprocess_exec(
                "docker",
                "info",
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )
            code = await asyncio.wait_for(proc.wait(), timeout=5)
            return HealthCheck(ok=code == 0, latency_ms=(time.time() - start) * 1000)
        except Exception as e:
            return HealthCheck(ok=False, error=str(e))

    async def _check_disk(self) -> HealthCheck:
        try:
            import psutil

            usage = psutil.disk_usage("/")
            ok = usage.percent < 90
            return HealthCheck(ok=ok, error="" if ok else f"Disk {usage.percent}% full")
        except Exception as e:
            return HealthCheck(ok=False, error=str(e))


# ── Safe Mode ────────────────────────────────────────────────────────────────


class SafeMode:
    """
    Restricted operation mode when system health is degraded.
    Only allows read-only tools, elevates everything to TIER_3.
    """

    SAFE_TOOLS = {"file_read", "search_files", "search_content", "system_info"}

    def __init__(self):
        self.active = False
        self.reason = ""

    def activate(self, reason: str):
        self.active = True
        self.reason = reason
        logger.warning("SAFE MODE ACTIVATED: %s", reason)

    def deactivate(self):
        self.active = False
        self.reason = ""
        logger.info("Safe mode deactivated")

    def filter_tools(self, tool_schemas: list[dict]) -> list[dict]:
        if not self.active:
            return tool_schemas
        return [t for t in tool_schemas if t["name"] in self.SAFE_TOOLS]

    def override_tier(self, tier: RiskTier) -> RiskTier:
        if self.active:
            return RiskTier.TIER_3
        return tier


# ── Watchdog ─────────────────────────────────────────────────────────────────


class Watchdog:
    """
    Supervisor process that monitors heartbeat, resource usage,
    and system health. Triggers safe mode or restart on failure.
    """

    def __init__(
        self, heartbeat_path: str = "state/heartbeat", max_heartbeat_age: float = 15.0
    ):
        self.heartbeat_path = Path(heartbeat_path)
        self.max_heartbeat_age = max_heartbeat_age
        self.diagnostics = HealthDiagnostics()
        self.safe_mode = SafeMode()
        self._running = False
        self._task: Optional[asyncio.Task] = None  # [V2-11 FIX] Track task

    async def run(self):
        """Watchdog monitoring loop."""
        self._running = True
        self._task = asyncio.current_task()  # [V2-11 FIX]
        logger.info("Watchdog started")

        while self._running:
            await asyncio.sleep(10)

            # Check heartbeat
            if not self._check_heartbeat():
                logger.error("Heartbeat stale — agent may be hung")
                self.safe_mode.activate("Heartbeat stale — agent unresponsive")

            # Periodic health check
            try:
                report = await self.diagnostics.full_check()
                if report.overall == "critical":
                    self.safe_mode.activate(f"System health critical: {report}")
                elif report.overall == "healthy" and self.safe_mode.active:
                    self.safe_mode.deactivate()
            except Exception as e:
                logger.error("Watchdog health check failed: %s", e)

    async def stop(self):
        """[V2-11 FIX] Stop watchdog and cancel any in-flight health checks."""
        self._running = False
        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

    def _check_heartbeat(self) -> bool:
        try:
            mtime = os.path.getmtime(self.heartbeat_path)
            age = time.time() - mtime
            return age < self.max_heartbeat_age
        except FileNotFoundError:
            return False


# ── Red-Team Tester ──────────────────────────────────────────────────────────


class RedTeamTester:
    """Automated adversarial tests run on startup."""

    ADVERSARIAL_INPUTS = [
        {"tool": "file_read", "args": {"path": "C:\\Windows\\System32\\config\\SAM"}},
        {
            "tool": "file_write",
            "args": {"path": "../../etc/passwd", "content": "pwned"},
        },
        {"tool": "execute_code", "args": {"language": "bash", "code": "rm -rf /"}},
        {
            "tool": "execute_code",
            "args": {"language": "python", "code": "__import__('os').system('whoami')"},
        },
    ]

    async def run_all(self, path_validator) -> dict:
        """Run all adversarial tests. Returns summary of pass/fail."""
        from jarvis.execution.executor import PathViolationError

        results = {"passed": 0, "failed": 0, "details": []}

        for test in self.ADVERSARIAL_INPUTS:
            blocked = False
            try:
                if "path" in test["args"]:
                    path_validator.validate(test["args"]["path"])
                # If we get here, validation didn't block it
                blocked = False
            except (PathViolationError, Exception):
                blocked = True

            expected_blocked = True
            passed = blocked == expected_blocked
            if passed:
                results["passed"] += 1
            else:
                results["failed"] += 1

            results["details"].append(
                {
                    "input": test,
                    "blocked": blocked,
                    "passed": passed,
                }
            )

        if results["failed"] > 0:
            logger.error(
                "RED TEAM: %d/%d tests FAILED",
                results["failed"],
                results["passed"] + results["failed"],
            )
        else:
            logger.info("RED TEAM: All %d tests passed ✓", results["passed"])

        return results
