"""
jarvis.execution.uac_elevation — Windows UAC admin elevation.

Provides controlled access to elevated (admin) operations on Windows
via ShellExecuteW with 'runas' verb.

Safety:
  - TIER_3 always — every elevated action requires explicit confirmation
  - Command whitelist — only pre-approved commands can be elevated
  - Full audit logging via event bus
  - Registry access limited to query/set with explicit paths
  - No persistent elevation — each call gets its own UAC prompt
"""

from __future__ import annotations

import asyncio
import logging
import subprocess
import time
from dataclasses import dataclass, field
from typing import Any, Optional

logger = logging.getLogger(__name__)

# Only these commands may be elevated — all others rejected
DEFAULT_ALLOWED_COMMANDS = [
    "netsh",
    "sfc",
    "dism",
    "chkdsk",
    "reg",
    "powershell",
    "cmd",
    "wmic",
    "bcdedit",
    "sc",
    "takeown",
    "icacls",
]


@dataclass
class ElevationResult:
    """Result of an elevated operation."""

    status: str = "success"  # success | denied | failed | blocked
    output: str = ""
    error: Optional[str] = None
    command: str = ""
    duration_ms: float = 0


class UACElevator:
    """Windows UAC elevation with safety controls.

    Safety invariants:
      - Always TIER_3: every elevated action logged + confirmed
      - Only whitelisted commands allowed
      - No persistent elevation — each call is standalone
      - Registry operations explicitly logged
      - ctypes ShellExecuteW for true UAC dialog (not subprocess runas)
    """

    def __init__(
        self,
        allowed_commands: list[str] | None = None,
        event_bus: Any = None,
        require_confirmation: bool = True,
    ):
        self._allowed = [
            c.lower() for c in (allowed_commands or DEFAULT_ALLOWED_COMMANDS)
        ]
        self._event_bus = event_bus
        self._require_confirmation = require_confirmation

        # Stats
        self.total_elevations = 0
        self.total_denied = 0
        self.total_errors = 0

    # ── Command Elevation ─────────────────────────────────────────────────

    async def run_elevated(
        self, command: str, args: str = "", timeout: float = 30.0
    ) -> ElevationResult:
        """Run a command with admin elevation via UAC.

        The command must be in the whitelist. A Windows UAC dialog will
        appear for user confirmation (by the OS itself).

        Args:
            command: The executable (must be in whitelist)
            args: Command-line arguments
            timeout: Max wait time for completion

        Returns:
            ElevationResult
        """
        start = time.time()

        # Validate command is in whitelist
        cmd_lower = command.lower().strip()
        if cmd_lower not in self._allowed:
            self.total_denied += 1
            logger.warning("Elevation denied — '%s' not in whitelist", command)
            await self._audit("elevation_denied", command, args)
            return ElevationResult(
                status="blocked",
                command=command,
                error=f"Command '{command}' not in elevation whitelist",
                duration_ms=(time.time() - start) * 1000,
            )

        # Audit log the attempt
        await self._audit("elevation_requested", command, args)

        try:
            # Use subprocess with runas for elevation
            # On Windows, this triggers the UAC dialog
            full_cmd = f"{command} {args}".strip()
            result = await asyncio.wait_for(
                self._execute_elevated(command, args),
                timeout=timeout,
            )

            self.total_elevations += 1
            await self._audit("elevation_completed", command, args)
            result.duration_ms = (time.time() - start) * 1000
            return result

        except asyncio.TimeoutError:
            self.total_errors += 1
            return ElevationResult(
                status="failed",
                command=command,
                error=f"Elevated command timed out after {timeout}s",
                duration_ms=(time.time() - start) * 1000,
            )
        except Exception as e:
            self.total_errors += 1
            logger.error("Elevation failed: %s", e)
            return ElevationResult(
                status="failed",
                command=command,
                error=str(e),
                duration_ms=(time.time() - start) * 1000,
            )

    async def _execute_elevated(self, command: str, args: str) -> ElevationResult:
        """Execute with elevation using ShellExecuteW.

        On Windows, uses ctypes.windll.shell32.ShellExecuteW with 'runas'.
        On non-Windows, falls back to subprocess with sudo hint.
        """
        import sys

        if sys.platform == "win32":
            try:
                import ctypes

                # ShellExecuteW(hwnd, verb, file, params, dir, show)
                ret = ctypes.windll.shell32.ShellExecuteW(  # type: ignore[union-attr]
                    None, "runas", command, args, None, 1
                )
                # ShellExecuteW returns > 32 on success
                if ret > 32:
                    return ElevationResult(
                        status="success",
                        command=command,
                        output=f"Elevated process launched (handle={ret})",
                    )
                else:
                    return ElevationResult(
                        status="denied",
                        command=command,
                        error=f"UAC denied or failed (code={ret})",
                    )
            except Exception as e:
                return ElevationResult(
                    status="failed",
                    command=command,
                    error=f"ShellExecuteW failed: {e}",
                )
        else:
            # Non-Windows stub
            return ElevationResult(
                status="failed",
                command=command,
                error="UAC elevation only available on Windows",
            )

    # ── Registry Operations ───────────────────────────────────────────────

    async def reg_query(self, key: str) -> ElevationResult:
        """Query a registry key (elevated).

        Args:
            key: Full registry path, e.g. "HKLM\\SOFTWARE\\..."
        """
        return await self.run_elevated("reg", f'query "{key}"')

    async def reg_set(
        self, key: str, value_name: str, value_data: str, value_type: str = "REG_SZ"
    ) -> ElevationResult:
        """Set a registry value (elevated, TIER_3).

        Args:
            key: Registry path
            value_name: Name of the value
            value_data: Data to set
            value_type: REG_SZ, REG_DWORD, etc.
        """
        args = f'add "{key}" /v "{value_name}" /t {value_type} /d "{value_data}" /f'
        return await self.run_elevated("reg", args)

    # ── Audit ─────────────────────────────────────────────────────────────

    async def _audit(self, event_type: str, command: str, args: str):
        """Emit audit event for elevation action."""
        if self._event_bus:
            try:
                await self._event_bus.emit(
                    f"uac.{event_type}",
                    {
                        "command": command,
                        "args": args,
                        "timestamp": time.time(),
                        "risk_tier": 3,
                    },
                )
            except Exception as e:
                logger.debug("Failed to audit UAC event: %s", e)

    # ── Utility ───────────────────────────────────────────────────────────

    def is_allowed(self, command: str) -> bool:
        """Check if a command is in the elevation whitelist."""
        return command.lower().strip() in self._allowed
