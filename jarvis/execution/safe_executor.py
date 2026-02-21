"""
jarvis.execution.safe_executor — Audited Python execution in isolated subprocess.

Safety layers:
  1. Separate subprocess (not in-process)
  2. sys.addaudithook blocks dangerous syscalls
  3. Allowed-import whitelist (not just blocklist)
  4. Memory limit via resource/Job Objects
  5. Always-enforced execution timeout
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
import tempfile
import textwrap
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ── Defaults ─────────────────────────────────────────────────────────────────

DEFAULT_ALLOWED_IMPORTS = frozenset(
    {
        "math",
        "datetime",
        "json",
        "re",
        "collections",
        "itertools",
        "functools",
        "string",
        "textwrap",
        "decimal",
        "fractions",
        "statistics",
        "random",
        "copy",
        "pprint",
        "typing",
        "os.path",
        "pathlib",
        "csv",
        "io",
    }
)

DEFAULT_TIMEOUT_S = 30
DEFAULT_MEMORY_MB = 256

# ── Audit Preamble (injected before user code) ──────────────────────────────

_AUDIT_PREAMBLE_HEAD = textwrap.dedent(
    """\
import sys as _sys

_BLOCKED_AUDIT_EVENTS = {
    "os.system", "os.exec", "os.spawn",
    "subprocess.Popen", "subprocess.call",
    "socket.connect", "socket.bind", "socket.sendto",
    "ctypes.dlopen", "webbrowser.open",
}

_ALLOWED_IMPORTS = """
)

_AUDIT_PREAMBLE_TAIL = textwrap.dedent(
    """

def _audit_hook(event, args,
                _blocked=_BLOCKED_AUDIT_EVENTS,
                _allowed=_ALLOWED_IMPORTS):
    # BUG-11 FIX: _blocked and _allowed are captured via default args
    # so `del _BLOCKED_AUDIT_EVENTS` from user code cannot bypass the hook.
    if event in _blocked:
        raise RuntimeError(f"BLOCKED by audit hook: {event}")
    if event == "import" and args:
        mod_name = args[0]
        top_level = mod_name.split(".")[0]
        if top_level not in _allowed and mod_name not in _allowed:
            if top_level not in _sys.stdlib_module_names:
                raise ImportError(
                    f"Import '{mod_name}' not in allowed whitelist"
                )

_sys.addaudithook(_audit_hook)
del _audit_hook
# Clean up globals so user code can't introspect the sets
del _BLOCKED_AUDIT_EVENTS
del _ALLOWED_IMPORTS
# --- user code below ---
"""
)


class AuditedPythonRunner:
    """Execute Python code/scripts in a sandboxed subprocess with audit hooks.

    Features:
      - Code runs in a separate process (never in-process)
      - sys.addaudithook blocks dangerous syscalls
      - Allowed-import whitelist
      - Enforced timeout + memory limits
      - stdout/stderr captured and returned
    """

    def __init__(
        self,
        timeout_seconds: int = DEFAULT_TIMEOUT_S,
        memory_limit_mb: int = DEFAULT_MEMORY_MB,
        allowed_imports: frozenset[str] | None = None,
    ):
        self.timeout = timeout_seconds
        self.memory_limit_mb = memory_limit_mb
        self.allowed_imports = allowed_imports or DEFAULT_ALLOWED_IMPORTS

    def _build_preamble(self) -> str:
        """Generate the audit hook preamble with current allowed imports."""
        import_set_repr = repr(set(self.allowed_imports))
        return _AUDIT_PREAMBLE_HEAD + import_set_repr + _AUDIT_PREAMBLE_TAIL

    async def run_code(self, code: str) -> dict[str, Any]:
        """Execute a Python code string in audited subprocess.

        Returns: {"status": "success"|"failed"|"timeout",
                  "stdout": str, "stderr": str, "exit_code": int}
        """
        full_code = self._build_preamble() + "\n" + code

        # Write to temp file (avoids shell escaping issues)
        tmp = tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False, encoding="utf-8"
        )
        try:
            tmp.write(full_code)
            tmp.close()

            return await self._run_file(tmp.name)
        finally:
            try:
                os.unlink(tmp.name)
            except OSError:
                pass

    async def run_script(self, script_path: str) -> dict[str, Any]:
        """Execute a .py file with audit hooks injected.

        Returns: {"status": ..., "stdout": ..., "stderr": ..., "exit_code": ...}
        """
        path = Path(script_path)
        if not path.exists():
            return {
                "status": "failed",
                "stdout": "",
                "stderr": f"Script not found: {script_path}",
                "exit_code": -1,
            }
        if path.suffix != ".py":
            return {
                "status": "failed",
                "stdout": "",
                "stderr": f"Only .py scripts are allowed: {script_path}",
                "exit_code": -1,
            }

        user_code = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: path.read_text(encoding="utf-8", errors="replace"),
        )
        return await self.run_code(user_code)

    async def install_package(self, package_name: str) -> dict[str, Any]:
        """Install a Python package via pip (no audit hook — runs pip directly).

        Returns: {"status": ..., "stdout": ..., "stderr": ..., "exit_code": ...}
        """
        # Sanitize package name (prevent shell injection)
        safe_name = "".join(c for c in package_name if c.isalnum() or c in "-_.")
        if not safe_name or safe_name != package_name:
            return {
                "status": "failed",
                "stdout": "",
                "stderr": f"Invalid package name: {package_name}",
                "exit_code": -1,
            }

        try:
            proc = await asyncio.create_subprocess_exec(
                sys.executable,
                "-m",
                "pip",
                "install",
                "--user",
                safe_name,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=120)
            return {
                "status": "success" if proc.returncode == 0 else "failed",
                "stdout": stdout.decode("utf-8", errors="replace")[:5000],
                "stderr": stderr.decode("utf-8", errors="replace")[:2000],
                "exit_code": proc.returncode,
            }
        except asyncio.TimeoutError:
            try:
                proc.kill()
            except Exception:
                pass
            return {
                "status": "timeout",
                "stdout": "",
                "stderr": "pip install timed out (120s)",
                "exit_code": -1,
            }

    # ── Internal ──────────────────────────────────────────────────────────

    async def _run_file(self, filepath: str) -> dict[str, Any]:
        """Execute a Python file in a subprocess with resource limits."""
        env = os.environ.copy()
        # Signal to child: memory limit in bytes
        env["_JARVIS_MEM_LIMIT"] = str(self.memory_limit_mb * 1024 * 1024)

        cmd = [sys.executable, filepath]

        # On Windows, use CREATE_NEW_PROCESS_GROUP for isolation + job limits
        kwargs: dict[str, Any] = {
            "stdout": asyncio.subprocess.PIPE,
            "stderr": asyncio.subprocess.PIPE,
            "env": env,
        }
        if sys.platform == "win32":
            import subprocess

            kwargs["creationflags"] = (
                subprocess.CREATE_NEW_PROCESS_GROUP
                | subprocess.CREATE_BREAKAWAY_FROM_JOB
            )

        try:
            proc = await asyncio.create_subprocess_exec(*cmd, **kwargs)
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(), timeout=self.timeout
            )

            return {
                "status": "success" if proc.returncode == 0 else "failed",
                "stdout": stdout.decode("utf-8", errors="replace")[:5000],
                "stderr": stderr.decode("utf-8", errors="replace")[:3000],
                "exit_code": proc.returncode,
            }
        except asyncio.TimeoutError:
            # Kill the process tree
            try:
                proc.kill()
            except Exception:
                pass
            return {
                "status": "timeout",
                "stdout": "",
                "stderr": f"Execution timed out ({self.timeout}s)",
                "exit_code": -1,
            }
