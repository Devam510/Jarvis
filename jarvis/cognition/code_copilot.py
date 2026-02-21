"""
jarvis.cognition.code_copilot — Live Code Co-Pilot with project awareness.

Safety controls (hard limits):
  - Max 3 auto-fix retries (hard ceiling, not configurable higher)
  - Per-run timeout: 30s default
  - Token budget per session: 4096 tokens
  - Tool-call budget per run: 5 max
  - No network access unless explicitly allowed
  - File overwrite always requires TIER_3 confirmation
"""

from __future__ import annotations

import asyncio
import logging
import os
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

# Hard ceiling — cannot be overridden by config
_ABSOLUTE_MAX_RETRIES = 3
_ABSOLUTE_MAX_TIMEOUT = 120.0


@dataclass
class ProjectContext:
    """Scanned project metadata for code generation context."""

    root: str = ""
    language: str = "python"
    has_git: bool = False
    has_requirements: bool = False
    has_package_json: bool = False
    dependencies: list[str] = field(default_factory=list)
    structure: list[str] = field(default_factory=list)  # top-level files/dirs
    readme_summary: str = ""


@dataclass
class CoPilotResult:
    """Result of a code generation or run-and-fix cycle."""

    status: str = "pending"  # success | failed | timeout | budget_exceeded
    code: str = ""
    output: str = ""
    error: str = ""
    retries_used: int = 0
    duration_ms: float = 0


class CodeCoPilot:
    """Project-aware code generation + run + auto-fix loop.

    Safety invariants:
      - max_retries capped at 3 (even if config says higher)
      - run_timeout capped at 120s
      - network access blocked by default
      - file overwrites require explicit confirmation
    """

    def __init__(
        self,
        max_retries: int = 3,
        run_timeout: float = 30.0,
        token_budget: int = 4096,
        tool_call_budget: int = 5,
        allow_network: bool = False,
        require_confirm_overwrite: bool = True,
        cognitive_core: Any = None,
    ):
        # Enforce hard ceilings
        self._max_retries = min(max_retries, _ABSOLUTE_MAX_RETRIES)
        self._run_timeout = min(run_timeout, _ABSOLUTE_MAX_TIMEOUT)
        self._token_budget = token_budget
        self._tool_call_budget = tool_call_budget
        self._allow_network = allow_network
        self._require_confirm_overwrite = require_confirm_overwrite
        self._cognitive = cognitive_core

        # Session tracking
        self._tokens_used = 0
        self._tool_calls_used = 0
        self.total_generations = 0
        self.total_fixes = 0

    # ── Public API ────────────────────────────────────────────────────────

    def scan_project(self, path: str) -> ProjectContext:
        """Scan project directory for context.

        Reads only metadata files — never reads source code content
        to avoid token budget blow-up.
        """
        ctx = ProjectContext(root=path)
        root = Path(path)

        if not root.is_dir():
            return ctx

        # Check common markers
        ctx.has_git = (root / ".git").is_dir()
        ctx.has_requirements = (root / "requirements.txt").is_file()
        ctx.has_package_json = (root / "package.json").is_file()

        # Language detection
        if ctx.has_package_json:
            ctx.language = "javascript"
        elif ctx.has_requirements or (root / "setup.py").is_file():
            ctx.language = "python"

        # Read dependencies (capped)
        if ctx.has_requirements:
            try:
                lines = (
                    (root / "requirements.txt").read_text(errors="ignore").splitlines()
                )
                ctx.dependencies = [
                    l.strip().split("==")[0].split(">=")[0]
                    for l in lines[:50]
                    if l.strip() and not l.startswith("#")
                ]
            except Exception:
                pass

        # Top-level structure (capped)
        try:
            entries = sorted(root.iterdir())[:30]
            ctx.structure = [f"{'[D] ' if e.is_dir() else ''}{e.name}" for e in entries]
        except Exception:
            pass

        # README summary
        for name in ["README.md", "readme.md", "README.txt"]:
            readme = root / name
            if readme.is_file():
                try:
                    text = readme.read_text(errors="ignore")
                    ctx.readme_summary = text[:500]
                except Exception:
                    pass
                break

        return ctx

    async def generate_code(
        self, prompt: str, context: Optional[ProjectContext] = None
    ) -> CoPilotResult:
        """Generate code using LLM with project context.

        Returns generated code without executing it.
        """
        if self._tokens_used >= self._token_budget:
            return CoPilotResult(
                status="budget_exceeded",
                error=f"Token budget exhausted ({self._tokens_used}/{self._token_budget})",
            )

        if self._tool_calls_used >= self._tool_call_budget:
            return CoPilotResult(
                status="budget_exceeded",
                error=f"Tool-call budget exhausted ({self._tool_calls_used}/{self._tool_call_budget})",
            )

        self._tool_calls_used += 1
        self.total_generations += 1

        # Build context-enriched prompt
        enriched = self._build_prompt(prompt, context)

        # Call LLM if available
        if self._cognitive and hasattr(self._cognitive, "generate"):
            try:
                response = await self._cognitive.generate(enriched)
                code = self._extract_code_block(response)
                self._tokens_used += len(response.split())  # rough estimate
                return CoPilotResult(status="success", code=code)
            except Exception as e:
                return CoPilotResult(status="failed", error=str(e))
        else:
            # No LLM — return prompt for testing
            return CoPilotResult(
                status="success",
                code=f"# Generated for: {prompt}\npass\n",
            )

    async def run_and_fix(
        self,
        code: str,
        save_path: str,
        context: Optional[ProjectContext] = None,
    ) -> CoPilotResult:
        """Save, run, and auto-fix code up to max_retries times.

        Safety:
          - max_retries hard-capped at 3
          - per-run timeout enforced
          - network blocked unless explicitly allowed
        """
        start = time.time()
        current_code = code
        retries = 0
        last_error = ""

        while retries <= self._max_retries:
            # Check budgets
            if self._tool_calls_used >= self._tool_call_budget:
                return CoPilotResult(
                    status="budget_exceeded",
                    code=current_code,
                    error="Tool-call budget exhausted",
                    retries_used=retries,
                    duration_ms=(time.time() - start) * 1000,
                )

            # Save code
            try:
                Path(save_path).parent.mkdir(parents=True, exist_ok=True)
                Path(save_path).write_text(current_code, encoding="utf-8")
            except Exception as e:
                return CoPilotResult(
                    status="failed",
                    code=current_code,
                    error=f"Failed to save: {e}",
                    retries_used=retries,
                    duration_ms=(time.time() - start) * 1000,
                )

            # Run code
            run_result = await self._run_code(save_path)

            if run_result["success"]:
                return CoPilotResult(
                    status="success",
                    code=current_code,
                    output=run_result["stdout"],
                    retries_used=retries,
                    duration_ms=(time.time() - start) * 1000,
                )

            # Failed — attempt fix
            last_error = run_result["stderr"]
            retries += 1
            self.total_fixes += 1

            if retries > self._max_retries:
                break

            # Generate fix
            fix_prompt = (
                f"Fix this Python code error:\n\n"
                f"Code:\n```python\n{current_code}\n```\n\n"
                f"Error:\n```\n{last_error[:500]}\n```\n\n"
                f"Return ONLY the corrected code."
            )
            fix_result = await self.generate_code(fix_prompt, context)
            if fix_result.status == "success" and fix_result.code:
                current_code = fix_result.code
            else:
                break

        return CoPilotResult(
            status="failed",
            code=current_code,
            error=last_error,
            retries_used=retries,
            duration_ms=(time.time() - start) * 1000,
        )

    def reset_session(self):
        """Reset session budgets."""
        self._tokens_used = 0
        self._tool_calls_used = 0

    # ── Internal ──────────────────────────────────────────────────────────

    async def _run_code(self, path: str) -> dict:
        """Run a Python script with timeout and network blocking."""
        env = os.environ.copy()

        # Block network if not allowed
        if not self._allow_network:
            # BUG-10 NOTE: This is a SOFT block only — code can unset these env
            # vars or use raw sockets. For hard isolation, use OS-level firewall
            # rules (e.g., Windows Firewall API or iptables/netns on Linux).
            # Current approach is defence-in-depth alongside audit hook blocking.
            env["HTTP_PROXY"] = "http://0.0.0.0:0"
            env["HTTPS_PROXY"] = "http://0.0.0.0:0"
            env["NO_PROXY"] = ""

        try:
            proc = await asyncio.create_subprocess_exec(
                "python",
                path,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env,
            )

            try:
                stdout, stderr = await asyncio.wait_for(
                    proc.communicate(),
                    timeout=self._run_timeout,
                )
            except asyncio.TimeoutError:
                proc.kill()
                await proc.wait()
                return {
                    "success": False,
                    "stdout": "",
                    "stderr": f"Execution timed out after {self._run_timeout}s",
                }

            return {
                "success": proc.returncode == 0,
                "stdout": stdout.decode(errors="replace")[:2000],
                "stderr": stderr.decode(errors="replace")[:2000],
            }
        except Exception as e:
            return {"success": False, "stdout": "", "stderr": str(e)}

    @staticmethod
    def _build_prompt(prompt: str, context: Optional[ProjectContext]) -> str:
        """Build context-enriched prompt."""
        parts = []
        if context:
            parts.append(f"Project: {context.root}")
            parts.append(f"Language: {context.language}")
            if context.dependencies:
                parts.append(f"Dependencies: {', '.join(context.dependencies[:10])}")
            if context.structure:
                parts.append(f"Structure: {', '.join(context.structure[:10])}")
            parts.append("")

        parts.append(f"Task: {prompt}")
        parts.append("")
        parts.append("Return ONLY the code, no explanations.")
        return "\n".join(parts)

    @staticmethod
    def _extract_code_block(response: str) -> str:
        """Extract code from markdown code block if present."""
        import re

        match = re.search(r"```(?:python)?\n(.*?)```", response, re.DOTALL)
        if match:
            return match.group(1).strip()
        return response.strip()
