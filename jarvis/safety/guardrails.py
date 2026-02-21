"""
jarvis.safety.guardrails — Deterministic guardrails (§29).

Hard rules that override all AI logic. Checked BEFORE the execution
dispatcher runs any tool. Uses sliding-window rate counters (no external deps).
"""

from __future__ import annotations

import logging
import re
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


class GuardrailViolation(Exception):
    """Raised when a deterministic guardrail blocks an action."""

    def __init__(self, rule: str, detail: str = ""):
        self.rule = rule
        self.detail = detail
        super().__init__(f"Guardrail violation [{rule}]: {detail}")


# ── Protected path patterns ──────────────────────────────────────────────

_PROTECTED_PATHS = [
    re.compile(p, re.IGNORECASE)
    for p in [
        r"C:\\Windows\\System32\\config",  # SAM / SECURITY hives
        r"\\boot\.ini$",
        r"\\bootmgr$",
        r"\\ntldr$",
        r"C:\\Windows\\System32\\drivers\\etc\\hosts",
        r"C:\\Recovery",
        r"\\\.ssh[\\/]",  # SSH keys
        r"\\\.gnupg[\\/]",  # GPG keys
        r"HKLM\\",  # Registry hive paths
        r"/etc/(passwd|shadow|sudoers)",  # Linux system files
        r"/boot/",
    ]
]

_EXFIL_DOMAIN_PATTERN = re.compile(r"https?://([^/]+)", re.IGNORECASE)


@dataclass
class GuardrailConfig:
    """Configuration for deterministic guardrails."""

    enabled: bool = True
    max_file_deletes_per_min: int = 50
    max_file_ops_per_min: int = 200
    max_code_exec_per_min: int = 20
    domain_whitelist: list = field(
        default_factory=lambda: [
            "google.com",
            "github.com",
            "stackoverflow.com",
            "wikipedia.org",
            "python.org",
            "pypi.org",
            "npmjs.com",
            "docs.microsoft.com",
            "learn.microsoft.com",
        ]
    )
    extra_protected_paths: list = field(default_factory=list)


class _SlidingCounter:
    """Sliding window counter — O(1) amortised."""

    __slots__ = ("_window_secs", "_events")

    def __init__(self, window_secs: float = 60.0):
        self._window_secs = window_secs
        self._events: deque[float] = deque()

    def record(self):
        now = time.time()
        self._events.append(now)
        self._prune(now)

    def count(self) -> int:
        self._prune(time.time())
        return len(self._events)

    def _prune(self, now: float):
        cutoff = now - self._window_secs
        while self._events and self._events[0] < cutoff:
            self._events.popleft()


class GuardrailEngine:
    """Pre-execution hard-rule checker.

    Returns None if OK, raises GuardrailViolation if blocked.
    """

    def __init__(self, config: Optional[GuardrailConfig] = None):
        self._config = config or GuardrailConfig()

        # Rate counters (60-second sliding window)
        self._delete_counter = _SlidingCounter(60.0)
        self._file_op_counter = _SlidingCounter(60.0)
        self._code_exec_counter = _SlidingCounter(60.0)

        # Compile extra protected paths
        self._extra_protected = [
            re.compile(p, re.IGNORECASE) for p in self._config.extra_protected_paths
        ]

        self._total_violations = 0

    # ── Public API ────────────────────────────────────────────────────────

    def check(self, tool_name: str, args: dict) -> None:
        """Check all guardrails before execution. Raises GuardrailViolation."""
        if not self._config.enabled:
            return

        self._check_protected_paths(tool_name, args)
        self._check_rate_limits(tool_name)
        self._check_domain_whitelist(tool_name, args)

    @property
    def total_violations(self) -> int:
        return self._total_violations

    # ── Protected Path Check ──────────────────────────────────────────────

    def _check_protected_paths(self, tool_name: str, args: dict):
        """Block ops on protected system paths."""
        path_keys = ["path", "source", "destination", "src", "dst", "filepath"]
        paths = [str(args.get(k, "")) for k in path_keys if args.get(k)]

        for p in paths:
            for pattern in _PROTECTED_PATHS + self._extra_protected:
                if pattern.search(p):
                    self._total_violations += 1
                    raise GuardrailViolation(
                        "PROTECTED_PATH",
                        f"Tool '{tool_name}' targets protected path: {p}",
                    )

    # ── Rate Limit Check ──────────────────────────────────────────────────

    def _check_rate_limits(self, tool_name: str):
        """Enforce per-minute rate caps."""

        # File deletes — BUG-13 FIX: check BEFORE record
        if tool_name == "file_delete":
            if self._delete_counter.count() >= self._config.max_file_deletes_per_min:
                self._total_violations += 1
                raise GuardrailViolation(
                    "RATE_LIMIT_DELETE",
                    f">{self._config.max_file_deletes_per_min} file deletes in 60s",
                )
            self._delete_counter.record()

        # All file operations
        if tool_name in {
            "file_read",
            "file_write",
            "file_delete",
            "file_move",
            "file_copy",
            "search_files",
        }:
            if self._file_op_counter.count() >= self._config.max_file_ops_per_min:
                self._total_violations += 1
                raise GuardrailViolation(
                    "RATE_LIMIT_FILE_OPS",
                    f">{self._config.max_file_ops_per_min} file ops in 60s",
                )
            self._file_op_counter.record()

        # Code execution
        if tool_name in {"execute_code", "run_script"}:
            if self._code_exec_counter.count() >= self._config.max_code_exec_per_min:
                self._total_violations += 1
                raise GuardrailViolation(
                    "RATE_LIMIT_CODE_EXEC",
                    f">{self._config.max_code_exec_per_min} code executions in 60s",
                )
            self._code_exec_counter.record()

    # ── Domain Whitelist ──────────────────────────────────────────────────

    def _check_domain_whitelist(self, tool_name: str, args: dict):
        """Block browser navigation to non-whitelisted domains."""
        if tool_name != "browser_navigate":
            return

        url = args.get("url", "")
        match = _EXFIL_DOMAIN_PATTERN.search(url)
        if not match:
            return

        domain = match.group(1).lower()
        # Strip port
        if ":" in domain:
            domain = domain.split(":")[0]

        # Check against whitelist (allow subdomains)
        allowed = False
        for whitelisted in self._config.domain_whitelist:
            if domain == whitelisted or domain.endswith("." + whitelisted):
                allowed = True
                break

        if not allowed:
            self._total_violations += 1
            raise GuardrailViolation(
                "DOMAIN_BLOCKED",
                f"Domain '{domain}' not in whitelist",
            )
