"""
jarvis.observability.audit_ledger — Cryptographic audit trail (§31).

Tamper-proof append-only log where each entry is chained via SHA-256:
    hash = SHA256(action + timestamp + prev_hash)

Supports chain verification and forensic queries.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

_GENESIS_HASH = "0" * 64  # Genesis block hash


@dataclass
class AuditEntry:
    """Single entry in the audit chain."""

    seq: int
    timestamp: float
    action: str  # "tool_executed", "guardrail_violation", etc.
    tool: str
    args_hash: str  # SHA-256 of json(args) — not raw args (privacy)
    result_summary: str
    risk_tier: str
    prev_hash: str
    hash: str


class AuditLedger:
    """Append-only cryptographic audit log.

    Each entry's hash chains to the previous entry, forming a tamper-proof
    sequence. The ledger is persisted as JSONL.
    """

    # [V2-05] Max ledger file size before rotation (10 MB default)
    _MAX_FILE_SIZE_BYTES = 10 * 1024 * 1024

    def __init__(
        self,
        ledger_path: str = "logs/audit.ledger",
        enabled: bool = True,
        max_file_size_mb: float = 10.0,
    ):
        self._path = str(Path(ledger_path).resolve())
        self._enabled = enabled
        self._seq = 0
        self._prev_hash = _GENESIS_HASH
        self._total_entries = 0
        self._MAX_FILE_SIZE_BYTES = int(max_file_size_mb * 1024 * 1024)

        Path(self._path).parent.mkdir(parents=True, exist_ok=True)

        # Resume from existing ledger
        self._resume()

    # ── Public API ────────────────────────────────────────────────────────

    async def record(
        self,
        action: str,
        tool: str,
        args: dict,
        result_summary: str = "",
        risk_tier: str = "TIER_1",
    ) -> Optional[AuditEntry]:
        """Append a new entry to the audit chain.

        Returns the entry, or None if disabled.
        """
        if not self._enabled:
            return None

        self._seq += 1
        timestamp = time.time()

        # Hash the args for privacy (don't store raw args)
        args_hash = hashlib.sha256(
            json.dumps(args, sort_keys=True, default=str).encode()
        ).hexdigest()

        # Compute chain hash: SHA256(action + timestamp_str + prev_hash)
        chain_input = f"{action}|{timestamp}|{self._prev_hash}"
        entry_hash = hashlib.sha256(chain_input.encode()).hexdigest()

        entry = AuditEntry(
            seq=self._seq,
            timestamp=timestamp,
            action=action,
            tool=tool,
            args_hash=args_hash,
            result_summary=result_summary[:200],  # cap summary
            risk_tier=risk_tier,
            prev_hash=self._prev_hash,
            hash=entry_hash,
        )

        # [V2-01 FIX] Non-blocking append via executor
        await self._append(entry)
        self._prev_hash = entry_hash
        self._total_entries += 1

        return entry

    def verify(self) -> tuple[bool, list[str]]:
        """Verify the entire chain integrity.

        Returns (is_valid, list_of_errors).
        """
        errors = []
        entries = self._load_all()

        if not entries:
            return True, []

        expected_prev = _GENESIS_HASH

        for i, entry in enumerate(entries):
            # Check prev_hash chain
            if entry["prev_hash"] != expected_prev:
                errors.append(
                    f"seq={entry['seq']}: prev_hash mismatch "
                    f"(expected {expected_prev[:12]}..., got {entry['prev_hash'][:12]}...)"
                )

            # Recompute hash
            chain_input = f"{entry['action']}|{entry['timestamp']}|{entry['prev_hash']}"
            expected_hash = hashlib.sha256(chain_input.encode()).hexdigest()

            if entry["hash"] != expected_hash:
                errors.append(
                    f"seq={entry['seq']}: hash mismatch "
                    f"(expected {expected_hash[:12]}..., got {entry['hash'][:12]}...)"
                )

            expected_prev = entry["hash"]

        is_valid = len(errors) == 0
        if is_valid:
            logger.info("Audit ledger verified: %d entries, chain intact", len(entries))
        else:
            logger.error("Audit ledger TAMPERED: %d errors", len(errors))

        return is_valid, errors

    def query(
        self,
        tool: Optional[str] = None,
        action: Optional[str] = None,
        since: Optional[float] = None,
        until: Optional[float] = None,
        limit: int = 100,
    ) -> list[dict]:
        """Query the ledger with optional filters."""
        entries = self._load_all()
        results = []

        for entry in entries:
            if tool and entry.get("tool") != tool:
                continue
            if action and entry.get("action") != action:
                continue
            if since and entry.get("timestamp", 0) < since:
                continue
            if until and entry.get("timestamp", 0) > until:
                continue
            results.append(entry)
            if len(results) >= limit:
                break

        return results

    @property
    def total_entries(self) -> int:
        return self._total_entries

    # ── Internal ──────────────────────────────────────────────────────────

    async def _append(self, entry: AuditEntry):
        """Append entry to JSONL file (non-blocking).

        [V2-01 FIX] File I/O runs in executor to avoid blocking the event loop.
        [V2-05 FIX] Rotates file when size exceeds max.
        """
        data = {
            "seq": entry.seq,
            "timestamp": entry.timestamp,
            "action": entry.action,
            "tool": entry.tool,
            "args_hash": entry.args_hash,
            "result_summary": entry.result_summary,
            "risk_tier": entry.risk_tier,
            "prev_hash": entry.prev_hash,
            "hash": entry.hash,
        }
        line = json.dumps(data) + "\n"
        await asyncio.get_event_loop().run_in_executor(None, self._write_line, line)

    def _write_line(self, line: str):
        """Blocking write — always called inside run_in_executor."""
        # [V2-05 FIX] Rotate if file exceeds max size
        try:
            if os.path.exists(self._path):
                size = os.path.getsize(self._path)
                if size > self._MAX_FILE_SIZE_BYTES:
                    bak = self._path + ".bak"
                    if os.path.exists(bak):
                        os.remove(bak)
                    os.rename(self._path, bak)
                    logger.info(
                        "Audit ledger rotated (%.1f MB > %.1f MB limit)",
                        size / (1024 * 1024),
                        self._MAX_FILE_SIZE_BYTES / (1024 * 1024),
                    )
        except OSError as e:
            logger.warning("Ledger rotation failed: %s", e)

        with open(self._path, "a", encoding="utf-8") as f:
            f.write(line)

    def _load_all(self) -> list[dict]:
        """Load all entries from the ledger file."""
        if not os.path.exists(self._path):
            return []

        entries = []
        with open(self._path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        entries.append(json.loads(line))
                    except json.JSONDecodeError:
                        logger.warning("Corrupt ledger line: %s", line[:80])
        return entries

    def _resume(self):
        """Resume sequence and hash chain from existing ledger."""
        entries = self._load_all()
        if entries:
            last = entries[-1]
            self._seq = last.get("seq", 0)
            self._prev_hash = last.get("hash", _GENESIS_HASH)
            self._total_entries = len(entries)
            logger.info(
                "Audit ledger resumed at seq=%d (%d entries)",
                self._seq,
                self._total_entries,
            )
