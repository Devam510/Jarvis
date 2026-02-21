"""
jarvis.execution.transaction — Transactional rollback system (§30).

Every multi-step plan generates a transaction log. On failure, completed
steps are rolled back in LIFO order using pre-computed reverse operations.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import shutil
import time
import uuid
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

# ── Reverse operation map ────────────────────────────────────────────────

_REVERSE_OPS = {
    "file_write": "file_delete",  # new file → delete it
    "file_delete": "file_write",  # deleted file → restore from backup
    "file_move": "file_move",  # swap src/dst
    "file_copy": "file_delete",  # copied file → delete the copy
    "install_package": "uninstall_package",
}


@dataclass
class TransactionStep:
    """One step in a transaction."""

    seq: int
    tool: str
    args: dict
    reverse_tool: Optional[str] = None
    reverse_args: Optional[dict] = None
    backup_path: Optional[str] = None
    status: str = "pending"  # pending | completed | failed | rolled_back
    result: Optional[str] = None
    timestamp: float = 0.0


@dataclass
class TransactionManifest:
    """Full transaction manifest for a multi-step plan."""

    tx_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    created_at: float = field(default_factory=time.time)
    steps: list[TransactionStep] = field(default_factory=list)
    status: str = "active"  # active | committed | rolled_back | partial_rollback


class TransactionLog:
    """Manages transaction manifests with backup and rollback.

    Usage:
        tx = TransactionLog(persist_dir="state/transactions")
        manifest = tx.begin()
        tx.record_step(manifest, tool="file_write", args={...})
        tx.mark_completed(manifest, 0, result="ok")  # step 0 done
        # On failure:
        await tx.rollback(manifest, executor_fn)
    """

    def __init__(
        self,
        persist_dir: str = "state/transactions",
        backup_dir: str = "state/tx_backups",
    ):
        self._persist_dir = Path(persist_dir).resolve()
        self._backup_dir = Path(backup_dir).resolve()
        self._persist_dir.mkdir(parents=True, exist_ok=True)
        self._backup_dir.mkdir(parents=True, exist_ok=True)

    # ── Public API ────────────────────────────────────────────────────────

    def begin(self) -> TransactionManifest:
        """Start a new transaction."""
        manifest = TransactionManifest()
        logger.info("Transaction %s started", manifest.tx_id)
        return manifest

    def record_step(
        self,
        manifest: TransactionManifest,
        tool: str,
        args: dict,
    ) -> int:
        """Record a planned step and compute its reverse operation.

        Returns the step index.
        """
        seq = len(manifest.steps)
        reverse_tool = _REVERSE_OPS.get(tool)
        reverse_args = self._compute_reverse_args(tool, args, manifest.tx_id, seq)
        backup_path = None

        # Create backup of target file if we're about to overwrite/delete
        if tool in ("file_write", "file_delete"):
            target = args.get("path", "")
            if target and os.path.exists(target):
                backup_path = str(
                    self._backup_dir / f"{manifest.tx_id}_step{seq}_{Path(target).name}"
                )
                try:
                    shutil.copy2(target, backup_path)
                    logger.debug("Backed up %s → %s", target, backup_path)
                except Exception as e:
                    logger.warning("Backup failed for %s: %s", target, e)
                    backup_path = None

                # If overwriting, reverse is restore, not delete
                if tool == "file_write" and backup_path:
                    reverse_tool = "file_write"
                    reverse_args = {
                        "path": target,
                        "content": "__RESTORE_FROM_BACKUP__",
                    }

        step = TransactionStep(
            seq=seq,
            tool=tool,
            args=args,
            reverse_tool=reverse_tool,
            reverse_args=reverse_args,
            backup_path=backup_path,
            timestamp=time.time(),
        )
        manifest.steps.append(step)
        self._persist_manifest(manifest)
        return seq

    def mark_completed(self, manifest: TransactionManifest, seq: int, result: str = ""):
        """Mark a step as completed."""
        if seq < len(manifest.steps):
            manifest.steps[seq].status = "completed"
            manifest.steps[seq].result = result
            self._persist_manifest(manifest)

    def mark_failed(self, manifest: TransactionManifest, seq: int, error: str = ""):
        """Mark a step as failed."""
        if seq < len(manifest.steps):
            manifest.steps[seq].status = "failed"
            manifest.steps[seq].result = error
            self._persist_manifest(manifest)

    async def rollback(
        self,
        manifest: TransactionManifest,
        executor_fn=None,
    ) -> list[dict]:
        """Rollback completed steps in LIFO order.

        Args:
            manifest: The transaction to rollback.
            executor_fn: Optional async callable(tool, args) -> result.
                         If None, performs file-level rollbacks directly.

        Returns:
            List of rollback results.
        """
        completed = [
            s for s in manifest.steps if s.status == "completed" and s.reverse_tool
        ]
        completed.reverse()  # LIFO

        results = []
        all_ok = True

        for step in completed:
            try:
                result = await self._execute_reverse(step, executor_fn)
                step.status = "rolled_back"
                results.append(
                    {"seq": step.seq, "status": "rolled_back", "detail": result}
                )
                logger.info("Rolled back step %d (%s)", step.seq, step.tool)
            except Exception as e:
                all_ok = False
                results.append(
                    {"seq": step.seq, "status": "rollback_failed", "error": str(e)}
                )
                logger.error("Rollback failed for step %d: %s", step.seq, e)

        manifest.status = "rolled_back" if all_ok else "partial_rollback"
        self._persist_manifest(manifest)
        return results

    def commit(self, manifest: TransactionManifest):
        """Mark transaction as committed (cleanup backups)."""
        manifest.status = "committed"
        self._persist_manifest(manifest)
        self._cleanup_backups(manifest)

    # ── Internal ──────────────────────────────────────────────────────────

    def _compute_reverse_args(
        self, tool: str, args: dict, tx_id: str, seq: int
    ) -> Optional[dict]:
        """Compute reverse operation arguments."""
        if tool == "file_write":
            # New file → reverse is delete
            return {"path": args.get("path", "")}

        if tool == "file_delete":
            # Delete → reverse is write (from backup, handled in rollback)
            return {"path": args.get("path", ""), "content": "__RESTORE_FROM_BACKUP__"}

        if tool == "file_move":
            return {
                "source": args.get("destination", ""),
                "destination": args.get("source", ""),
            }

        if tool == "file_copy":
            return {"path": args.get("destination", "")}

        if tool == "install_package":
            return {"package": args.get("package", "")}

        return None

    async def _execute_reverse(self, step: TransactionStep, executor_fn=None) -> str:
        """Execute the reverse operation for a single step."""

        # Special: restore from backup
        if step.backup_path and os.path.exists(step.backup_path):
            target = step.args.get("path", "") or (
                step.reverse_args.get("path", "") if step.reverse_args else ""
            )
            if target:
                shutil.copy2(step.backup_path, target)
                return f"restored {target} from backup"

        # If we have a custom executor, use it
        if executor_fn and step.reverse_tool and step.reverse_args:
            result = await executor_fn(step.reverse_tool, step.reverse_args)
            return str(result)

        # Direct file operations as fallback
        if step.reverse_tool == "file_delete" and step.reverse_args:
            path = step.reverse_args.get("path", "")
            if path and os.path.exists(path):
                os.remove(path)
                return f"deleted {path}"

        if step.reverse_tool == "file_move" and step.reverse_args:
            src = step.reverse_args.get("source", "")
            dst = step.reverse_args.get("destination", "")
            if src and dst and os.path.exists(src):
                shutil.move(src, dst)
                return f"moved {src} → {dst}"

        return "no-op"

    def _persist_manifest(self, manifest: TransactionManifest):
        """Write manifest to disk (sync helper — called from thread)."""
        path = self._persist_dir / f"{manifest.tx_id}.json"
        data = {
            "tx_id": manifest.tx_id,
            "created_at": manifest.created_at,
            "status": manifest.status,
            "steps": [
                {
                    "seq": s.seq,
                    "tool": s.tool,
                    "args": s.args,
                    "reverse_tool": s.reverse_tool,
                    "reverse_args": s.reverse_args,
                    "backup_path": s.backup_path,
                    "status": s.status,
                    "result": s.result,
                    "timestamp": s.timestamp,
                }
                for s in manifest.steps
            ],
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, default=str)

    async def persist_manifest_async(self, manifest: TransactionManifest):
        """BUG-05 FIX: Non-blocking manifest persist via thread pool."""
        await asyncio.get_event_loop().run_in_executor(
            None, self._persist_manifest, manifest
        )

    def _cleanup_backups(self, manifest: TransactionManifest):
        """Remove backup files after successful commit."""
        for step in manifest.steps:
            if step.backup_path and os.path.exists(step.backup_path):
                try:
                    os.remove(step.backup_path)
                except Exception as e:
                    logger.warning("Backup cleanup failed: %s", e)
