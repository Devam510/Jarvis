"""
jarvis.core.state_store — Atomic, crash-safe, observable state persistence.

Architecture:
  - All mutations guarded by asyncio.Lock
  - Atomic writes: write .tmp → fsync → os.replace → fsync parent dir
  - Version increments ONLY inside locked mutation paths (never auto-persist)
  - Events emitted AFTER lock release to prevent deadlocks
  - snapshot() returns deep copy for isolation
  - Corrupt files preserved as .corrupt.<timestamp>
  - Auto-persist loop handles CancelledError and flushes once on shutdown
"""

from __future__ import annotations

import asyncio
import copy
import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Callable, Coroutine, Optional

from jarvis.core.event_bus import AsyncEventBus

logger = logging.getLogger(__name__)


class StateStore:
    """Observable, crash-safe key-value state with atomic persistence.

    Thread safety: all mutations and persistence are serialized via asyncio.Lock.
    File safety: write-tmp → fsync → replace → fsync-dir (atomic on NTFS/ext4).
    """

    def __init__(
        self,
        event_bus: AsyncEventBus,
        persist_path: str = "state/checkpoint.json",
        auto_persist_seconds: int = 30,
        max_versions: int = 5,
    ):
        self.event_bus = event_bus
        self._path = str(Path(persist_path).resolve())
        self._auto_persist_seconds = auto_persist_seconds
        self._max_versions = max_versions

        self._data: dict[str, Any] = {}
        self._version: int = 0
        self._updated_at: float = 0.0

        self._lock = asyncio.Lock()
        self._subscribers: list[Callable[[dict], Coroutine]] = []
        self._persist_task: Optional[asyncio.Task] = None
        self._dirty = False  # tracks if state changed since last persist

        # Ensure directory exists
        Path(self._path).parent.mkdir(parents=True, exist_ok=True)

    # ── Public API ────────────────────────────────────────────────────────

    def get(self, key: str, default: Any = None) -> Any:
        """Get a state value. Lock-free — reads are safe on the event loop."""
        return self._data.get(key, default)

    async def set(self, key: str, value: Any):
        """Set a single key. Version increments under lock, event emits after."""
        async with self._lock:
            self._data[key] = value
            self._version += 1
            self._updated_at = time.time()
            self._dirty = True
            event_data = {
                "key": key,
                "value": value,
                "version": self._version,
            }

        # Emit OUTSIDE lock — prevents deadlocks if handlers call back
        await self._notify(event_data)

    async def patch(self, updates: dict[str, Any]):
        """Bulk-set multiple keys. Single version bump, single event."""
        if not updates:
            return

        async with self._lock:
            self._data.update(updates)
            self._version += 1
            self._updated_at = time.time()
            self._dirty = True
            event_data = {
                "keys": list(updates.keys()),
                "updates": updates,
                "version": self._version,
            }

        await self._notify(event_data)

    def snapshot(self) -> dict[str, Any]:
        """Return a deep copy of the full state. Isolated from mutations."""
        return copy.deepcopy(
            {
                "version": self._version,
                "updated_at": self._updated_at,
                "data": self._data,
            }
        )

    def subscribe_changes(self, callback: Callable[[dict], Coroutine]):
        """Register a callback invoked on every mutation (after unlock)."""
        self._subscribers.append(callback)

    # ── Persistence ───────────────────────────────────────────────────────

    async def flush(self):
        """Persist current state to disk immediately (for shutdown)."""
        await self._persist()

    async def load(self):
        """Restore state from disk. Handles corrupt files gracefully."""
        if not os.path.exists(self._path):
            logger.info("StateStore: no checkpoint at %s — starting fresh", self._path)
            return

        try:
            raw = await asyncio.get_event_loop().run_in_executor(None, self._read_file)
            parsed = json.loads(raw)
            async with self._lock:
                self._data = parsed.get("data", {})
                self._version = parsed.get("version", 0)
                self._updated_at = parsed.get("updated_at", 0.0)
            logger.info(
                "StateStore: loaded v%d (%d keys) from %s",
                self._version,
                len(self._data),
                self._path,
            )
        except (json.JSONDecodeError, ValueError, KeyError) as e:
            # Preserve corrupt file for forensics
            ts = int(time.time())
            corrupt_path = f"{self._path}.corrupt.{ts}"
            try:
                os.rename(self._path, corrupt_path)
                logger.error(
                    "StateStore: corrupt checkpoint moved to %s: %s",
                    corrupt_path,
                    e,
                )
            except OSError as rename_err:
                logger.error(
                    "StateStore: corrupt file AND rename failed: %s / %s",
                    e,
                    rename_err,
                )
        except OSError as e:
            logger.error("StateStore: failed to read %s: %s", self._path, e)

    def _read_file(self) -> str:
        """Blocking read — always called inside run_in_executor."""
        with open(self._path, "r", encoding="utf-8") as f:
            return f.read()

    async def _persist(self):
        """Atomic persist: snapshot under lock → write in executor."""
        async with self._lock:
            if not self._dirty and os.path.exists(self._path):
                return  # nothing changed
            snap = {
                "version": self._version,
                "updated_at": self._updated_at,
                "data": copy.deepcopy(self._data),
            }
            self._rotate_backups_locked()
            self._dirty = False

        # File I/O outside lock — non-blocking
        await asyncio.get_event_loop().run_in_executor(None, self._atomic_write, snap)
        logger.debug("StateStore: persisted v%d to %s", snap["version"], self._path)

    def _atomic_write(self, snapshot: dict):
        """Write-tmp → fsync → os.replace → fsync-dir. Crash-safe."""
        tmp_path = self._path + ".tmp"
        try:
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(snapshot, f, indent=2, default=str)
                f.flush()
                os.fsync(f.fileno())

            os.replace(tmp_path, self._path)

            # fsync parent directory for full crash safety
            self._fsync_directory(os.path.dirname(self._path))

        except OSError as e:
            logger.error("StateStore: atomic write failed: %s", e)
            # Clean up temp file if it exists
            try:
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)
            except OSError:
                pass

    @staticmethod
    def _fsync_directory(dir_path: str):
        """fsync a directory to ensure rename/replace is durable."""
        try:
            fd = os.open(dir_path, os.O_RDONLY)
            try:
                os.fsync(fd)
            finally:
                os.close(fd)
        except OSError:
            # Windows may not support O_RDONLY on directories in all cases
            pass

    def _rotate_backups_locked(self):
        """Rotate checkpoint backups. MUST be called under self._lock."""
        if self._max_versions <= 0:
            return
        if not os.path.exists(self._path):
            return

        # Shift existing backups: .bak.4 → .bak.5, .bak.3 → .bak.4, etc.
        for i in range(self._max_versions - 1, 0, -1):
            src = f"{self._path}.bak.{i}"
            dst = f"{self._path}.bak.{i + 1}"
            try:
                if os.path.exists(dst):
                    os.unlink(dst)
                if os.path.exists(src):
                    os.rename(src, dst)
            except OSError:
                pass

        # Current checkpoint → .bak.1
        bak1 = f"{self._path}.bak.1"
        try:
            if os.path.exists(bak1):
                os.unlink(bak1)
            if os.path.exists(self._path):
                # Copy instead of rename — we still need the current file
                import shutil

                shutil.copy2(self._path, bak1)
        except OSError:
            pass

        # Remove excess backups
        for i in range(self._max_versions + 1, self._max_versions + 5):
            excess = f"{self._path}.bak.{i}"
            try:
                if os.path.exists(excess):
                    os.unlink(excess)
            except OSError:
                pass

    # ── Auto-Persist Loop ─────────────────────────────────────────────────

    async def start_auto_persist(self):
        """Start the background auto-persist loop."""
        if self._persist_task is not None:
            return
        self._persist_task = asyncio.create_task(self._auto_persist_loop())
        logger.info(
            "StateStore: auto-persist started (every %ds)", self._auto_persist_seconds
        )

    async def _auto_persist_loop(self):
        """Background loop — persists dirty state every N seconds.

        Handles CancelledError gracefully: flushes once on shutdown.
        """
        try:
            while True:
                await asyncio.sleep(self._auto_persist_seconds)
                try:
                    await self._persist()
                except Exception as e:
                    logger.error("StateStore: auto-persist error: %s", e)
        except asyncio.CancelledError:
            # Final flush on shutdown
            try:
                await self._persist()
                logger.info("StateStore: final flush on shutdown complete")
            except Exception as e:
                logger.error("StateStore: final flush failed: %s", e)
            raise  # Re-raise so task shows as cancelled

    async def stop(self):
        """Stop auto-persist and do a final flush."""
        if self._persist_task is not None:
            self._persist_task.cancel()
            try:
                await self._persist_task
            except asyncio.CancelledError:
                pass
            self._persist_task = None

        # Extra safety: flush even if task was already done
        await self._persist()
        logger.info("StateStore: stopped")

    # ── Event Notification ────────────────────────────────────────────────

    async def _notify(self, event_data: dict):
        """Emit state.changed event and call subscribers. MUST be called OUTSIDE lock."""
        try:
            await self.event_bus.emit("state.changed", event_data)
        except Exception as e:
            logger.error("StateStore: event emission error: %s", e)

        for cb in self._subscribers:
            try:
                await cb(event_data)
            except Exception as e:
                logger.error("StateStore: subscriber callback error: %s", e)
