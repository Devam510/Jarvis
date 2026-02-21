"""
jarvis.perception.behavioral_memory — Action pattern learning and prediction.

Safety guarantees:
  [S3] Memory write throttling — debounce, batch queue, max N writes/min
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import time
from collections import defaultdict
from pathlib import Path
from typing import Optional

from jarvis.utils.config import BehavioralMemoryConfig
from jarvis.utils.types import BehaviorPattern, ScreenContext, Suggestion

logger = logging.getLogger(__name__)


def _context_hash(hour: int, day_of_week: int, app: str) -> str:
    """Create a deterministic hash from time bucket + active app."""
    key = f"{hour}:{day_of_week}:{app.lower().strip()}"
    return hashlib.sha256(key.encode()).hexdigest()[:16]


class BehavioralMemory:
    """
    Stores action patterns keyed by (time_bucket, active_app).

    Uses ChromaDB when available, falls back to JSON file.

    [S3] All writes are debounced, batched, and rate-limited:
    - Debounce: ignores updates within `debounce_seconds` of last write
    - Batch: accumulates writes and flushes periodically
    - Rate limit: hard cap of `max_writes_per_minute`
    """

    def __init__(self, config: BehavioralMemoryConfig):
        self._config = config
        self._collection = None  # ChromaDB collection (lazy)
        self._use_chromadb = False

        # [S3] Write throttling state
        # [V2-03 FIX] Use asyncio.Queue instead of plain list for async safety
        self._write_queue: asyncio.Queue = asyncio.Queue(maxsize=500)
        self._last_write_time: float = 0.0
        self._writes_this_minute: int = 0
        self._minute_start: float = time.time()
        self._flush_task: Optional[asyncio.Task] = None

        # In-memory pattern cache (for JSON fallback + fast lookup)
        self._patterns: dict[str, BehaviorPattern] = {}
        self._json_path = Path(config.persist_dir) / "patterns.json"

    async def initialize(self):
        """Initialize storage backend."""
        Path(self._config.persist_dir).mkdir(parents=True, exist_ok=True)

        try:
            import chromadb

            client = chromadb.PersistentClient(path=self._config.persist_dir)
            self._collection = client.get_or_create_collection("behavior_patterns")
            self._use_chromadb = True
            logger.info("BehavioralMemory initialized with ChromaDB")
        except ImportError:
            logger.info("ChromaDB not available — using JSON fallback")
            self._use_chromadb = False
            await self._load_json()
        except Exception as e:
            logger.warning("ChromaDB init failed, using JSON fallback: %s", e)
            self._use_chromadb = False
            await self._load_json()

        # Start batch flush loop
        self._flush_task = asyncio.create_task(self._flush_loop())

    async def record_action(self, context: ScreenContext, action: str):
        """
        Record a user action in the current context.

        [S3] Debounced — ignores rapid updates within debounce window.
        """
        if not self._config.enabled:
            return

        now = time.time()

        # [S3] Debounce check
        if now - self._last_write_time < self._config.debounce_seconds:
            return

        # Build pattern
        import datetime

        dt = datetime.datetime.now()
        ctx_hash = _context_hash(dt.hour, dt.weekday(), context.active_app)

        pattern = BehaviorPattern(
            context_hash=ctx_hash,
            action=action,
            count=1,
            last_seen=now,
            metadata={
                "app": context.active_app,
                "hour": dt.hour,
                "day": dt.weekday(),
            },
        )

        # [S3] Add to batch queue (not written immediately)
        # [V2-03 FIX] Use put_nowait for async-safe queuing
        try:
            self._write_queue.put_nowait(pattern)
        except asyncio.QueueFull:
            logger.debug("Behavioral memory write queue full — dropping pattern")
            return
        self._last_write_time = now

    async def on_context_update(self, context: ScreenContext):
        """
        Handle perception context update for passive learning.
        Records active app as implicit action.
        """
        if context.is_blacked_out or not context.active_app:
            return
        await self.record_action(context, f"using:{context.active_app}")

    async def predict_action(self, context: ScreenContext) -> Optional[Suggestion]:
        """
        Predict what the user might want based on current context.
        Returns a suggestion if confidence is high enough.
        """
        if not self._config.enabled or context.is_blacked_out:
            return None

        import datetime

        dt = datetime.datetime.now()
        ctx_hash = _context_hash(dt.hour, dt.weekday(), context.active_app)

        # Look up patterns for this context
        pattern = self._patterns.get(ctx_hash)
        if pattern and pattern.count >= self._config.min_pattern_count:
            # Calculate confidence based on frequency
            confidence = min(1.0, pattern.count / (self._config.min_pattern_count * 3))
            if confidence >= self._config.prediction_threshold:
                return Suggestion(
                    trigger="behavior_prediction",
                    message=f"Based on your usual routine, would you like me to {pattern.action}?",
                    confidence=confidence,
                    action_hint=pattern.action,
                )

        return None

    async def _flush_loop(self):
        """[S3] Periodic batch flush with rate limiting."""
        try:
            while True:
                await asyncio.sleep(self._config.batch_flush_interval)
                await self._flush_queue()
        except asyncio.CancelledError:
            # Final flush on shutdown
            await self._flush_queue()

    async def _flush_queue(self):
        """[S3] Flush pending writes, respecting rate limits.

        [V2-03 FIX] Drains from asyncio.Queue instead of slicing a list.
        """
        if self._write_queue.empty():
            return

        now = time.time()

        # Reset minute counter
        if now - self._minute_start >= 60.0:
            self._writes_this_minute = 0
            self._minute_start = now

        # [S3] Hard cap: max writes per minute
        remaining_budget = self._config.max_writes_per_minute - self._writes_this_minute
        if remaining_budget <= 0:
            logger.debug(
                "Write rate limit reached (%d/min), deferring %d items",
                self._config.max_writes_per_minute,
                self._write_queue.qsize(),
            )
            return

        # Drain up to remaining_budget items from queue
        batch: list[BehaviorPattern] = []
        while len(batch) < remaining_budget and not self._write_queue.empty():
            try:
                batch.append(self._write_queue.get_nowait())
            except asyncio.QueueEmpty:
                break

        # Merge into in-memory patterns
        for pattern in batch:
            existing = self._patterns.get(pattern.context_hash)
            if existing:
                existing.count += pattern.count
                existing.last_seen = pattern.last_seen
            else:
                # [S4] Check max patterns cap
                if len(self._patterns) >= self._config.max_patterns:
                    # Evict oldest pattern
                    oldest_key = min(
                        self._patterns, key=lambda k: self._patterns[k].last_seen
                    )
                    del self._patterns[oldest_key]
                self._patterns[pattern.context_hash] = pattern

        self._writes_this_minute += len(batch)

        # Persist
        if self._use_chromadb:
            await self._persist_chromadb(batch)
        else:
            await self._persist_json()

    async def _persist_chromadb(self, patterns: list[BehaviorPattern]):
        """Write patterns to ChromaDB."""
        if not self._collection:
            return
        try:
            for p in patterns:
                self._collection.upsert(
                    ids=[p.context_hash],
                    documents=[p.action],
                    metadatas=[
                        {
                            "count": str(p.count),
                            "last_seen": str(p.last_seen),
                            **(p.metadata or {}),
                        }
                    ],
                )
        except Exception as e:
            logger.error("ChromaDB write failed: %s", e)

    async def _persist_json(self):
        """Write all patterns to JSON file."""
        try:
            data = {
                k: {
                    "context_hash": v.context_hash,
                    "action": v.action,
                    "count": v.count,
                    "last_seen": v.last_seen,
                    "metadata": v.metadata,
                    "id": v.id,
                }
                for k, v in self._patterns.items()
            }
            await asyncio.get_event_loop().run_in_executor(None, self._write_json, data)
        except Exception as e:
            logger.error("JSON persist failed: %s", e)

    def _write_json(self, data: dict):
        """Atomic JSON write (runs in executor)."""
        tmp = self._json_path.with_suffix(".tmp")
        with open(tmp, "w") as f:
            json.dump(data, f, indent=2)
        tmp.replace(self._json_path)

    async def _load_json(self):
        """Load patterns from JSON file."""
        if not self._json_path.exists():
            return
        try:
            with open(self._json_path, "r") as f:
                data = json.load(f)
            for key, val in data.items():
                self._patterns[key] = BehaviorPattern(
                    context_hash=val["context_hash"],
                    action=val["action"],
                    count=val.get("count", 1),
                    last_seen=val.get("last_seen", 0.0),
                    metadata=val.get("metadata", {}),
                    id=val.get("id", key),
                )
            logger.info("Loaded %d behavior patterns from JSON", len(self._patterns))
        except Exception as e:
            logger.error("JSON load failed: %s", e)

    async def stop(self):
        """Graceful shutdown — flush remaining writes."""
        if self._flush_task:
            self._flush_task.cancel()
            try:
                await self._flush_task
            except asyncio.CancelledError:
                pass
        # Final flush
        await self._flush_queue()
        logger.info(
            "BehavioralMemory stopped, %d patterns persisted", len(self._patterns)
        )
