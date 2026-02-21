"""
jarvis.memory.memory_governance — PII filtering, TTL enforcement, right-to-forget (§32).

Privacy layer that sits between the application and memory storage:
  1. PIIFilter:      Regex-based sanitization before ChromaDB/JSON storage.
  2. TTLEnforcer:    Async background task that expires old entries.
  3. ForgetHandler:  Processes "forget" commands → vector search + delete.
"""

from __future__ import annotations

import asyncio
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


# ── PII Patterns ─────────────────────────────────────────────────────────

_PII_PATTERNS: list[tuple[str, re.Pattern]] = [
    (
        "CREDIT_CARD",
        re.compile(
            r"\b(?:4[0-9]{12}(?:[0-9]{3})?|5[1-5][0-9]{14}|"
            r"3[47][0-9]{13}|6(?:011|5[0-9]{2})[0-9]{12})\b"
        ),
    ),
    (
        "SSN",
        re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    ),
    (
        "PHONE",
        re.compile(r"\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"),
    ),
    (
        "EMAIL",
        re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"),
    ),
    (
        "IP_ADDRESS",
        re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"),
    ),
]


@dataclass
class MemoryGovernanceConfig:
    """Configuration for memory governance."""

    pii_enabled: bool = True
    ttl_hours: float = 24.0  # Session memory expiry
    ttl_check_interval: float = 300.0  # 5-minute check interval
    custom_pii_patterns: list = field(default_factory=list)


class PIIFilter:
    """Regex-based PII sanitization.

    Replaces detected patterns with `[REDACTED:TYPE]` tokens.
    """

    def __init__(self, config: Optional[MemoryGovernanceConfig] = None):
        self._config = config or MemoryGovernanceConfig()
        self._patterns = list(_PII_PATTERNS)

        # Add custom patterns
        for p in self._config.custom_pii_patterns:
            if isinstance(p, dict) and "name" in p and "pattern" in p:
                try:
                    self._patterns.append((p["name"], re.compile(p["pattern"])))
                except re.error as e:
                    logger.warning("Invalid custom PII pattern '%s': %s", p["name"], e)

        self._total_redactions = 0

    def sanitize(self, text: str) -> str:
        """Sanitize text by replacing PII with redaction tokens.

        Returns the sanitized text.
        """
        if not self._config.pii_enabled or not text:
            return text

        result = text
        for pii_type, pattern in self._patterns:
            count = len(pattern.findall(result))
            if count > 0:
                result = pattern.sub(f"[REDACTED:{pii_type}]", result)
                self._total_redactions += count

        return result

    @property
    def total_redactions(self) -> int:
        return self._total_redactions


class TTLEnforcer:
    """Background task that expires old memory entries.

    Calls the provided `delete_fn(entry_id)` for entries older than `ttl_hours`.
    """

    def __init__(
        self,
        ttl_hours: float = 24.0,
        check_interval: float = 300.0,
        query_fn=None,
        delete_fn=None,
    ):
        self._ttl_seconds = ttl_hours * 3600
        self._check_interval = check_interval
        self._query_fn = query_fn  # async fn() -> list[{id, timestamp}]
        self._delete_fn = delete_fn  # async fn(entry_id) -> None
        self._task: Optional[asyncio.Task] = None
        self._total_expired = 0

    async def start(self):
        """Start the background expiry loop."""
        if self._task is not None:
            return
        self._task = asyncio.create_task(self._loop())
        logger.info(
            "TTL enforcer started (ttl=%.0fh, interval=%.0fs)",
            self._ttl_seconds / 3600,
            self._check_interval,
        )

    async def stop(self):
        """Stop the background loop."""
        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

    async def enforce_once(self) -> int:
        """Run one expiry pass. Returns count of expired entries."""
        if not self._query_fn or not self._delete_fn:
            return 0

        try:
            entries = await self._query_fn()
        except Exception as e:
            logger.error("TTL query failed: %s", e)
            return 0

        now = time.time()
        cutoff = now - self._ttl_seconds
        expired = 0

        for entry in entries:
            ts = entry.get("timestamp", now)
            if ts < cutoff:
                try:
                    await self._delete_fn(entry["id"])
                    expired += 1
                except Exception as e:
                    logger.warning("TTL delete failed for %s: %s", entry.get("id"), e)

        self._total_expired += expired
        if expired > 0:
            logger.info("TTL enforcer expired %d entries", expired)
        return expired

    @property
    def total_expired(self) -> int:
        return self._total_expired

    async def _loop(self):
        """Background loop."""
        try:
            while True:
                await asyncio.sleep(self._check_interval)
                await self.enforce_once()
        except asyncio.CancelledError:
            pass


class ForgetHandler:
    """Processes 'forget' commands — vector search + batch delete.

    Usage:
        handler = ForgetHandler(search_fn, delete_fn)
        count = await handler.forget("Project X")
    """

    def __init__(self, search_fn=None, delete_fn=None):
        self._search_fn = search_fn  # async fn(topic) -> list[{id, ...}]
        self._delete_fn = delete_fn  # async fn(entry_id) -> None
        self._total_forgotten = 0

    async def forget(self, topic: str) -> int:
        """Delete all memories matching the topic.

        Returns the number of entries deleted.
        """
        if not self._search_fn or not self._delete_fn:
            logger.warning("ForgetHandler not connected to memory backend")
            return 0

        try:
            matches = await self._search_fn(topic)
        except Exception as e:
            logger.error("Forget search failed: %s", e)
            return 0

        deleted = 0
        for entry in matches:
            try:
                await self._delete_fn(entry["id"])
                deleted += 1
            except Exception as e:
                logger.warning("Forget delete failed for %s: %s", entry.get("id"), e)

        self._total_forgotten += deleted
        logger.info("Forgot %d entries about '%s'", deleted, topic)
        return deleted

    @property
    def total_forgotten(self) -> int:
        return self._total_forgotten
