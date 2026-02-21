"""
tests.test_behavioral_memory — Phase 4 tests for BehavioralMemory.

Tests cover safety requirement S3 (write throttling).
"""

import asyncio
import time

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from jarvis.perception.behavioral_memory import BehavioralMemory, _context_hash
from jarvis.utils.config import BehavioralMemoryConfig
from jarvis.utils.types import BehaviorPattern, ScreenContext


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def config(tmp_path):
    return BehavioralMemoryConfig(
        enabled=True,
        persist_dir=str(tmp_path / "behavior_db"),
        min_pattern_count=3,
        prediction_threshold=0.3,  # Low threshold so 3 recordings pass (conf=0.33)
        max_patterns=100,
        debounce_seconds=0.1,  # Fast for tests
        max_writes_per_minute=6,
        batch_flush_interval=0.5,  # Fast for tests
    )


@pytest.fixture
def memory(config):
    mem = BehavioralMemory(config)
    # Don't start the flush loop for most tests — we'll flush manually
    from pathlib import Path

    Path(config.persist_dir).mkdir(parents=True, exist_ok=True)
    yield mem


def _make_context(app: str = "VS Code") -> ScreenContext:
    return ScreenContext(
        active_app=app,
        window_title=f"{app} - main.py",
        visible_text="",
        detected_errors=(),
        timestamp=time.time(),
    )


# ══════════════════════════════════════════════════════════════════════════════
# Core Functionality
# ══════════════════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_record_and_predict(memory):
    """3+ patterns with same context → prediction returned."""
    ctx = _make_context("VS Code")

    # Record same action 3 times (with debounce gaps)
    for i in range(3):
        memory._last_write_time = 0  # Reset debounce for test
        await memory.record_action(ctx, "open_terminal")
        # Flush immediately
        await memory._flush_queue()

    # Now predict
    prediction = await memory.predict_action(ctx)
    assert prediction is not None
    assert prediction.trigger == "behavior_prediction"
    assert prediction.confidence > 0


@pytest.mark.asyncio
async def test_below_threshold_no_prediction(memory):
    """<3 occurrences → no prediction."""
    ctx = _make_context("VS Code")

    memory._last_write_time = 0
    await memory.record_action(ctx, "open_terminal")
    await memory._flush_queue()

    prediction = await memory.predict_action(ctx)
    assert prediction is None


@pytest.mark.asyncio
async def test_json_fallback(config):
    """Works when ChromaDB is not installed."""
    # Mock ChromaDB import to fail
    with patch.dict("sys.modules", {"chromadb": None}):
        mem = BehavioralMemory(config)
        # initialize should fall through to JSON
        with patch("builtins.__import__", side_effect=ImportError):
            await mem._load_json()
        assert mem._use_chromadb is False

    if mem._flush_task and not mem._flush_task.done():
        mem._flush_task.cancel()
        try:
            await mem._flush_task
        except asyncio.CancelledError:
            pass


@pytest.mark.asyncio
async def test_json_persist_and_load(config):
    """Persist patterns to JSON and reload them."""
    mem1 = BehavioralMemory(config)
    from pathlib import Path

    Path(config.persist_dir).mkdir(parents=True, exist_ok=True)

    # Add a pattern directly
    pattern = BehaviorPattern(
        context_hash="abc123",
        action="open_browser",
        count=5,
        metadata={"app": "VS Code"},
    )
    mem1._patterns["abc123"] = pattern
    await mem1._persist_json()

    # Load into new instance
    mem2 = BehavioralMemory(config)
    await mem2._load_json()
    assert "abc123" in mem2._patterns
    assert mem2._patterns["abc123"].action == "open_browser"
    assert mem2._patterns["abc123"].count == 5


# ══════════════════════════════════════════════════════════════════════════════
# S3: Memory Write Throttling
# ══════════════════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_s3_debounce(memory):
    """S3: Rapid updates within debounce window → only first accepted."""
    ctx = _make_context("VS Code")

    memory._config.debounce_seconds = 10.0  # Long debounce
    memory._last_write_time = time.time()  # Mark as "just wrote"

    await memory.record_action(ctx, "open_terminal")
    assert memory._write_queue.qsize() == 0  # Debounced — nothing queued


@pytest.mark.asyncio
async def test_s3_debounce_allows_after_window(memory):
    """S3: Write allowed after debounce window passes."""
    ctx = _make_context("VS Code")

    memory._config.debounce_seconds = 0.1
    memory._last_write_time = time.time() - 1.0  # 1 second ago

    await memory.record_action(ctx, "open_terminal")
    assert memory._write_queue.qsize() == 1  # Accepted


@pytest.mark.asyncio
async def test_s3_max_writes_per_minute(memory):
    """S3: More than max_writes_per_minute → excess deferred."""
    memory._config.max_writes_per_minute = 3
    memory._minute_start = time.time()

    # Add 5 items to queue
    for i in range(5):
        memory._write_queue.put_nowait(
            BehaviorPattern(
                context_hash=f"hash_{i}",
                action=f"action_{i}",
            )
        )

    # Flush — should only process 3
    await memory._flush_queue()
    assert memory._writes_this_minute == 3
    assert memory._write_queue.qsize() == 2  # 2 remaining


@pytest.mark.asyncio
async def test_s3_batch_accumulation(memory):
    """S3: Multiple record_action calls accumulate in batch queue."""
    ctx = _make_context("VS Code")

    for i in range(3):
        memory._last_write_time = 0  # Reset debounce
        await memory.record_action(ctx, f"action_{i}")

    assert memory._write_queue.qsize() == 3


@pytest.mark.asyncio
async def test_max_patterns_eviction(memory):
    """Oldest patterns evicted when max capacity reached."""
    memory._config.max_patterns = 3

    # Fill to capacity
    for i in range(3):
        memory._patterns[f"hash_{i}"] = BehaviorPattern(
            context_hash=f"hash_{i}",
            action=f"action_{i}",
            last_seen=time.time() - (100 - i),  # hash_0 is oldest
        )

    # Add one more via queue
    memory._write_queue.put_nowait(
        BehaviorPattern(context_hash="hash_new", action="new_action")
    )
    await memory._flush_queue()

    # hash_0 (oldest) should have been evicted
    assert "hash_0" not in memory._patterns
    assert "hash_new" in memory._patterns
    assert len(memory._patterns) == 3


# ══════════════════════════════════════════════════════════════════════════════
# Context Handling
# ══════════════════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_skips_blacked_out_context(memory):
    """Does not record patterns from blacked-out contexts."""
    ctx = ScreenContext(
        active_app="KeePass",
        is_blacked_out=True,
        timestamp=time.time(),
    )
    memory._last_write_time = 0
    await memory.on_context_update(ctx)
    assert memory._write_queue.qsize() == 0


@pytest.mark.asyncio
async def test_skips_empty_app(memory):
    """Does not record patterns when no app is active."""
    ctx = ScreenContext(active_app="", timestamp=time.time())
    memory._last_write_time = 0
    await memory.on_context_update(ctx)
    assert memory._write_queue.qsize() == 0


def test_context_hash_deterministic():
    """Context hash is deterministic for same inputs."""
    h1 = _context_hash(9, 1, "VS Code")
    h2 = _context_hash(9, 1, "VS Code")
    assert h1 == h2


def test_context_hash_varies():
    """Context hash differs for different inputs."""
    h1 = _context_hash(9, 1, "VS Code")
    h2 = _context_hash(14, 1, "VS Code")
    h3 = _context_hash(9, 1, "Chrome")
    assert h1 != h2
    assert h1 != h3
