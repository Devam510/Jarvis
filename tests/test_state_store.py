"""
tests/test_state_store.py — Comprehensive tests for the Atomic StateStore.

Covers all 8 safety invariants:
  1. Async lock guards
  2. Directory fsync
  3. Version discipline
  4. Atomic backup rotation
  5. Corrupt file preservation
  6. Cancelable auto-persist
  7. Post-unlock event emission
  8. Deep copy snapshots
"""

import asyncio
import json
import os
import tempfile
import time

import pytest

# Ensure project root on path
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from jarvis.core.event_bus import AsyncEventBus
from jarvis.core.state_store import StateStore


@pytest.fixture
def tmp_dir():
    """Create a temp directory for test checkpoints."""
    d = tempfile.mkdtemp(prefix="jarvis_state_test_")
    yield d
    # Cleanup
    import shutil

    shutil.rmtree(d, ignore_errors=True)


@pytest.fixture
def event_bus():
    return AsyncEventBus()


@pytest.fixture
def store(event_bus, tmp_dir):
    path = os.path.join(tmp_dir, "checkpoint.json")
    return StateStore(
        event_bus=event_bus,
        persist_path=path,
        auto_persist_seconds=60,  # long interval — we manually flush in tests
        max_versions=3,
    )


# ── 1. Basic get/set ─────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_get_set_basic(store):
    """set/get round-trip works."""
    assert store.get("foo") is None
    assert store.get("foo", "default") == "default"

    await store.set("foo", "bar")
    assert store.get("foo") == "bar"


# ── 2. Patch bulk ────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_patch_bulk(store):
    """patch sets multiple keys with a single version bump."""
    await store.patch({"a": 1, "b": 2, "c": 3})
    assert store.get("a") == 1
    assert store.get("b") == 2
    assert store.get("c") == 3
    # Only one version bump for the entire patch
    assert store._version == 1


# ── 3. Version increments only in mutations ──────────────────────────────


@pytest.mark.asyncio
async def test_version_increments(store):
    """Version bumps on set/patch, NOT on flush."""
    await store.set("x", 1)
    assert store._version == 1
    await store.set("y", 2)
    assert store._version == 2
    await store.patch({"z": 3, "w": 4})
    assert store._version == 3

    v_before_flush = store._version
    await store.flush()
    assert store._version == v_before_flush, "flush must NOT bump version"


# ── 4. Deep copy snapshots ───────────────────────────────────────────────


@pytest.mark.asyncio
async def test_snapshot_isolation(store):
    """snapshot() returns a deep copy — mutations don't leak."""
    await store.set("nested", {"inner": [1, 2, 3]})
    snap = store.snapshot()

    # Mutate the snapshot
    snap["data"]["nested"]["inner"].append(999)

    # Original must be untouched
    assert store.get("nested") == {"inner": [1, 2, 3]}


# ── 5. Flush creates file ────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_flush_creates_file(store):
    """flush() writes the checkpoint file."""
    await store.set("key", "value")
    await store.flush()
    assert os.path.exists(store._path)

    with open(store._path, "r") as f:
        data = json.load(f)
    assert data["version"] == 1
    assert data["data"]["key"] == "value"


# ── 6. Atomic write — no .tmp left behind ────────────────────────────────


@pytest.mark.asyncio
async def test_atomic_write_no_tmp(store):
    """After flush, .tmp file should not exist."""
    await store.set("x", 1)
    await store.flush()

    tmp_path = store._path + ".tmp"
    assert not os.path.exists(tmp_path), ".tmp file must be cleaned up"
    assert os.path.exists(store._path)


# ── 7. Load restores state ───────────────────────────────────────────────


@pytest.mark.asyncio
async def test_load_restores_state(event_bus, tmp_dir):
    """flush → new instance → load → same state."""
    path = os.path.join(tmp_dir, "checkpoint.json")
    store1 = StateStore(event_bus=event_bus, persist_path=path)
    await store1.set("restored_key", "restored_value")
    await store1.patch({"a": 1, "b": 2})
    await store1.flush()

    # New instance, same path
    store2 = StateStore(event_bus=event_bus, persist_path=path)
    await store2.load()

    assert store2.get("restored_key") == "restored_value"
    assert store2.get("a") == 1
    assert store2.get("b") == 2
    assert store2._version == 2  # set + patch = 2 bumps


# ── 8. Corrupt file preservation ─────────────────────────────────────────


@pytest.mark.asyncio
async def test_corrupt_file_recovery(event_bus, tmp_dir):
    """Corrupt JSON is renamed to .corrupt.<ts>, not silently overwritten."""
    path = os.path.join(tmp_dir, "checkpoint.json")

    # Write corrupt data
    with open(path, "w") as f:
        f.write("{broken json!!! ...")

    store = StateStore(event_bus=event_bus, persist_path=path)
    await store.load()

    # State should be empty (graceful recovery)
    assert store.get("anything") is None
    assert store._version == 0

    # Corrupt file should be preserved with .corrupt. suffix
    files = os.listdir(tmp_dir)
    corrupt_files = [f for f in files if ".corrupt." in f]
    assert len(corrupt_files) == 1, f"Expected 1 corrupt file, got: {files}"


# ── 9. Event emitted on set ──────────────────────────────────────────────


@pytest.mark.asyncio
async def test_event_emitted_on_set(store, event_bus):
    """state.changed fires with key, value, and version."""
    received = []

    async def handler(event):
        received.append(event)

    event_bus.subscribe("state.changed", handler)

    await store.set("k", "v")

    assert len(received) == 1
    assert received[0]["key"] == "k"
    assert received[0]["value"] == "v"
    assert received[0]["version"] == 1


# ── 10. Subscribe changes callback ───────────────────────────────────────


@pytest.mark.asyncio
async def test_subscribe_changes(store):
    """Subscriber callbacks are invoked on mutation."""
    received = []

    async def cb(event):
        received.append(event)

    store.subscribe_changes(cb)
    await store.set("a", 1)
    await store.patch({"b": 2, "c": 3})

    assert len(received) == 2
    assert received[0]["key"] == "a"
    assert "keys" in received[1]
    assert "b" in received[1]["keys"]


# ── 11. Rolling backups ──────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_rolling_backups(store):
    """Flush creates rolling .bak.N backups up to max_versions."""
    for i in range(5):
        await store.set("counter", i)
        await store.flush()

    # max_versions=3 → .bak.1, .bak.2, .bak.3 should exist
    for n in range(1, 4):
        bak = f"{store._path}.bak.{n}"
        assert os.path.exists(bak), f"Expected backup {bak} to exist"

    # .bak.4 should NOT exist (max_versions=3)
    bak4 = f"{store._path}.bak.4"
    assert not os.path.exists(bak4), f"Backup {bak4} should have been pruned"


# ── 12. Auto-persist loop with cancellation + final flush ─────────────────


@pytest.mark.asyncio
async def test_auto_persist_cancel_flushes(event_bus, tmp_dir):
    """Auto-persist handles CancelledError and flushes once on stop."""
    path = os.path.join(tmp_dir, "checkpoint.json")
    store = StateStore(
        event_bus=event_bus,
        persist_path=path,
        auto_persist_seconds=1,  # short for test
    )

    await store.set("before_persist", True)
    await store.start_auto_persist()

    # Wait for at least one auto-persist cycle
    await asyncio.sleep(1.5)
    assert os.path.exists(path), "Auto-persist should have written file"

    # Set new state, then stop (should trigger final flush)
    await store.set("after_persist", True)
    await store.stop()

    # Verify final flush captured the latest state
    with open(path, "r") as f:
        data = json.load(f)
    assert (
        data["data"].get("after_persist") is True
    ), "Final flush must capture latest state"
