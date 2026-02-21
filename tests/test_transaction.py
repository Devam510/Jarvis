"""
Tests for jarvis.execution.transaction — Phase 6B: Transactional Rollback.
"""

import asyncio
import json
import os
import shutil
import tempfile

import pytest

from jarvis.execution.transaction import TransactionLog, TransactionManifest


@pytest.fixture
def tmp_dirs(tmp_path):
    persist = str(tmp_path / "transactions")
    backup = str(tmp_path / "backups")
    return persist, backup


@pytest.fixture
def tx_log(tmp_dirs):
    return TransactionLog(persist_dir=tmp_dirs[0], backup_dir=tmp_dirs[1])


# ── Transaction Basics ───────────────────────────────────────────────────


class TestTransactionBasics:
    def test_begin_creates_manifest(self, tx_log):
        m = tx_log.begin()
        assert isinstance(m, TransactionManifest)
        assert m.status == "active"
        assert len(m.steps) == 0

    def test_record_step_computes_reverse(self, tx_log):
        m = tx_log.begin()
        seq = tx_log.record_step(
            m, "file_move", {"source": "a.txt", "destination": "b.txt"}
        )
        assert seq == 0
        step = m.steps[0]
        assert step.reverse_tool == "file_move"
        assert step.reverse_args == {"source": "b.txt", "destination": "a.txt"}

    def test_record_new_file_write_reverse_is_delete(self, tx_log):
        m = tx_log.begin()
        tx_log.record_step(m, "file_write", {"path": "/nonexistent/new_file.txt"})
        step = m.steps[0]
        assert step.reverse_tool == "file_delete"
        assert step.reverse_args["path"] == "/nonexistent/new_file.txt"

    def test_record_copy_reverse_is_delete_copy(self, tx_log):
        m = tx_log.begin()
        tx_log.record_step(m, "file_copy", {"source": "a", "destination": "b"})
        assert m.steps[0].reverse_tool == "file_delete"
        assert m.steps[0].reverse_args == {"path": "b"}

    def test_manifest_persists_to_disk(self, tx_log, tmp_dirs):
        m = tx_log.begin()
        tx_log.record_step(m, "file_write", {"path": "test.txt"})
        path = os.path.join(tmp_dirs[0], f"{m.tx_id}.json")
        assert os.path.exists(path)
        with open(path) as f:
            data = json.load(f)
        assert data["tx_id"] == m.tx_id
        assert len(data["steps"]) == 1


# ── Mark Status ──────────────────────────────────────────────────────────


class TestMarkStatus:
    def test_mark_completed(self, tx_log):
        m = tx_log.begin()
        tx_log.record_step(m, "file_write", {"path": "x"})
        tx_log.mark_completed(m, 0, result="ok")
        assert m.steps[0].status == "completed"

    def test_mark_failed(self, tx_log):
        m = tx_log.begin()
        tx_log.record_step(m, "file_write", {"path": "x"})
        tx_log.mark_failed(m, 0, error="boom")
        assert m.steps[0].status == "failed"
        assert m.steps[0].result == "boom"


# ── Backup & Rollback ───────────────────────────────────────────────────


class TestRollback:
    @pytest.mark.asyncio
    async def test_rollback_new_file_write(self, tx_log, tmp_path):
        """Rollback of a new file_write should delete the created file."""
        target = str(tmp_path / "created_file.txt")
        m = tx_log.begin()
        # File doesn't exist yet → reverse is file_delete
        tx_log.record_step(m, "file_write", {"path": target})
        tx_log.mark_completed(m, 0)

        # Simulate that the file was created
        with open(target, "w") as f:
            f.write("new content")

        results = await tx_log.rollback(m)
        assert len(results) == 1
        assert results[0]["status"] == "rolled_back"
        assert not os.path.exists(target)

    @pytest.mark.asyncio
    async def test_rollback_overwrite_restores_backup(self, tx_log, tmp_path):
        """Rollback of overwrite restores original content from backup."""
        target = str(tmp_path / "existing.txt")
        # Create the original file
        with open(target, "w") as f:
            f.write("ORIGINAL CONTENT")

        m = tx_log.begin()
        tx_log.record_step(m, "file_write", {"path": target})
        tx_log.mark_completed(m, 0)

        # Simulate overwrite
        with open(target, "w") as f:
            f.write("OVERWRITTEN")

        results = await tx_log.rollback(m)
        assert results[0]["status"] == "rolled_back"

        with open(target) as f:
            assert f.read() == "ORIGINAL CONTENT"

    @pytest.mark.asyncio
    async def test_rollback_file_move(self, tx_log, tmp_path):
        """Rollback of file_move moves the file back."""
        src = str(tmp_path / "src.txt")
        dst = str(tmp_path / "dst.txt")

        # Create src
        with open(src, "w") as f:
            f.write("data")

        m = tx_log.begin()
        tx_log.record_step(m, "file_move", {"source": src, "destination": dst})
        tx_log.mark_completed(m, 0)

        # Simulate the move
        shutil.move(src, dst)
        assert os.path.exists(dst)
        assert not os.path.exists(src)

        results = await tx_log.rollback(m)
        assert results[0]["status"] == "rolled_back"
        assert os.path.exists(src)
        assert not os.path.exists(dst)

    @pytest.mark.asyncio
    async def test_lifo_rollback_order(self, tx_log, tmp_path):
        """Completed steps should be rolled back in LIFO order."""
        f1 = str(tmp_path / "f1.txt")
        f2 = str(tmp_path / "f2.txt")

        m = tx_log.begin()
        tx_log.record_step(m, "file_write", {"path": f1})  # step 0
        tx_log.record_step(m, "file_write", {"path": f2})  # step 1
        tx_log.mark_completed(m, 0)
        tx_log.mark_completed(m, 1)

        # Create both files
        for f in [f1, f2]:
            with open(f, "w") as fh:
                fh.write("x")

        results = await tx_log.rollback(m)
        # LIFO: step 1 rolled back first, then step 0
        assert results[0]["seq"] == 1
        assert results[1]["seq"] == 0

    @pytest.mark.asyncio
    async def test_only_completed_steps_rolled_back(self, tx_log, tmp_path):
        """Failed and pending steps should not be rolled back."""
        m = tx_log.begin()
        tx_log.record_step(m, "file_write", {"path": str(tmp_path / "a.txt")})
        tx_log.record_step(m, "file_write", {"path": str(tmp_path / "b.txt")})
        tx_log.record_step(m, "file_write", {"path": str(tmp_path / "c.txt")})

        tx_log.mark_completed(m, 0)
        tx_log.mark_failed(m, 1, error="fail")
        # step 2 remains pending

        # Create file for step 0 only
        with open(str(tmp_path / "a.txt"), "w") as f:
            f.write("data")

        results = await tx_log.rollback(m)
        assert len(results) == 1  # only step 0
        assert results[0]["seq"] == 0


# ── Commit ───────────────────────────────────────────────────────────────


class TestCommit:
    def test_commit_marks_status(self, tx_log, tmp_path):
        target = str(tmp_path / "f.txt")
        with open(target, "w") as f:
            f.write("data")

        m = tx_log.begin()
        tx_log.record_step(m, "file_write", {"path": target})
        tx_log.mark_completed(m, 0)
        tx_log.commit(m)

        assert m.status == "committed"

    def test_commit_cleans_backups(self, tx_log, tmp_path):
        target = str(tmp_path / "f.txt")
        with open(target, "w") as f:
            f.write("data")

        m = tx_log.begin()
        tx_log.record_step(m, "file_write", {"path": target})
        # A backup was created
        backup = m.steps[0].backup_path
        assert backup and os.path.exists(backup)

        tx_log.commit(m)
        assert not os.path.exists(backup)
