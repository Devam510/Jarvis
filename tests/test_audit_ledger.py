"""
Tests for jarvis.observability.audit_ledger — Phase 6C: Cryptographic Audit Ledger.
"""

import json
import os

import pytest

from jarvis.observability.audit_ledger import AuditLedger


@pytest.fixture
def ledger(tmp_path):
    path = str(tmp_path / "test_audit.ledger")
    return AuditLedger(ledger_path=path, enabled=True)


# ── Recording ────────────────────────────────────────────────────────────


class TestRecording:
    @pytest.mark.asyncio
    async def test_record_returns_entry(self, ledger):
        entry = await ledger.record(
            action="tool_executed",
            tool="file_write",
            args={"path": "test.txt"},
            result_summary="success",
        )
        assert entry is not None
        assert entry.seq == 1
        assert entry.tool == "file_write"
        assert entry.action == "tool_executed"
        assert len(entry.hash) == 64  # SHA-256 hex
        assert len(entry.prev_hash) == 64

    @pytest.mark.asyncio
    async def test_sequential_numbering(self, ledger):
        e1 = await ledger.record(action="a", tool="t1", args={})
        e2 = await ledger.record(action="b", tool="t2", args={})
        e3 = await ledger.record(action="c", tool="t3", args={})
        assert e1.seq == 1
        assert e2.seq == 2
        assert e3.seq == 3

    @pytest.mark.asyncio
    async def test_hash_chain_links(self, ledger):
        e1 = await ledger.record(action="a", tool="t1", args={})
        e2 = await ledger.record(action="b", tool="t2", args={})
        assert e2.prev_hash == e1.hash

    @pytest.mark.asyncio
    async def test_genesis_prev_hash(self, ledger):
        entry = await ledger.record(action="a", tool="t", args={})
        assert entry.prev_hash == "0" * 64

    @pytest.mark.asyncio
    async def test_args_hashed_not_stored(self, ledger, tmp_path):
        """Raw args should NOT appear in the ledger file."""
        await ledger.record(
            action="a",
            tool="t",
            args={"secret": "my_password_123"},
        )
        path = str(tmp_path / "test_audit.ledger")
        with open(path) as f:
            content = f.read()
        assert "my_password_123" not in content

    @pytest.mark.asyncio
    async def test_disabled_returns_none(self, tmp_path):
        l = AuditLedger(
            ledger_path=str(tmp_path / "disabled.ledger"),
            enabled=False,
        )
        result = await l.record(action="a", tool="t", args={})
        assert result is None

    @pytest.mark.asyncio
    async def test_total_entries(self, ledger):
        assert ledger.total_entries == 0
        await ledger.record(action="a", tool="t", args={})
        await ledger.record(action="b", tool="t", args={})
        assert ledger.total_entries == 2


# ── Chain Verification ───────────────────────────────────────────────────


class TestVerification:
    @pytest.mark.asyncio
    async def test_verify_intact_chain(self, ledger):
        for i in range(10):
            await ledger.record(action=f"action_{i}", tool="t", args={"i": i})
        valid, errors = ledger.verify()
        assert valid is True
        assert errors == []

    def test_verify_empty_ledger(self, ledger):
        valid, errors = ledger.verify()
        assert valid is True

    @pytest.mark.asyncio
    async def test_verify_detects_tamper(self, ledger, tmp_path):
        for i in range(5):
            await ledger.record(action=f"a{i}", tool="t", args={})

        # Tamper with entry 3: change the hash
        path = str(tmp_path / "test_audit.ledger")
        with open(path) as f:
            lines = f.readlines()

        entry = json.loads(lines[2])
        entry["hash"] = "deadbeef" * 8
        lines[2] = json.dumps(entry) + "\n"

        with open(path, "w") as f:
            f.writelines(lines)

        # Re-create ledger to reload
        l2 = AuditLedger(ledger_path=path)
        valid, errors = l2.verify()
        assert valid is False
        assert len(errors) > 0


# ── Query ────────────────────────────────────────────────────────────────


class TestQuery:
    @pytest.mark.asyncio
    async def test_query_by_tool(self, ledger):
        await ledger.record(action="a", tool="file_write", args={})
        await ledger.record(action="a", tool="file_read", args={})
        await ledger.record(action="a", tool="file_write", args={})

        results = ledger.query(tool="file_write")
        assert len(results) == 2

    @pytest.mark.asyncio
    async def test_query_by_action(self, ledger):
        await ledger.record(action="tool_executed", tool="t", args={})
        await ledger.record(action="guardrail_violation", tool="t", args={})

        results = ledger.query(action="guardrail_violation")
        assert len(results) == 1

    @pytest.mark.asyncio
    async def test_query_with_limit(self, ledger):
        for i in range(20):
            await ledger.record(action="a", tool="t", args={})
        results = ledger.query(limit=5)
        assert len(results) == 5


# ── Resume ───────────────────────────────────────────────────────────────


class TestResume:
    @pytest.mark.asyncio
    async def test_resume_continues_chain(self, tmp_path):
        path = str(tmp_path / "resume.ledger")

        l1 = AuditLedger(ledger_path=path)
        e1 = await l1.record(action="a", tool="t", args={})
        e2 = await l1.record(action="b", tool="t", args={})

        # New instance resumes
        l2 = AuditLedger(ledger_path=path)
        e3 = await l2.record(action="c", tool="t", args={})
        assert e3.seq == 3
        assert e3.prev_hash == e2.hash

        # Full chain still valid
        valid, errors = l2.verify()
        assert valid is True
