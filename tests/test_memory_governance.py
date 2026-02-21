"""
Tests for jarvis.memory.memory_governance — Phase 6D: Memory Governance.
"""

import asyncio
import pytest

from jarvis.memory.memory_governance import (
    PIIFilter,
    TTLEnforcer,
    ForgetHandler,
    MemoryGovernanceConfig,
)


# ── PII Filter Tests ────────────────────────────────────────────────────


class TestPIIFilter:
    def test_redacts_credit_card_visa(self):
        f = PIIFilter()
        text = "Payment card: 4111111111111111"
        result = f.sanitize(text)
        assert "4111111111111111" not in result
        assert "[REDACTED:CREDIT_CARD]" in result

    def test_redacts_credit_card_mastercard(self):
        f = PIIFilter()
        text = "Card: 5500000000000004"
        result = f.sanitize(text)
        assert "[REDACTED:CREDIT_CARD]" in result

    def test_redacts_ssn(self):
        f = PIIFilter()
        text = "SSN is 123-45-6789"
        result = f.sanitize(text)
        assert "123-45-6789" not in result
        assert "[REDACTED:SSN]" in result

    def test_redacts_phone_number(self):
        f = PIIFilter()
        text = "Call me at (555) 123-4567"
        result = f.sanitize(text)
        assert "[REDACTED:PHONE]" in result

    def test_redacts_email(self):
        f = PIIFilter()
        text = "Contact user@example.com for details"
        result = f.sanitize(text)
        assert "user@example.com" not in result
        assert "[REDACTED:EMAIL]" in result

    def test_redacts_ip_address(self):
        f = PIIFilter()
        text = "Server at 192.168.1.100"
        result = f.sanitize(text)
        assert "[REDACTED:IP_ADDRESS]" in result

    def test_multiple_pii_in_one_string(self):
        f = PIIFilter()
        text = "SSN: 123-45-6789, email: a@b.com, card: 4111111111111111"
        result = f.sanitize(text)
        assert "[REDACTED:SSN]" in result
        assert "[REDACTED:EMAIL]" in result
        assert "[REDACTED:CREDIT_CARD]" in result

    def test_no_pii_passes_through(self):
        f = PIIFilter()
        text = "The weather is nice today"
        assert f.sanitize(text) == text

    def test_empty_string(self):
        f = PIIFilter()
        assert f.sanitize("") == ""

    def test_disabled_skips_redaction(self):
        config = MemoryGovernanceConfig(pii_enabled=False)
        f = PIIFilter(config)
        text = "SSN: 123-45-6789"
        assert f.sanitize(text) == text

    def test_redaction_counter(self):
        f = PIIFilter()
        f.sanitize("SSN: 123-45-6789 and 987-65-4321")
        assert f.total_redactions == 2

    def test_custom_pattern(self):
        config = MemoryGovernanceConfig(
            custom_pii_patterns=[{"name": "LICENSE", "pattern": r"DL-\d{8}"}]
        )
        f = PIIFilter(config)
        result = f.sanitize("License: DL-12345678")
        assert "[REDACTED:LICENSE]" in result


# ── TTL Enforcer Tests ───────────────────────────────────────────────────


class TestTTLEnforcer:
    @pytest.mark.asyncio
    async def test_enforce_expires_old_entries(self):
        import time

        now = time.time()
        entries = [
            {"id": "old_1", "timestamp": now - 90000},  # ~25h ago
            {"id": "old_2", "timestamp": now - 100000},  # ~28h ago
            {"id": "new_1", "timestamp": now - 3600},  # 1h ago
        ]
        deleted_ids = []

        async def query_fn():
            return entries

        async def delete_fn(entry_id):
            deleted_ids.append(entry_id)

        enforcer = TTLEnforcer(
            ttl_hours=24.0,
            query_fn=query_fn,
            delete_fn=delete_fn,
        )
        count = await enforcer.enforce_once()
        assert count == 2
        assert "old_1" in deleted_ids
        assert "old_2" in deleted_ids
        assert "new_1" not in deleted_ids

    @pytest.mark.asyncio
    async def test_enforce_no_expired(self):
        import time

        now = time.time()

        async def query_fn():
            return [{"id": "x", "timestamp": now - 100}]

        async def delete_fn(entry_id):
            pass

        enforcer = TTLEnforcer(
            ttl_hours=24.0,
            query_fn=query_fn,
            delete_fn=delete_fn,
        )
        count = await enforcer.enforce_once()
        assert count == 0

    @pytest.mark.asyncio
    async def test_enforce_no_backend(self):
        enforcer = TTLEnforcer()
        count = await enforcer.enforce_once()
        assert count == 0

    @pytest.mark.asyncio
    async def test_total_expired_counter(self):
        import time

        now = time.time()

        async def query_fn():
            return [{"id": "x", "timestamp": now - 200000}]

        async def delete_fn(entry_id):
            pass

        enforcer = TTLEnforcer(
            ttl_hours=24.0,
            query_fn=query_fn,
            delete_fn=delete_fn,
        )
        await enforcer.enforce_once()
        assert enforcer.total_expired == 1

    @pytest.mark.asyncio
    async def test_start_and_stop(self):
        enforcer = TTLEnforcer(check_interval=0.1)
        await enforcer.start()
        assert enforcer._task is not None
        await enforcer.stop()
        assert enforcer._task is None


# ── Forget Handler Tests ─────────────────────────────────────────────────


class TestForgetHandler:
    @pytest.mark.asyncio
    async def test_forget_deletes_matches(self):
        deleted_ids = []

        async def search_fn(topic):
            return [{"id": "mem_1"}, {"id": "mem_2"}]

        async def delete_fn(entry_id):
            deleted_ids.append(entry_id)

        handler = ForgetHandler(search_fn=search_fn, delete_fn=delete_fn)
        count = await handler.forget("Project X")
        assert count == 2
        assert "mem_1" in deleted_ids
        assert "mem_2" in deleted_ids

    @pytest.mark.asyncio
    async def test_forget_no_matches(self):
        async def search_fn(topic):
            return []

        async def delete_fn(entry_id):
            pass

        handler = ForgetHandler(search_fn=search_fn, delete_fn=delete_fn)
        count = await handler.forget("nonexistent")
        assert count == 0

    @pytest.mark.asyncio
    async def test_forget_no_backend(self):
        handler = ForgetHandler()
        count = await handler.forget("anything")
        assert count == 0

    @pytest.mark.asyncio
    async def test_total_forgotten_counter(self):
        async def search_fn(topic):
            return [{"id": "a"}, {"id": "b"}, {"id": "c"}]

        async def delete_fn(entry_id):
            pass

        handler = ForgetHandler(search_fn=search_fn, delete_fn=delete_fn)
        await handler.forget("topic1")
        await handler.forget("topic2")
        assert handler.total_forgotten == 6
