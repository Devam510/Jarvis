"""
Tests for the event-driven voice confirmation loop (Phase 1A).

Covers: ConfirmationContext lifecycle, receive_confirmation guards,
classify_confirmation phrase matching, and timeout behaviour.
"""

import asyncio
import pytest
import uuid

from jarvis.risk.risk_engine import (
    RiskEngine,
    ConfirmationContext,
    classify_confirmation,
    compute_heuristic_score,
    score_to_tier,
)
from jarvis.core.event_bus import AsyncEventBus
from jarvis.utils.config import SafetyConfig
from jarvis.utils.enums import ConfirmStatus, RiskTier
from jarvis.utils.types import PlannedAction


# ── Helpers ──────────────────────────────────────────────────────────────────


class FakeTTS:
    """Records spoken messages instead of playing audio."""

    def __init__(self):
        self.spoken: list[str] = []

    async def speak(self, text: str):
        self.spoken.append(text)


def _make_engine(timeout: float = 2.0, tts=None) -> RiskEngine:
    config = SafetyConfig(confirmation_timeout_seconds=timeout)
    bus = AsyncEventBus()
    return RiskEngine(config, bus, tts=tts)


def _make_action(tool: str = "file_delete", **kwargs) -> PlannedAction:
    return PlannedAction(tool_name=tool, arguments=kwargs)


# ── classify_confirmation ────────────────────────────────────────────────────


class TestClassifyConfirmation:
    def test_confirm_yes(self):
        assert classify_confirmation("yes") == ConfirmStatus.CONFIRMED

    def test_confirm_go_ahead(self):
        assert classify_confirmation("go ahead") == ConfirmStatus.CONFIRMED

    def test_confirm_sure(self):
        assert classify_confirmation("sure do it") == ConfirmStatus.CONFIRMED

    def test_deny_no(self):
        assert classify_confirmation("no") == ConfirmStatus.DENIED

    def test_deny_cancel(self):
        assert classify_confirmation("cancel") == ConfirmStatus.DENIED

    def test_deny_stop(self):
        assert classify_confirmation("stop abort") == ConfirmStatus.DENIED

    def test_ambiguous_mixed(self):
        assert classify_confirmation("yes no maybe") == ConfirmStatus.AMBIGUOUS

    def test_ambiguous_unrelated(self):
        assert classify_confirmation("tell me a joke") == ConfirmStatus.AMBIGUOUS


# ── ConfirmationContext ──────────────────────────────────────────────────────


class TestConfirmationContext:
    def test_defaults(self):
        action = _make_action(path="/tmp/test")
        ctx = ConfirmationContext(
            action_id="abc-123",
            action=action,
            description="delete /tmp/test",
        )
        assert ctx.result is None
        assert ctx.expired is False
        assert not ctx.event.is_set()


# ── receive_confirmation guards ──────────────────────────────────────────────


class TestReceiveConfirmation:
    def test_no_pending_request(self):
        engine = _make_engine()
        assert engine.receive_confirmation("fake-id", True) is False

    def test_wrong_action_id(self):
        engine = _make_engine()
        action = _make_action(path="/tmp/x")
        engine._pending_confirm = ConfirmationContext(
            action_id="correct-id",
            action=action,
            description="delete /tmp/x",
        )
        assert engine.receive_confirmation("wrong-id", True) is False
        assert engine._pending_confirm.result is None  # unchanged

    def test_expired_request(self):
        engine = _make_engine()
        action = _make_action(path="/tmp/x")
        ctx = ConfirmationContext(
            action_id="id-1",
            action=action,
            description="delete /tmp/x",
        )
        ctx.expired = True
        engine._pending_confirm = ctx
        assert engine.receive_confirmation("id-1", True) is False

    def test_valid_confirmation_confirmed(self):
        engine = _make_engine()
        action = _make_action(path="/tmp/x")
        ctx = ConfirmationContext(
            action_id="id-1",
            action=action,
            description="delete /tmp/x",
        )
        engine._pending_confirm = ctx
        assert engine.receive_confirmation("id-1", True) is True
        assert ctx.result is True
        assert ctx.event.is_set()

    def test_valid_confirmation_denied(self):
        engine = _make_engine()
        action = _make_action(path="/tmp/x")
        ctx = ConfirmationContext(
            action_id="id-2",
            action=action,
            description="delete /tmp/x",
        )
        engine._pending_confirm = ctx
        assert engine.receive_confirmation("id-2", False) is True
        assert ctx.result is False
        assert ctx.event.is_set()


# ── _confirm_action integration ──────────────────────────────────────────────


class TestConfirmActionIntegration:
    @pytest.mark.asyncio
    async def test_confirm_action_approved(self):
        tts = FakeTTS()
        engine = _make_engine(timeout=5.0, tts=tts)
        action = _make_action(path="/tmp/test")

        # Run _confirm_action in background, then deliver confirmation
        async def _confirm_after_delay():
            await asyncio.sleep(0.1)
            ctx = engine._pending_confirm
            assert ctx is not None
            engine.receive_confirmation(ctx.action_id, True)

        confirm_task = asyncio.create_task(_confirm_after_delay())
        result = await engine._confirm_action(action)
        await confirm_task

        assert result is True
        assert engine._pending_confirm is None

    @pytest.mark.asyncio
    async def test_confirm_action_denied(self):
        tts = FakeTTS()
        engine = _make_engine(timeout=5.0, tts=tts)
        action = _make_action(path="/tmp/test")

        async def _deny_after_delay():
            await asyncio.sleep(0.1)
            ctx = engine._pending_confirm
            assert ctx is not None
            engine.receive_confirmation(ctx.action_id, False)

        deny_task = asyncio.create_task(_deny_after_delay())
        result = await engine._confirm_action(action)
        await deny_task

        assert result is False

    @pytest.mark.asyncio
    async def test_confirm_action_timeout(self):
        tts = FakeTTS()
        engine = _make_engine(timeout=0.3, tts=tts)
        action = _make_action(path="/tmp/test")

        result = await engine._confirm_action(action)

        assert result is False
        # TTS should have spoken the timeout message
        assert any("timed out" in msg.lower() for msg in tts.spoken)

    @pytest.mark.asyncio
    async def test_late_confirmation_after_timeout(self):
        tts = FakeTTS()
        engine = _make_engine(timeout=0.2, tts=tts)
        action = _make_action(path="/tmp/test")

        # Let it timeout
        result = await engine._confirm_action(action)
        assert result is False

        # Now try to confirm with a stale action_id — should be rejected
        assert engine.receive_confirmation("stale-id", True) is False


# ── Heuristic scoring (sanity check) ────────────────────────────────────────


class TestHeuristicScoring:
    def test_file_delete_is_tier3(self):
        score, _ = compute_heuristic_score(
            "file_delete", {"path": "/tmp/test", "recursive": True}
        )
        assert score_to_tier(score) == RiskTier.TIER_3

    def test_file_read_is_tier1(self):
        score, _ = compute_heuristic_score("file_read", {"path": "test.txt"})
        assert score_to_tier(score) == RiskTier.TIER_1
