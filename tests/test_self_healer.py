"""Tests for jarvis.observability.self_healer — Self-Healing Supervisor."""

import asyncio
import time
from unittest.mock import AsyncMock

import pytest

from jarvis.observability.self_healer import (
    SelfHealer,
    _MAX_RESTARTS_PER_HOUR,
)


class TestRestartBudget:
    """Safety: max 3 restarts per component per hour."""

    def test_restarts_remaining_fresh(self):
        sh = SelfHealer()
        assert sh.restarts_remaining("stt") == _MAX_RESTARTS_PER_HOUR

    def test_restarts_decrease(self):
        sh = SelfHealer()
        sh._restart_history["stt"] = [time.time()]
        assert sh.restarts_remaining("stt") == _MAX_RESTARTS_PER_HOUR - 1

    def test_old_restarts_expire(self):
        sh = SelfHealer()
        old = time.time() - 7200  # 2 hours ago
        sh._restart_history["stt"] = [old, old, old]
        assert sh.restarts_remaining("stt") == _MAX_RESTARTS_PER_HOUR


class TestMetricReporting:
    def test_report_healthy(self):
        sh = SelfHealer(stt_latency_threshold=2000)
        sh.report_metric("stt", 500)
        assert sh._metrics["stt"].is_healthy

    def test_report_degraded(self):
        sh = SelfHealer(stt_latency_threshold=2000)
        sh.report_metric("stt", 3000)
        assert not sh._metrics["stt"].is_healthy

    def test_report_error(self):
        sh = SelfHealer()
        sh.report_error("stt")
        sh.report_error("stt")
        assert sh._metrics["stt_errors"].current_value == 2


class TestHealing:
    @pytest.mark.asyncio
    async def test_healing_calls_handler(self):
        handler = AsyncMock()
        bus = AsyncMock()
        sh = SelfHealer(event_bus=bus)
        sh.register_handler("stt", handler)
        sh.report_metric("stt", 5000)  # degraded

        await sh._check_health()
        handler.assert_called_once()
        assert sh.total_restarts == 1

    @pytest.mark.asyncio
    async def test_healing_respects_budget(self):
        handler = AsyncMock()
        sh = SelfHealer()
        sh.register_handler("stt", handler)
        sh.report_metric("stt", 5000)

        # Fill budget
        sh._restart_history["stt"] = [time.time()] * _MAX_RESTARTS_PER_HOUR

        await sh._check_health()
        handler.assert_not_called()  # budget exhausted

    @pytest.mark.asyncio
    async def test_healing_logs_failure(self):
        handler = AsyncMock(side_effect=Exception("restart failed"))
        sh = SelfHealer()
        sh.register_handler("stt", handler)
        sh.report_metric("stt", 5000)

        await sh._check_health()
        assert sh.total_restart_failures == 1

    @pytest.mark.asyncio
    async def test_emit_before_act(self):
        handler = AsyncMock()
        bus = AsyncMock()
        sh = SelfHealer(event_bus=bus)
        sh.register_handler("stt", handler)
        sh.report_metric("stt", 5000)

        await sh._check_health()
        # Event bus should be called BEFORE handler
        assert bus.emit.call_count >= 1
        first_event = bus.emit.call_args_list[0][0][0]
        assert first_event == "subsystem.degraded"


class TestLifecycle:
    @pytest.mark.asyncio
    async def test_start_stop(self):
        sh = SelfHealer(check_interval=100)
        await sh.start()
        assert sh._running
        await sh.stop()
        assert not sh._running

    @pytest.mark.asyncio
    async def test_double_start(self):
        sh = SelfHealer(check_interval=100)
        await sh.start()
        await sh.start()  # no duplicate task
        await sh.stop()


class TestHealthSummary:
    def test_summary_empty(self):
        sh = SelfHealer()
        assert sh.get_health_summary() == {}

    def test_summary_with_metrics(self):
        sh = SelfHealer()
        sh.report_metric("stt", 500)
        summary = sh.get_health_summary()
        assert "stt" in summary
        assert summary["stt"]["healthy"]


class TestHealingLog:
    def test_empty_log(self):
        sh = SelfHealer()
        assert sh.get_healing_log() == []

    @pytest.mark.asyncio
    async def test_log_records_actions(self):
        handler = AsyncMock()
        sh = SelfHealer()
        sh.register_handler("stt", handler)
        sh.report_metric("stt", 5000)

        await sh._check_health()
        log = sh.get_healing_log()
        assert len(log) == 1
        assert log[0]["component"] == "stt"
        assert log[0]["success"] is True


class TestConfig:
    def test_config_defaults(self):
        from jarvis.utils.config import SelfHealerConfig

        c = SelfHealerConfig()
        assert c.enabled is False
        assert c.check_interval == 30.0
        assert c.stt_latency_threshold_ms == 2000.0
