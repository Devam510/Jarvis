"""Tests for jarvis.audio.barge_in — Barge-in detection."""

from unittest.mock import AsyncMock, patch

import pytest

from jarvis.audio.barge_in import BargeInDetector


class TestDebounce:
    """Safety: debounce prevents false triggers."""

    def test_single_frame_no_trigger(self):
        d = BargeInDetector(debounce_ms=300)
        d.start_monitoring()
        assert d.on_vad_frame(0.9) is False

    @patch("jarvis.audio.barge_in.time")
    def test_sustained_speech_triggers(self, mock_time):
        """Two frames separated by time > debounce should trigger."""
        mock_time.time.side_effect = [1000.0, 1001.0]  # 1s apart
        d = BargeInDetector(debounce_ms=100)
        d.start_monitoring()
        d.on_vad_frame(0.9)  # sets start @ t=1000
        assert d.on_vad_frame(0.9) is True  # elapsed=1000ms > 100ms

    def test_speech_then_silence_resets(self):
        d = BargeInDetector(debounce_ms=300)
        d.start_monitoring()
        d.on_vad_frame(0.9)  # start
        d.on_vad_frame(0.1)  # silence — resets
        # Next speech frame should start fresh (not trigger immediately)
        result = d.on_vad_frame(0.9)
        assert result is False  # just sets start again

    def test_below_threshold_no_trigger(self):
        d = BargeInDetector(vad_threshold=0.6, debounce_ms=300)
        d.start_monitoring()
        d.on_vad_frame(0.3)
        assert d.on_vad_frame(0.3) is False


class TestMonitoringState:
    def test_not_monitoring_no_trigger(self):
        d = BargeInDetector(debounce_ms=300)
        assert d.on_vad_frame(0.9) is False

    def test_start_stop(self):
        d = BargeInDetector()
        d.start_monitoring()
        assert d.is_monitoring
        d.stop_monitoring()
        assert not d.is_monitoring

    def test_no_trigger_after_stop(self):
        d = BargeInDetector(debounce_ms=300)
        d.start_monitoring()
        d.on_vad_frame(0.9)
        d.stop_monitoring()
        assert d.on_vad_frame(0.9) is False

    @patch("jarvis.audio.barge_in.time")
    def test_only_triggers_once(self, mock_time):
        mock_time.time.side_effect = [1000.0, 1001.0, 1002.0]
        d = BargeInDetector(debounce_ms=100)
        d.start_monitoring()
        d.on_vad_frame(0.9)  # sets start
        assert d.on_vad_frame(0.9) is True  # triggers
        # After trigger, should not trigger again
        assert d.on_vad_frame(0.9) is False


class TestStats:
    @patch("jarvis.audio.barge_in.time")
    def test_barge_in_counter(self, mock_time):
        mock_time.time.side_effect = [1000.0, 1001.0]
        d = BargeInDetector(debounce_ms=100)
        d.start_monitoring()
        d.on_vad_frame(0.9)
        d.on_vad_frame(0.9)
        assert d.total_barge_ins == 1


class TestEventEmission:
    @pytest.mark.asyncio
    async def test_emit_barge_in(self):
        bus = AsyncMock()
        d = BargeInDetector(event_bus=bus)
        d.total_barge_ins = 1
        await d.emit_barge_in()
        bus.emit.assert_called_once()
        assert bus.emit.call_args[0][0] == "barge_in.detected"

    @pytest.mark.asyncio
    async def test_emit_without_bus(self):
        d = BargeInDetector(event_bus=None)
        await d.emit_barge_in()  # no crash
