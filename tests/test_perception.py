"""
tests.test_perception — Phase 4 tests for ScreenWatcher + OCR.

Tests cover safety requirements S1, S2, S4, S6, S7.
"""

import asyncio
import re
import time

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from jarvis.core.event_bus import AsyncEventBus
from jarvis.perception.ocr_analyzer import OCRAnalyzer
from jarvis.perception.screen_watcher import ScreenWatcher
from jarvis.utils.config import PerceptionConfig
from jarvis.utils.types import ScreenContext


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def config():
    return PerceptionConfig(
        enabled=True,
        capture_fps=10.0,  # Fast for tests
        change_threshold=0.15,
        max_ocr_per_minute=10,
        cpu_disable_threshold=95.0,
        privacy_apps=["KeePass", "1Password", "Signal"],
        privacy_title_patterns=[
            r"(?i)bank",
            r"(?i)password",
            r"(?i)auth",
        ],
    )


@pytest.fixture
def disabled_config():
    return PerceptionConfig(enabled=False)


@pytest.fixture
def event_bus():
    return AsyncEventBus()


@pytest.fixture
def watcher(config, event_bus):
    return ScreenWatcher(config, event_bus)


# ══════════════════════════════════════════════════════════════════════════════
# S1: Opt-In Toggle
# ══════════════════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_s1_opt_in_disabled_by_default():
    """S1: PerceptionConfig defaults to enabled=False."""
    cfg = PerceptionConfig()
    assert cfg.enabled is False


@pytest.mark.asyncio
async def test_s1_watcher_does_not_start_when_disabled(disabled_config, event_bus):
    """S1: ScreenWatcher.run() returns immediately when disabled."""
    watcher = ScreenWatcher(disabled_config, event_bus)
    # Should return without blocking
    await asyncio.wait_for(watcher.run(), timeout=2.0)
    assert not watcher.is_running


@pytest.mark.asyncio
async def test_s1_logs_activation(config, event_bus, caplog):
    """S1: Activation is logged at WARNING level."""
    watcher = ScreenWatcher(config, event_bus)

    # Mock capture to just return immediately (clean exit breaks loop)
    async def _mock_capture_loop():
        pass

    with patch.object(watcher, "_capture_loop", _mock_capture_loop):
        import logging

        with caplog.at_level(logging.WARNING):
            await asyncio.wait_for(watcher.run(), timeout=2.0)

    assert any("ACTIVATED" in r.message for r in caplog.records)


# ══════════════════════════════════════════════════════════════════════════════
# S2: Hard Privacy Blackout
# ══════════════════════════════════════════════════════════════════════════════


def test_s2_privacy_app_blackout(watcher):
    """S2: App in privacy list → blacked out."""
    assert watcher._is_blacked_out("KeePass", "KeePass - Passwords") is True
    assert watcher._is_blacked_out("1Password", "1Password — Vault") is True
    assert watcher._is_blacked_out("Signal", "Signal Desktop") is True


def test_s2_privacy_title_regex(watcher):
    """S2: Window title matching regex → blacked out."""
    assert watcher._is_blacked_out("Chrome", "My Bank - Online Banking") is True
    assert watcher._is_blacked_out("Firefox", "Change Password | Settings") is True
    assert watcher._is_blacked_out("Edge", "Two-Factor Auth - Google") is True


def test_s2_normal_app_not_blacked_out(watcher):
    """S2: Normal apps are not blacked out."""
    assert watcher._is_blacked_out("VS Code", "main.py - Jarvis") is False
    assert watcher._is_blacked_out("Chrome", "GitHub - Pull Requests") is False


# ══════════════════════════════════════════════════════════════════════════════
# S4: Resource Budget
# ══════════════════════════════════════════════════════════════════════════════


def test_s4_ocr_rate_limit(watcher):
    """S4: OCR rate limit allows calls up to max, then blocks."""
    watcher._ocr_minute_start = time.time()
    for i in range(10):
        assert watcher._ocr_rate_ok() is True
        watcher._ocr_calls_this_minute += 1
    # 11th call should be blocked
    assert watcher._ocr_rate_ok() is False


def test_s4_ocr_rate_resets_after_minute(watcher):
    """S4: OCR rate limit resets after 60 seconds."""
    watcher._ocr_calls_this_minute = 10
    watcher._ocr_minute_start = time.time() - 61.0  # 61 seconds ago
    assert watcher._ocr_rate_ok() is True
    assert watcher._ocr_calls_this_minute == 0  # Counter reset


def test_s4_disk_usage_check(watcher, tmp_path):
    """S4: Disk usage check respects max_screenshot_disk_mb."""
    watcher._config.screenshot_dir = str(tmp_path)
    watcher._config.max_screenshot_disk_mb = 1  # 1MB cap

    # Empty dir → OK
    assert watcher._disk_usage_ok() is True

    # Create 2MB file → should exceed
    big_file = tmp_path / "big.png"
    big_file.write_bytes(b"\x00" * (2 * 1024 * 1024))
    assert watcher._disk_usage_ok() is False


# ══════════════════════════════════════════════════════════════════════════════
# S6: Context Immutability
# ══════════════════════════════════════════════════════════════════════════════


def test_s6_screen_context_is_frozen():
    """S6: ScreenContext is a frozen dataclass — cannot mutate after creation."""
    ctx = ScreenContext(
        active_app="VS Code",
        window_title="main.py",
        visible_text="hello world",
    )
    with pytest.raises(AttributeError):
        ctx.active_app = "Chrome"  # type: ignore

    with pytest.raises(AttributeError):
        ctx.visible_text = "hacked"  # type: ignore


# ══════════════════════════════════════════════════════════════════════════════
# S7: Crash Isolation
# ══════════════════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_s7_crash_restart_once(config, event_bus):
    """S7: First crash → restart. Second crash → disable permanently."""
    watcher = ScreenWatcher(config, event_bus)
    call_count = 0

    async def _crashing_loop():
        nonlocal call_count
        call_count += 1
        raise RuntimeError(f"Simulated crash #{call_count}")

    with patch.object(watcher, "_capture_loop", _crashing_loop):
        await asyncio.wait_for(watcher.run(), timeout=5.0)

    # Should have attempted twice (initial + 1 restart)
    assert call_count == 2
    assert watcher.is_permanently_disabled is True


@pytest.mark.asyncio
async def test_s7_crash_emits_disabled_event(config, event_bus):
    """S7: Permanent disable emits perception.disabled event."""
    watcher = ScreenWatcher(config, event_bus)
    disabled_events = []

    async def _capture(event):
        disabled_events.append(event)

    event_bus.subscribe("perception.disabled", _capture)

    async def _crashing_loop():
        raise RuntimeError("boom")

    with patch.object(watcher, "_capture_loop", _crashing_loop):
        await asyncio.wait_for(watcher.run(), timeout=5.0)

    assert len(disabled_events) == 1
    assert "repeated_crash" in disabled_events[0]["reason"]


# ══════════════════════════════════════════════════════════════════════════════
# OCR Analyzer
# ══════════════════════════════════════════════════════════════════════════════


def test_ocr_error_detection_traceback():
    """OCR analyzer detects Python tracebacks."""
    ocr = OCRAnalyzer()
    text = "some output\nTraceback (most recent call last):\n  File 'test.py'"
    errors = ocr.detect_errors(text)
    assert len(errors) >= 1
    assert any("Traceback" in e for e in errors)


def test_ocr_error_detection_js_errors():
    """OCR analyzer detects JavaScript errors."""
    ocr = OCRAnalyzer()
    text = "console output: Uncaught TypeError: cannot read property"
    errors = ocr.detect_errors(text)
    assert len(errors) >= 1


def test_ocr_error_detection_no_errors():
    """OCR analyzer returns empty list for clean text."""
    ocr = OCRAnalyzer()
    text = "Hello world, everything is working fine!"
    errors = ocr.detect_errors(text)
    assert errors == []


def test_ocr_error_detection_empty():
    """OCR analyzer handles empty text."""
    ocr = OCRAnalyzer()
    assert ocr.detect_errors("") == []


# ══════════════════════════════════════════════════════════════════════════════
# Change Detection
# ══════════════════════════════════════════════════════════════════════════════


def test_change_detection_first_frame(watcher):
    """First frame always reports maximum change."""
    change = watcher._compute_change(b"\x00" * 100)
    assert change == 1.0


def test_change_detection_identical_frames(watcher):
    """Identical frames report zero change."""
    data = b"\x42" * 100
    watcher._compute_change(data)  # First frame
    change = watcher._compute_change(data)  # Same frame
    assert change == 0.0


def test_change_detection_different_frames(watcher):
    """Different frames report non-zero change."""
    watcher._compute_change(b"\x00" * 100)
    change = watcher._compute_change(b"\xff" * 100)
    assert change > 0.0
