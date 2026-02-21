"""Tests for jarvis.ui — GUI Dashboard (headless, no display required)."""

import pytest
from unittest.mock import MagicMock, patch


# ── Bridge (No-Op when PySide6 missing) ───────────────────────────────────


class TestBridgeNoop:
    """GUI Isolation: bridge degrades gracefully without PySide6."""

    def test_bridge_import_without_pyside6(self):
        """Should import without error even without PySide6."""
        from jarvis.ui.bridge import GUIBridge

        bridge = GUIBridge()
        # All methods should be no-op
        bridge.emit_state("IDLE")
        bridge.emit_log({"tool": "test"})
        bridge.emit_metrics({"cpu": 50})
        bridge.emit_confirmation({"action": "test"})
        bridge.emit_anomaly({"type": "test"})


# ── Theme ─────────────────────────────────────────────────────────────────


class TestTheme:
    def test_dark_theme_exists(self):
        from jarvis.ui.theme import DARK_THEME, LIGHT_THEME

        assert "background-color" in DARK_THEME
        assert len(DARK_THEME) > 100

    def test_light_theme_exists(self):
        from jarvis.ui.theme import LIGHT_THEME

        assert "background-color" in LIGHT_THEME

    def test_risk_tier_colors(self):
        from jarvis.ui.theme import RISK_TIER_COLORS

        assert "TIER_1" in RISK_TIER_COLORS
        assert "TIER_2" in RISK_TIER_COLORS
        assert "TIER_3" in RISK_TIER_COLORS

    def test_state_colors(self):
        from jarvis.ui.theme import STATE_COLORS

        assert "IDLE" in STATE_COLORS
        assert "LISTENING" in STATE_COLORS
        assert "PROCESSING" in STATE_COLORS


# ── Dashboard Crash Isolation ─────────────────────────────────────────────


class TestDashboardIsolation:
    """GUI Isolation: dashboard handles missing PySide6 gracefully."""

    def test_dashboard_create_without_pyside6(self):
        """Should return False when PySide6 not available."""
        from jarvis.ui.dashboard import JarvisDashboard, PYSIDE6_AVAILABLE

        d = JarvisDashboard()
        if not PYSIDE6_AVAILABLE:
            result = d.create()
            assert result is False

    def test_dashboard_close_without_create(self):
        """Should not crash if close called before create."""
        from jarvis.ui.dashboard import JarvisDashboard

        d = JarvisDashboard()
        d.close()  # no crash

    def test_dashboard_bridge_none_before_create(self):
        from jarvis.ui.dashboard import JarvisDashboard

        d = JarvisDashboard()
        assert d.bridge is None

    def test_dashboard_mode_stored(self):
        from jarvis.ui.dashboard import JarvisDashboard

        d = JarvisDashboard(mode="admin", theme="light")
        assert d._mode == "admin"
        assert d._theme == "light"


# ── Config Dataclasses ────────────────────────────────────────────────────


class TestGUIConfig:
    """Capability gating: config defaults."""

    def test_gui_config_disabled_by_default(self):
        from jarvis.utils.config import GUIConfig

        c = GUIConfig()
        assert c.enabled is False

    def test_gui_config_modes(self):
        from jarvis.utils.config import GUIConfig

        c = GUIConfig(mode="admin")
        assert c.mode == "admin"


class TestFastSearchConfig:
    def test_disabled_by_default(self):
        from jarvis.utils.config import FastSearchConfig

        c = FastSearchConfig()
        assert c.enabled is False

    def test_timeout_default(self):
        from jarvis.utils.config import FastSearchConfig

        c = FastSearchConfig()
        assert c.timeout_seconds == 3.0

    def test_absolute_max(self):
        from jarvis.utils.config import FastSearchConfig

        c = FastSearchConfig()
        assert c.absolute_max_results == 200


class TestCoPilotConfig:
    def test_disabled_by_default(self):
        from jarvis.utils.config import CoPilotConfig

        c = CoPilotConfig()
        assert c.enabled is False

    def test_max_retries_default(self):
        from jarvis.utils.config import CoPilotConfig

        c = CoPilotConfig()
        assert c.max_retries == 3

    def test_network_blocked_default(self):
        from jarvis.utils.config import CoPilotConfig

        c = CoPilotConfig()
        assert c.allow_network is False

    def test_overwrite_requires_confirm(self):
        from jarvis.utils.config import CoPilotConfig

        c = CoPilotConfig()
        assert c.require_confirm_overwrite is True


class TestProcessGraphConfig:
    def test_disabled_by_default(self):
        from jarvis.utils.config import ProcessGraphConfig

        c = ProcessGraphConfig()
        assert c.enabled is False

    def test_min_interval(self):
        from jarvis.utils.config import ProcessGraphConfig

        c = ProcessGraphConfig()
        assert c.min_interval == 5.0
