"""
jarvis.ui.dashboard — Main PySide6 GUI Dashboard.

Safety:
  - Runs in QThread — never blocks EventBus
  - All data via Qt signals (thread-safe bridge)
  - Crash isolated: exception → log + continue headless
  - Can be started/stopped independently of agent
"""

from __future__ import annotations

import logging
import time
from typing import Any, Optional

logger = logging.getLogger(__name__)

try:
    from PySide6.QtCore import Qt, QTimer
    from PySide6.QtWidgets import (
        QApplication,
        QMainWindow,
        QWidget,
        QVBoxLayout,
        QHBoxLayout,
        QLabel,
        QTextEdit,
        QProgressBar,
        QPushButton,
        QGroupBox,
        QSplitter,
        QFrame,
    )

    from jarvis.ui.theme import DARK_THEME, LIGHT_THEME, STATE_COLORS, RISK_TIER_COLORS
    from jarvis.ui.bridge import GUIBridge

    PYSIDE6_AVAILABLE = True
except ImportError:
    PYSIDE6_AVAILABLE = False
    logger.info("PySide6 not installed — GUI disabled")


class JarvisDashboard:
    """Main GUI controller — creates and manages all panels.

    Safety invariants:
      - Never blocks the async event loop
      - All EventBus data arrives via GUIBridge signals
      - Crash → log error + agent continues headless
    """

    def __init__(
        self,
        mode: str = "developer",
        theme: str = "dark",
        width: int = 1200,
        height: int = 800,
    ):
        self._mode = mode
        self._theme = theme
        self._width = width
        self._height = height
        self._app: Optional[Any] = None
        self._window: Optional[Any] = None
        self._bridge: Optional[GUIBridge] = None
        self._running = False

    @property
    def bridge(self) -> Optional[GUIBridge]:
        return self._bridge

    def create(self) -> bool:
        """Create the GUI (must be called from GUI thread).

        Returns True if GUI was created, False if PySide6 not available.
        """
        if not PYSIDE6_AVAILABLE:
            return False

        try:
            # Create QApplication if not exists
            self._app = QApplication.instance()
            if self._app is None:
                self._app = QApplication([])

            self._bridge = GUIBridge()
            self._window = _MainWindow(
                bridge=self._bridge,
                mode=self._mode,
                theme=self._theme,
            )
            self._window.resize(self._width, self._height)
            self._window.setWindowTitle("Jarvis Control Interface")

            # Connect bridge signals
            self._bridge.state_changed.connect(self._window.on_state_changed)
            self._bridge.log_entry.connect(self._window.on_log_entry)
            self._bridge.system_metrics.connect(self._window.on_system_metrics)
            self._bridge.confirmation_required.connect(
                self._window.on_confirmation_required
            )
            self._bridge.agent_stopped.connect(self._window.on_agent_stopped)

            self._running = True
            logger.info("GUI dashboard created (mode=%s)", self._mode)
            return True

        except Exception as e:
            logger.error("GUI creation failed: %s", e)
            return False

    def show(self):
        """Show the window."""
        if self._window:
            try:
                self._window.show()
            except Exception as e:
                logger.error("GUI show failed: %s", e)

    def close(self):
        """Close the GUI gracefully."""
        self._running = False
        if self._window:
            try:
                self._window.close()
            except Exception:
                pass


if PYSIDE6_AVAILABLE:

    class _MainWindow(QMainWindow):
        """Internal main window with multi-panel layout."""

        def __init__(self, bridge: GUIBridge, mode: str, theme: str):
            super().__init__()
            self._bridge = bridge
            self._mode = mode

            # Apply theme
            stylesheet = DARK_THEME if theme == "dark" else LIGHT_THEME
            self.setStyleSheet(stylesheet)

            # Central widget
            central = QWidget()
            self.setCentralWidget(central)
            main_layout = QVBoxLayout(central)
            main_layout.setContentsMargins(12, 12, 12, 12)
            main_layout.setSpacing(8)

            # ── Top: Status Panel ─────────────────────────────────────
            self._status_label = QLabel("● IDLE")
            self._status_label.setObjectName("status_label")
            self._status_label.setAlignment(Qt.AlignCenter)

            self._tool_label = QLabel("No active tool")
            self._tool_label.setObjectName("active_tool_label")
            self._tool_label.setAlignment(Qt.AlignCenter)

            status_group = QGroupBox("Agent Status")
            status_layout = QVBoxLayout(status_group)
            status_layout.addWidget(self._status_label)
            status_layout.addWidget(self._tool_label)
            main_layout.addWidget(status_group)

            # ── Middle: Splitter (Log | System) ───────────────────────
            splitter = QSplitter(Qt.Horizontal)

            # Log Panel
            log_group = QGroupBox("Execution Log")
            log_layout = QVBoxLayout(log_group)
            self._log_view = QTextEdit()
            self._log_view.setObjectName("log_view")
            self._log_view.setReadOnly(True)
            log_layout.addWidget(self._log_view)
            splitter.addWidget(log_group)

            # System Panel (developer/admin only)
            if self._mode in ("developer", "admin"):
                sys_group = QGroupBox("System Monitor")
                sys_layout = QVBoxLayout(sys_group)

                self._cpu_bar = QProgressBar()
                self._cpu_bar.setFormat("CPU: %p%")
                sys_layout.addWidget(QLabel("CPU"))
                sys_layout.addWidget(self._cpu_bar)

                self._ram_bar = QProgressBar()
                self._ram_bar.setFormat("RAM: %p%")
                sys_layout.addWidget(QLabel("RAM"))
                sys_layout.addWidget(self._ram_bar)

                self._disk_bar = QProgressBar()
                self._disk_bar.setFormat("Disk: %p%")
                sys_layout.addWidget(QLabel("Disk"))
                sys_layout.addWidget(self._disk_bar)

                sys_layout.addStretch()
                splitter.addWidget(sys_group)

            splitter.setStretchFactor(0, 3)
            if splitter.count() > 1:
                splitter.setStretchFactor(1, 1)

            main_layout.addWidget(splitter, stretch=1)

            # ── Bottom: Kill Switch + Confirmation ────────────────────
            bottom_layout = QHBoxLayout()

            self._confirm_btn = QPushButton("✓ Confirm")
            self._confirm_btn.setObjectName("confirm_btn")
            self._confirm_btn.setVisible(False)
            self._confirm_btn.clicked.connect(self._on_confirm)

            self._abort_btn = QPushButton("✗ Abort")
            self._abort_btn.setObjectName("abort_btn")
            self._abort_btn.setVisible(False)
            self._abort_btn.clicked.connect(self._on_abort)

            self._kill_btn = QPushButton("⬛ STOP JARVIS")
            self._kill_btn.setObjectName("kill_btn")
            self._kill_btn.clicked.connect(self._on_kill)

            bottom_layout.addWidget(self._confirm_btn)
            bottom_layout.addWidget(self._abort_btn)
            bottom_layout.addStretch()
            bottom_layout.addWidget(self._kill_btn)

            main_layout.addLayout(bottom_layout)

            # Pending confirmation state
            self._pending_confirmation: Optional[dict] = None

        # ── Signal Handlers ───────────────────────────────────────────

        def on_state_changed(self, state: str):
            """Update status display."""
            color = STATE_COLORS.get(state, "#888888")
            self._status_label.setText(f"● {state}")
            self._status_label.setStyleSheet(f"color: {color};")

        def on_log_entry(self, entry: dict):
            """Append a log entry to the log view."""
            ts = time.strftime("%H:%M:%S")
            tool = entry.get("tool", "?")
            risk = entry.get("risk", "TIER_1")
            result = entry.get("result", "")
            color = RISK_TIER_COLORS.get(risk, "#cccccc")

            html = (
                f'<span style="color:#666">{ts}</span> '
                f'<span style="color:{color}; font-weight:bold">[{risk}]</span> '
                f'<span style="color:#aaa">{tool}</span>: '
                f"{result[:200]}"
            )
            self._log_view.append(html)

            # Update active tool
            self._tool_label.setText(f"Last: {tool}")

        def on_system_metrics(self, metrics: dict):
            """Update system gauges."""
            if not hasattr(self, "_cpu_bar"):
                return

            cpu = int(metrics.get("cpu_percent", 0))
            ram = int(metrics.get("memory_percent", 0))
            disk = int(metrics.get("disk_percent", 0))

            self._cpu_bar.setValue(cpu)
            self._ram_bar.setValue(ram)
            self._disk_bar.setValue(disk)

            # Color-code based on severity
            for bar, val in [
                (self._cpu_bar, cpu),
                (self._ram_bar, ram),
                (self._disk_bar, disk),
            ]:
                if val >= 90:
                    bar.setObjectName("critical")
                elif val >= 70:
                    bar.setObjectName("warn")
                else:
                    bar.setObjectName("")
                bar.style().unpolish(bar)
                bar.style().polish(bar)

        def on_confirmation_required(self, data: dict):
            """Show confirm/abort buttons for TIER 3 action."""
            self._pending_confirmation = data
            action = data.get("action", "Unknown action")
            risk = data.get("risk", "TIER_3")

            self._confirm_btn.setVisible(True)
            self._abort_btn.setVisible(True)
            self._confirm_btn.setText(f"✓ Confirm: {action}")

            # Flash status
            self._status_label.setText(f"⚠ AWAITING CONFIRMATION: {action}")
            self._status_label.setStyleSheet(f"color: {RISK_TIER_COLORS['TIER_3']};")

        def on_agent_stopped(self):
            """Handle agent shutdown notification."""
            self._status_label.setText("● STOPPED")
            self._status_label.setStyleSheet("color: #666666;")
            self._tool_label.setText("Agent has stopped")

        def _on_confirm(self):
            """User confirmed via GUI button."""
            self._confirm_btn.setVisible(False)
            self._abort_btn.setVisible(False)
            self._bridge.confirmation_resolved.emit(True)
            self._pending_confirmation = None

        def _on_abort(self):
            """User aborted via GUI button."""
            self._confirm_btn.setVisible(False)
            self._abort_btn.setVisible(False)
            self._bridge.confirmation_resolved.emit(False)
            self._pending_confirmation = None

        def _on_kill(self):
            """Emergency kill switch — emits stop signal."""
            self._bridge.agent_stopped.emit()
            self._status_label.setText("● KILL SIGNAL SENT")
            self._status_label.setStyleSheet("color: #ff0000;")
