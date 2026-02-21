"""
jarvis.ui.bridge — Thread-safe EventBus ↔ Qt signal bridge.

Safety: GUI runs in QThread — bridge uses Qt signals (thread-safe)
to pass data from async EventBus to GUI thread. Never blocks EventBus.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

# Try importing PySide6 — gracefully degrade if not installed
try:
    from PySide6.QtCore import QObject, Signal

    class GUIBridge(QObject):
        """Bridge between async EventBus and Qt GUI thread.

        Signals are thread-safe in Qt — they queue across threads
        automatically. The EventBus handler calls bridge.emit_*() from
        the async thread, and Qt delivers the signal in the GUI thread.
        """

        # Signals for GUI updates
        state_changed = Signal(str)  # new state name
        log_entry = Signal(dict)  # {timestamp, tool, risk, result, ...}
        system_metrics = Signal(dict)  # {cpu, ram, disk, ...}
        confirmation_required = Signal(dict)  # {action, risk, correlation_id}
        confirmation_resolved = Signal(bool)  # True=confirmed, False=aborted
        anomaly_detected = Signal(dict)  # {type, metric, value, ...}
        agent_stopped = Signal()  # agent shutdown notification

        def __init__(self):
            super().__init__()
            self._connected = False

        def emit_state(self, state: str):
            """Emit state change to GUI (thread-safe)."""
            try:
                self.state_changed.emit(state)
            except Exception as e:
                logger.debug("Bridge state emit failed: %s", e)

        def emit_log(self, entry: dict):
            """Emit log entry to GUI (thread-safe)."""
            try:
                self.log_entry.emit(entry)
            except Exception as e:
                logger.debug("Bridge log emit failed: %s", e)

        def emit_metrics(self, metrics: dict):
            """Emit system metrics to GUI (thread-safe)."""
            try:
                self.system_metrics.emit(metrics)
            except Exception as e:
                logger.debug("Bridge metrics emit failed: %s", e)

        def emit_confirmation(self, data: dict):
            """Emit confirmation request to GUI (thread-safe)."""
            try:
                self.confirmation_required.emit(data)
            except Exception as e:
                logger.debug("Bridge confirmation emit failed: %s", e)

        def emit_anomaly(self, data: dict):
            """Emit anomaly alert to GUI (thread-safe)."""
            try:
                self.anomaly_detected.emit(data)
            except Exception as e:
                logger.debug("Bridge anomaly emit failed: %s", e)

except ImportError:
    # PySide6 not installed — provide a no-op bridge
    class GUIBridge:  # type: ignore[no-redef]
        """No-op bridge when PySide6 is not installed."""

        def __init__(self):
            self._connected = False

        def emit_state(self, state: str):
            pass

        def emit_log(self, entry: dict):
            pass

        def emit_metrics(self, metrics: dict):
            pass

        def emit_confirmation(self, data: dict):
            pass

        def emit_anomaly(self, data: dict):
            pass
