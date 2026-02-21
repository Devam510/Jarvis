"""
jarvis.ui.theme — Dark cinematic stylesheet for PySide6 dashboard.
"""

DARK_THEME = """
QMainWindow, QWidget {
    background-color: #0a0a0f;
    color: #e0e0e4;
    font-family: 'Segoe UI', 'Inter', sans-serif;
    font-size: 13px;
}

/* Status Panel */
QLabel#status_label {
    font-size: 28px;
    font-weight: 700;
    color: #00d4ff;
    padding: 10px;
}

QLabel#active_tool_label {
    font-size: 14px;
    color: #a0a0b0;
    padding: 4px 10px;
}

/* Log Panel */
QTextEdit#log_view {
    background-color: #12121a;
    border: 1px solid #2a2a3a;
    border-radius: 8px;
    padding: 8px;
    font-family: 'Cascadia Code', 'Consolas', monospace;
    font-size: 12px;
    color: #c8c8d0;
    selection-background-color: #1a3a5a;
}

/* System Panel Gauges */
QProgressBar {
    border: 1px solid #2a2a3a;
    border-radius: 6px;
    background-color: #12121a;
    text-align: center;
    color: #e0e0e4;
    height: 22px;
}

QProgressBar::chunk {
    border-radius: 5px;
    background: qlineargradient(
        x1:0, y1:0, x2:1, y2:0,
        stop:0 #00d4ff, stop:1 #0088ff
    );
}

QProgressBar#warn::chunk {
    background: qlineargradient(
        x1:0, y1:0, x2:1, y2:0,
        stop:0 #ff8800, stop:1 #ff4400
    );
}

QProgressBar#critical::chunk {
    background: qlineargradient(
        x1:0, y1:0, x2:1, y2:0,
        stop:0 #ff2200, stop:1 #cc0000
    );
}

/* Buttons */
QPushButton {
    background-color: #1a1a2a;
    border: 1px solid #3a3a4a;
    border-radius: 6px;
    padding: 8px 20px;
    color: #e0e0e4;
    font-weight: 600;
}

QPushButton:hover {
    background-color: #2a2a3a;
    border-color: #00d4ff;
}

QPushButton#confirm_btn {
    background-color: #004400;
    border-color: #00aa00;
    color: #00ff44;
}

QPushButton#confirm_btn:hover {
    background-color: #006600;
}

QPushButton#abort_btn {
    background-color: #440000;
    border-color: #aa0000;
    color: #ff4444;
}

QPushButton#abort_btn:hover {
    background-color: #660000;
}

QPushButton#kill_btn {
    background-color: #660000;
    border: 2px solid #ff0000;
    color: #ff2222;
    font-size: 16px;
    font-weight: 800;
    padding: 12px 30px;
    border-radius: 10px;
}

QPushButton#kill_btn:hover {
    background-color: #880000;
}

/* Group Boxes */
QGroupBox {
    border: 1px solid #2a2a3a;
    border-radius: 8px;
    margin-top: 14px;
    padding-top: 18px;
    font-weight: 600;
    color: #8888aa;
}

QGroupBox::title {
    subcontrol-origin: margin;
    left: 12px;
    padding: 0 6px;
}

/* Scroll Bars */
QScrollBar:vertical {
    background: #0a0a0f;
    width: 8px;
    border-radius: 4px;
}

QScrollBar::handle:vertical {
    background: #3a3a4a;
    border-radius: 4px;
    min-height: 30px;
}

QScrollBar::handle:vertical:hover {
    background: #00d4ff;
}

QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
    height: 0;
}
"""

LIGHT_THEME = """
QMainWindow, QWidget {
    background-color: #f5f5fa;
    color: #1a1a2a;
    font-family: 'Segoe UI', 'Inter', sans-serif;
    font-size: 13px;
}
"""

RISK_TIER_COLORS = {
    "TIER_1": "#00cc44",  # green — safe
    "TIER_2": "#ffaa00",  # amber — sensitive
    "TIER_3": "#ff2222",  # red — critical
}

STATE_COLORS = {
    "IDLE": "#4488aa",
    "LISTENING": "#00d4ff",
    "PROCESSING": "#ffaa00",
    "SPEAKING": "#00cc44",
    "AWAITING_CONFIRM": "#ff2222",
    "EXECUTING": "#ff8800",
}
