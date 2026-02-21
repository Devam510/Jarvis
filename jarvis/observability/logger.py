"""
jarvis.observability.logger — Structured JSON logging and trace collection.
"""

from __future__ import annotations

import json
import logging
import os
import time
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Optional, Any

from jarvis.utils.config import ObservabilityConfig


class JSONFormatter(logging.Formatter):
    """Emit each log record as a single JSON line."""

    def format(self, record: logging.LogRecord) -> str:
        entry = {
            "timestamp": datetime.utcfromtimestamp(record.created).isoformat() + "Z",
            "level": record.levelname,
            "component": record.name,
            "message": record.getMessage(),
        }
        if hasattr(record, "correlation_id"):
            entry["correlation_id"] = record.correlation_id
        if record.exc_info and record.exc_info[0]:
            entry["exception"] = self.formatException(record.exc_info)
        return json.dumps(entry, default=str)


def setup_logging(config: ObservabilityConfig):
    """Configure root logger with JSON file handler and console handler."""
    log_dir = Path(config.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    # File handler — structured JSON
    file_handler = logging.FileHandler(str(log_dir / config.log_file), encoding="utf-8")
    file_handler.setFormatter(JSONFormatter())
    file_handler.setLevel(logging.DEBUG)

    # Console handler — human readable
    console_handler = logging.StreamHandler()
    console_fmt = logging.Formatter(
        "%(asctime)s │ %(levelname)-7s │ %(name)-28s │ %(message)s",
        datefmt="%H:%M:%S",
    )
    console_handler.setFormatter(console_fmt)
    console_handler.setLevel(logging.INFO)

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    root.handlers.clear()
    root.addHandler(file_handler)
    root.addHandler(console_handler)


class TraceCollector:
    """Collects execution traces per correlation_id."""

    def __init__(self, trace_dir: str = "logs/traces"):
        self.trace_dir = Path(trace_dir)
        self.trace_dir.mkdir(parents=True, exist_ok=True)
        self._traces: dict[str, list[dict]] = {}

    def start_stage(
        self, correlation_id: str, stage_name: str, input_summary: str = ""
    ):
        if correlation_id not in self._traces:
            self._traces[correlation_id] = []
        self._traces[correlation_id].append(
            {
                "stage": stage_name,
                "started_at": time.time(),
                "input": input_summary[:500],
                "status": "in_progress",
            }
        )

    def complete_stage(
        self, correlation_id: str, output_summary: str = "", status: str = "success"
    ):
        if correlation_id in self._traces and self._traces[correlation_id]:
            stage = self._traces[correlation_id][-1]
            stage["completed_at"] = time.time()
            stage["output"] = output_summary[:500]
            stage["status"] = status
            stage["duration_ms"] = (stage["completed_at"] - stage["started_at"]) * 1000

    def save(self, correlation_id: str):
        if correlation_id in self._traces:
            path = self.trace_dir / f"{correlation_id}.json"
            with open(path, "w") as f:
                json.dump(self._traces[correlation_id], f, indent=2, default=str)
            del self._traces[correlation_id]


class RiskAuditLog:
    """Immutable append-only log of all risk classification decisions."""

    def __init__(self, log_path: str = "logs/risk_audit.jsonl"):
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def record(
        self,
        correlation_id: str,
        tool_name: str,
        arguments: dict,
        heuristic_score: float,
        llm_risk_score: float,
        tier: str,
        authorization: str,
        reasoning: str = "",
        execution_status: Optional[str] = None,
    ):
        entry = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "correlation_id": correlation_id,
            "tool_name": tool_name,
            "arguments_hash": hashlib.sha256(
                json.dumps(arguments, sort_keys=True).encode()
            ).hexdigest(),
            "heuristic_score": heuristic_score,
            "llm_risk_score": llm_risk_score,
            "tier": tier,
            "authorization": authorization,
            "reasoning": reasoning,
            "execution_status": execution_status,
        }
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, default=str) + "\n")
