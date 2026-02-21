"""Tests for jarvis.execution.process_graph — System State Graph."""

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from jarvis.execution.process_graph import (
    ProcessGraph,
    ProcessInfo,
    Anomaly,
    MIN_INTERVAL_S,
)


# ── Collection Interval Enforcement ──────────────────────────────────────


class TestIntervalEnforcement:
    """Safety control: collection frequency capped."""

    def test_interval_enforced_minimum(self):
        pg = ProcessGraph(collection_interval=1.0)
        assert pg._interval >= MIN_INTERVAL_S

    def test_interval_above_minimum_kept(self):
        pg = ProcessGraph(collection_interval=30.0)
        assert pg._interval == 30.0


# ── Exception Safety ─────────────────────────────────────────────────────


class TestExceptionSafety:
    """Safety control: all psutil calls wrapped in per-call try/except."""

    def test_collect_without_psutil(self):
        pg = ProcessGraph()
        with patch.dict("sys.modules", {"psutil": None}):
            # Should not crash, returns None when psutil unavailable
            result = pg._collect()
            assert result is None

    def test_collect_with_mock_psutil(self):
        mock_psutil = MagicMock()
        proc_mock = MagicMock()
        proc_mock.info = {
            "pid": 1234,
            "name": "python.exe",
            "cpu_percent": 25.0,
            "memory_info": MagicMock(rss=100 * 1024 * 1024),
            "status": "running",
            "ppid": 1,
        }
        mock_psutil.process_iter.return_value = [proc_mock]

        with patch.dict("sys.modules", {"psutil": mock_psutil}):
            pg = ProcessGraph()
            result = pg._collect()
            pg._process_map = result  # V2-02: assign on caller side
            assert 1234 in pg._process_map
            assert pg._process_map[1234].name == "python.exe"
            assert pg._process_map[1234].memory_mb == pytest.approx(100.0, abs=1)

    def test_collect_handles_process_exception(self):
        """Per-process exception safety — bad process doesn't crash collection."""
        mock_psutil = MagicMock()
        good_proc = MagicMock()
        good_proc.info = {
            "pid": 1,
            "name": "good",
            "cpu_percent": 10,
            "memory_info": MagicMock(rss=50 * 1024 * 1024),
            "status": "running",
            "ppid": 0,
        }
        bad_proc = MagicMock()
        bad_proc.info.__getitem__ = MagicMock(side_effect=Exception("access denied"))

        mock_psutil.process_iter.return_value = [good_proc, bad_proc]

        with patch.dict("sys.modules", {"psutil": mock_psutil}):
            pg = ProcessGraph()
            result = pg._collect()
            pg._process_map = result  # V2-02: assign on caller side
            # Good process should still be captured
            assert 1 in pg._process_map


# ── Resource Hog Detection ───────────────────────────────────────────────


class TestResourceHogs:
    def test_find_hogs_by_cpu(self):
        pg = ProcessGraph()
        pg._process_map = {
            1: ProcessInfo(pid=1, name="chrome", cpu_percent=80, memory_mb=500),
            2: ProcessInfo(pid=2, name="idle", cpu_percent=1, memory_mb=10),
        }
        hogs = pg.find_resource_hogs(cpu_threshold=50, mem_mb_threshold=9999)
        assert len(hogs) == 1
        assert hogs[0].name == "chrome"

    def test_find_hogs_by_memory(self):
        pg = ProcessGraph()
        pg._process_map = {
            1: ProcessInfo(pid=1, name="vscode", cpu_percent=5, memory_mb=3000),
            2: ProcessInfo(pid=2, name="notepad", cpu_percent=0, memory_mb=10),
        }
        hogs = pg.find_resource_hogs(cpu_threshold=99, mem_mb_threshold=2048)
        assert len(hogs) == 1
        assert hogs[0].name == "vscode"

    def test_find_hogs_sorted_descending(self):
        pg = ProcessGraph()
        pg._process_map = {
            1: ProcessInfo(pid=1, name="low", cpu_percent=10, memory_mb=100),
            2: ProcessInfo(pid=2, name="high", cpu_percent=90, memory_mb=5000),
        }
        hogs = pg.find_resource_hogs(cpu_threshold=5, mem_mb_threshold=50)
        assert hogs[0].name == "high"

    def test_find_hogs_empty_map(self):
        pg = ProcessGraph()
        hogs = pg.find_resource_hogs()
        assert hogs == []


# ── Anomaly Detection ────────────────────────────────────────────────────


class TestAnomalyDetection:
    """Safety control: anomalies are emit-only, never auto-act."""

    def test_cpu_anomaly_requires_sustained(self):
        mock_psutil = MagicMock()
        mock_psutil.cpu_percent.return_value = 95.0
        mock_psutil.virtual_memory.return_value = MagicMock(percent=50)
        mock_psutil.disk_usage.return_value = MagicMock(percent=50)

        with patch.dict("sys.modules", {"psutil": mock_psutil}):
            pg = ProcessGraph(cpu_threshold=80, sustained_seconds=0)
            # First call — sets anomaly_start but doesn't emit yet
            anomalies = pg._detect_anomalies()
            # With sustained_seconds=0, second call should emit
            anomalies = pg._detect_anomalies()
            cpu_anomalies = [a for a in anomalies if a.type == "cpu_sustained"]
            assert len(cpu_anomalies) >= 1

    def test_memory_anomaly(self):
        mock_psutil = MagicMock()
        mock_psutil.cpu_percent.return_value = 10.0
        mock_psutil.virtual_memory.return_value = MagicMock(percent=92)
        mock_psutil.disk_usage.return_value = MagicMock(percent=50)

        with patch.dict("sys.modules", {"psutil": mock_psutil}):
            pg = ProcessGraph(mem_threshold=85, sustained_seconds=0)
            pg._detect_anomalies()  # set start
            anomalies = pg._detect_anomalies()
            mem_anomalies = [a for a in anomalies if a.type == "memory_high"]
            assert len(mem_anomalies) >= 1

    def test_disk_anomaly_immediate(self):
        mock_psutil = MagicMock()
        mock_psutil.cpu_percent.return_value = 10
        mock_psutil.virtual_memory.return_value = MagicMock(percent=50)
        mock_psutil.disk_usage.return_value = MagicMock(percent=95)

        with patch.dict("sys.modules", {"psutil": mock_psutil}):
            pg = ProcessGraph(disk_threshold=90)
            anomalies = pg._detect_anomalies()
            disk_anomalies = [a for a in anomalies if a.type == "disk_full"]
            assert len(disk_anomalies) == 1

    def test_no_anomaly_when_healthy(self):
        mock_psutil = MagicMock()
        mock_psutil.cpu_percent.return_value = 30
        mock_psutil.virtual_memory.return_value = MagicMock(percent=50)
        mock_psutil.disk_usage.return_value = MagicMock(percent=40)

        with patch.dict("sys.modules", {"psutil": mock_psutil}):
            pg = ProcessGraph()
            anomalies = pg._detect_anomalies()
            assert len(anomalies) == 0


# ── Emit Only (Never Auto-Act) ───────────────────────────────────────────


class TestEmitOnly:
    """Verify anomalies are emitted via EventBus — never auto-remediated."""

    @pytest.mark.asyncio
    async def test_emit_anomaly_calls_event_bus(self):
        bus = AsyncMock()
        pg = ProcessGraph(event_bus=bus)
        anomaly = Anomaly(
            type="test",
            metric="cpu",
            value=99,
            threshold=80,
            message="test anomaly",
            timestamp=time.time(),
        )
        await pg._emit_anomaly(anomaly)
        bus.emit.assert_called_once()
        call_args = bus.emit.call_args
        assert call_args[0][0] == "system.anomaly_detected"
        assert call_args[0][1]["type"] == "test"
        assert pg.total_anomalies_emitted == 1

    @pytest.mark.asyncio
    async def test_emit_anomaly_no_event_bus(self):
        """Should not crash without event bus."""
        pg = ProcessGraph(event_bus=None)
        anomaly = Anomaly(
            type="test",
            metric="cpu",
            value=99,
            threshold=80,
            message="test",
            timestamp=time.time(),
        )
        await pg._emit_anomaly(anomaly)  # no crash
        assert pg.total_anomalies_emitted == 1


# ── Natural Summary ──────────────────────────────────────────────────────


class TestNaturalSummary:
    def test_summary_without_psutil(self):
        pg = ProcessGraph()
        with patch.dict("sys.modules", {"psutil": None}):
            summary = pg.get_natural_summary()
            assert "unavailable" in summary.lower()

    def test_summary_with_mock_psutil(self):
        mock_psutil = MagicMock()
        mock_psutil.cpu_percent.return_value = 45.0
        mock_psutil.virtual_memory.return_value = MagicMock(
            percent=60, used=8 * 1024**3, total=16 * 1024**3
        )
        mock_psutil.disk_usage.return_value = MagicMock(
            percent=55, used=200 * 1024**3, total=500 * 1024**3
        )

        with patch.dict("sys.modules", {"psutil": mock_psutil}):
            pg = ProcessGraph()
            summary = pg.get_natural_summary()
            assert "CPU:" in summary
            assert "RAM:" in summary
            assert "Disk:" in summary


# ── Lifecycle ─────────────────────────────────────────────────────────────


class TestLifecycle:
    @pytest.mark.asyncio
    async def test_start_stop(self):
        pg = ProcessGraph()
        await pg.start()
        assert pg._running
        await pg.stop()
        assert not pg._running

    @pytest.mark.asyncio
    async def test_double_start(self):
        pg = ProcessGraph()
        await pg.start()
        await pg.start()  # should not create duplicate tasks
        await pg.stop()

    def test_build_graph(self):
        pg = ProcessGraph()
        pg._process_map = {1: ProcessInfo(pid=1, name="test")}
        graph = pg.build_graph()
        assert 1 in graph
        assert graph[1].name == "test"
