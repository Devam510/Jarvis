"""Tests for jarvis.cognition.code_copilot — Code Co-Pilot guardrails."""

import asyncio
import os
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from jarvis.cognition.code_copilot import (
    CodeCoPilot,
    CoPilotResult,
    ProjectContext,
    _ABSOLUTE_MAX_RETRIES,
    _ABSOLUTE_MAX_TIMEOUT,
)


# ── Retry Hard Cap ────────────────────────────────────────────────────────


class TestRetryCap:
    """Safety control: max retries hard-capped at 3."""

    def test_max_retries_default(self):
        cp = CodeCoPilot()
        assert cp._max_retries == 3

    def test_max_retries_cannot_exceed_ceiling(self):
        cp = CodeCoPilot(max_retries=100)
        assert cp._max_retries == _ABSOLUTE_MAX_RETRIES

    def test_max_retries_can_be_lower(self):
        cp = CodeCoPilot(max_retries=1)
        assert cp._max_retries == 1


# ── Timeout Cap ───────────────────────────────────────────────────────────


class TestTimeoutCap:
    """Safety control: per-run timeout capped."""

    def test_timeout_default(self):
        cp = CodeCoPilot()
        assert cp._run_timeout == 30.0

    def test_timeout_capped_at_absolute(self):
        cp = CodeCoPilot(run_timeout=999)
        assert cp._run_timeout == _ABSOLUTE_MAX_TIMEOUT


# ── Token Budget ──────────────────────────────────────────────────────────


class TestTokenBudget:
    """Safety control: token budget per session."""

    @pytest.mark.asyncio
    async def test_token_budget_exceeded(self):
        cp = CodeCoPilot(token_budget=10)
        cp._tokens_used = 10  # simulate exhaustion
        result = await cp.generate_code("test")
        assert result.status == "budget_exceeded"
        assert "Token budget" in result.error

    @pytest.mark.asyncio
    async def test_tool_call_budget_exceeded(self):
        cp = CodeCoPilot(tool_call_budget=1)
        cp._tool_calls_used = 1  # simulate exhaustion
        result = await cp.generate_code("test")
        assert result.status == "budget_exceeded"
        assert "Tool-call budget" in result.error

    def test_reset_session(self):
        cp = CodeCoPilot()
        cp._tokens_used = 100
        cp._tool_calls_used = 3
        cp.reset_session()
        assert cp._tokens_used == 0
        assert cp._tool_calls_used == 0


# ── Project Scanning ──────────────────────────────────────────────────────


class TestProjectScan:
    """Project awareness: scan directory for context."""

    def test_scan_empty_dir(self, tmp_path):
        cp = CodeCoPilot()
        ctx = cp.scan_project(str(tmp_path))
        assert ctx.root == str(tmp_path)
        assert not ctx.has_git
        assert not ctx.has_requirements

    def test_scan_python_project(self, tmp_path):
        (tmp_path / ".git").mkdir()
        (tmp_path / "requirements.txt").write_text("flask>=2.0\nrequests\n")
        (tmp_path / "README.md").write_text("# My Project\nA test project.")
        (tmp_path / "main.py").write_text("print('hello')")

        cp = CodeCoPilot()
        ctx = cp.scan_project(str(tmp_path))
        assert ctx.has_git
        assert ctx.has_requirements
        assert ctx.language == "python"
        assert "flask" in ctx.dependencies
        assert "requests" in ctx.dependencies
        assert "My Project" in ctx.readme_summary

    def test_scan_javascript_project(self, tmp_path):
        (tmp_path / "package.json").write_text('{"name": "test"}')

        cp = CodeCoPilot()
        ctx = cp.scan_project(str(tmp_path))
        assert ctx.language == "javascript"
        assert ctx.has_package_json

    def test_scan_nonexistent_dir(self):
        cp = CodeCoPilot()
        ctx = cp.scan_project("/fake/nonexistent/path")
        assert ctx.root == "/fake/nonexistent/path"
        assert not ctx.has_git

    def test_scan_caps_structure(self, tmp_path):
        for i in range(50):
            (tmp_path / f"file_{i}.txt").write_text("data")

        cp = CodeCoPilot()
        ctx = cp.scan_project(str(tmp_path))
        assert len(ctx.structure) <= 30


# ── Code Generation ──────────────────────────────────────────────────────


class TestCodeGeneration:
    """Code generation without LLM — returns stub code."""

    @pytest.mark.asyncio
    async def test_generate_without_llm(self):
        cp = CodeCoPilot()
        result = await cp.generate_code("print hello world")
        assert result.status == "success"
        assert "Generated for:" in result.code
        assert cp.total_generations == 1

    @pytest.mark.asyncio
    async def test_generate_tracks_tool_calls(self):
        cp = CodeCoPilot()
        await cp.generate_code("test 1")
        await cp.generate_code("test 2")
        assert cp._tool_calls_used == 2

    @pytest.mark.asyncio
    async def test_generate_with_context(self, tmp_path):
        (tmp_path / "requirements.txt").write_text("numpy\n")
        cp = CodeCoPilot()
        ctx = cp.scan_project(str(tmp_path))
        result = await cp.generate_code("create array", ctx)
        assert result.status == "success"


# ── Run and Fix ──────────────────────────────────────────────────────────


class TestRunAndFix:
    """Run code and auto-fix on error."""

    @pytest.mark.asyncio
    async def test_run_successful_code(self, tmp_path):
        save_path = str(tmp_path / "test_script.py")
        code = "print('hello world')"

        cp = CodeCoPilot(run_timeout=10)
        result = await cp.run_and_fix(code, save_path)
        assert result.status == "success"
        assert "hello world" in result.output
        assert result.retries_used == 0

    @pytest.mark.asyncio
    async def test_run_failing_code_retries(self, tmp_path):
        save_path = str(tmp_path / "bad_script.py")
        code = "import nonexistent_module_xyz"

        cp = CodeCoPilot(max_retries=2, run_timeout=10)
        result = await cp.run_and_fix(code, save_path)
        # Should attempt retries (but without LLM, the fix is still bad)
        assert result.retries_used <= 3
        assert result.duration_ms > 0

    @pytest.mark.asyncio
    async def test_run_timeout(self, tmp_path):
        save_path = str(tmp_path / "slow_script.py")
        code = "import time; time.sleep(100)"

        cp = CodeCoPilot(max_retries=0, run_timeout=1)
        result = await cp.run_and_fix(code, save_path)
        assert result.status == "failed"
        assert "timed out" in result.error

    @pytest.mark.asyncio
    async def test_run_budget_check(self, tmp_path):
        save_path = str(tmp_path / "test.py")
        cp = CodeCoPilot(tool_call_budget=0)
        result = await cp.run_and_fix("print(1)", save_path)
        assert result.status == "budget_exceeded"


# ── Code Extraction ──────────────────────────────────────────────────────


class TestCodeExtraction:
    def test_extract_markdown_block(self):
        response = "Here is the code:\n```python\nprint('hello')\n```\nDone."
        code = CodeCoPilot._extract_code_block(response)
        assert code == "print('hello')"

    def test_extract_plain_code(self):
        response = "print('hello')"
        code = CodeCoPilot._extract_code_block(response)
        assert code == "print('hello')"


# ── Network Blocking ─────────────────────────────────────────────────────


class TestNetworkBlocking:
    """Safety control: network blocked by default."""

    def test_network_blocked_default(self):
        cp = CodeCoPilot()
        assert cp._allow_network is False

    def test_network_allowed_explicit(self):
        cp = CodeCoPilot(allow_network=True)
        assert cp._allow_network is True

    def test_overwrite_requires_confirm(self):
        cp = CodeCoPilot()
        assert cp._require_confirm_overwrite is True
