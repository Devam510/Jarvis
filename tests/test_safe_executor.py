"""
tests/test_safe_executor.py — Tests for AuditedPythonRunner.

Covers:
  - Basic code execution
  - Audit hook blocks dangerous calls
  - Import whitelist enforcement
  - Timeout enforcement
  - Script execution
  - Package name sanitization
"""

import asyncio
import os
import sys
import tempfile

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from jarvis.execution.safe_executor import AuditedPythonRunner


@pytest.fixture
def runner():
    return AuditedPythonRunner(timeout_seconds=10, memory_limit_mb=128)


# ── Basic Execution ──────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_basic_code_runs(runner):
    """Simple print statement should succeed."""
    result = await runner.run_code('print("Hello from sandbox")')
    assert result["status"] == "success"
    assert "Hello from sandbox" in result["stdout"]


@pytest.mark.asyncio
async def test_math_allowed(runner):
    """Math import should be allowed by default whitelist."""
    result = await runner.run_code("import math\nprint(math.pi)")
    assert result["status"] == "success"
    assert "3.14" in result["stdout"]


@pytest.mark.asyncio
async def test_json_allowed(runner):
    """json import should be allowed by default whitelist."""
    result = await runner.run_code('import json\nprint(json.dumps({"key": "value"}))')
    assert result["status"] == "success"
    assert "key" in result["stdout"]


# ── Audit Hook Blocks ────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_os_system_blocked(runner):
    """os.system() should be blocked by audit hook."""
    result = await runner.run_code('import os\nos.system("echo pwned")')
    assert result["status"] == "failed"
    assert "BLOCKED" in result["stderr"] or "audit" in result["stderr"].lower()


@pytest.mark.asyncio
async def test_subprocess_blocked(runner):
    """subprocess should be blocked by audit hook."""
    result = await runner.run_code(
        'import subprocess\nsubprocess.run(["echo", "pwned"])'
    )
    assert result["status"] == "failed"


# ── Timeout ──────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_timeout_enforced():
    """Code that runs forever should be killed by timeout."""
    runner = AuditedPythonRunner(timeout_seconds=2)
    result = await runner.run_code("import time\nwhile True: time.sleep(0.1)")
    assert result["status"] == "timeout"


# ── Script Execution ─────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_run_script(runner):
    """run_script should execute a .py file."""
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False, encoding="utf-8"
    ) as f:
        f.write('print("Script executed")')
        f.flush()
        path = f.name

    try:
        result = await runner.run_script(path)
        assert result["status"] == "success"
        assert "Script executed" in result["stdout"]
    finally:
        os.unlink(path)


@pytest.mark.asyncio
async def test_run_script_not_found(runner):
    """run_script with nonexistent path should fail gracefully."""
    result = await runner.run_script("/nonexistent/fake_script.py")
    assert result["status"] == "failed"
    assert "not found" in result["stderr"].lower()


@pytest.mark.asyncio
async def test_run_script_wrong_extension(runner):
    """run_script with non-.py extension should fail."""
    # Create a real file with wrong extension so it hits the extension check
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".sh", delete=False, encoding="utf-8"
    ) as f:
        f.write("echo test")
        path = f.name

    try:
        result = await runner.run_script(path)
        assert result["status"] == "failed"
        assert ".py" in result["stderr"]
    finally:
        os.unlink(path)


# ── Package Name Sanitization ────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_invalid_package_name(runner):
    """Package names with shell characters should be rejected."""
    result = await runner.install_package("numpy; rm -rf /")
    assert result["status"] == "failed"
    assert "Invalid" in result["stderr"]
