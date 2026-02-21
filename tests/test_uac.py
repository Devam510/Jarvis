"""Tests for jarvis.execution.uac_elevation — UAC elevation controls."""

import pytest
from unittest.mock import AsyncMock

from jarvis.execution.uac_elevation import (
    UACElevator,
    ElevationResult,
    DEFAULT_ALLOWED_COMMANDS,
)


class TestWhitelist:
    """Safety: command whitelist enforcement."""

    def test_default_whitelist(self):
        uac = UACElevator()
        assert uac.is_allowed("netsh")
        assert uac.is_allowed("sfc")
        assert uac.is_allowed("reg")

    def test_not_in_whitelist(self):
        uac = UACElevator()
        assert not uac.is_allowed("rm")
        assert not uac.is_allowed("format")
        assert not uac.is_allowed("del")

    def test_case_insensitive(self):
        uac = UACElevator()
        assert uac.is_allowed("NETSH")
        assert uac.is_allowed("Reg")

    def test_custom_whitelist(self):
        uac = UACElevator(allowed_commands=["custom_tool"])
        assert uac.is_allowed("custom_tool")
        assert not uac.is_allowed("netsh")


class TestBlocking:
    @pytest.mark.asyncio
    async def test_blocked_command(self):
        uac = UACElevator()
        result = await uac.run_elevated("rm", "-rf /")
        assert result.status == "blocked"
        assert "not in elevation whitelist" in result.error
        assert uac.total_denied == 1

    @pytest.mark.asyncio
    async def test_blocked_format(self):
        uac = UACElevator()
        result = await uac.run_elevated("format", "C:")
        assert result.status == "blocked"


class TestAuditLogging:
    @pytest.mark.asyncio
    async def test_blocked_emits_audit(self):
        bus = AsyncMock()
        uac = UACElevator(event_bus=bus)
        await uac.run_elevated("rm", "-rf /")
        bus.emit.assert_called()
        # Should have logged elevation_denied
        calls = [c[0][0] for c in bus.emit.call_args_list]
        assert "uac.elevation_denied" in calls

    @pytest.mark.asyncio
    async def test_without_bus_no_crash(self):
        uac = UACElevator(event_bus=None)
        result = await uac.run_elevated("rm", "-rf /")
        assert result.status == "blocked"


class TestRegOps:
    @pytest.mark.asyncio
    async def test_reg_query_uses_reg(self):
        uac = UACElevator()
        result = await uac.reg_query("HKLM\\SOFTWARE\\Test")
        # reg is in whitelist so it proceeds to execution
        assert result.command == "reg"

    @pytest.mark.asyncio
    async def test_reg_set_uses_reg(self):
        uac = UACElevator()
        result = await uac.reg_set("HKLM\\SOFTWARE\\Test", "key", "val")
        assert result.command == "reg"


class TestStats:
    @pytest.mark.asyncio
    async def test_denial_stats(self):
        uac = UACElevator()
        await uac.run_elevated("hack", "")
        await uac.run_elevated("exploit", "")
        assert uac.total_denied == 2
