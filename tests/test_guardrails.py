"""
Tests for jarvis.safety.guardrails — Phase 6A: Deterministic Guardrails.
"""

import time
import pytest
from jarvis.safety.guardrails import (
    GuardrailEngine,
    GuardrailConfig,
    GuardrailViolation,
)


# ── Helpers ──────────────────────────────────────────────────────────────


def make_engine(**overrides) -> GuardrailEngine:
    cfg = GuardrailConfig(**overrides)
    return GuardrailEngine(cfg)


# ── Protected Path Tests ─────────────────────────────────────────────────


class TestProtectedPaths:
    def test_blocks_system32_config(self):
        engine = make_engine()
        with pytest.raises(GuardrailViolation, match="PROTECTED_PATH"):
            engine.check("file_write", {"path": r"C:\Windows\System32\config\SAM"})

    def test_blocks_ssh_keys(self):
        engine = make_engine()
        with pytest.raises(GuardrailViolation, match="PROTECTED_PATH"):
            engine.check("file_read", {"path": r"C:\Users\test\.ssh\id_rsa"})

    def test_blocks_boot_ini(self):
        engine = make_engine()
        with pytest.raises(GuardrailViolation, match="PROTECTED_PATH"):
            engine.check("file_delete", {"path": r"C:\boot.ini"})

    def test_allows_normal_path(self):
        engine = make_engine()
        engine.check("file_write", {"path": r"D:\Projects\test.txt"})

    def test_extra_protected_paths(self):
        engine = make_engine(extra_protected_paths=[r"D:\\secrets"])
        with pytest.raises(GuardrailViolation, match="PROTECTED_PATH"):
            engine.check("file_read", {"path": r"D:\secrets\api_key.txt"})

    def test_checks_both_source_and_destination(self):
        engine = make_engine()
        with pytest.raises(GuardrailViolation, match="PROTECTED_PATH"):
            engine.check(
                "file_move",
                {
                    "source": r"D:\temp.txt",
                    "destination": r"C:\Windows\System32\config\evil",
                },
            )


# ── Rate Limit Tests ─────────────────────────────────────────────────────


class TestRateLimits:
    def test_file_delete_rate_limit(self):
        engine = make_engine(max_file_deletes_per_min=3)
        for i in range(3):
            engine.check("file_delete", {"path": f"D:\\tmp\\file{i}.txt"})
        with pytest.raises(GuardrailViolation, match="RATE_LIMIT_DELETE"):
            engine.check("file_delete", {"path": r"D:\tmp\file_too_many.txt"})

    def test_file_ops_rate_limit(self):
        engine = make_engine(max_file_ops_per_min=5)
        for i in range(5):
            engine.check("file_read", {"path": f"D:\\tmp\\file{i}.txt"})
        with pytest.raises(GuardrailViolation, match="RATE_LIMIT_FILE_OPS"):
            engine.check("file_write", {"path": r"D:\tmp\overflow.txt"})

    def test_code_exec_rate_limit(self):
        engine = make_engine(max_code_exec_per_min=2)
        engine.check("execute_code", {"code": "print(1)"})
        engine.check("execute_code", {"code": "print(2)"})
        with pytest.raises(GuardrailViolation, match="RATE_LIMIT_CODE_EXEC"):
            engine.check("execute_code", {"code": "print(3)"})

    def test_non_file_tool_not_rate_limited(self):
        engine = make_engine(max_file_ops_per_min=1)
        engine.check("file_read", {"path": "D:\\tmp\\a.txt"})
        # app_launch is not a file operation, should pass
        engine.check("app_launch", {"target": "chrome"})


# ── Domain Whitelist Tests ───────────────────────────────────────────────


class TestDomainWhitelist:
    def test_blocks_unknown_domain(self):
        engine = make_engine(domain_whitelist=["google.com"])
        with pytest.raises(GuardrailViolation, match="DOMAIN_BLOCKED"):
            engine.check("browser_navigate", {"url": "https://evil.com/steal"})

    def test_allows_whitelisted_domain(self):
        engine = make_engine(domain_whitelist=["google.com"])
        engine.check("browser_navigate", {"url": "https://google.com/search"})

    def test_allows_subdomain(self):
        engine = make_engine(domain_whitelist=["google.com"])
        engine.check("browser_navigate", {"url": "https://docs.google.com/doc"})

    def test_non_browser_tool_ignores_whitelist(self):
        engine = make_engine(domain_whitelist=[])
        engine.check("file_read", {"url": "https://evil.com/x"})

    def test_no_url_in_args_passes(self):
        engine = make_engine()
        engine.check("browser_navigate", {"action": "screenshot"})


# ── Disabled Mode ────────────────────────────────────────────────────────


class TestDisabledMode:
    def test_disabled_skips_all_checks(self):
        engine = make_engine(enabled=False)
        # Would normally fail — protected path
        engine.check("file_write", {"path": r"C:\Windows\System32\config\SAM"})

    def test_violation_counter(self):
        engine = make_engine(max_file_deletes_per_min=1)
        engine.check("file_delete", {"path": "D:\\tmp\\a.txt"})
        assert engine.total_violations == 0
        with pytest.raises(GuardrailViolation):
            engine.check("file_delete", {"path": "D:\\tmp\\b.txt"})
        assert engine.total_violations == 1
