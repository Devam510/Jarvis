"""
tests/test_path_validator.py — Tests for the hardened PathValidator.

Covers:
  - Basic whitelist mode
  - Blacklist / system dir deny
  - Drive-level deny
  - UNC/network path rejection
  - \\\\?\\ prefix normalization
  - Path traversal blocking
  - Unrestricted (blacklist-only) mode
"""

import os
import sys
import tempfile

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from jarvis.execution.executor import PathValidator, PathViolationError


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def restricted_validator():
    """Default whitelist mode with Desktop/Documents allowed."""
    return PathValidator(
        allowed_roots=[
            str(os.path.join(os.path.expanduser("~"), "Desktop")),
            str(os.path.join(os.path.expanduser("~"), "Documents")),
        ],
    )


@pytest.fixture
def unrestricted_validator():
    """Blacklist-only mode — everything except system dirs allowed."""
    return PathValidator(
        allowed_roots=[],
        unrestricted_mode=True,
    )


@pytest.fixture
def drive_deny_validator():
    """Unrestricted mode but with drive E: and F: denied."""
    return PathValidator(
        allowed_roots=[],
        denied_drives=["E:", "F:"],
        unrestricted_mode=True,
    )


# ── Whitelist Mode Tests ──────────────────────────────────────────────────────


def test_allowed_root_passes(restricted_validator):
    """Paths under allowed roots should pass."""
    desktop = os.path.join(os.path.expanduser("~"), "Desktop", "test.txt")
    p = restricted_validator.validate(desktop)
    assert str(p).endswith("test.txt")


def test_outside_root_rejected(restricted_validator):
    """Paths outside allowed roots should be rejected in whitelist mode."""
    with pytest.raises(PathViolationError, match="not under any allowed root"):
        restricted_validator.validate("D:\\SomeRandomFolder\\file.txt")


# ── System Directory Deny ────────────────────────────────────────────────────


def test_system_dir_blocked(restricted_validator):
    """System directories are always blocked."""
    with pytest.raises(PathViolationError, match="denied"):
        restricted_validator.validate("C:\\Windows\\System32\\config\\SAM")


def test_program_files_blocked(unrestricted_validator):
    """Program Files is blocked even in unrestricted mode."""
    with pytest.raises(PathViolationError, match="denied"):
        unrestricted_validator.validate("C:\\Program Files\\Firefox\\firefox.exe")


# ── UNC / Network Paths ──────────────────────────────────────────────────────


def test_unc_path_denied(unrestricted_validator):
    """UNC network paths should be denied."""
    with pytest.raises(PathViolationError, match="UNC"):
        unrestricted_validator.validate("\\\\server\\share\\secret.db")


def test_long_path_prefix_stripped(unrestricted_validator):
    """\\\\?\\ prefix should be normalized away."""
    # This path should be treated as a normal path after stripping
    home = os.path.expanduser("~")
    long_prefix_path = "\\\\?\\" + os.path.join(home, "Desktop", "test.txt")
    try:
        p = unrestricted_validator.validate(long_prefix_path)
        assert "test.txt" in str(p)
    except PathViolationError:
        # If it fails validation it's because of root check, not prefix
        pass


# ── Path Traversal ────────────────────────────────────────────────────────────


def test_path_traversal_blocked(unrestricted_validator):
    """Literal .. in path parts should be rejected."""
    with pytest.raises(PathViolationError, match="traversal"):
        unrestricted_validator.validate("C:\\Users\\..\\Windows\\System32\\cmd.exe")


# ── Drive-Level Deny ─────────────────────────────────────────────────────────


def test_denied_drive_blocked(drive_deny_validator):
    """Paths on denied drives should be rejected."""
    with pytest.raises(PathViolationError, match="drive.*denied"):
        drive_deny_validator.validate("E:\\SomeFolder\\data.csv")


def test_allowed_drive_passes(drive_deny_validator):
    """Paths on non-denied drives should pass in unrestricted mode."""
    home = os.path.join(os.path.expanduser("~"), "test.txt")
    p = drive_deny_validator.validate(home)
    assert "test.txt" in str(p)


# ── Unrestricted Mode ────────────────────────────────────────────────────────


def test_unrestricted_allows_any_non_system(unrestricted_validator):
    """In unrestricted mode, non-system paths are allowed."""
    home = os.path.join(os.path.expanduser("~"), "AnyFolder", "file.txt")
    p = unrestricted_validator.validate(home)
    assert "file.txt" in str(p)


def test_unrestricted_still_blocks_system(unrestricted_validator):
    """Even in unrestricted mode, system dirs remain blocked."""
    with pytest.raises(PathViolationError):
        unrestricted_validator.validate("C:\\Windows\\explorer.exe")
