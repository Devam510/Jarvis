"""
jarvis.cognition.context_fusion — Multi-modal context fusion.

Combines voice transcript with visual screen context to resolve
ambiguous references like "this", "here", "that error".

Uses existing ScreenContext from perception layer.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Optional

logger = logging.getLogger(__name__)


@dataclass
class FusedContext:
    """Enriched intent with resolved references."""

    original_transcript: str = ""
    enriched_transcript: str = ""
    resolved_references: dict[str, str] = field(default_factory=dict)
    screen_app: str = ""
    screen_title: str = ""
    active_file: str = ""
    has_visual_context: bool = False


# Patterns that suggest visual/temporal references
_DEICTIC_PATTERNS = [
    (r"\bthis\b", "this"),
    (r"\bthat\b", "that"),
    (r"\bhere\b", "here"),
    (r"\bthese\b", "these"),
    (r"\bthose\b", "those"),
    (r"\bthat error\b", "that_error"),
    (r"\bthis file\b", "this_file"),
    (r"\bthis page\b", "this_page"),
    (r"\bthis app\b", "this_app"),
    (r"\bthis window\b", "this_window"),
    (r"\bcurrent\b", "current"),
    (r"\bopen\s+file\b", "open_file"),
]


class ContextFusion:
    """Fuse voice transcript with visual screen context.

    Resolves ambiguous references:
      - "this"/"this file" → currently active file from ScreenContext
      - "here" → active directory or cursor location
      - "that error" → last detected error in OCR context
      - "this app" → active application window
      - "this page" → browser tab title/URL

    Safety:
      - Read-only: never modifies anything, only enriches context
      - Falls back gracefully without screen context
      - No persistent state — each fusion is independent
    """

    def __init__(self):
        # Stats
        self.total_fusions = 0
        self.total_references_resolved = 0

    def fuse(
        self,
        transcript: str,
        screen_context: Any = None,
        recent_errors: list[str] | None = None,
    ) -> FusedContext:
        """Combine voice transcript with screen context.

        Args:
            transcript: Raw voice transcript
            screen_context: ScreenContext instance (or dict with similar fields)
            recent_errors: List of recently detected errors from perception

        Returns:
            FusedContext with enriched transcript and resolved references
        """
        self.total_fusions += 1

        result = FusedContext(
            original_transcript=transcript,
            enriched_transcript=transcript,
        )

        if screen_context is None:
            return result

        # Extract screen info
        app_name = self._get_attr(screen_context, "app_name", "")
        window_title = self._get_attr(screen_context, "window_title", "")
        ocr_text = self._get_attr(screen_context, "ocr_text", "")

        result.screen_app = app_name
        result.screen_title = window_title
        result.has_visual_context = bool(app_name or window_title)

        # Extract active file (from window title or screen context)
        active_file = self._extract_active_file(window_title, app_name)
        result.active_file = active_file

        # Detect and resolve deictic references
        found_refs = self._detect_references(transcript)
        enriched = transcript

        for ref_type in found_refs:
            resolved = self._resolve_reference(
                ref_type,
                app_name,
                window_title,
                active_file,
                ocr_text,
                recent_errors,
            )
            if resolved:
                result.resolved_references[ref_type] = resolved
                self.total_references_resolved += 1

        # Build enriched transcript
        if result.resolved_references:
            context_suffix = "; ".join(
                f"[{k}={v}]" for k, v in result.resolved_references.items()
            )
            result.enriched_transcript = f"{transcript} (context: {context_suffix})"

        return result

    # ── Reference Detection ───────────────────────────────────────────────

    def _detect_references(self, transcript: str) -> list[str]:
        """Detect deictic/ambiguous references in transcript."""
        found = []
        lower = transcript.lower()
        for pattern, ref_type in _DEICTIC_PATTERNS:
            if re.search(pattern, lower):
                found.append(ref_type)
        return found

    # ── Reference Resolution ──────────────────────────────────────────────

    def _resolve_reference(
        self,
        ref_type: str,
        app_name: str,
        window_title: str,
        active_file: str,
        ocr_text: str,
        recent_errors: list[str] | None,
    ) -> str:
        """Resolve a deictic reference using screen context."""

        if ref_type == "this_file" or ref_type == "open_file":
            return active_file or window_title

        if ref_type == "this_app" or ref_type == "this_window":
            return app_name or window_title

        if ref_type == "this_page":
            # Browser tab title
            if any(
                b in app_name.lower() for b in ["chrome", "firefox", "edge", "brave"]
            ):
                return window_title
            return ""

        if ref_type == "that_error":
            if recent_errors:
                return recent_errors[-1]
            # Try OCR for error detection
            error_match = self._find_error_in_text(ocr_text)
            return error_match or ""

        if ref_type == "here":
            return active_file or f"active window: {window_title}"

        if ref_type in ("this", "that", "current"):
            # Generic — use active file or window
            return active_file or window_title

        if ref_type in ("these", "those"):
            return f"context in {window_title}"

        return ""

    # ── Helpers ───────────────────────────────────────────────────────────

    @staticmethod
    def _extract_active_file(window_title: str, app_name: str) -> str:
        """Attempt to extract active file path from window title.

        Common patterns:
          - "file.py - Visual Studio Code"
          - "file.py - Notepad++"
          - "C:\\path\\file.py - Editor"
        """
        if not window_title:
            return ""

        # Pattern: "filename - AppName"
        parts = window_title.split(" - ")
        if len(parts) >= 2:
            candidate = parts[0].strip()
            # Check if it looks like a filename
            if "." in candidate or "/" in candidate or "\\" in candidate:
                return candidate

        return ""

    @staticmethod
    def _find_error_in_text(text: str) -> str:
        """Search for error patterns in OCR/screen text."""
        if not text:
            return ""

        error_patterns = [
            r"(TypeError:.*?)(?:\n|$)",
            r"(ValueError:.*?)(?:\n|$)",
            r"(ImportError:.*?)(?:\n|$)",
            r"(Traceback.*?)(?:\n|$)",
            r"(FAILED.*?)(?:\n|$)",
            r"(Exception:.*?)(?:\n|$)",
            r"(Error:.*?)(?:\n|$)",
        ]

        for pattern in error_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1).strip()[:200]

        return ""

    @staticmethod
    def _get_attr(obj: Any, attr: str, default: Any = "") -> Any:
        """Get attribute from object or dict safely."""
        if isinstance(obj, dict):
            return obj.get(attr, default)
        return getattr(obj, attr, default)
