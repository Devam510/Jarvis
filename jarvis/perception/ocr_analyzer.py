"""
jarvis.perception.ocr_analyzer — EasyOCR wrapper with error pattern detection.

Runs OCR in a thread executor to avoid blocking the event loop.
Returns empty string during privacy blackout (double-guard).
"""

from __future__ import annotations

import asyncio
import logging
import re
from typing import Optional

logger = logging.getLogger(__name__)

# Pre-compiled error patterns for fast matching
_ERROR_PATTERNS = [
    re.compile(r"Traceback \(most recent call last\)", re.IGNORECASE),
    re.compile(r"Error:\s+.+", re.IGNORECASE),
    re.compile(r"Exception:\s+.+", re.IGNORECASE),
    re.compile(r"FAILED", re.IGNORECASE),
    re.compile(r"Permission\s*denied", re.IGNORECASE),
    re.compile(r"Uncaught\s+(TypeError|ReferenceError|SyntaxError)", re.IGNORECASE),
    re.compile(r"FATAL", re.IGNORECASE),
    re.compile(r"Segmentation fault", re.IGNORECASE),
    re.compile(r"Access\s+(is\s+)?denied", re.IGNORECASE),
    re.compile(r"ModuleNotFoundError", re.IGNORECASE),
    re.compile(r"ImportError", re.IGNORECASE),
    re.compile(r"FileNotFoundError", re.IGNORECASE),
]


class OCRAnalyzer:
    """Thin wrapper around EasyOCR with error pattern detection."""

    def __init__(self, language: str = "en"):
        self._language = language
        self._reader = None  # Lazy-loaded
        self._loop = asyncio.get_event_loop()

    async def _ensure_reader(self):
        """Lazy-load EasyOCR reader on first use.

        [V2-09 FIX] Init runs in executor to avoid blocking the event loop
        during model loading + GPU init (can take 2-5 seconds).
        """
        if self._reader is None:
            try:
                import easyocr

                lang = self._language
                self._reader = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: easyocr.Reader([lang], gpu=True, verbose=False),
                )
                logger.info("EasyOCR reader initialized (lang=%s)", self._language)
            except ImportError:
                logger.warning("easyocr not installed — OCR disabled")
                raise
            except Exception as e:
                logger.error("EasyOCR init failed: %s", e)
                raise

    async def extract_text(self, image_path: str) -> str:
        """Extract text from image. Runs in executor to avoid blocking."""
        try:
            await self._ensure_reader()
            results = await asyncio.get_event_loop().run_in_executor(
                None, self._reader.readtext, image_path
            )
            # EasyOCR returns list of (bbox, text, confidence)
            texts = [r[1] for r in results if len(r) >= 2]
            return " ".join(texts)
        except ImportError:
            return ""
        except Exception as e:
            logger.error("OCR extraction failed: %s", e)
            return ""

    def detect_errors(self, text: str) -> list[str]:
        """Scan text for common error patterns. Returns matched snippets."""
        if not text:
            return []
        errors = []
        for pattern in _ERROR_PATTERNS:
            matches = pattern.findall(text)
            for m in matches:
                snippet = m.strip() if isinstance(m, str) else str(m).strip()
                if snippet and snippet not in errors:
                    errors.append(snippet)
        return errors
