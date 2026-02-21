"""
jarvis.tts.ssml_processor — Emotive SSML tag injection.

Analyzes text sentiment/punctuation and injects SSML prosody tags
for more natural TTS output. Works with edge-tts (SSML-capable);
pyttsx3 strips tags gracefully.

Rules:
  - Questions → rising pitch
  - Exclamations → emphasis + slightly faster
  - Ellipsis → 400ms pause
  - List items → short pauses between items
  - Long sentences → natural breathing pauses at commas
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass
class SSMLConfig:
    """Configuration for SSML processing."""

    enabled: bool = True
    question_pitch: str = "+10%"
    exclamation_rate: str = "+5%"
    ellipsis_pause_ms: int = 400
    comma_pause_ms: int = 200
    sentence_pause_ms: int = 300


class SSMLProcessor:
    """Inject SSML prosody tags based on text analysis.

    Designed to work with edge-tts which supports SSML.
    For pyttsx3/Piper fallback, the strip() method removes all tags.
    """

    def __init__(self, config: SSMLConfig | None = None):
        self._config = config or SSMLConfig()

    def process(self, text: str) -> str:
        """Analyze text and inject SSML tags.

        Returns SSML-annotated text (or plain text if disabled).
        """
        if not self._config.enabled or not text.strip():
            return text

        sentences = self._split_for_ssml(text)
        parts = []

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            parts.append(self._annotate_sentence(sentence))

        return " ".join(parts)

    def strip(self, text: str) -> str:
        """Remove all SSML tags, returning plain text.

        Use this for backends that don't support SSML (pyttsx3, Piper).
        """
        # Remove XML/SSML tags
        clean = re.sub(r"<[^>]+>", "", text)
        # Collapse whitespace
        clean = re.sub(r"\s+", " ", clean).strip()
        return clean

    def _annotate_sentence(self, sentence: str) -> str:
        """Apply prosody/emphasis based on sentence punctuation."""
        cfg = self._config

        # Detect sentence type
        if sentence.rstrip().endswith("?"):
            # Question → rising pitch
            return (
                f'<prosody pitch="{cfg.question_pitch}">'
                f"{sentence}</prosody>"
                f'<break time="{cfg.sentence_pause_ms}ms"/>'
            )

        if sentence.rstrip().endswith("!"):
            # Exclamation → emphasis + faster
            return (
                f'<prosody rate="{cfg.exclamation_rate}">'
                f"<emphasis>{sentence}</emphasis></prosody>"
                f'<break time="{cfg.sentence_pause_ms}ms"/>'
            )

        if sentence.rstrip().endswith("...") or sentence.rstrip().endswith("…"):
            # Ellipsis → trailing pause
            return f"{sentence}" f'<break time="{cfg.ellipsis_pause_ms}ms"/>'

        # Insert comma pauses for long sentences
        if "," in sentence and len(sentence) > 60:
            sentence = self._add_comma_pauses(sentence)

        return f'{sentence}<break time="{cfg.sentence_pause_ms}ms"/>'

    def _add_comma_pauses(self, sentence: str) -> str:
        """Insert short pauses after commas in long sentences."""
        pause = f'<break time="{self._config.comma_pause_ms}ms"/>'
        # Only add pauses after commas followed by a space
        return re.sub(r",\s+", f", {pause}", sentence)

    @staticmethod
    def _split_for_ssml(text: str) -> list[str]:
        """Split text into sentences for SSML processing.

        Preserves punctuation unlike typical sentence splitters.
        """
        # Split on sentence-ending punctuation while keeping the punctuation
        parts = re.split(r"(?<=[.!?…])\s+", text)
        return [p for p in parts if p.strip()]
