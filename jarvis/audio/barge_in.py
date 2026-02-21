"""
jarvis.audio.barge_in — Barge-in detection during TTS playback.

Monitors VAD during playback. If sustained user speech is detected
(>300ms above threshold), emits barge_in.detected and signals the
TTS engine to stop immediately.

Safety:
  - Debounce: 300ms sustained speech before triggering (anti-flicker)
  - Configurable VAD threshold
  - Never blocks main loop
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Optional

logger = logging.getLogger(__name__)

# Minimum sustained speech duration before triggering barge-in (ms)
_DEBOUNCE_MS = 300


class BargeInDetector:
    """Detect user speech during TTS playback and interrupt.

    Usage:
        detector = BargeInDetector(event_bus=bus, vad=vad_engine)
        detector.start_monitoring()  # called when TTS starts playing
        ... VAD frames fed via on_vad_frame() ...
        detector.stop_monitoring()   # called when TTS finishes or is interrupted
    """

    def __init__(
        self,
        event_bus: Any = None,
        vad_threshold: float = 0.6,
        debounce_ms: int = _DEBOUNCE_MS,
    ):
        self._event_bus = event_bus
        self._vad_threshold = vad_threshold
        self._debounce_ms = debounce_ms
        self._monitoring = False
        self._speech_start: Optional[float] = None
        self._triggered = False

        # Stats
        self.total_barge_ins = 0

    def start_monitoring(self):
        """Begin monitoring for barge-in (called when TTS starts playing)."""
        self._monitoring = True
        self._speech_start = None
        self._triggered = False

    def stop_monitoring(self):
        """Stop monitoring (called when TTS finishes or was interrupted)."""
        self._monitoring = False
        self._speech_start = None
        self._triggered = False

    @property
    def is_monitoring(self) -> bool:
        return self._monitoring

    def on_vad_frame(self, speech_prob: float) -> bool:
        """Process a VAD frame during playback.

        Returns True if barge-in should be triggered (sustained speech detected).
        The caller (audio pipeline) should then stop TTS.

        Args:
            speech_prob: VAD probability [0.0, 1.0]

        Returns:
            True if barge-in detected, False otherwise
        """
        if not self._monitoring or self._triggered:
            return False

        now = time.time()

        if speech_prob >= self._vad_threshold:
            # Speech detected
            if self._speech_start is None:
                self._speech_start = now

            elapsed_ms = (now - self._speech_start) * 1000
            if elapsed_ms > self._debounce_ms:
                # Sustained speech — trigger barge-in
                self._triggered = True
                self.total_barge_ins += 1
                logger.info(
                    "Barge-in detected after %.0fms sustained speech",
                    elapsed_ms,
                )
                return True
        else:
            # No speech — reset debounce
            self._speech_start = None

        return False

    async def emit_barge_in(self):
        """Emit barge-in event via EventBus."""
        if self._event_bus:
            try:
                await self._event_bus.emit(
                    "barge_in.detected",
                    {
                        "timestamp": time.time(),
                        "count": self.total_barge_ins,
                    },
                )
            except Exception as e:
                logger.debug("Failed to emit barge-in event: %s", e)
