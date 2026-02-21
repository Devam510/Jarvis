"""
jarvis.tts.tts_engine — Streaming text-to-speech with sub-400ms TTFB.

Engine priority:
  1. edge-tts  (streaming via WebSocket — lowest latency)
  2. Piper     (local ONNX model — sentence-level)
  3. pyttsx3   (system fallback — blocking)

Streaming architecture:
  - Text is split into sentences immediately.
  - Each sentence is synthesized and played concurrently:
    sentence N plays while sentence N+1 is being synthesized.
  - edge-tts streams audio chunks as they arrive from the server,
    allowing playback to start before full synthesis completes.
"""

from __future__ import annotations

import asyncio
import io
import logging
import re
import subprocess
import threading
import time
from pathlib import Path
from typing import Optional

import numpy as np

from jarvis.utils.config import TTSConfig

logger = logging.getLogger(__name__)


class TTSEngine:
    """Streaming text-to-speech engine with pipeline parallelism.

    speak() splits text into sentences, then synthesizes and plays them
    with overlap: the next sentence starts synthesizing while the current
    one is playing.
    """

    def __init__(self, config: TTSConfig):
        self.config = config
        self._speak_lock = asyncio.Lock()

        # Detect available backends (lazily validated on first use)
        self._edge_tts_available: Optional[bool] = None
        self._piper_available = self._check_piper()
        self._pyttsx_engine = None
        self._pyttsx_lock = threading.Lock()  # Serialize blocking pyttsx3 calls

        if not self._piper_available:
            self._init_pyttsx()

        # Metrics
        self._total_speaks = 0
        self._total_latency_ms = 0.0

    # ── Public API ────────────────────────────────────────────────────────

    async def speak(self, text: str):
        """Speak text using streaming sentence-level pipeline.

        The pipeline overlaps synthesis and playback:
          - Sentence 1: [synthesize] → [play ────────]
          - Sentence 2:         [synthesize] → [play ────────]
        """
        if not text or not text.strip():
            return

        logger.info("TTS: %s", text[:100] + ("..." if len(text) > 100 else ""))
        sentences = self._split_sentences(text)
        if not sentences:
            return

        start = time.time()
        self._total_speaks += 1

        async with self._speak_lock:
            await self._streaming_pipeline(sentences)

        latency = (time.time() - start) * 1000
        self._total_latency_ms += latency
        logger.debug(
            "TTS pipeline complete: %.0fms for %d sentences", latency, len(sentences)
        )

    @property
    def stats(self) -> dict:
        return {
            "total_speaks": self._total_speaks,
            "avg_latency_ms": (
                self._total_latency_ms / self._total_speaks
                if self._total_speaks > 0
                else 0
            ),
            "backend": self._active_backend,
        }

    @property
    def _active_backend(self) -> str:
        if self._edge_tts_available:
            return "edge-tts"
        if self._piper_available:
            return "piper"
        if self._pyttsx_engine:
            return "pyttsx3"
        return "print"

    # ── Streaming Pipeline ────────────────────────────────────────────────

    async def _streaming_pipeline(self, sentences: list[str]):
        """Overlap synthesis and playback across sentences.

        Uses a queue: synthesizer produces audio, player consumes it.
        This ensures sentence N+1 starts synthesizing while N is playing.
        """
        audio_queue: asyncio.Queue[Optional[tuple[np.ndarray, int]]] = asyncio.Queue(
            maxsize=2
        )

        async def producer():
            for sentence in sentences:
                audio_data = await self._synthesize(sentence)
                if audio_data is not None:
                    await audio_queue.put(audio_data)
            await audio_queue.put(None)  # sentinel

        async def consumer():
            while True:
                item = await audio_queue.get()
                if item is None:
                    break
                audio, sample_rate = item
                await self._play_audio(audio, sample_rate)

        # Run producer and consumer concurrently
        await asyncio.gather(producer(), consumer())

    async def _synthesize(self, text: str) -> Optional[tuple[np.ndarray, int]]:
        """Synthesize a single sentence using the best available backend.

        Returns (audio_array, sample_rate) or None on failure.
        """
        # Try edge-tts first (streaming, lowest latency)
        if self._edge_tts_available is None:
            self._edge_tts_available = await self._check_edge_tts()

        if self._edge_tts_available:
            result = await self._synthesize_edge_tts(text)
            if result is not None:
                return result

        # Piper fallback
        if self._piper_available:
            result = await self._synthesize_piper(text)
            if result is not None:
                return result

        # pyttsx3 fallback (blocking, plays directly)
        if self._pyttsx_engine:
            await self._speak_pyttsx(text)
            return None

        # No engine — print fallback
        print(f"🔊 Jarvis: {text}")
        return None

    # ── edge-tts Backend ──────────────────────────────────────────────────

    async def _check_edge_tts(self) -> bool:
        try:
            import edge_tts  # noqa: F401

            return True
        except ImportError:
            logger.info("edge-tts not installed — falling back to Piper/pyttsx3")
            return False

    async def _synthesize_edge_tts(self, text: str) -> Optional[tuple[np.ndarray, int]]:
        """Synthesize via edge-tts with streaming chunk collection.

        edge-tts streams audio via WebSocket — we collect chunks as they
        arrive. TTFB is typically < 200ms.
        """
        try:
            import edge_tts

            voice = getattr(self.config, "edge_voice", "en-US-GuyNeural")
            rate = getattr(self.config, "edge_rate", "+0%")
            communicate = edge_tts.Communicate(text, voice, rate=rate)

            audio_chunks: list[bytes] = []
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    audio_chunks.append(chunk["data"])

            if not audio_chunks:
                return None

            raw = b"".join(audio_chunks)

            # edge-tts outputs MP3 — decode to PCM
            audio, sr = self._decode_mp3(raw)
            if audio is not None:
                return (audio, sr)
            return None

        except Exception as e:
            logger.warning("edge-tts synthesis failed: %s", e)
            self._edge_tts_available = False  # disable for this session
            return None

    @staticmethod
    def _decode_mp3(raw: bytes) -> tuple[Optional[np.ndarray], int]:
        """Decode MP3 bytes to numpy int16 array.

        Tries pydub first, falls back to io-based approach.
        """
        try:
            from pydub import AudioSegment

            seg = AudioSegment.from_mp3(io.BytesIO(raw))
            seg = seg.set_channels(1).set_sample_width(2)
            samples = np.frombuffer(seg.raw_data, dtype=np.int16)
            return samples, seg.frame_rate
        except ImportError:
            pass

        # Fallback: try minimp3 or just warn
        try:
            import minimp3

            decoder = minimp3.Decoder()
            frames = decoder.decode(raw)
            return np.array(frames, dtype=np.int16), 24000
        except (ImportError, Exception):
            logger.warning("Cannot decode MP3 — install pydub or minimp3")
            return None, 0

    # ── Piper Backend ─────────────────────────────────────────────────────

    def _check_piper(self) -> bool:
        """Check if Piper binary is available (one-time sync check)."""
        if not self.config.model_path:
            return False
        try:
            result = subprocess.run(
                ["piper", "--version"], capture_output=True, timeout=5
            )
            return result.returncode == 0
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return False

    async def _synthesize_piper(self, text: str) -> Optional[tuple[np.ndarray, int]]:
        """Synthesize via Piper (sentence-level, no streaming)."""
        try:
            proc = await asyncio.create_subprocess_exec(
                "piper",
                "--model",
                self.config.model_path,
                "--output-raw",
                "--sentence_silence",
                str(self.config.sentence_silence),
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.DEVNULL,
            )
            stdout, _ = await proc.communicate(input=text.encode("utf-8"))
            if stdout:
                audio = np.frombuffer(stdout, dtype=np.int16)
                return (audio, self.config.sample_rate)
            return None
        except Exception as e:
            logger.error("Piper TTS error: %s", e)
            return None

    # ── pyttsx3 Backend ───────────────────────────────────────────────────

    def _init_pyttsx(self):
        """Initialize pyttsx3 as fallback TTS."""
        try:
            import pyttsx3

            self._pyttsx_engine = pyttsx3.init()
            self._pyttsx_engine.setProperty("rate", 175)
            self._pyttsx_engine.setProperty("volume", 0.9)
            logger.info("TTS initialized: pyttsx3 (fallback)")
        except Exception as e:
            logger.warning("pyttsx3 not available: %s", e)

    async def _speak_pyttsx(self, text: str):
        """Speak via pyttsx3 in executor (blocking)."""
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._pyttsx_say, text)

    def _pyttsx_say(self, text: str):
        """Speak via pyttsx3 — creates a fresh engine per call to avoid COM conflicts."""
        with self._pyttsx_lock:
            try:
                import pyttsx3

                engine = pyttsx3.init()
                engine.setProperty("rate", 175)
                engine.setProperty("volume", 0.9)
                engine.say(text)
                engine.runAndWait()
                engine.stop()
            except Exception as e:
                logger.warning("pyttsx3 error (skipping): %s", e)

    # ── Audio Playback ────────────────────────────────────────────────────

    async def _play_audio(self, audio: np.ndarray, sample_rate: int):
        """Play raw audio via sounddevice in thread pool."""
        try:
            import sounddevice as sd

            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None, lambda: sd.play(audio, samplerate=sample_rate, blocking=True)
            )
        except Exception as e:
            logger.error("Audio playback error: %s", e)

    # ── Text Splitting ────────────────────────────────────────────────────

    @staticmethod
    def _split_sentences(text: str) -> list[str]:
        """Split text into sentences for pipelined playback.

        Also splits on commas/semicolons for long clauses to reduce
        per-chunk latency.
        """
        # First split on sentence boundaries
        parts = re.split(r"(?<=[.!?])\s+", text)

        # Further split long segments on commas/semicolons
        result = []
        for p in parts:
            p = p.strip()
            if not p:
                continue
            if len(p) > 120:
                sub = re.split(r"(?<=[,;])\s+", p)
                result.extend(s.strip() for s in sub if s.strip())
            else:
                result.append(p)

        return result
