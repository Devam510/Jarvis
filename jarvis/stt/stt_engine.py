"""
jarvis.stt.stt_engine — Speech-to-text via Faster-Whisper.

Receives preprocessed float32 audio from the audio system,
runs Whisper inference, applies quality guards, and emits transcript events.
"""

from __future__ import annotations

import asyncio
import logging
import math
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

import numpy as np

from jarvis.core.event_bus import AsyncEventBus
from jarvis.utils.config import STTConfig
from jarvis.utils.types import SpeechSegment, TranscriptEvent, EmptyTranscript

logger = logging.getLogger(__name__)

# ── Known Whisper hallucination phrases ──────────────────────────────────────

HALLUCINATION_PHRASES = frozenset(
    {
        "thank you",
        "thanks for watching",
        "thanks for listening",
        "subscribe",
        "like and subscribe",
        "please subscribe",
        "bye",
        "goodbye",
        "see you",
        "see you next time",
        "you",
        "the end",
        "...",
        ".",
        "",
        "i",
        "a",
        "ah",
        "um",
        "uh",
        "music",
        "applause",
        "laughter",
        "it's good",
        "let's do that",
        "so",
        "okay",
        "ok",
        "right",
        "yes",
        "no",
        "hmm",
        "oh",
        "ha",
        "huh",
    }
)


class _TranscribeResult:
    """Internal result container for synchronous transcription."""

    __slots__ = ("text", "confidence", "language", "avg_logprob", "segments_count")

    def __init__(
        self,
        text: str,
        confidence: float,
        language: str,
        avg_logprob: float = 0.0,
        segments_count: int = 0,
    ):
        self.text = text
        self.confidence = confidence
        self.language = language
        self.avg_logprob = avg_logprob
        self.segments_count = segments_count


class STTEngine:
    """Speech-to-text engine using faster-whisper.

    Features:
      - Accepts float32 audio (preferred) or raw int16 bytes (backward compat)
      - Peak-normalizes audio as safety net
      - Quality guards: silence rejection, clipping detection
      - Hallucination filtering
      - Per-transcription diagnostic logging
      - Automatic model fallback on low confidence
    """

    def __init__(self, config: STTConfig, event_bus: AsyncEventBus):
        self.config = config
        self.event_bus = event_bus
        self._primary_model = None
        self._fallback_model = None
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="stt")
        self._load_models()

    # ── Model Loading ────────────────────────────────────────────────────

    def _load_models(self):
        from faster_whisper import WhisperModel

        device = self.config.device
        compute_type = self.config.compute_type

        if device == "cuda":
            self._setup_cuda_dll_paths()
            try:
                logger.info("Loading STT model on CUDA (%s)...", compute_type)
                model = WhisperModel(
                    self.config.model_size,
                    device="cuda",
                    compute_type=compute_type,
                )
                # Smoke test
                test_audio = np.zeros(16000, dtype=np.float32)
                segments, _ = model.transcribe(test_audio, language="en")
                list(segments)
                self._primary_model = model
                logger.info("✅ STT model loaded on CUDA (%s)", compute_type)
                return
            except Exception as e:
                logger.warning("CUDA STT failed (%s), falling back to CPU", e)

        # CPU fallback
        try:
            logger.info("Loading STT model on CPU (int8)...")
            self._primary_model = WhisperModel(
                self.config.model_size,
                device="cpu",
                compute_type="int8",
            )
            logger.info("✅ STT model loaded on CPU (int8)")
        except Exception as e:
            logger.error("Failed to load STT model: %s", e)

    @staticmethod
    def _setup_cuda_dll_paths():
        """Add nvidia pip package DLL directories to PATH."""
        import os
        import sys

        for site_dir in sys.path:
            nvidia_dir = os.path.join(site_dir, "nvidia")
            if not os.path.isdir(nvidia_dir):
                continue
            for subdir in os.listdir(nvidia_dir):
                for lib_dir in ("bin", "lib"):
                    dll_path = os.path.join(nvidia_dir, subdir, lib_dir)
                    if os.path.isdir(dll_path) and dll_path not in os.environ.get(
                        "PATH", ""
                    ):
                        os.environ["PATH"] = (
                            dll_path + os.pathsep + os.environ.get("PATH", "")
                        )
                        if hasattr(os, "add_dll_directory"):
                            try:
                                os.add_dll_directory(dll_path)
                            except OSError:
                                pass

    def _load_fallback(self):
        if self._fallback_model is not None:
            return
        try:
            from faster_whisper import WhisperModel

            logger.info("Loading fallback STT model: %s", self.config.fallback_model)
            self._fallback_model = WhisperModel(
                self.config.fallback_model,
                device=self.config.device,
                compute_type=self.config.compute_type,
            )
            logger.info("Fallback STT model loaded")
        except Exception as e:
            logger.error("Failed to load fallback model: %s", e)

    # ── Audio Preprocessing ──────────────────────────────────────────────

    def _prepare_audio(self, speech: SpeechSegment) -> Optional[np.ndarray]:
        """Convert SpeechSegment audio to clean float32 for Whisper.

        Handles both float32 numpy arrays and raw int16 bytes.
        Applies peak normalization as a safety net.
        """
        if isinstance(speech.audio, np.ndarray):
            audio = speech.audio.astype(np.float32)
        elif isinstance(speech.audio, bytes):
            # Legacy path: raw int16 PCM bytes
            raw = np.frombuffer(speech.audio, dtype=np.int16)
            audio = raw.astype(np.float32) / 32768.0
        else:
            logger.error("Unknown audio type: %s", type(speech.audio))
            return None

        # Safety-net peak normalization (in case audio system didn't normalize)
        peak = float(np.max(np.abs(audio)))
        if peak > 0.01:
            target = 0.8
            if peak < 0.3 or peak > 0.95:
                # Normalize — audio is either too quiet or clipping
                scale = target / peak
                audio = audio * scale
                audio = np.clip(audio, -1.0, 1.0)
                logger.debug(
                    "STT safety normalization: peak=%.3f → %.3f (scale=%.2f)",
                    peak,
                    target,
                    scale,
                )

        return audio

    # ── Quality Guards ───────────────────────────────────────────────────

    def _check_audio_quality(self, audio: np.ndarray) -> Optional[str]:
        """Check audio quality. Returns rejection reason or None if OK."""
        rms = float(np.sqrt(np.mean(audio**2)))
        peak = float(np.max(np.abs(audio)))

        # Too quiet — silence
        if rms < 0.005:
            return f"Audio too quiet (RMS={rms:.4f})"

        # Severe clipping — more than 5% of samples at ±1.0
        clip_count = int(np.sum(np.abs(audio) > 0.99))
        clip_pct = clip_count / len(audio) * 100
        if clip_pct > 5.0:
            return f"Audio severely clipped ({clip_pct:.1f}% samples)"

        return None

    # ── Main Transcription Pipeline ──────────────────────────────────────

    async def transcribe(self, speech: SpeechSegment):
        """Transcribe a speech segment and emit transcript event.

        Full pipeline: prepare → quality check → transcribe → guard → emit.
        """
        correlation_id = speech.correlation_id
        start_time = time.time()

        # Prepare audio
        audio = self._prepare_audio(speech)
        if audio is None:
            await self._emit_empty(correlation_id)
            return

        # Quality check
        rejection = self._check_audio_quality(audio)
        if rejection:
            logger.debug("Audio rejected: %s", rejection)
            await self._emit_empty(correlation_id)
            return

        # Log input diagnostics
        in_peak = float(np.max(np.abs(audio)))
        in_rms = float(np.sqrt(np.mean(audio**2)))
        logger.info(
            "STT input: duration=%.0fms, peak=%.3f, RMS=%.3f, speech_prob=%.2f",
            speech.duration_ms,
            in_peak,
            in_rms,
            speech.avg_speech_prob,
        )

        # Transcribe with primary model
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            self._executor, self._transcribe_sync, audio, self._primary_model
        )

        # Fallback if confidence below threshold
        if result and result.confidence < self.config.confidence_fallback_threshold:
            logger.info(
                "Low confidence (%.2f) — trying fallback model", result.confidence
            )
            self._load_fallback()
            if self._fallback_model:
                fallback_result = await loop.run_in_executor(
                    self._executor, self._transcribe_sync, audio, self._fallback_model
                )
                if fallback_result and fallback_result.confidence > result.confidence:
                    result = fallback_result
                    logger.info(
                        "Fallback model improved: conf=%.2f → %.2f",
                        result.confidence,
                        fallback_result.confidence,
                    )

        elapsed = (time.time() - start_time) * 1000

        # Process result
        if result and result.text.strip():
            clean_text = result.text.strip()

            # Log full diagnostics
            logger.info(
                "STT result: text='%s', confidence=%.3f, avg_logprob=%.3f, "
                "language=%s, segments=%d, latency=%.0fms",
                clean_text,
                result.confidence,
                result.avg_logprob,
                result.language,
                result.segments_count,
                elapsed,
            )

            # Filter hallucinations
            if clean_text.lower().rstrip(".!?,") in HALLUCINATION_PHRASES:
                logger.info("Filtered hallucination: '%s'", clean_text)
                await self._emit_empty(correlation_id)
                return

            # Filter repetitive phrases
            words = clean_text.split()
            if len(words) >= 4:
                sentences = [
                    s.strip().lower().rstrip(".!?,")
                    for s in clean_text.split(".")
                    if s.strip()
                ]
                if len(sentences) >= 3 and len(set(sentences)) <= 2:
                    logger.info("Filtered repetitive transcript: '%s'", clean_text)
                    await self._emit_empty(correlation_id)
                    return

            # Filter low confidence
            min_conf = getattr(self.config, "min_confidence", 0.35)
            if result.confidence < min_conf:
                logger.info(
                    "Discarding low-confidence transcript (%.2f < %.2f): '%s'",
                    result.confidence,
                    min_conf,
                    clean_text,
                )
                await self._emit_empty(correlation_id)
                return

            # Emit successful transcript
            await self.event_bus.emit(
                "stt.transcript",
                TranscriptEvent(
                    text=clean_text,
                    confidence=result.confidence,
                    language=result.language,
                    correlation_id=correlation_id,
                ),
            )
        else:
            logger.info("STT produced empty transcript (latency=%.0fms)", elapsed)
            await self._emit_empty(correlation_id)

    def _transcribe_sync(self, audio: np.ndarray, model) -> Optional[_TranscribeResult]:
        """Synchronous Whisper transcription (runs in executor thread)."""
        if model is None:
            return None

        try:
            segments, info = model.transcribe(
                audio,
                beam_size=self.config.beam_size,
                language=self.config.language,
                no_speech_threshold=self.config.no_speech_threshold,
                log_prob_threshold=self.config.log_prob_threshold,
                compression_ratio_threshold=self.config.compression_ratio_threshold,
                vad_filter=False,
            )

            full_text = ""
            total_prob = 0.0
            seg_count = 0

            for segment in segments:
                full_text += segment.text
                total_prob += segment.avg_logprob
                seg_count += 1

            if seg_count == 0:
                return None

            avg_prob = total_prob / seg_count
            confidence = math.exp(avg_prob)

            return _TranscribeResult(
                text=full_text.strip(),
                confidence=confidence,
                language=info.language,
                avg_logprob=avg_prob,
                segments_count=seg_count,
            )
        except Exception as e:
            logger.error("Transcription error: %s", e)
            return None

    async def _emit_empty(self, correlation_id: str):
        """Emit an empty transcript event."""
        await self.event_bus.emit(
            "stt.empty_transcript",
            EmptyTranscript(correlation_id=correlation_id),
        )

    # ── Confirmation Listening ───────────────────────────────────────────

    async def listen_for_response(
        self, timeout: float = 15.0
    ) -> Optional[TranscriptEvent]:
        """Listen for a spoken response (used by confirmation protocol)."""
        result_future: asyncio.Future = asyncio.get_event_loop().create_future()

        async def _capture(event: TranscriptEvent):
            if not result_future.done():
                result_future.set_result(event)

        self.event_bus.subscribe("stt.transcript", _capture)
        try:
            return await asyncio.wait_for(result_future, timeout=timeout)
        except asyncio.TimeoutError:
            return None
        finally:
            self.event_bus.unsubscribe("stt.transcript", _capture)
