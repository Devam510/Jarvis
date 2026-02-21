"""
jarvis.audio.audio_system — Microphone capture, VAD, and speech collection.

Architecture:
  - A dedicated thread runs the sounddevice InputStream callback.
  - The callback pushes raw int16 frames into a thread-safe queue.
  - An async consumer loop (on the event loop) drains the queue,
    normalizes amplitude, runs VAD, collects speech, and emits events.
  - Zero shared mutable state between threads — all coordination via queue.
"""

from __future__ import annotations

import asyncio
import logging
import queue
import threading
import time
from collections import deque
from typing import Optional

import numpy as np

from jarvis.core.event_bus import AsyncEventBus
from jarvis.utils.config import AudioConfig
from jarvis.utils.enums import AudioState
from jarvis.utils.types import SpeechSegment

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
#  Amplitude Normalizer
# ═══════════════════════════════════════════════════════════════════════════════


class AmplitudeNormalizer:
    """Normalize int16 audio to a target peak amplitude.

    - If peak > noise_floor: scale to target_peak * 32768
    - If peak <= noise_floor: signal is silence — return zeros
    - Clipping is prevented via np.clip
    """

    def __init__(self, target_peak: float = 0.8, noise_floor: int = 50):
        self.target_peak_i16 = int(32768 * target_peak)
        self.noise_floor = noise_floor

    def normalize_int16(
        self, frames: np.ndarray
    ) -> tuple[np.ndarray, float, float, float]:
        """Normalize int16 frames. Returns (normalized_int16, peak, rms, noise_floor_est).

        All returned metrics are in int16 scale (0–32768).
        """
        peak = int(np.max(np.abs(frames)))
        rms = float(np.sqrt(np.mean(frames.astype(np.float32) ** 2)))

        # Estimate noise floor from the quietest 10% of frame energies
        frame_sz = 160  # 10ms at 16kHz
        energies = []
        for i in range(0, len(frames) - frame_sz, frame_sz):
            chunk = frames[i : i + frame_sz]
            energies.append(float(np.sqrt(np.mean(chunk.astype(np.float32) ** 2))))
        if energies:
            energies.sort()
            noise_est = float(np.mean(energies[: max(1, len(energies) // 10)]))
        else:
            noise_est = 0.0

        if peak <= self.noise_floor:
            # Near-silence — don't amplify noise
            return frames, float(peak), rms, noise_est

        scale = self.target_peak_i16 / peak
        if abs(scale - 1.0) > 0.05:
            normalized = np.clip(
                frames.astype(np.float32) * scale, -32768, 32767
            ).astype(np.int16)
        else:
            normalized = frames

        norm_rms = float(np.sqrt(np.mean(normalized.astype(np.float32) ** 2)))
        return normalized, float(peak), norm_rms, noise_est

    def int16_to_float32(self, frames: np.ndarray) -> np.ndarray:
        """Convert int16 to float32 in [-1, 1] range."""
        return frames.astype(np.float32) / 32768.0


# ═══════════════════════════════════════════════════════════════════════════════
#  VAD Engine
# ═══════════════════════════════════════════════════════════════════════════════


class VADEngine:
    """Voice Activity Detection using Silero VAD with energy fallback."""

    def __init__(self):
        self._model = None
        self._init_model()

    def _init_model(self):
        try:
            import torch

            model, utils = torch.hub.load(
                repo_or_dir="snakers4/silero-vad",
                model="silero_vad",
                force_reload=False,
                onnx=True,
            )
            self._model = model
            logger.info("Silero VAD loaded (ONNX)")
        except Exception as e:
            logger.warning("Silero VAD not available, using energy fallback: %s", e)
            self._model = None

    def process(self, frames_int16: np.ndarray) -> float:
        """Return speech probability [0.0, 1.0] for the given int16 frame."""
        if self._model is not None:
            try:
                import torch

                # Silero expects float32 in [-1, 1]
                audio_tensor = torch.from_numpy(
                    frames_int16.astype(np.float32) / 32768.0
                )
                prob = self._model(audio_tensor, 16000).item()
                return prob
            except Exception:
                return self._energy_fallback(frames_int16)
        return self._energy_fallback(frames_int16)

    def _energy_fallback(self, frames: np.ndarray) -> float:
        energy = np.sqrt(np.mean(frames.astype(np.float32) ** 2))
        return min(energy / 3000.0, 1.0)


# ═══════════════════════════════════════════════════════════════════════════════
#  Wake Word Detector
# ═══════════════════════════════════════════════════════════════════════════════


class WakeWordDetector:
    """Wake word detection using openwakeword."""

    def __init__(self, wake_word: str = "jarvis"):
        self.wake_word = wake_word
        self._model = None
        self._model_key = None
        self._init_model()

    def _init_model(self):
        try:
            from openwakeword.model import Model

            hey_name = f"hey_{self.wake_word}"
            try:
                self._model = Model(
                    wakeword_models=[hey_name],
                    inference_framework="onnx",
                )
                self._model_key = hey_name
                logger.info("OpenWakeWord loaded: '%s'", hey_name)
            except Exception:
                try:
                    self._model = Model(inference_framework="onnx")
                    for key in self._model.models:
                        if self.wake_word in key.lower():
                            self._model_key = key
                            break
                    logger.info(
                        "OpenWakeWord loaded default models (key=%s)", self._model_key
                    )
                except Exception as e2:
                    logger.warning("OpenWakeWord default models failed: %s", e2)
                    self._model = None
        except Exception as e:
            logger.warning(
                "OpenWakeWord not available, using energy-based fallback: %s", e
            )
            self._model = None

    def process(self, frames: np.ndarray) -> bool:
        if self._model is not None:
            try:
                prediction = self._model.predict(frames)
                if self._model_key and self._model_key in prediction:
                    score = prediction[self._model_key]
                    if score > 0.5:
                        logger.info(
                            "Wake word '%s' detected: score=%.2f",
                            self._model_key,
                            score,
                        )
                        return True
                else:
                    for key, score in prediction.items():
                        if score > 0.5:
                            logger.info(
                                "Wake word detected (model): %s score=%.2f", key, score
                            )
                            return True
                return False
            except Exception:
                return False
        else:
            energy = np.sqrt(np.mean(frames.astype(np.float32) ** 2))
            return energy > 1500

    def reset(self):
        """Reset model internal state."""
        if self._model is not None:
            try:
                self._model.reset()
            except Exception:
                pass


# ═══════════════════════════════════════════════════════════════════════════════
#  Speech Collector
# ═══════════════════════════════════════════════════════════════════════════════


class SpeechCollector:
    """Accumulates speech frames with pre-roll and end-of-speech detection.

    Operates entirely on the async consumer side — no thread safety needed.
    """

    def __init__(self, config: AudioConfig, normalizer: AmplitudeNormalizer):
        self.pre_roll_samples = int(config.pre_roll_ms * config.sample_rate / 1000)
        self.max_samples = int(config.max_speech_ms * config.sample_rate / 1000)
        self.min_samples = int(config.min_speech_ms * config.sample_rate / 1000)
        self.silence_samples = int(
            config.silence_timeout_ms * config.sample_rate / 1000
        )
        self.speech_threshold = config.speech_threshold
        self.sample_rate = config.sample_rate
        self.normalizer = normalizer
        self.reset()

    def reset(self):
        self.frames: list[np.ndarray] = []
        self.speech_probs: list[float] = []
        self.silence_counter = 0
        self.total_samples = 0
        self.speech_samples = 0  # samples with speech_prob >= threshold
        self.active = False

    def start(self, pre_roll: np.ndarray):
        """Start collecting with pre-roll audio prepended."""
        self.reset()
        if len(pre_roll) > 0:
            self.frames.append(pre_roll.copy())
            self.total_samples = len(pre_roll)
        self.active = True

    def add_frames(self, frames: np.ndarray, speech_prob: float):
        """Add a frame of audio with its VAD probability."""
        self.frames.append(frames.copy())
        self.speech_probs.append(speech_prob)
        self.total_samples += len(frames)

        if speech_prob < self.speech_threshold:
            self.silence_counter += len(frames)
        else:
            self.silence_counter = 0
            self.speech_samples += len(frames)

    def end_of_speech_detected(self) -> bool:
        """Check if we've accumulated enough silence after enough speech."""
        return (
            self.silence_counter >= self.silence_samples
            and self.speech_samples >= self.min_samples
        )

    def max_duration_exceeded(self) -> bool:
        return self.total_samples >= self.max_samples

    def finalize(self) -> SpeechSegment:
        """Concatenate, normalize, and return the speech segment.

        Returns a SpeechSegment with float32 audio ready for Whisper.
        """
        # Concatenate all collected int16 frames
        raw_audio = np.concatenate(self.frames)

        # Trim trailing silence (keep up to 200ms of tail)
        tail_keep = int(0.2 * self.sample_rate)
        if self.silence_counter > tail_keep:
            trim = self.silence_counter - tail_keep
            if trim < len(raw_audio):
                raw_audio = raw_audio[: len(raw_audio) - trim]

        # Peak-normalize the entire segment
        normalized, peak, rms, noise_est = self.normalizer.normalize_int16(raw_audio)

        # Convert to float32 for Whisper
        audio_f32 = self.normalizer.int16_to_float32(normalized)

        avg_prob = float(np.mean(self.speech_probs)) if self.speech_probs else 0.0

        logger.info(
            "Speech segment: duration=%.0fms, peak=%d, RMS=%.0f, noise=%.0f, "
            "speech_prob=%.2f, speech_samples=%d",
            len(raw_audio) / self.sample_rate * 1000,
            peak,
            rms,
            noise_est,
            avg_prob,
            self.speech_samples,
        )

        segment = SpeechSegment(
            audio=audio_f32,
            sample_rate=self.sample_rate,
            duration_ms=len(audio_f32) / self.sample_rate * 1000,
            avg_speech_prob=avg_prob,
            peak=float(peak) / 32768.0,
            rms=rms / 32768.0,
            noise_floor=noise_est / 32768.0,
        )
        self.reset()
        return segment


# ═══════════════════════════════════════════════════════════════════════════════
#  Audio System (Main Subsystem)
# ═══════════════════════════════════════════════════════════════════════════════


class AudioSystem:
    """Complete audio subsystem.

    Architecture:
      - Capture thread: sounddevice callback → thread-safe queue
      - Async consumer: queue → normalize → VAD → collect → emit event

    No shared mutable state between threads.
    """

    def __init__(self, config: AudioConfig, event_bus: AsyncEventBus):
        self.config = config
        self.event_bus = event_bus
        self.normalizer = AmplitudeNormalizer(
            target_peak=config.target_peak,
            noise_floor=config.noise_floor,
        )
        self.vad = VADEngine()
        self.collector = SpeechCollector(config, self.normalizer)

        # Thread-safe queue for frame transfer: capture thread → async consumer
        self._frame_queue: queue.Queue[Optional[np.ndarray]] = queue.Queue(maxsize=500)

        # Pre-roll buffer — rolling window of recent audio (async side only)
        self._pre_roll_maxlen = int(config.pre_roll_ms * config.sample_rate / 1000)
        self._pre_roll: deque[np.ndarray] = deque()
        self._pre_roll_samples = 0

        # State
        self.state = AudioState.WAITING_FOR_WAKE
        self._running = False
        self._capture_thread: Optional[threading.Thread] = None
        self._last_segment_time: float = 0.0
        self._speech_started = False

    async def run(self):
        """Start capture thread and async consumer loop."""
        self._running = True
        logger.info(
            "Audio system starting (sample_rate=%d, frame_size=%d, "
            "target_peak=%.1f, noise_floor=%d)",
            self.config.sample_rate,
            self.config.frame_size,
            self.config.target_peak,
            self.config.noise_floor,
        )

        # Start capture in a dedicated thread
        self._capture_thread = threading.Thread(
            target=self._capture_loop,
            name="audio-capture",
            daemon=True,
        )
        self._capture_thread.start()

        # Run the async consumer on the event loop
        await self._async_consumer()

    def stop(self):
        """Signal both the capture thread and consumer to stop."""
        self._running = False
        # Push a sentinel to unblock the consumer
        try:
            self._frame_queue.put_nowait(None)
        except queue.Full:
            pass
        # [V2-10 FIX] Join capture thread for orderly shutdown
        if hasattr(self, "_capture_thread") and self._capture_thread is not None:
            self._capture_thread.join(timeout=2.0)

    # ── Capture Thread ──────────────────────────────────────────────────

    def _capture_loop(self):
        """Blocking audio capture — runs in a dedicated thread.

        Pushes raw int16 frames into _frame_queue.
        The async consumer handles all processing.
        """
        try:
            import sounddevice as sd
        except ImportError:
            logger.error("sounddevice not installed — audio capture disabled")
            return

        def audio_callback(indata, frames_count, time_info, status):
            if status:
                logger.warning("Audio status: %s", status)
            if not self._running:
                return

            # Extract mono int16 and push to queue
            audio_frames = indata[:, 0].copy()
            if audio_frames.dtype != np.int16:
                audio_frames = (audio_frames * 32768).astype(np.int16)

            try:
                self._frame_queue.put_nowait(audio_frames)
            except queue.Full:
                pass  # Drop frame rather than block the audio thread

        try:
            with sd.InputStream(
                samplerate=self.config.sample_rate,
                channels=self.config.channels,
                dtype="int16",
                blocksize=self.config.frame_size,
                callback=audio_callback,
            ):
                logger.info("Microphone stream opened — listening (16kHz, mono, int16)")
                while self._running:
                    time.sleep(0.1)
        except Exception as e:
            logger.error("Audio capture error: %s", e)

    # ── Async Consumer ──────────────────────────────────────────────────

    async def _async_consumer(self):
        """Drain the frame queue and process frames on the event loop.

        All mutable state (pre_roll, collector, VAD) is accessed only here.
        """
        loop = asyncio.get_event_loop()

        while self._running:
            # Non-blocking drain: get all available frames
            frames_batch: list[np.ndarray] = []
            try:
                # Block briefly to avoid busy-spinning
                raw = await loop.run_in_executor(
                    None, lambda: self._frame_queue.get(timeout=0.05)
                )
                if raw is None:
                    break  # Sentinel — shutting down
                frames_batch.append(raw)
            except queue.Empty:
                continue

            # Drain any additional frames that arrived
            while not self._frame_queue.empty():
                try:
                    raw = self._frame_queue.get_nowait()
                    if raw is None:
                        self._running = False
                        break
                    frames_batch.append(raw)
                except queue.Empty:
                    break

            # Process each frame
            for frames in frames_batch:
                await self._process_frame(frames)

    async def _process_frame(self, frames: np.ndarray):
        """Process a single audio frame through VAD and speech collection.

        All state mutations happen here, on the async event loop — thread safe.

        CRITICAL: We normalize audio ONLY for VAD decisions.
        Raw (un-normalized) int16 frames go into the collector and pre-roll.
        The collector normalizes the full segment once in finalize().
        Per-frame normalization destroys speech dynamics and causes
        Whisper to hallucinate.
        """
        # Normalize ONLY for VAD — so VAD sees proper signal levels
        normalized, peak, rms, noise_est = self.normalizer.normalize_int16(frames)

        # Store RAW frames in pre-roll buffer (not normalized!)
        self._pre_roll.append(frames.copy())
        self._pre_roll_samples += len(frames)
        while (
            self._pre_roll_samples > self._pre_roll_maxlen and len(self._pre_roll) > 1
        ):
            removed = self._pre_roll.popleft()
            self._pre_roll_samples -= len(removed)

        # Run VAD on normalized audio (VAD needs good levels)
        speech_prob = self.vad.process(normalized)

        if not self._speech_started:
            # Cooldown: don't start a new segment within 1.5s of finishing one
            now = time.time()
            if now - self._last_segment_time < 1.5:
                return

            if speech_prob >= self.config.speech_threshold:
                # Speech detected — start collecting with RAW pre-roll
                self._speech_started = True

                pre_roll = (
                    np.concatenate(list(self._pre_roll))
                    if self._pre_roll
                    else np.array([], dtype=np.int16)
                )
                self.collector.start(pre_roll)
                self.collector.add_frames(frames, speech_prob)  # RAW frames
                logger.info("Speech detected (prob=%.2f) — recording...", speech_prob)
        else:
            # Collecting speech — store RAW frames
            self.collector.add_frames(frames, speech_prob)

            if (
                self.collector.end_of_speech_detected()
                or self.collector.max_duration_exceeded()
            ):
                segment = self.collector.finalize()
                self._speech_started = False
                self._last_segment_time = time.time()

                # Clear pre-roll to avoid re-using old audio
                self._pre_roll.clear()
                self._pre_roll_samples = 0

                # Emit event — we're already on the event loop, safe to await
                await self.event_bus.emit("audio.speech_segment", segment)
                logger.info(
                    "Speech segment emitted: %.0fms, peak=%.3f, rms=%.3f",
                    segment.duration_ms,
                    segment.peak,
                    segment.rms,
                )
