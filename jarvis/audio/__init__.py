"""
jarvis.audio — Audio capture, VAD, amplitude normalization, and speech collection.
"""

from jarvis.audio.audio_system import (
    AudioSystem,
    AmplitudeNormalizer,
    SpeechCollector,
    WakeWordDetector,
    VADEngine,
)

__all__ = [
    "AudioSystem",
    "AmplitudeNormalizer",
    "SpeechCollector",
    "WakeWordDetector",
    "VADEngine",
]
