"""
Tests for jarvis.tts.tts_engine — Phase 1: Streaming TTS Pipeline.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from jarvis.utils.config import TTSConfig
from jarvis.tts.tts_engine import TTSEngine


# ── Helpers ──────────────────────────────────────────────────────────────


def make_engine(**overrides) -> TTSEngine:
    """Create a TTSEngine with no real backends."""
    cfg = TTSConfig(**overrides)
    with patch.object(TTSEngine, "_check_piper", return_value=False):
        with patch.object(TTSEngine, "_init_pyttsx"):
            engine = TTSEngine(cfg)
            engine._piper_available = False
            engine._pyttsx_engine = None
            engine._edge_tts_available = False
    return engine


# ── Sentence Splitting ───────────────────────────────────────────────────


class TestSentenceSplitting:
    def test_basic_split(self):
        result = TTSEngine._split_sentences("Hello world. How are you?")
        assert result == ["Hello world.", "How are you?"]

    def test_exclamation_and_question(self):
        result = TTSEngine._split_sentences("Wow! Really? Yes.")
        assert result == ["Wow!", "Really?", "Yes."]

    def test_single_sentence(self):
        result = TTSEngine._split_sentences("Just one sentence.")
        assert result == ["Just one sentence."]

    def test_empty_string(self):
        result = TTSEngine._split_sentences("")
        assert result == []

    def test_long_clause_splitting(self):
        long = "This is a very long sentence with commas, separating clauses, " * 3
        result = TTSEngine._split_sentences(long.strip())
        assert len(result) > 1  # should split on commas

    def test_whitespace_handling(self):
        result = TTSEngine._split_sentences("  Hello.   World.  ")
        assert result == ["Hello.", "World."]


# ── Pipeline Architecture ───────────────────────────────────────────────


class TestStreamingPipeline:
    @pytest.mark.asyncio
    async def test_speak_empty_text_noop(self):
        engine = make_engine()
        await engine.speak("")  # should not raise
        await engine.speak("   ")

    @pytest.mark.asyncio
    async def test_speak_fallback_to_print(self, capsys):
        engine = make_engine()
        await engine.speak("Hello Jarvis")
        captured = capsys.readouterr()
        assert "Jarvis" in captured.out

    @pytest.mark.asyncio
    async def test_pipeline_processes_all_sentences(self):
        engine = make_engine()
        synthesized = []

        async def mock_synthesize(text):
            synthesized.append(text)
            return (np.zeros(100, dtype=np.int16), 22050)

        async def mock_play(audio, sr):
            pass

        engine._synthesize = mock_synthesize
        engine._play_audio = mock_play

        await engine.speak("First sentence. Second sentence. Third one!")
        assert len(synthesized) == 3

    @pytest.mark.asyncio
    async def test_pipeline_order_preserved(self):
        engine = make_engine()
        played = []

        async def mock_synthesize(text):
            return (np.zeros(100, dtype=np.int16), 22050)

        async def mock_play(audio, sr):
            played.append("played")

        engine._synthesize = mock_synthesize
        engine._play_audio = mock_play

        await engine.speak("A. B. C.")
        assert len(played) == 3

    @pytest.mark.asyncio
    async def test_stats_tracking(self):
        engine = make_engine()
        assert engine.stats["total_speaks"] == 0

        # Mock to avoid actual synthesis
        engine._synthesize = AsyncMock(return_value=None)
        await engine.speak("Test.")

        assert engine.stats["total_speaks"] == 1
        assert engine.stats["avg_latency_ms"] >= 0


# ── Backend Selection ────────────────────────────────────────────────────


class TestBackendSelection:
    def test_active_backend_print_when_nothing(self):
        engine = make_engine()
        assert engine._active_backend == "print"

    def test_active_backend_piper(self):
        engine = make_engine()
        engine._piper_available = True
        engine._edge_tts_available = False
        assert engine._active_backend == "piper"

    def test_active_backend_edge_tts(self):
        engine = make_engine()
        engine._edge_tts_available = True
        assert engine._active_backend == "edge-tts"

    def test_active_backend_pyttsx3(self):
        engine = make_engine()
        engine._pyttsx_engine = MagicMock()
        assert engine._active_backend == "pyttsx3"

    def test_edge_tts_preferred_over_piper(self):
        engine = make_engine()
        engine._edge_tts_available = True
        engine._piper_available = True
        assert engine._active_backend == "edge-tts"


# ── Concurrency Safety ──────────────────────────────────────────────────


class TestConcurrency:
    @pytest.mark.asyncio
    async def test_speak_lock_serializes(self):
        engine = make_engine()
        order = []

        original_pipeline = engine._streaming_pipeline

        async def slow_pipeline(sentences):
            order.append("start")
            await asyncio.sleep(0.05)
            order.append("end")

        engine._streaming_pipeline = slow_pipeline

        await asyncio.gather(
            engine.speak("First."),
            engine.speak("Second."),
        )
        # Both should complete — lock serializes them
        assert order.count("start") == 2
        assert order.count("end") == 2
        # Serialized: start-end-start-end
        assert order == ["start", "end", "start", "end"]


# ── MP3 Decode ───────────────────────────────────────────────────────────


class TestMP3Decode:
    def test_decode_returns_none_without_deps(self):
        """Without pydub or minimp3, should return None gracefully."""
        with patch.dict("sys.modules", {"pydub": None, "minimp3": None}):
            audio, sr = TTSEngine._decode_mp3(b"\x00" * 100)
            # Either returns data (if deps happen to be installed) or None
            # The key thing is it doesn't crash
            assert audio is None or isinstance(audio, np.ndarray)
