"""Tests for jarvis.tts.ssml_processor — Emotive SSML tags."""

from jarvis.tts.ssml_processor import SSMLProcessor, SSMLConfig


class TestSSMLAnnotation:
    def test_question_gets_pitch(self):
        proc = SSMLProcessor()
        result = proc.process("How are you?")
        assert "prosody" in result
        assert 'pitch="' in result

    def test_exclamation_gets_emphasis(self):
        proc = SSMLProcessor()
        result = proc.process("That is amazing!")
        assert "emphasis" in result

    def test_ellipsis_gets_pause(self):
        proc = SSMLProcessor()
        result = proc.process("Well...")
        assert "break" in result

    def test_normal_sentence_gets_break(self):
        proc = SSMLProcessor()
        result = proc.process("Hello world.")
        assert 'break time="300ms"' in result

    def test_long_comma_sentence_gets_pauses(self):
        proc = SSMLProcessor()
        long = "This is a very long sentence with commas, and more words, and even more content here."
        result = proc.process(long)
        assert 'break time="200ms"' in result


class TestSSMLStrip:
    def test_strip_removes_tags(self):
        proc = SSMLProcessor()
        tagged = '<prosody pitch="+10%">Hello?</prosody><break time="300ms"/>'
        clean = proc.strip(tagged)
        assert "<" not in clean
        assert "Hello?" in clean

    def test_strip_plain_text_unchanged(self):
        proc = SSMLProcessor()
        assert proc.strip("hello world") == "hello world"


class TestSSMLDisabled:
    def test_disabled_returns_plain(self):
        cfg = SSMLConfig(enabled=False)
        proc = SSMLProcessor(config=cfg)
        result = proc.process("How are you?")
        assert "<" not in result
        assert result == "How are you?"


class TestSSMLEdgeCases:
    def test_empty_text(self):
        proc = SSMLProcessor()
        assert proc.process("") == ""
        assert proc.process("   ") == "   "

    def test_unicode_ellipsis(self):
        proc = SSMLProcessor()
        result = proc.process("Hmm\u2026")
        assert "break" in result

    def test_multiple_sentences(self):
        proc = SSMLProcessor()
        result = proc.process("Hello. How are you? Great!")
        assert result.count("break") >= 2
