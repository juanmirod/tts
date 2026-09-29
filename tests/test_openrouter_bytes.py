"""Tests for the in-memory OpenRouter TTS entry point."""

from unittest.mock import patch

from tts import openrouter


class TestOpenrouterTtsBytes:
    def test_returns_bytes_from_chunk_bytes_with_resolved_voice(self):
        with patch.object(openrouter, "resolve_voice", return_value="af_heart") as rv, \
                patch.object(openrouter, "_chunk_bytes", return_value=b"mp3") as cb:
            out = openrouter.openrouter_tts_bytes("hello", "kokoro", "nope")
        assert out == b"mp3"
        rv.assert_called_once_with("kokoro", "nope")
        cb.assert_called_once_with("hello", "kokoro", "af_heart")

    def test_file_wrapper_writes_the_bytes(self, tmp_path):
        target = tmp_path / "out.mp3"
        with patch.object(openrouter, "openrouter_tts_bytes", return_value=b"abc"):
            path = openrouter.openrouter_tts("hi", str(target))
        assert path == str(target)
        assert target.read_bytes() == b"abc"
