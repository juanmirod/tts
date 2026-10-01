"""Tests for the in-memory OpenRouter TTS entry point."""

from unittest.mock import MagicMock, patch

import pytest
import requests

from tts import openrouter


class TestOpenrouterTtsBytes:
    def test_returns_bytes_from_chunk_bytes_with_resolved_voice(self):
        with patch.object(openrouter, "resolve_voice", return_value="af_heart") as rv, \
                patch.object(openrouter, "_chunk_bytes", return_value=b"mp3") as cb:
            out = openrouter.openrouter_tts_bytes("hello", "kokoro", "nope")
        assert out == b"mp3"
        rv.assert_called_once_with("kokoro", "nope")
        cb.assert_called_once_with("hello", "kokoro", "af_heart", style=None,
                                   timeout=openrouter.DEFAULT_TIMEOUT, retries=0)

    def test_file_wrapper_writes_the_bytes(self, tmp_path):
        target = tmp_path / "out.mp3"
        with patch.object(openrouter, "openrouter_tts_bytes", return_value=b"abc"):
            path = openrouter.openrouter_tts("hi", str(target))
        assert path == str(target)
        assert target.read_bytes() == b"abc"


CATALOG = {
    "gemini-3.8-lite": {"id": "google/gemini-3.8-flash-lite-tts",
                        "format": "mp3", "voices": ["Kore"],
                        "style_provider": "google-ai-studio",
                        "vocal_tags": True},
    "kokoro": {"id": "hexgrad/kokoro-82m", "format": "mp3",
               "voices": ["af_heart"]},
}


def _posted_body(model, style):
    """Run _chunk_bytes with a mocked HTTP call; return the JSON body sent."""
    response = MagicMock(status_code=200, content=b"mp3", headers={})
    with patch.object(openrouter, "load_or_models", return_value=CATALOG), \
            patch.object(openrouter, "or_api_key", return_value="k"), \
            patch.object(openrouter.requests, "post", return_value=response) as post:
        openrouter._chunk_bytes("hi", model, "Kore", style=style)
    return post.call_args.kwargs["json"]


class TestStyle:
    def test_style_goes_to_the_catalog_provider_options(self):
        body = _posted_body("gemini-3.8-lite", "whispering")
        assert body["provider"] == {"options": {"google-ai-studio": {
            "speech_metadata": {"style": "whispering"}}}}

    def test_no_style_sends_no_provider_options(self):
        assert "provider" not in _posted_body("gemini-3.8-lite", None)

    def test_style_on_unsupported_model_raises(self):
        with patch.object(openrouter, "load_or_models", return_value=CATALOG):
            with pytest.raises(RuntimeError, match="no soporta --style"):
                openrouter._chunk_bytes("hi", "kokoro", "af_heart", style="calm")

    def test_capability_helpers_read_the_catalog(self):
        with patch.object(openrouter, "load_or_models", return_value=CATALOG):
            assert openrouter.supports_style("gemini-3.8-lite")
            assert openrouter.supports_vocal_tags("gemini-3.8-lite")
            assert not openrouter.supports_style("kokoro")
            assert not openrouter.supports_vocal_tags("kokoro")
            assert not openrouter.supports_style("unknown")

    def test_file_wrapper_passes_style_through(self, tmp_path):
        with patch.object(openrouter, "_chunk_bytes", return_value=b"x") as cb:
            openrouter.openrouter_tts("hi", str(tmp_path / "o.mp3"),
                                      model="kokoro", voice="af_heart",
                                      style="calm")
        assert cb.call_args.kwargs["style"] == "calm"


class TestTimeoutAndRetries:
    OK = MagicMock(status_code=200, content=b"mp3", headers={})

    def _call(self, post_side_effect, **kwargs):
        with patch.object(openrouter, "load_or_models", return_value=CATALOG), \
                patch.object(openrouter, "or_api_key", return_value="k"), \
                patch.object(openrouter.requests, "post",
                             side_effect=post_side_effect) as post:
            try:
                return openrouter._chunk_bytes("hi", "kokoro", "af_heart", **kwargs), post
            except Exception as exc:
                return exc, post

    def test_timeout_is_passed_to_requests(self):
        out, post = self._call([self.OK], timeout=7)
        assert out == b"mp3"
        assert post.call_args.kwargs["timeout"] == 7

    def test_hung_call_is_retried(self):
        out, post = self._call([requests.ReadTimeout(), self.OK], retries=1)
        assert out == b"mp3"
        assert post.call_count == 2

    def test_gives_up_after_the_retries(self):
        out, post = self._call([requests.ReadTimeout(), requests.ConnectionError()],
                               retries=1)
        assert isinstance(out, requests.ConnectionError)
        assert post.call_count == 2

    def test_no_retry_by_default(self):
        out, post = self._call([requests.ReadTimeout(), self.OK])
        assert isinstance(out, requests.ReadTimeout)
        assert post.call_count == 1

    def test_http_errors_are_not_retried(self):
        bad = MagicMock(status_code=500, text="boom", headers={})
        out, post = self._call([bad, self.OK], retries=1)
        assert isinstance(out, RuntimeError)
        assert post.call_count == 1
