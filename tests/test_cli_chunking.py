"""Tests for provider-specific chunk sizing in the CLI.

OpenRouter chunks at 1500 chars (it rejects larger ones), all other
providers keep the 4000 char default.
"""

import sys
from unittest.mock import patch

from tts import tts as tts_mod
from tts.tts import get_chunks, OPENAI_MAX_CHUNK, OPENROUTER_MAX_CHUNK

# ~15000 chars of single-space-separated words, no paragraph breaks,
# so it exercises hard-splitting too.
LONG_TEXT = ("word " * 3000).strip()


def _run_main(tmp_path, argv, provider_patch):
    """Run main() with a temp input file and a mocked provider TTS.

    Returns the list of chunks that were passed to the provider.
    """
    input_file = tmp_path / "input.txt"
    input_file.write_text("ignored")
    with patch.object(sys, "argv", ["tts"] + argv + [str(input_file)]), \
            patch("tts.tts.parse_markdown", return_value=LONG_TEXT), \
            patch("tts.tts.combine_chunks"), \
            provider_patch as provider:
        tts_mod.main()
    return [call.kwargs["txt"] for call in provider.call_args_list]


class TestGetChunks:
    """Tests for the chunk-size selector."""

    def test_openrouter_uses_1500_limit(self):
        with patch("tts.tts.chunk_text", return_value=["x"]) as mock_chunk:
            get_chunks("some text", use_openrouter=True)
        mock_chunk.assert_called_once_with("some text", OPENROUTER_MAX_CHUNK)

    def test_default_uses_4000_limit(self):
        with patch("tts.tts.chunk_text", return_value=["x"]) as mock_chunk:
            get_chunks("some text", use_openrouter=False)
        mock_chunk.assert_called_once_with("some text", OPENAI_MAX_CHUNK)


class TestCLIChunkSizes:
    """End-to-end chunk sizes through main()."""

    def test_openrouter_chunks_at_1500(self, tmp_path):
        chunks = _run_main(
            tmp_path, ["-or"],
            patch("tts.tts.openrouter_tts", return_value="out.mp3"),
        )
        assert chunks
        assert all(len(c) <= OPENROUTER_MAX_CHUNK for c in chunks)
        assert " ".join(chunks) == LONG_TEXT

    def test_default_chunks_at_4000(self, tmp_path):
        chunks = _run_main(
            tmp_path, [],
            patch("tts.tts.openai_tts", return_value="out.mp3"),
        )
        assert chunks
        assert all(len(c) <= OPENAI_MAX_CHUNK for c in chunks)
        assert " ".join(chunks) == LONG_TEXT
        # With the 4000 limit we need fewer chunks than with 1500.
        assert 1 < len(chunks) < len(LONG_TEXT) // OPENROUTER_MAX_CHUNK

    def test_dry_run_openrouter_shows_1500_chunks(self, tmp_path):
        input_file = tmp_path / "input.txt"
        input_file.write_text("ignored")
        with patch.object(sys, "argv", ["tts", "-or", "-d", str(input_file)]), \
                patch("tts.tts.parse_markdown", return_value=LONG_TEXT), \
                patch("tts.tts.print") as mock_print:
            tts_mod.main()
        chunks = mock_print.call_args[0][0]
        assert all(len(c) <= OPENROUTER_MAX_CHUNK for c in chunks)
