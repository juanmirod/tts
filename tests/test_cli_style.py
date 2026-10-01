"""Tests for --style and vocal-tag handling in the CLI."""

import sys
from unittest.mock import patch

import pytest

from tts import tts as tts_mod


def _run(tmp_path, argv, text="Hello <sigh> there."):
    """Run main() with a temp input file and mocked OpenRouter TTS.
    Returns the mock so callers can inspect what was sent."""
    input_file = tmp_path / "input.md"
    input_file.write_text(text)
    with patch.object(sys, "argv", ["tts"] + argv + [str(input_file)]), \
            patch("tts.tts.combine_chunks"), \
            patch("tts.tts.openrouter_tts", return_value="out.mp3") as provider:
        tts_mod.main()
    return provider


def test_style_is_passed_to_every_chunk(tmp_path):
    provider = _run(tmp_path, ["-or", "--model", "gemini-3.8-lite",
                               "--style", "whispering"])
    assert provider.call_args_list
    assert all(c.kwargs["style"] == "whispering" for c in provider.call_args_list)


def test_no_style_by_default(tmp_path):
    provider = _run(tmp_path, ["-or", "--model", "gemini-3.8-lite"])
    assert provider.call_args.kwargs["style"] is None


@pytest.mark.parametrize("argv", [
    ["-or", "--model", "kokoro", "--style", "calm"],      # model can't do it
    ["-or", "--model", "gemini-flash", "--style", "calm"],  # 3.1 ignores style
    ["--style", "calm"],                                  # not OpenRouter
])
def test_style_rejected_when_unsupported(tmp_path, argv):
    with pytest.raises(SystemExit) as exc:
        _run(tmp_path, argv)
    assert exc.value.code == 2


def test_vocal_tags_kept_for_gemini_38(tmp_path):
    provider = _run(tmp_path, ["-or", "--model", "gemini-3.8-flash"])
    assert provider.call_args.kwargs["txt"] == "Hello <sigh> there."


def test_vocal_tags_stripped_for_other_models(tmp_path):
    provider = _run(tmp_path, ["-or", "--model", "kokoro"])
    assert provider.call_args.kwargs["txt"] == "Hello there."
