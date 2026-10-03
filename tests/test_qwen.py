"""Qwen3-TTS provider: speaker/language resolution and CLI wiring."""

import sys
from unittest.mock import patch

import pytest

from tts import qwen
from tts import tts as tts_mod
from tts.text_parser import chunk_text


@pytest.mark.parametrize("voice,expected", [
    ("ryan", "Ryan"), ("Vivian", "Vivian"), ("uncle_fu", "Uncle_Fu"),
    ("nova", "Ryan"),   # the CLI's OpenAI default isn't a Qwen speaker
    (None, "Ryan"),
])
def test_resolve_speaker(voice, expected):
    assert qwen.resolve_speaker(voice) == expected


@pytest.mark.parametrize("language,expected", [
    ("en", "English"), ("ES", "Spanish"), ("Spanish", "Spanish"), (None, "English"),
])
def test_resolve_language(language, expected):
    assert qwen.resolve_language(language) == expected


def _run(tmp_path, argv, text="Hello there."):
    input_file = tmp_path / "input.md"
    input_file.write_text(text)
    with patch.object(sys, "argv", ["tts"] + argv + [str(input_file)]), \
            patch("tts.tts.combine_chunks") as combine, \
            patch("tts.tts.qwen.qwen_tts", return_value="c.mp3") as provider:
        tts_mod.main()
    return provider, combine


def test_cli_passes_speaker_language_and_model(tmp_path):
    provider, combine = _run(tmp_path, ["--qwen", "-v", "aiden", "-l", "es",
                                        "-o", "out.mp3"])
    kwargs = provider.call_args.kwargs
    assert kwargs["voice"] == "Aiden"
    assert kwargs["language"] == "es"
    assert kwargs["model_name"] == qwen.DEFAULT_MODEL
    assert combine.call_args.args[1] == "out_Aiden.mp3"


def test_cli_defaults_to_ryan(tmp_path):
    provider, _ = _run(tmp_path, ["--qwen"])
    assert provider.call_args.kwargs["voice"] == "Ryan"


def test_qwen_chunks_are_small_by_default(tmp_path):
    text = ("A sentence of about forty characters. " * 20).strip()
    provider, _ = _run(tmp_path, ["--qwen"], text=text)
    assert provider.call_count > 1
    assert all(len(c.args[0]) <= tts_mod.QWEN_MAX_CHUNK
               for c in provider.call_args_list)


def test_max_chars_overrides_the_provider_default(tmp_path):
    text = ("A sentence of about forty characters. " * 20).strip()
    provider, _ = _run(tmp_path, ["--qwen", "--max-chars", "1000"], text=text)
    assert provider.call_count == 1


def test_long_paragraph_splits_at_sentence_boundaries():
    text = "First sentence here. Second sentence here. Third one."
    assert chunk_text(text, 45) == ["First sentence here. Second sentence here.",
                                    "Third one."]


def test_sentence_longer_than_limit_falls_back_to_words():
    chunks = chunk_text("Short one. " + "word " * 10 + "end.", 20)
    assert chunks[0] == "Short one."
    assert all(len(c) <= 20 for c in chunks)
