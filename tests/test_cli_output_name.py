"""The output file name only carries a voice suffix when a voice applies."""

import sys
from unittest.mock import patch

from tts import tts as tts_mod


def _output_path(tmp_path, argv):
    input_file = tmp_path / "input.md"
    input_file.write_text("Hello there.")
    with patch.object(sys, "argv", ["tts"] + argv + [str(input_file)]), \
            patch("tts.tts.combine_chunks") as combine, \
            patch("tts.tts.local_tts", return_value="c.mp3"), \
            patch("tts.tts.openrouter_tts", return_value="c.mp3"):
        tts_mod.main()
    return combine.call_args.args[1]


def test_local_model_has_no_voice_suffix(tmp_path):
    out = _output_path(tmp_path, ["--local-model", "-o", "out.mp3"])
    assert out == "out.mp3"


def test_openrouter_uses_the_resolved_voice(tmp_path):
    # -v defaults to the OpenAI "nova"; kokoro resolves it to its own default.
    out = _output_path(tmp_path, ["-or", "--model", "kokoro", "-o", "out.mp3"])
    assert out == "out_af_heart.mp3"
