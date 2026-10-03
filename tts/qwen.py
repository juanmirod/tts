"""Local Qwen3-TTS provider.

Qwen3-TTS doesn't work with transformers.pipeline("text-to-speech"): it needs
its own `qwen-tts` package (pip install qwen-tts), which also pins a recent
transformers. Imports are lazy so the rest of the CLI works without it.
"""

import functools
import io
import os
from datetime import datetime

# 0.6B fits in ~4 GB of GPU memory; Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice is
# better but needs more. Both CustomVoice models share the speakers below.
DEFAULT_MODEL = "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice"
DEFAULT_SPEAKER = "Ryan"
SPEAKERS = [
    "Ryan", "Aiden",                                    # English
    "Vivian", "Serena", "Uncle_Fu", "Dylan", "Eric",    # Chinese
    "Ono_Anna",                                         # Japanese
    "Sohee",                                            # Korean
]
# ISO codes (the CLI's -l option) -> language names the model expects.
LANGUAGES = {
    "zh": "Chinese", "en": "English", "ja": "Japanese", "ko": "Korean",
    "de": "German", "fr": "French", "ru": "Russian", "pt": "Portuguese",
    "es": "Spanish", "it": "Italian",
}


def resolve_speaker(voice):
    """Speaker name (case-insensitive) from SPEAKERS, or DEFAULT_SPEAKER when
    the voice isn't one of them (e.g. the CLI's OpenAI default "nova")."""
    for name in SPEAKERS:
        if voice and voice.lower() == name.lower():
            return name
    return DEFAULT_SPEAKER


def resolve_language(language):
    """Language name for an ISO code ("es" -> "Spanish"); anything else is
    passed through so full names like "Spanish" work too."""
    return LANGUAGES.get((language or "en").lower(), language)


@functools.lru_cache(maxsize=1)
def load_model(model_name=DEFAULT_MODEL):
    """Loads the model once per process (GPU in bf16 if available)."""
    try:
        import torch
        from qwen_tts import Qwen3TTSModel
    except ImportError:
        raise RuntimeError(
            "Qwen3-TTS needs the qwen-tts package: pip install qwen-tts")
    cuda = torch.cuda.is_available()
    return Qwen3TTSModel.from_pretrained(
        model_name,
        device_map="cuda:0" if cuda else "cpu",
        dtype=torch.bfloat16 if cuda else torch.float32,
    )


def qwen_tts(txt, model_name=DEFAULT_MODEL, voice=DEFAULT_SPEAKER,
             language="en", index=0, speech_file_path=None):
    """Synthesizes `txt` locally with Qwen3-TTS and saves an mp3. Returns the
    path."""
    import soundfile as sf
    from pydub import AudioSegment

    model = load_model(model_name)
    wavs, rate = model.generate_custom_voice(
        text=txt, speaker=resolve_speaker(voice),
        language=resolve_language(language))
    wav = io.BytesIO()
    sf.write(wav, wavs[0], rate, format="WAV")
    wav.seek(0)
    if not speech_file_path:
        speech_file_path = "tmp/chunks/tts_qwen_%s_%d.mp3" % (
            datetime.now().strftime("%Y%m%d_%H%M%S"), index)
    os.makedirs(os.path.dirname(speech_file_path) or ".", exist_ok=True)
    AudioSegment.from_wav(wav).export(speech_file_path, format="mp3")
    return speech_file_path
