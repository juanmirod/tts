"""OpenRouter TTS provider (endpoint compatible con OpenAI Audio Speech).

Portado de tts_bot: permite usar cualquier modelo TTS del catálogo de
OpenRouter (kokoro, gemini-flash, aura-2, qwen, ...) con una sola API key
en `~/tts/models/openrouter.json` esta la lista de modelos + voces + precios.
"""

import io
import json
import os
import re

import requests

# Ruta del catálogo (modelos + voces + precios). Copia del de tts_bot.
MODELS_FILE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "models", "openrouter.json",
)
OR_API = "https://openrouter.ai/api/v1"

# Modelo por defecto para textos largos (barato: $0.62/M chars) y en inglés.
DEFAULT_MODEL = "kokoro"
# Voz inglesa por defecto de kokoro (recomendada en tts_bot).
DEFAULT_VOICE = "af_heart"


def load_or_models():
    """Carga el catálogo de modelos TTS de OpenRouter (dict slug -> info)."""
    try:
        with open(MODELS_FILE) as f:
            return json.load(f)
    except OSError:
        return {}


def or_api_key():
    """API key de OpenRouter desde el entorno (env o ~/.openrouter_key)."""
    key = os.environ.get("OPENROUTER_API_KEY")
    if key:
        return key.strip()
    try:
        with open(os.path.expanduser("~/.openrouter_key")) as f:
            return f.read().strip()
    except OSError:
        return None


def model_voices(model):
    """Voces válidas del modelo (del catálogo)."""
    OR_MODELS = load_or_models()
    info = OR_MODELS.get(model) or {}
    return info.get("voices", [])


def resolve_voice(model, voice, default=None):
    """Comprueba que la voz existe para el modelo. Si el usuario no la
    especifico (o la que dio no es valida), usa la default del modelo
    (af_heart para kokoro) y solo cae a la primera disponible si la
    default tampoco existe en el catalogo."""
    if default is None:
        default = DEFAULT_VOICE
    voices = model_voices(model)
    if voice and voice in voices:
        return voice
    if default in voices:
        return default
    if voices:
        return voices[0]
    return voice or default


def _chunk_bytes(chunk, model, voice):
    """POST a OpenRouter /audio/speech. Devuelve bytes mp3."""
    out_mod = load_or_models().get(model) or {}
    fmt = out_mod.get("format", "mp3")
    body = {
        "model": out_mod.get("id", model),
        "input": chunk,
        "voice": voice,
        "response_format": fmt,
    }
    key = or_api_key()
    if not key:
        raise RuntimeError(
            "OpenRouter no configurado: falta OPENROUTER_API_KEY "
            "(env o ~/.openrouter_key)"
        )
    headers = {"Authorization": "Bearer " + key, "Content-Type": "application/json"}
    r = requests.post(OR_API + "/audio/speech", json=body, headers=headers, timeout=120)
    if r.status_code != 200:
        raise RuntimeError(
            "OpenRouter TTS HTTP %d: %s" % (r.status_code, r.text[:300])
        )
    data = r.content
    if fmt == "pcm":
        data = _pcm_to_mp3(data, (r.headers.get("Content-Type") or ""))
    return data


def _pcm_to_mp3(pcm_bytes, ctype):
    """Convierte audio pcm (s16le) a mp3 con pydub (para modelos tipo Gemini)."""
    try:
        from pydub import AudioSegment
    except ImportError:
        raise RuntimeError("pcm -> mp3 requiere pydub/ffmpeg")
    rate, channels = 24000, 1
    m = re.search(r"rate=(\d+)", ctype)
    if m:
        rate = int(m.group(1))
    m = re.search(r"channels=(\d+)", ctype)
    if m:
        channels = int(m.group(1))
    seg = AudioSegment(data=pcm_bytes, sample_width=2,
                       frame_rate=rate, channels=channels)
    buf = io.BytesIO()
    seg.export(buf, format="mp3")
    return buf.getvalue()


def openrouter_tts(txt, speech_file_path, model=DEFAULT_MODEL,
                   voice=DEFAULT_VOICE, index=0):
    """Genera el audio de `txt` con OpenRouter y lo guarda en
    `speech_file_path`. Devuelve la ruta."""
    data = _chunk_bytes(txt, model, resolve_voice(model, voice))
    if not speech_file_path:
        from datetime import datetime
        speech_file_path = "tmp/chunks/tts_%s_%s_%d.mp3" % (
            model.replace("/", "_"),
            datetime.now().strftime("%Y%m%d_%H%M%S"),
            index,
        )
        os.makedirs(os.path.dirname(speech_file_path) or ".", exist_ok=True)
    with open(speech_file_path, "wb") as f:
        f.write(data)
    return speech_file_path