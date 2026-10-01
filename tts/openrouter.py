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
# Segundos de espera por chunk. Generoso para el CLI; readbuddy usa menos.
DEFAULT_TIMEOUT = 120


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


def supports_style(model):
    """True si el modelo acepta un estilo de locución (--style). El catálogo
    indica en `style_provider` el proveedor de OpenRouter al que va."""
    return bool((load_or_models().get(model) or {}).get("style_provider"))


def supports_vocal_tags(model):
    """True si el modelo interpreta etiquetas inline como <sigh> o
    <short pause> (Gemini 3.8), así que no hay que borrarlas del texto."""
    return bool((load_or_models().get(model) or {}).get("vocal_tags"))


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


def _chunk_bytes(chunk, model, voice, style=None, timeout=DEFAULT_TIMEOUT,
                 retries=0):
    """POST a OpenRouter /audio/speech. Devuelve bytes mp3. `style` es el
    estilo de locución (p.ej. "whispering") para modelos que lo soportan.
    Un timeout o fallo de conexión se reintenta `retries` veces (a veces el
    proveedor se cuelga y no responde); un error HTTP no se reintenta."""
    out_mod = load_or_models().get(model) or {}
    fmt = out_mod.get("format", "mp3")
    body = {
        "model": out_mod.get("id", model),
        "input": chunk,
        "voice": voice,
        "response_format": fmt,
    }
    if style:
        slug = out_mod.get("style_provider")
        if not slug:
            raise RuntimeError("El modelo %s no soporta --style" % model)
        body["provider"] = {
            "options": {slug: {"speech_metadata": {"style": style}}}
        }
    key = or_api_key()
    if not key:
        raise RuntimeError(
            "OpenRouter no configurado: falta OPENROUTER_API_KEY "
            "(env o ~/.openrouter_key)"
        )
    headers = {"Authorization": "Bearer " + key, "Content-Type": "application/json"}
    for attempt in range(retries + 1):
        try:
            r = requests.post(OR_API + "/audio/speech", json=body,
                              headers=headers, timeout=timeout)
            break
        except (requests.Timeout, requests.ConnectionError):
            if attempt == retries:
                raise
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


def openrouter_tts_bytes(txt, model=DEFAULT_MODEL, voice=DEFAULT_VOICE,
                         style=None, timeout=DEFAULT_TIMEOUT, retries=0):
    """Genera el audio de `txt` con OpenRouter y devuelve los bytes mp3
    (sin tocar disco). `txt` debe caber en un chunk (<= 1500 chars).
    `timeout` (segundos) y `retries`: ver _chunk_bytes."""
    return _chunk_bytes(txt, model, resolve_voice(model, voice), style=style,
                        timeout=timeout, retries=retries)


def openrouter_tts(txt, speech_file_path, model=DEFAULT_MODEL,
                   voice=DEFAULT_VOICE, index=0, style=None):
    """Genera el audio de `txt` con OpenRouter y lo guarda en
    `speech_file_path`. Devuelve la ruta."""
    data = openrouter_tts_bytes(txt, model, voice, style=style)
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