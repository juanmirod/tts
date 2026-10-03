# TTS_CLI

TTS_CLI is a text-to-speech (TTS) script that allows you to convert text into spoken words using various TTS engines.

## Installation

1. Clone the repository:

```shell
git clone https://github.com/juanmirod/tts.git
```

2. Navigate to the project directory:

```shell
cd tts
```

3. Create and activate a virtual environment:

```shell
python3 -m venv venv
source venv/bin/activate
```

4. Install the required dependencies:

```shell
pip install -r requirements.txt
```

5. Add your API keys to the .env file:

```shell
cp .env.sample .env
# open .env file and add your keys
```

### Use it as a library

The repo ships a `pyproject.toml`, so other projects can import it from an editable install (`pip install -e /path/to/tts`). The package only pulls in `requests` and `pydub` (plus `ffmpeg` on the system for PCM -> MP3 conversion); the CLI (`tts.tts`) still needs `requirements.txt` (gtts, openai, huggingface-hub, ...). See [Using it as a library](#using-it-as-a-library).

## Usage

tts is a command line command, you can run -h for help:

```shell
python -m tts.tts -h
```

Run it with the sample txt file and the default voice in OpenAI tts API:

```shell
python -m tts.tts sample.txt
```

Run it with google tts (free):

```shell
python -m tts.tts -g sample.txt
```

### OpenRouter TTS (many models & voices)

Use any TTS model from OpenRouter's catalog with a single API key. The catalog with models, voices and prices lives in `models/openrouter.json`.

Run with OpenRouter (default model: kokoro, cheap and great for long/English texts):

```shell
python -m tts.tts -or sample.txt
```

Text is automatically chunked at 1500 characters for OpenRouter (other providers chunk at 4000), since OpenRouter rejects larger chunks — long files keep working.

Pick a different model and voice:

```shell
# English, cheap ($0.62/M chars):
python -m tts.tts -or --model kokoro -v af_heart sample.txt
# Premium Gemini voice (needs PCM->MP3 conversion, handled automatically):
python -m tts.tts -or --model gemini-3.8-flash -v Aoede sample.txt
# Cheaper Gemini 3.8 tier (~$0.69/hour of audio vs ~$1.04 for flash):
python -m tts.tts -or --model gemini-3.8-lite -v Kore sample.txt
# Deepgram (90 voices):
python -m tts.tts -or --model aura-2 -v aura-2-agustina-es sample.txt
```

Gemini 3.8 models (`gemini-3.8-flash`, `gemini-3.8-lite`) also take a delivery style for the whole text, and act on inline vocal tags written in the input (for other models the tags are stripped like HTML):

```shell
python -m tts.tts -or --model gemini-3.8-flash -v Kore --style "calm, warm narrator" story.md
```

```markdown
I have a secret to tell you. <short pause> Nobody knows. <sigh>
```

Useful tags: `<laugh>`, `<sigh>`, `<gasp>`, `<cough>`, `<breath>`, `<short pause>`, `<long pause>`. Tags are lowercase words (`<laughs softly>` works too); HTML tags such as `<b>` or `<div>` are still stripped as usual.

`--style` is applied to every chunk. Only the Gemini 3.8 models support it: `gemini-flash` (3.1) accepts the field but ignores it, and the CLI exits with an error if you pass `--style` with a model that doesn't support it (or without `-or`). Gemini models return PCM only (OpenRouter rejects mp3 for them); it is converted to MP3 automatically with pydub/ffmpeg.

Available models and their voices:

```shell
python -c "import json,os; d=json.load(open('models/openrouter.json')); [print(m, '->', ', '.join(v['voices'][:6]), '...') for m,v in d.items()]"
```

Set your key in `.env` (or `~/.openrouter_key`):

```shell
OPENROUTER_API_KEY=your_key
```

`-v` still selects the voice; `--model` picks the model. The default stays OpenAI `tts-1` unless you pass `-or`. If the voice isn't in the model's catalog, the model's default voice is used instead (`af_heart` for kokoro).

Kokoro also has Spanish voices (`ef_dora`, `em_alex`):

```shell
python -m tts.tts -or --model kokoro -v ef_dora sample.txt
```

List available HuggingFace TTS models:

```shell
python -m tts.tts --list-hf-models
```

Example output:

```
Fetching top 20 HuggingFace TTS models...

Model ID                                   | Downloads | Likes
---------------------------------------------------------------------
espnet/eng_male_fgl                        |    123456 |   987
facebook/mms-tts-eng                       |    234567 |   876
...

Found 20 model(s).
```

Search for HuggingFace TTS models by keyword:

```shell
python -m tts.tts --search-hf-models hindi
```

Example output:

```
Searching for HuggingFace TTS models matching 'hindi'...

Model ID                              | Downloads | Likes
--------------------------------------------------------------------
espnet/hindi_male_fgl                 |     54321 |   432
...

Found 3 model(s).
```

Run a HuggingFace model locally (free, runs on your machine; needs `torch`, `transformers` and `scipy`, which the CLI offers to `pip install`):

```shell
python -m tts.tts --local-model --model-name facebook/mms-tts-eng sample.txt
```

> **Not every listed model works.** Local synthesis goes through `transformers.pipeline("text-to-speech")`, so only architectures your installed `transformers` supports can be loaded (e.g. `facebook/mms-tts-*`, `suno/bark-small`). The listing is sorted by likes, and many of the top entries ship their own libraries and fail with `Unrecognized model ...`. Two of them have a dedicated option here: Qwen3-TTS (`--qwen`, below) and Kokoro (via OpenRouter: `-or --model kokoro`).

If you get `401 ... OAuth token has expired`, your cached token in `~/.cache/huggingface/token` is stale: run `huggingface-cli login` again or delete that file (listing public models doesn't need a token).

### Qwen3-TTS (local, multilingual)

[Qwen3-TTS](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice) runs on your machine (GPU recommended, CPU works but is slow) and speaks 10 languages. It doesn't work through `pipeline("text-to-speech")`: it needs its own package, which also requires a recent `transformers` and `torch`:

```shell
pip install -U qwen-tts soundfile torch torchaudio
```

```shell
python -m tts.tts --qwen sample.txt
# Spanish text with another speaker:
python -m tts.tts --qwen -l es -v Aiden sample.txt
```

- `-v` picks the speaker: `Ryan` (default), `Aiden` (English), `Vivian`, `Serena`, `Uncle_Fu`, `Dylan`, `Eric` (Chinese), `Ono_Anna` (Japanese), `Sohee` (Korean). The speakers are grouped by native language, but other languages work too (an English speaker reading Spanish was tested).
- `-l` picks the language: `en`, `es`, `fr`, `de`, `it`, `pt`, `ru`, `zh`, `ja`, `ko`.
- `--qwen-model` picks the model. The default is `Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice` (about 2 GB of GPU memory); `Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice` sounds better but needs a bigger GPU. The first run downloads the model.
- Text is chunked at 120 characters, because the model's memory use grows with the length of each chunk (about 5 MB of GPU memory per character; a 4 GB GPU ran out of memory at ~190). With more GPU memory, raise it with `--max-chars 400` for fewer, more natural-sounding chunks. `--max-chars` works with any provider.
- Speed on a 4 GB laptop GPU: roughly real time (140 s of audio in about 3 minutes).

### Using it as a library

`tts.openrouter` can be imported without the CLI. `openrouter_tts_bytes` returns MP3 bytes without touching disk; the text must fit in one chunk (<= 1500 chars), use `chunk_text` from `tts.text_parser` to split longer texts.

```python
from tts.openrouter import openrouter_tts_bytes
from tts.text_parser import parse_markdown, chunk_text, strip_vocal_tags

text = parse_markdown(open("story.md").read(), keep_vocal_tags=True)
for chunk in chunk_text(text, 1500):
    mp3 = openrouter_tts_bytes(
        chunk,
        model="gemini-3.8-flash",
        voice="Kore",
        style="calm, warm narrator",  # Gemini 3.8 only
        timeout=30,                   # seconds per call (default 120)
        retries=2,                    # default 0
    )
```

- `timeout` / `retries`: timeouts and connection errors are retried `retries` times; HTTP errors are not. Useful because some providers (Kokoro on OpenRouter) occasionally never answer.
- `parse_markdown(..., keep_vocal_tags=True)` keeps inline tags like `<sigh>`; use `strip_vocal_tags(text)` when sending tagged text to a model that would read them aloud.
- `supports_style(model)`, `supports_vocal_tags(model)`, `model_voices(model)` and `resolve_voice(model, voice)` query the catalog.

## Tests

```shell
pip install pytest
python -m pytest
```

## Contributing

Contributions are welcome! If you encounter any issues or have suggestions for improvements, please open an issue or submit a pull request.

## License

This project is licensed under the [MIT License](LICENSE).
