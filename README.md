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

Use a HuggingFace TTS model:

```shell
python -m tts.tts -hf --hf-model espnet/eng_male_fgl sample.txt
```

## Contributing

Contributions are welcome! If you encounter any issues or have suggestions for improvements, please open an issue or submit a pull request.

## License

This project is licensed under the [MIT License](LICENSE).
