# TTS_CLI

TTS_CLI is a text-to-speech (TTS) script that allows you to convert text into spoken words using various TTS engines.

## Installation

1. Clone the repository:

```shell
git clone https://github.com/juanmirod/tts.git
```

2. Navigate to the project directory:

```shell
cd tts-whisper
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

5. Add your openAI key to the .env file:

```shell
cp .env.example .env
# open .env file and add your key
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

## Contributing

Contributions are welcome! If you encounter any issues or have suggestions for improvements, please open an issue or submit a pull request.

## License

This project is licensed under the [MIT License](LICENSE).
