from gtts import gTTS
from datetime import datetime
import argparse
from pydub import AudioSegment
from openai import OpenAI
from dotenv import load_dotenv
from .text_parser import parse_markdown, chunk_text
import importlib.util
import subprocess
import sys
import os
import numpy as np

load_dotenv()

def openai_tts(txt, speech_file_path=None, voice='nova', index=0):
    if speech_file_path is None:
        speech_file_path = f"tmp/chunks/tts_{voice}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{index}.mp3"
    client = OpenAI()
    response = client.audio.speech.create(model="tts-1", voice=voice, input=txt)

    response.stream_to_file(speech_file_path)
    return speech_file_path


def google_tts(txt, lang='en', voice='co.uk', index=0):
    tts = gTTS(text=txt, lang=lang, tld=voice, slow=False)
    filename = f"tmp/chunks/tts_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{index}.mp3"
    tts.save(filename)
    return filename

def check_install_dependencies():
    missing = []
    for package in ['torch', 'transformers', 'scipy']:
        if importlib.util.find_spec(package) is None:
            missing.append(package)
    
    if missing:
        print(f"Installing required dependencies for local TTS: {', '.join(missing)}")
        try:
            subprocess.check_call([sys.executable, '-m', 'pip', 'install'] + missing)
            return True
        except subprocess.CalledProcessError:
            print("Failed to install dependencies. Please install manually:")
            print(f"pip install {' '.join(missing)}")
            return False
    return True

def local_tts(txt, model_name, index=0):
    wav_filename = f"tmp/chunks/tts_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{index}.wav"
    mp3_filename = f"tmp/chunks/tts_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{index}.mp3"
    
    if not check_install_dependencies():
        raise RuntimeError("Required dependencies not available for local TTS")
    
    try:
        from transformers import pipeline
        from scipy.io import wavfile
        synthesizer = pipeline("text-to-speech", model=model_name)
        result = synthesizer(txt)
        
        # The audio data comes as normalized float32 (-1 to 1)
        audio_array = np.frombuffer(result["audio"], dtype=np.float32)
        
        # Convert to int16 format (required for wav)
        audio_int16 = (audio_array * 32767).astype(np.int16)
        
        # Save as WAV using the correct sampling rate from the model output
        wavfile.write(wav_filename, result["sampling_rate"], audio_int16)
        
        # Convert WAV to MP3
        audio = AudioSegment.from_wav(wav_filename)
        audio.export(mp3_filename, format="mp3")
        
        # Clean up temporary WAV file
        os.remove(wav_filename)
        
        return mp3_filename
    except Exception as e:
        print(f"Error using local model: {str(e)}")
        raise

def main():
    # Create the arguments parser
    default_output_file = f"tmp/tts_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp3"
    parser = argparse.ArgumentParser(
        description="Takes a markdown file and returns an mp3 file with the tts audio transcription.")
    parser.add_argument(
        'input_file',
        type=str,
        help='The input file to process.')
    parser.add_argument('-v', '--voice', type=str,
                        default='nova',
                        help='The voice. Valid values for openAI: nova, shimmer, echo, onyx, fable, alloy. Defaults to "nova".')
    parser.add_argument('-o', '--output', type=str,
                        default=default_output_file,
                        help='The output file to write to. Defaults to "tmp/tts_yyyymmdd_hhmmss.mp3".')
    parser.add_argument('-d', '--dry-run', action='store_true',
                        help='If true, it will output what would be send to the tts. Defaults to false.')
    parser.add_argument('-g', '--google-tts', action='store_true',
                        help='Use google tts (free but more robotic, you have to specify the language).')
    parser.add_argument('-l', '--language', type=str,
                        default='en',
                        help='The language to use with google tts. Defaults to "en".')
    parser.add_argument('--local-model', action='store_true',
                      help='Use a local HuggingFace model for TTS.')
    parser.add_argument('--model-name', type=str,
                      default='facebook/mms-tts-eng',
                      help='HuggingFace model name to use for local TTS. Default: facebook/mms-tts-eng')

    # Parse the arguments
    args = parser.parse_args()

    with open(args.input_file, 'r') as f:
        text = f.read()
    text = parse_markdown(text)
    chunks = chunk_text(text, max_length=4000)
    if args.dry_run:
        print(chunks)
    else:
        index = 0
        audio_chunks = []
        args.output = args.output.replace(".mp3", f"_{args.voice}.mp3")
        for chunk in chunks:
            print(f"Processing chunk {index} of {len(chunks)}, please wait...")
            if args.local_model:
                chunk_file_path = local_tts(chunk, args.model_name, index)
            elif args.google_tts:
                if args.voice == 'nova':
                    args.voice = 'co.uk'
                chunk_file_path = google_tts(chunk, voice=args.voice, index=index, lang=args.language)
            else:
                chunk_file_path = openai_tts(txt=chunk, voice=args.voice, index=index)
            audio_chunks.append(chunk_file_path)
            index += 1
        combine_chunks(audio_chunks, args.output)
        print(f"Done. Output file: {args.output}")

def combine_chunks(chunks, outputFile):
    combined = AudioSegment.empty()
    for chunk in chunks:
        audio = AudioSegment.from_file(chunk)
        combined += audio
    
    # Export the result
    combined.export(outputFile, format="mp3")

if __name__ == '__main__':
    main()
