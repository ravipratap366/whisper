import datetime
import whisper

import pyaudio
from langchain.llms import OpenAI
import wave
from pydub import AudioSegment
import streamlit as st
from googletrans import Translator
from gtts import gTTS
import os
import pyttsx3
import os

os.environ["OPEN_API_KEY"] = "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"

llm = OpenAI(openai_api_key=os.environ["OPEN_API_KEY"], temperature=0.6)

CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100


def speak(message, voice="default", speed=200):
    engine = pyttsx3.init()
    voices = engine.getProperty('voices')
    
    if voice == "male":
        engine.setProperty('voice', voices[0].id)
    elif voice == "female":
        engine.setProperty('voice', voices[1].id)
    else:
        engine.setProperty('voice', voices[0].id)  # Default to male voice

    engine.setProperty('rate', speed)  # Set the speed (words per minute)

    engine.say(message)
    engine.runAndWait()

# Audio recording function
def record_audio():
    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)

    st.sidebar.info("Start recording")
    frames = []
    seconds = 5

    for i in range(0, int(RATE / CHUNK * seconds)):
        data = stream.read(CHUNK)
        frames.append(data)

    st.sidebar.info("Recording stopped")
    stream.stop_stream()
    stream.close()
    p.terminate()

    # Save the recorded audio to a WAV file
    wave_output_filename = "recorded_output.wav"
    mp3_output_filename = "recorded_output.mp3"

    wf = wave.open(wave_output_filename, "wb")
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()

    # Convert WAV to MP3
    audio = AudioSegment.from_wav(wave_output_filename)
    audio.export(mp3_output_filename, format="mp3")

    return wave_output_filename

# Translation function
def translate_to_english(text):
    translator = Translator()
    translation = translator.translate(text, src='auto', dest='en')
    return translation.text




# Streamlit app setup
st.set_page_config(
    page_title="Whisper",
    page_icon="https://revoquant.com/assets/img/logo/logo-dark.png"
)

st.sidebar.title("Video to Text Transcription with Whisper")

# Button to start recording and transcription
if st.sidebar.button("Speak"):
    try:
        audio_filename = record_audio()

        # Load the Whisper ASR model
        model = whisper.load_model('base')

        st.sidebar.info("Starting transcription...")

        output = model.transcribe(audio_filename)

        st.sidebar.info("Transcription complete")
        # translated_text="hi there"


        # Display the transcription results
        st.write("Transcription Result:")
        st.write(output['text'])
        
        # Translate to English
        translated_text = translate_to_english(output['text'])
        st.write("Translation to English:")
        st.write(translated_text)

        # Get AI response
        response = llm.predict(translated_text)
        st.write("AI Response:")
        st.write(response)
        speak(response)


        # Speak the response


    except Exception as e:
        st.sidebar.error(f"An error occurred during transcription: {str(e)}")
