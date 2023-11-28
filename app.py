import datetime
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import google.generativeai as palm
from langchain.embeddings import GooglePalmEmbeddings
from langchain.llms import GooglePalm
from langchain.vectorstores import FAISS
import whisper
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
import pyaudio
from langchain.llms import OpenAI
import wave
from pydub import AudioSegment
import streamlit as st
from googletrans import Translator
from sklearn.feature_extraction.text import TfidfVectorizer
from gtts import gTTS
import os
import pyttsx3
from sklearn.metrics.pairwise import cosine_similarity
import os
import numpy as np


os.environ['GOOGLE_API_KEY'] =  'AIzaSyB8cWDofkHVWOOFMbiUwr74pAEFnySQAHw'

CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100

def calculate_cosine_similarity(input_text, greetings):
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform([input_text] + greetings)
    similarity_scores = cosine_similarity(vectors[0], vectors[1:]).flatten()
    return similarity_scores


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
    seconds = 7

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


def get_pdf_text(pdf_docs):
    text=""
    for pdf in pdf_docs:
        pdf_reader= PdfReader(pdf)
        for page in pdf_reader.pages:
            text+= page.extract_text()
    return  text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=20)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    embeddings = GooglePalmEmbeddings()
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    return vector_store

def get_conversational_chain(vector_store):
    llm=OpenAI()
    memory = ConversationBufferMemory(memory_key = "chat_history", return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=vector_store.as_retriever(), memory=memory)
    return conversation_chain


# def user_input(user_question):
#     if st.session_state.conversation is not None:
#         response = st.session_state.conversation({'question': user_question})
#         st.session_state.chatHistory = response.get('chat_history', [])
#         for i, message in enumerate(st.session_state.chatHistory):
#             if i % 2 == 0:
#                 st.write(user_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
#             else:
#                 st.write(bot_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
#                 speak(message.content)  # Add this line to speak the bot's response
#     else:
#         st.warning("Conversation not initialized. Please upload PDFs and start the conversation.")



def user_input(user_question):
    if st.session_state.conversation is not None:
        response = st.session_state.conversation({'question': user_question})
        st.session_state.chatHistory = response.get('chat_history', [])
        
        if st.session_state.chatHistory:
            # Display only the current response
            bot_response = st.session_state.chatHistory[-1].content
            st.write(bot_response, unsafe_allow_html=True)
            
            # Speak the current response
            speak(bot_response)
    else:
        st.warning("Conversation not initialized. Please upload PDFs and start the conversation.")





def main():
    st.set_page_config("InfoWhisper Using Multiple PDFs")
    st.header("InfoWhisper")
    st.sidebar.header("InfoWhisper Using Multiple PDFs")


    if st.button("Speak"):
        while True:  # Add a loop to continuously capture and process user speech
            audio_filename = record_audio()



            model = whisper.load_model('small')
            st.sidebar.info("Starting transcription...")
            output = model.transcribe(audio_filename)

            st.write("Transcription Result:")
            st.write(output['text'])

            translated_text = translate_to_english(output['text'])
            st.write("Translation to English:")
            st.write(translated_text)
            user_question = translated_text

            if "conversation" not in st.session_state:
                st.session_state.conversation = None
            if "chatHistory" not in st.session_state:
                st.session_state.chatHistory = None

            greetings = ["hi", "hello", "hey", "greetings", "howdy", "hi there", "yo", "what's up"]
            similarity_scores = calculate_cosine_similarity(user_question.lower(), greetings)
            similarity_threshold = 0.5

            if np.max(similarity_scores) > similarity_threshold:
                speak("Hello, How can I assist you?")
            else:
                user_input(user_question)
    with st.sidebar:
        pdf_docs = st.file_uploader(" ", accept_multiple_files=True)
        if st.button("Process"):
            with st.spinner("Processing"):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                vector_store = get_vector_store(text_chunks)
                st.session_state.conversation = get_conversational_chain(vector_store)
                st.success("Done")



if __name__ == "__main__":
    main()
