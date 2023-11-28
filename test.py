import speech_recognition as sr

def display_text_from_audio(audio_file_path):
    # Initialize the recognizer
    recognizer = sr.Recognizer()

    # Load the audio file
    audio_file = sr.AudioFile(audio_file_path)

    # Read audio data from the file
    with audio_file as source:
        audio_data = recognizer.record(source)

    try:
        # Use the recognizer to convert speech to text
        text = recognizer.recognize_google(audio_data)
        print("{}".format(text))
    except sr.UnknownValueError:
        print("Could not understand audio")
    except sr.RequestError as e:
        print("Error with the speech recognition service; {0}".format(e))

# Replace 'your_audio_file.wav' with the path to your audio file
audio_file_path = 'recorded_output.wav'
display_text_from_audio(audio_file_path)


