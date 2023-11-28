# InfoWhisper

Created an interactive chatbot using Google's Palm, incorporating speaking functionality. This chatbot begins by taking a PDF as input from the user. In the backend, embeddings are generated using Google Palm embeddings. Subsequently, speaking functionality has been added.

The chatbot operates by having the user click on the "speak" button in the user interface. Utilizing PyAudio, it records the user's voice in MP3 format. If the user speaks English, the system processes the language accordingly. Otherwise, using OpenAI Whisper, transcriptions are generated. To handle multiple languages, we first convert them into English using Google Translator.

With Google Palm serving as a large language model, we predict responses to the user's questions. The speaking functionality is achieved through the pyttsx3 library.

I have implemented this particular project using two large language model service providers. The first one is OpenAI (closed source), and the second one is Hugging Face (open source).

## Steps to Run it
### 1. Cloning the Repository
```bash
git clone https://github.com/MANMEET75/InfoWhisper.git
```
### 2. Creating the virtual environment using anaconda
```bash
conda create -p venv python=3.11 -y
```

### 3. Activate the virtual environment
```bash
conda activate venv/
```

### 4. Installing the dependencies
```bash
pip install -r requirements.txt
```

### 5. Run the following commands in your anaconda prompt one by one

```bash
conda install -c conda-forge ffmpeg
```


```bash
pip install google-generativeai
```
```bash
sudo apt-get install portaudio19-dev
```


## 6. Check the InfoWhisper
```bash
streamlit run app.py
```
Enjoy Coding!
