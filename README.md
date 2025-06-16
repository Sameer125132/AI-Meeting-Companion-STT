# Business AI Meeting Companion 🚀

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Framework: Gradio](https://img.shields.io/badge/Framework-Gradio-orange)](https://gradio.app/)
[![AI: Whisper | Llama 3](https://img.shields.io/badge/AI-Whisper%20%7C%20Llama%203-blueviolet)](https://platform.openai.com/docs/models/whisper)

---

## 📝 Introduction

**Business AI Meeting Companion** is an advanced AI application that captures meeting conversations, transcribes them with high accuracy using **OpenAI's Whisper**, and provides a concise summary with key points and decisions using **IBM WatsonX with Llama 3**. The entire application is wrapped in an intuitive user interface built with **Hugging Face Gradio**.

---

## 🎯 Learning Objectives

By working through this project, you will learn how to:

- 🧑‍💻 Create a Python script to generate text using a large language model (LLM).
- 🗣️ Use OpenAI's Whisper for high-accuracy speech-to-text conversion.
- 🤖 Implement IBM Watson's AI (Llama 3 on WatsonX) to summarize transcribed text and extract key points.
- 🖥️ Build an intuitive and user-friendly interface using Hugging Face Gradio.
- 🔗 Utilize LangChain to orchestrate prompts and interactions with LLMs.

---

## 🛠️ Core Technologies

| Technology         | Purpose                                   |
|-------------------|-------------------------------------------|
| Whisper           | Speech-to-Text (ASR)                      |
| IBM WatsonX (Llama 3) | Language Model for Summarization     |
| Gradio            | User Interface                            |
| LangChain         | Prompt Orchestration                      |
| Python            | Programming Language                      |

---

## ⚙️ Preparing the Environment

> **Tip:** It's highly recommended to use a virtual environment to manage project dependencies.

### 1️⃣ Create a Python Virtual Environment

```bash
# Install virtualenv if you haven't already
pip3 install virtualenv

# Create a virtual environment named 'my_env'
virtualenv my_env

# Activate the virtual environment
source my_env/bin/activate  # On Linux/Mac
# .\my_env\Scripts\activate  # On Windows
```

---

### 2️⃣ Install Required Libraries

```bash
pip install transformers==4.36.0 torch==2.1.1 gradio==4.23.0 langchain==0.0.343 ibm_watson_machine_learning==1.0.335 huggingface-hub==0.20.1
```

☕ _Have a cup of coffee, this may take a few minutes!_

---

### 3️⃣ Install FFmpeg

FFmpeg is required for processing audio files in Python.

```bash
# Update package list (Linux)
sudo apt update

# Install ffmpeg (Linux)
sudo apt install ffmpeg -y
```

On Windows, download FFmpeg from [ffmpeg.org](https://ffmpeg.org/download.html) and add it to your PATH.

---

## 🚦 Project Steps & Usage

### Step 1: Simple Speech-to-Text Test

Download a sample audio file (e.g., your own `.mp3`). Create a Python script `simple_speech2text.py`:

```python
import torch
from transformers import pipeline

# Initialize the speech-to-text pipeline from Hugging Face Transformers
# This uses the "openai/whisper-tiny.en" model for automatic speech recognition (ASR)
pipe = pipeline(
  "automatic-speech-recognition",
  model="openai/whisper-tiny.en",
  chunk_length_s=30,
)

# Define the path to the audio file that needs to be transcribed
# Make sure to place the downloaded audio file in the same directory
sample = 'Testing speech to text.mp3' 

# Perform speech recognition on the audio file
prediction = pipe(sample, batch_size=8)["text"]

# Print the transcribed text to the console
print("Transcription:")
print(prediction)
```

Run the script:

```bash
python3 simple_speech2text.py
```

---

### Step 2: Creating an Audio Transcription App with Gradio

Create `speech2text_app.py`:

```python
import torch
from transformers import pipeline
import gradio as gr

# Initialize the speech recognition pipeline (loaded once)
pipe = pipeline(
    "automatic-speech-recognition",
    model="openai/whisper-tiny.en",
    chunk_length_s=30,
)

# Function to transcribe audio using the pre-loaded Whisper model
def transcript_audio(audio_file):
    # Transcribe the audio file and return the result
    result = pipe(audio_file, batch_size=8)["text"]
    return result

# Set up Gradio interface
audio_input = gr.Audio(sources=["upload"], type="filepath", label="Upload Audio File")
output_text = gr.Textbox(label="Transcription")

# Create the Gradio interface
iface = gr.Interface(
    fn=transcript_audio, 
    inputs=audio_input, 
    outputs=output_text, 
    title="Audio Transcription App",
    description="Upload an audio file (e.g., MP3, WAV) to transcribe it to text using OpenAI's Whisper."
)

# Launch the Gradio app
iface.launch(server_name="0.0.0.0", server_port=7860)
```

Run the app:

```bash
python3 speech2text_app.py
```

Open your browser to [http://0.0.0.0:7860](http://0.0.0.0:7860) to use the app.

---

### Step 3: Integrating the Language Model (WatsonX Llama 3)

Test your connection to the IBM WatsonX LLM. Create `simple_llm.py`:

```python
from ibm_watson_machine_learning.foundation_models import Model
from ibm_watson_machine_learning.foundation_models.extensions.langchain import WatsonxLLM
from ibm_watson_machine_learning.metanames import GenTextParamsMetaNames as GenParams

# --- Credentials and Parameters ---
# Note: The credentials are set up for the IBM Skills Network environment.
# For local use, you would need your own IBM Cloud API key and project ID.
my_credentials = {
    "url"    : "https://us-south.ml.cloud.ibm.com"
}

params = {
    GenParams.MAX_NEW_TOKENS: 700,
    GenParams.TEMPERATURE: 0.1,
}

# --- Initialize the Llama 3 Model ---
LLAMA3_model = Model(
        model_id='meta-llama/llama-3-8b-instruct', 
        credentials=my_credentials,
        params=params,
        project_id="skills-network",  
)

llm = WatsonxLLM(LLAMA3_model)

# --- Generate and Print Response ---
question = "How to read a book effectively?"
print(f"Question: {question}")
print("Answer:")
print(llm(question))
```

Run the script:

```bash
python3 simple_llm.py
```

---

### Step 4: Putting It All Together: The Speech Analyzer App

Create the final application file `speech_analyzer.py`:

```python
import gradio as gr
from transformers import pipeline
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from ibm_watson_machine_learning.foundation_models import Model
from ibm_watson_machine_learning.foundation_models.extensions.langchain import WatsonxLLM
from ibm_watson_machine_learning.metanames import GenTextParamsMetaNames as GenParams

#######------------- LLM Setup-------------####
# Note: The credentials are set up for the IBM Skills Network environment.
my_credentials = {
    "url": "https://us-south.ml.cloud.ibm.com"
}

params = {
    GenParams.MAX_NEW_TOKENS: 800,
    GenParams.TEMPERATURE: 0.1,
}

LLAMA3_model = Model(
    model_id='meta-llama/llama-3-8b-instruct',
    credentials=my_credentials,
    params=params,
    project_id="skills-network",
)

llm = WatsonxLLM(LLAMA3_model)

#######------------- Prompt Template-------------####
# This template is structured for Llama 3 instruct models.
temp = """
<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are a helpful assistant that summarizes text and extracts key points.

<|eot_id|><|start_header_id|>user<|end_header_id|>

Summarize the following text and list the key points with details.

Text: "{context}"

<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""

pt = PromptTemplate(
    input_variables=["context"],
    template=temp
)

prompt_to_llm = LLMChain(llm=llm, prompt=pt)

#######------------- Speech-to-Text Pipeline-------------####
# Initialize the speech recognition pipeline
speech_pipe = pipeline(
    "automatic-speech-recognition",
    model="openai/whisper-tiny.en",
    chunk_length_s=30,
)

#######------------- Main Function-------------####
def analyze_audio(audio_file):
    # 1. Transcribe the audio file
    print("Transcribing audio...")
    transcript_txt = speech_pipe(audio_file, batch_size=8)["text"]
    print("Transcription complete.")
    
    # 2. Run the LLM chain to summarize and find key points
    print("Analyzing text with LLM...")
    result = prompt_to_llm.run(transcript_txt)
    print("Analysis complete.")
    
    return result

#######------------- Gradio Interface-------------####
audio_input = gr.Audio(sources="upload", type="filepath", label="Upload Meeting Audio")
output_text = gr.Textbox(label="Meeting Summary and Key Points")

iface = gr.Interface(
    fn=analyze_audio,
    inputs=audio_input,
    outputs=output_text,
    title="Business AI Meeting Companion",
    description="Upload an audio recording of a meeting. The app will transcribe it and provide a summary with key points."
)

iface.launch(server_name="0.0.0.0", server_port=7860)
```

Run your final application:

```bash
python3 speech_analyzer.py
```

Open your browser to [http://0.0.0.0:7860](http://0.0.0.0:7860), upload a meeting recording, and see the AI-generated summary and key points!

---

## 📄 License

[![MIT License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

The code and models for OpenAI's Whisper are released under the MIT License. The code in this repository is provided as-is for educational purposes.
