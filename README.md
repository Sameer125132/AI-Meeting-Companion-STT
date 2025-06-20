# Business AI Meeting Companion 🚀

Hi! I'm Garbii, and this is my personal project: **Business AI Meeting Companion**.

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Framework: Gradio](https://img.shields.io/badge/Framework-Gradio-orange)](https://gradio.app/)
[![AI: Whisper | Llama 3](https://img.shields.io/badge/AI-Whisper%20%7C%20Llama%203-blueviolet)](https://platform.openai.com/docs/models/whisper)

---

## 📝 About This Project

This is a personal project I built to experiment with AI-powered meeting tools. The app captures meeting conversations, transcribes them using **OpenAI's Whisper**, and then summarizes the transcript and extracts key points using **IBM WatsonX with Llama 3**. The interface is built with **Gradio** for easy use.

---

## 🎯 What I Learned

Working on this project helped me:

- 🧑‍💻 Write Python scripts that use large language models (LLMs)
- 🗣️ Integrate OpenAI's Whisper for accurate speech-to-text
- 🤖 Use IBM WatsonX (Llama 3) to summarize and extract key points from text
- 🖥️ Build a user-friendly web UI with Gradio
- 🔗 Orchestrate LLM prompts and workflows with LangChain

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

## ⚙️ How to Run It

> **Tip:** I recommend using a virtual environment for Python projects.

### 1️⃣ Set Up Your Environment

```bash
pip3 install virtualenv
virtualenv my_env
# On Linux/Mac
source my_env/bin/activate
# On Windows
.\my_env\Scripts\activate
```

### 2️⃣ Install Dependencies

```bash
pip install transformers==4.36.0 torch==2.1.1 gradio==4.23.0 langchain==0.0.343 ibm_watson_machine_learning==1.0.335 huggingface-hub==0.20.1
```

### 3️⃣ Install FFmpeg

- **Linux:**
  ```bash
  sudo apt update
  sudo apt install ffmpeg -y
  ```
- **Windows:** Download from [ffmpeg.org](https://ffmpeg.org/download.html) and add to your PATH.

---

## 🚦 Usage & Demo

### 1. Test Speech-to-Text

Run:

```bash
python3 simple_speech2text.py
```

### 2. Try the Gradio Transcription App

Run:

```bash
python3 speech2text_app.py
```

Then open [http://0.0.0.0:7860](http://0.0.0.0:7860) in your browser.

### 3. Test Llama 3 Summarization

Run:

```bash
python3 simple_llm.py
```

### 4. Full Meeting Analyzer App

Run:

```bash
python3 speech_analyzer.py
```

Then open [http://0.0.0.0:7860](http://0.0.0.0:7860), upload a meeting recording, and see the AI-generated summary and key points!

---

## 📸 Screenshots

<!-- Add your screenshots here! -->

---

## 📄 License

[![MIT License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

This project is open source under the MIT License. The code and models for OpenAI's Whisper are released under the MIT License. Everything here is for educational and personal use.
