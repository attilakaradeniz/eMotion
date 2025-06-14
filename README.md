# Voice Emotion Analysis

This program records audio from your system microphone and analyzes the emotional content of the speaker's voice.

## Features
- Records 10 seconds of audio from the default microphone
- Analyzes the emotional content of the recorded voice
- Detects various emotions including happiness, sadness, anger, etc.
- Shows confidence level of the emotion detection

## Setup

1. Make sure you have Python 3.7+ installed
2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

3. Additional setup for pyAudioAnalysis:
```bash
# Download the pre-trained emotion recognition model
mkdir -p data/models
# You'll need to download the emotion recognition model separately
```

## Usage

Simply run the main script:
```bash
python main.py
```

The program will:
1. Give you a 3-second countdown before recording
2. Record your voice for 10 seconds
3. Analyze the emotional content
4. Display the detected emotion and confidence level

## Requirements
- A working microphone
- Python 3.7+
- Required Python packages (see requirements.txt)

## Note
Make sure your system microphone is properly configured and has the necessary permissions enabled.
