# Voice Emotion Analysis

This program records audio from your system microphone and analyzes the emotional content of the speaker's voice. It provides real-time feedback about the detected emotions and their confidence levels.

## Features
- Real-time audio recording from system microphone
- Emotion detection with multiple emotion categories
- Confidence scoring for emotion detection
- Recording management with automatic file naming
- Logging system for debugging and tracking
- Test suite for verification

## Setup

1. Make sure you have Python 3.7+ installed
2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

3. The project uses a pre-trained emotion recognition model that is included in the repository

## Project Structure
- `main.py`: Main application file
- `requirements.txt`: Python dependencies
- `test_main.py`: Test suite
- `recordings/`: Directory for audio recordings
- `logs/`: Directory for application logs

## Usage

Run the main script:
```bash
python main.py
```

The program will:
1. Start recording from your microphone
2. Analyze the emotional content in real-time
3. Display detected emotions and confidence levels
4. Save recordings with timestamp-based filenames

## Requirements
- A working microphone
- Python 3.7+
- Required Python packages (see requirements.txt)

## Testing

To run the test suite:
```bash
pytest test_main.py
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
