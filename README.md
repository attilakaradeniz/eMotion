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
1. Ask if you want to record a new audio or analyze an existing file
2. For new recordings:
   - Prompt you to enter the recording duration in seconds
   - Show a 3-second countdown before recording starts
   - Display a progress bar during recording
   - Save the recording with a timestamp-based filename
3. For existing files:
   - Prompt you to enter the path to the audio file
4. Analyze the audio and display:
   - Detected emotions
   - Confidence levels
   - Audio features used for analysis
   - Save all analysis details to a log file

## User Interface Features

- Interactive command-line interface
- 3-second countdown before recording starts
- Progress bar during recording
- Flexible recording duration (user-specified)
- Option to analyze existing audio files
- Detailed logging of analysis results
- Automatic timestamp-based file naming

## Audio Analysis Features
- Energy analysis
- Spectral centroid calculation
- Zero crossing rate detection
- Pitch analysis
- Energy variance measurement
- Emotion classification based on multiple features

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
1. Ask if you want to record a new audio or analyze an existing file
2. For new recordings:
   - Prompt you to enter the recording duration in seconds
   - Show a 3-second countdown before recording starts
   - Display a progress bar during recording
   - Save the recording with a timestamp-based filename
3. For existing files:
   - Prompt you to enter the path to the audio file
4. Analyze the audio and display:
   - Detected emotions
   - Confidence levels
   - Audio features used for analysis
   - Save all analysis details to a log file

## Requirements
- A working microphone
- Python 3.7+
- Required Python packages (see requirements.txt)

## Note
Make sure your system microphone is properly configured and has the necessary permissions enabled.

---

## Developer Notes (Branch-Specific)

### Branch: `feature/mfcc-extraction`

This branch introduces an early stage of modularizing the audio feature extraction process. The goal is to prepare the foundation for transitioning from rule-based to ML-based emotion analysis in future branches.

####  Key Changes:
- `extract_features()` now returns a dictionary with two keys:
  - `main`: core audio features (energy, spectral centroid, ZCR, pitch, energy variance)
  - `mfcc`: first 3 MFCCs
- `analyze_emotion()` currently works with `features["main"]` (rule-based logic remains untouched)
- MFCCs are extracted but not yet used in emotion calculation (will be activated in future ML models)
- Return structure updated for flexibility in future model training
- Logging expanded to include both feature groups

####  Next Steps:
- Add a machine learning model (likely in `feature/ml-classifier`)
- Enable training with stored features + manual emotion labels
- Introduce analysis with MFCC-based classifiers
- Build a dashboard for visualizing features and predictions
- Improve modularity and separation of feature engineering from classification logic

####  Notes:
- This branch maintains full compatibility with rule-based analysis.
- The code is written in a way to easily test old and new approaches side by side.

