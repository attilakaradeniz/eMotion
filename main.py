import winsound
import time
import os
import subprocess
import numpy as np
from scipy.io import wavfile
from scipy import signal
from sklearn.preprocessing import StandardScaler
import sounddevice as sd
import soundfile as sf
from datetime import datetime
from tqdm import tqdm
import librosa
import logging
from extract_features import extract_features
from record_audio import record_audio

# Set up logging
def setup_logging():
    """
    Set up logging configuration to write to both file and console
    """
    # Create logs directory if it doesn't exist
    os.makedirs("logs", exist_ok=True)
    
    # Create log filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = os.path.join("logs", f"emotion_analysis_{timestamp}.log")
    
    # Configure logging to write to both file and console
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename),
            logging.StreamHandler()
        ]
    )
    logging.info(f"Starting new session. Log file: {log_filename}")
    return log_filename

# 
# removed extract features function
def analyze_emotion(features):
    """
    Analyze emotions based on extracted features
    """
    try:
        # Debug print raw features
        logging.info("Raw features before scaling: %s", features)
        
        # Define typical ranges for normalization
        feature_ranges = {
            'energy': (0.0, 0.2),        # RMS energy typical range
            'spectral': (500, 2500),     # Spectral centroid typical range in Hz
            'zcr': (0.0, 0.3),          # Zero crossing rate typical range
            'pitch': (50, 400),          # Pitch typical range in Hz
            'energy_var': (0.0, 0.1)     # Energy variation typical range
        }
        
        # Manual min-max scaling to [-1, 1] range
        def scale_feature(value, min_val, max_val):
            if max_val == min_val:
                return 0
            return 2 * ((value - min_val) / (max_val - min_val)) - 1
        
        # Scale each feature
        energy = scale_feature(features[0], *feature_ranges['energy'])
        spectral = scale_feature(features[1], *feature_ranges['spectral'])
        zcr = scale_feature(features[2], *feature_ranges['zcr'])
        pitch = scale_feature(features[3], *feature_ranges['pitch'])
        energy_var = scale_feature(features[4], *feature_ranges['energy_var'])
        
        features_scaled = np.array([energy, spectral, zcr, pitch, energy_var])
        logging.info("Scaled features: %s", features_scaled)
        
        # Initialize emotion scores
        emotions = {
            'happy': 0.0,
            'sad': 0.0,
            'angry': 0.0,
            'neutral': 0.3  # Small baseline for neutral
        }
        
        # Energy contribution
        if energy > 0.3:
            emotions['happy'] += 0.8
            emotions['angry'] += 0.6
        elif energy < -0.3:
            emotions['sad'] += 0.8
        
        # Spectral centroid contribution
        if spectral > 0.3:
            emotions['happy'] += 0.7
            if energy > 0:
                emotions['angry'] += 0.5
        elif spectral < -0.3:
            emotions['sad'] += 0.7
        
        # Zero crossing rate contribution
        if zcr > 0.3:
            if energy > 0:
                emotions['angry'] += 0.7
            else:
                emotions['happy'] += 0.6
        elif zcr < -0.3:
            emotions['sad'] += 0.5
        
        # Pitch contribution
        if pitch > 0.3:
            emotions['happy'] += 0.6
            if energy > 0:
                emotions['angry'] += 0.5
        elif pitch < -0.3:
            emotions['sad'] += 0.6
        
        # Energy variance contribution
        if energy_var > 0.3:
            emotions['angry'] += 0.6
            emotions['happy'] += 0.4
        elif energy_var < -0.3:
            emotions['sad'] += 0.5
        
        # Log emotion scores before normalization
        logging.info("\nEmotion scores before normalization:")
        for emotion, score in emotions.items():
            logging.info(f"{emotion}: {score}")
        
        # Normalize scores to percentages
        total = sum(emotions.values())
        if total > 0:  # Prevent division by zero
            for emotion in emotions:
                emotions[emotion] = (emotions[emotion] / total) * 100
        
        # Log final percentages
        logging.info("\nFinal emotion percentages:")
        for emotion, percentage in emotions.items():
            logging.info(f"{emotion}: {percentage:.2f}%")
        
        return emotions
        
    except Exception as e:
        logging.error(f"Error in emotion analysis: {e}")
        raise

def main():
    try:
        # Set up logging
        log_file = setup_logging()
        logging.info("Welcome to Voice Emotion Analysis!")
        
        while True:
            # Ask user for choice
            logging.info("\nChoose an option:")
            logging.info("1. Record new audio")
            logging.info("2. Analyze existing audio file")
            logging.info("3. Exit")
            choice = input("\nEnter your choice (1-3): ")
            
            if choice == '1':
                # Get recording duration from user
                while True:
                    try:
                        duration = input("\nEnter recording duration in seconds (1-30): ")
                        duration = int(duration)
                        if 1 <= duration <= 30:
                            break
                        else:
                            logging.warning("Please enter a duration between 1 and 30 seconds.")
                    except ValueError:
                        logging.warning("Please enter a valid number.")
                
                # Record new audio
                audio_path = record_audio(duration=duration)
            elif choice == '2':
                # Get existing audio file
                logging.info("\nEnter the path to the audio file:")
                audio_path = input().strip('"')  # Remove quotes if present
                if not os.path.exists(audio_path):
                    logging.error("File not found!")
                    continue
            elif choice == '3':
                logging.info("Goodbye!")
                break
            else:
                logging.warning("Invalid choice! Please try again.")
                continue
            
            # Extract features
            features = extract_features(audio_path)
            
            # Analyze emotions
            # the former approach (without mfcc)
            emotions = analyze_emotion(features["main"])

            # new approach, included MFCC ← (not in use right now)
            # emotions = analyze_emotion(features["mfcc"])  
            
            # Display results
            logging.info("\nAnalysis Results:")
            for emotion, percentage in emotions.items():
                logging.info(f"{emotion.capitalize()}: {percentage:.2f}%")
            
            logging.info(f"\nFull analysis has been saved to: {log_file}")
            
    except Exception as e:
        logging.error(f"An error occurred: {e}")
        raise

if __name__ == "__main__":
    main()
