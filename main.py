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

def record_audio(duration=10):
    """
    Record audio using sounddevice
    """
    try:
        # Create recordings directory if it doesn't exist
        os.makedirs("recordings", exist_ok=True)
        
        # Create filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.abspath(os.path.join("recordings", f"recording_{timestamp}.wav"))
        
        # Recording parameters
        samplerate = 44100  # CD quality audio
        channels = 1        # Mono
        
        logging.info("\nRecording will start in:")
        for i in range(3, 0, -1):
            logging.info(f"{i}...")
            time.sleep(1)
        
        # Play a beep to indicate start of recording
        winsound.Beep(1000, 500)  # 1000 Hz for 500 milliseconds
        logging.info("\nRecording... Speak now!")
        
        # Record audio
        recording = sd.rec(int(duration * samplerate),
                         samplerate=samplerate,
                         channels=channels,
                         dtype=np.int16)
        
        # Show recording progress with progress bar
        with tqdm(total=duration, desc="Recording", unit="sec", bar_format="{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} seconds") as pbar:
            for _ in range(duration):
                time.sleep(1)
                pbar.update(1)
        sd.wait()  # Wait for any remaining recording time
        
        # Play a beep to indicate end of recording
        winsound.Beep(1000, 500)
        logging.info("\nRecording finished!")
        
        # Save as WAV file
        sf.write(filename, recording, samplerate)
        
        logging.info(f"\nSuccess! Recording saved to: {filename}")
        return filename
            
    except Exception as e:
        logging.error(f"\nError during recording: {e}")
        logging.info("\nTroubleshooting tips:")
        logging.info("1. Make sure your microphone is connected and working")
        logging.info("2. Check Windows sound settings")
        logging.info("3. Try running the program as administrator")
        raise

def extract_features(audio_path):
    """
    Extract audio features from the file
    """
    try:
        # Load the audio file using librosa
        logging.info(f"Loading audio file: {audio_path}")
        audio_data, sample_rate = librosa.load(audio_path)
        
        # Calculate features
        energy = np.mean(librosa.feature.rms(y=audio_data))
        spectral_centroids = librosa.feature.spectral_centroid(y=audio_data, sr=sample_rate)[0]
        zero_crossing_rate = np.mean(librosa.feature.zero_crossing_rate(y=audio_data)[0])
        pitch, _ = librosa.piptrack(y=audio_data, sr=sample_rate)
        pitch_mean = np.mean(pitch[pitch > 0])
        energy_var = np.std(librosa.feature.rms(y=audio_data))
        
        features = np.array([energy, np.mean(spectral_centroids), zero_crossing_rate, pitch_mean, energy_var])
        
        # Debug print
        logging.info("\nExtracted Features:")
        logging.info(f"Energy: {energy}")
        logging.info(f"Spectral Centroid: {np.mean(spectral_centroids)}")
        logging.info(f"Zero Crossing Rate: {zero_crossing_rate}")
        logging.info(f"Pitch Mean: {pitch_mean}")
        logging.info(f"Energy Variance: {energy_var}\n")
        
        return features
        
    except Exception as e:
        logging.error(f"Error extracting features: {e}")
        raise

def analyze_emotion(features):
    """
    Analyze emotions based on extracted features
    """
    try:
        # Debug print raw features
        logging.info("Raw features before scaling: %s", features)
        
        # Normalize features
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features.reshape(1, -1))
        
        # Debug print scaled features
        logging.info("Scaled features: %s", features_scaled[0])
        
        # Extract normalized features
        energy = features_scaled[0][0]      # Energy (RMS)
        pitch = features_scaled[0][1]       # Spectral centroid
        zcr = features_scaled[0][2]         # Zero crossing rate
        pitch_var = features_scaled[0][3]   # Pitch
        energy_var = features_scaled[0][4]  # Energy variation
        
        # Initialize emotion scores with zero baseline
        emotions = {
            'happy': 0.0,
            'sad': 0.0,
            'angry': 0.0,
            'neutral': 0.0
        }
        
        # Much more aggressive emotion detection
        # Energy contribution
        if energy > 0.05:  # Even more sensitive threshold
            emotions['happy'] += 0.8
            emotions['angry'] += 0.6
        elif energy < -0.05:
            emotions['sad'] += 0.8
        
        # Pitch contribution
        if pitch > 0.05:
            emotions['happy'] += 0.7
            if energy > 0:
                emotions['angry'] += 0.5
        elif pitch < -0.05:
            emotions['sad'] += 0.7
        
        # Zero crossing rate contribution
        if zcr > 0.05:
            if energy > 0:
                emotions['angry'] += 0.7
            else:
                emotions['happy'] += 0.6
        elif zcr < -0.05:
            emotions['sad'] += 0.5
        
        # Pitch variance contribution
        if pitch_var > 0.05:
            emotions['happy'] += 0.6
            if energy > 0:
                emotions['angry'] += 0.5
        elif pitch_var < -0.05:
            emotions['sad'] += 0.6
        
        # Energy variance contribution
        if energy_var > 0.05:
            emotions['angry'] += 0.6
            emotions['happy'] += 0.4
        elif energy_var < -0.05:
            emotions['sad'] += 0.5
        
        # Neutral is now much harder to achieve
        total_emotion = sum(emotions.values())
        if total_emotion < 0.3:  # Much lower threshold
            emotions['neutral'] = 0.3
        else:
            emotions['neutral'] = 0.05  # Very small baseline
        
        # Debug print emotion scores before normalization
        logging.info("\nEmotion scores before normalization:")
        for emotion, score in emotions.items():
            logging.info(f"{emotion}: {score}")
        
        # Find the dominant emotion
        dominant_emotion = max(emotions.items(), key=lambda x: x[1])
        total_score = sum(emotions.values())
        confidence = (dominant_emotion[1] / total_score) * 100 if total_score > 0 else 0
        
        # Convert emotion scores to percentages
        emotions = {k: (v / total_score) * 100 if total_score > 0 else 0 for k, v in emotions.items()}
        
        # Debug print final percentages
        logging.info("\nFinal emotion percentages:")
        for emotion, percentage in emotions.items():
            logging.info(f"{emotion}: {percentage:.2f}%")
        
        return dominant_emotion[0], confidence, emotions
        
    except Exception as e:
        logging.error(f"Error analyzing emotions: {e}")
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
            emotion, confidence, all_emotions = analyze_emotion(features)
            
            # Display results
            logging.info("\nAnalysis Results:")
            logging.info(f"Dominant Emotion: {emotion.upper()}")
            logging.info(f"Confidence: {confidence:.2f}%")
            
            logging.info("\nDetailed Emotion Breakdown:")
            for emotion, percentage in all_emotions.items():
                logging.info(f"{emotion.capitalize()}: {percentage:.2f}%")
            
            logging.info(f"\nFull analysis has been saved to: {log_file}")
            
    except Exception as e:
        logging.error(f"An error occurred: {e}")
        raise

if __name__ == "__main__":
    main()
