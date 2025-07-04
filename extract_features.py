import numpy as np
import librosa
import logging

def extract_features(audio_path):
    """
    Extract audio features from the file, ıncluding MFCCs
    """
    try:
        # Load the audio file using librosa
        logging.info(f"Loading audio file: {audio_path}")
        audio_data, sample_rate = librosa.load(audio_path)
        
        # Calculate basic features
        energy = np.mean(librosa.feature.rms(y=audio_data))
        spectral_centroids = librosa.feature.spectral_centroid(y=audio_data, sr=sample_rate)[0]
        zero_crossing_rate = np.mean(librosa.feature.zero_crossing_rate(y=audio_data)[0])
        pitch, _ = librosa.piptrack(y=audio_data, sr=sample_rate)
        pitch_mean = np.mean(pitch[pitch > 0])
        energy_var = np.std(librosa.feature.rms(y=audio_data))

       # Calculate MFCCs 
        mfccs = librosa.feature.mfcc(y=audio_data, sr=sample_rate, n_mfcc=13)
        mfccs_mean = np.mean(mfccs, axis=1)
        mfccs_features = mfccs_mean[:3] # Use only the first 3 MFCCs

        # combine all features
        features = np.array([energy, np.mean(spectral_centroids), zero_crossing_rate, pitch_mean, energy_var])
        features = np.concatenate((features, mfccs_features))
        
        # Debug print
        logging.info("\nExtracted Features:")
        logging.info(f"Energy: {energy}")
        logging.info(f"Spectral Centroid: {np.mean(spectral_centroids)}")
        logging.info(f"Zero Crossing Rate: {zero_crossing_rate}")
        logging.info(f"Pitch Mean: {pitch_mean}")
        logging.info(f"Energy Variance: {energy_var}\n")
        logging.info(f"MFCCs (first 3) : {mfccs_features}\n")
        
        #return features
        return {
            "main": features[:5],         # Energy, Spectral, ZCR, Pitch, Energy Var
            "mfcc": mfccs_features        # MFCC1, MFCC2, MFCC3
        }
        
    except Exception as e:
        logging.error(f"Error extracting features: {e}")
        raise
