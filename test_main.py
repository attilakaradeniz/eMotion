import unittest
import numpy as np
import os
import logging
from main import extract_features, analyze_emotion, setup_logging

class TestEmotionAnalysis(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Set up logging for tests
        cls.log_file = setup_logging()
        
        # Create a test audio file with a sine wave
        cls.test_audio_path = "test_audio.wav"
        if not os.path.exists(cls.test_audio_path):
            import soundfile as sf
            sample_rate = 44100
            duration = 3  # seconds
            t = np.linspace(0, duration, int(sample_rate * duration))
            # Generate a 440 Hz sine wave
            audio_data = np.sin(2 * np.pi * 440 * t)
            sf.write(cls.test_audio_path, audio_data, sample_rate)
    
    def test_feature_extraction(self):
        """Test that feature extraction returns the expected shape and non-zero values"""
        features = extract_features(self.test_audio_path)
        self.assertEqual(len(features), 5)  # Should return 5 features
        self.assertTrue(np.all(features != 0))  # Features should not be zero
    
    def test_emotion_analysis(self):
        """Test that emotion analysis returns valid emotion scores"""
        # Create some test features
        test_features = np.array([0.5, 0.6, 0.7, 0.8, 0.9])  # High energy/pitch features
        emotions = analyze_emotion(test_features)
        
        # Check that we get all expected emotion keys
        expected_emotions = {'happy', 'sad', 'angry', 'neutral'}
        self.assertEqual(set(emotions.keys()), expected_emotions)
        
        # Check that all emotion values are between 0 and 1
        for emotion, score in emotions.items():
            self.assertGreaterEqual(score, 0.0)
            self.assertLessEqual(score, 1.0)
    
    @classmethod
    def tearDownClass(cls):
        # Clean up test files
        if os.path.exists(cls.test_audio_path):
            os.remove(cls.test_audio_path)
        if os.path.exists(cls.log_file):
            os.remove(cls.log_file)

if __name__ == '__main__':
    unittest.main() 