import os
import csv
import logging
from extract_features import extract_features
from record_audio import record_audio
from datetime import datetime

DATASET_PATH = "dataset.csv"
LOG_DIR = "logs"

# Set up logging
os.makedirs(LOG_DIR, exist_ok=True)
logging.basicConfig(
    filename=os.path.join(LOG_DIR, f"prepare_dataset_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def save_to_csv(features, label, filepath=DATASET_PATH):
    header = [f'feature_{i}' for i in range(len(features))] + ['label']
    file_exists = os.path.exists(filepath)

    with open(filepath, mode='a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(header)
        writer.writerow(list(features) + [label])
    logging.info(f"Saved example to {filepath} with label '{label}'")

def main():
    print("\n🎤 Dataset Prep Tool")
    print("----------------------")

    while True:
        choice = input("New sample? (y/n): ").strip().lower()
        if choice != 'y':
            break

        input_method = input("Use microphone or existing file? (m/f): ").strip().lower()

        if input_method == 'm':
            try:
                duration = int(input("Recording duration (sec): "))
            except ValueError:
                print("Invalid duration. Try again.")
                continue
            audio_path = record_audio(duration=duration)
        elif input_method == 'f':
            audio_path = input("Enter full path to .wav file: ").strip('"')
            if not os.path.isfile(audio_path):
                print("File not found. Try again.")
                continue
        else:
            print("Invalid input method. Try again.")
            continue

        # Extract features (you can switch between 'main' and 'mfcc')
        features = extract_features(audio_path)["main"]

        print("\nExtracted features:")
        print(features)

        label = input("Label this emotion (e.g. happy/sad/angry/neutral/excited/fearful/stressed): ").strip().lower()
        save_to_csv(features, label)

if __name__ == "__main__":
    main()
