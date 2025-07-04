import os
import pandas as pd
import logging
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from joblib import dump
from datetime import datetime
import numpy as np


# Constants
DATASET_PATH = "dataset.csv"
MODEL_DIR = "models"
MODEL_FILENAME = "emotion_model.pkl"
LOG_DIR = "logs"

# Setup logging
os.makedirs(LOG_DIR, exist_ok=True)
logging.basicConfig(
    filename=os.path.join(LOG_DIR, f"train_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def load_dataset():
    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(f"Dataset not found at {DATASET_PATH}")

    df = pd.read_csv(DATASET_PATH)
    logging.info(f"Dataset loaded. Shape: {df.shape}")
    return df

def preprocess_data(df):
    X = df.drop(columns=["label"]).values
    y = df["label"].values

    # Normalize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Encode labels
    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)

    logging.info(f"Classes found: {list(encoder.classes_)}")

    return X_scaled, y_encoded, scaler, encoder

def train_and_save_model(X, y, scaler, encoder):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    
    # Get all unique labels from both train and test sets
    all_unique_labels = np.unique(np.concatenate((y_train, y_test)))
    # Get corresponding class names for all labels
    all_class_names = encoder.inverse_transform(all_unique_labels)
    
    report = classification_report(y_test, y_pred, target_names=all_class_names, labels=all_unique_labels)
    logging.info("Model Evaluation Report:\n" + report)
    print("Model Evaluation Report:\n", report)

    # Save model
    os.makedirs(MODEL_DIR, exist_ok=True)
    dump(model, os.path.join(MODEL_DIR, MODEL_FILENAME))
    dump(scaler, os.path.join(MODEL_DIR, "scaler.pkl"))
    dump(encoder, os.path.join(MODEL_DIR, "label_encoder.pkl"))

    logging.info(f"Model saved to {MODEL_DIR}/{MODEL_FILENAME}")

def main():
    print("\n🧠 Training Emotion Recognition Model...")
    df = load_dataset()
    X, y, scaler, encoder = preprocess_data(df)
    train_and_save_model(X, y, scaler, encoder)

if __name__ == "__main__":
    main()
