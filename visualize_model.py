# visualize_model.py

import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split

# Constants
DATASET_PATH = "dataset.csv"
MODEL_PATH = os.path.join("models", "emotion_model.pkl")
SCALER_PATH = os.path.join("models", "scaler.pkl")
ENCODER_PATH = os.path.join("models", "label_encoder.pkl")

def load_data():
    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(f"Dataset not found at {DATASET_PATH}")
    return pd.read_csv(DATASET_PATH)

def visualize_confusion_matrix(y_true, y_pred, labels):
    cm = confusion_matrix(y_true, y_pred, labels=range(len(labels)))
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
                xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.show()

def visualize_classification_report(y_true, y_pred, labels):
    report_dict = classification_report(y_true, y_pred, target_names=labels, output_dict=True)
    df = pd.DataFrame(report_dict).transpose()

    df = df.loc[labels]  # keep only emotion classes
    df[["precision", "recall", "f1-score"]].plot(kind="bar", figsize=(10, 6))
    plt.title("Classification Report Metrics per Class")
    plt.ylabel("Score")
    plt.ylim(0, 1.1)
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def main():
    print("\n📊 Visualizing Model Performance...\n")

    df = load_data()

    # Split features and labels
    X = df.drop(columns=["label"]).values
    y = df["label"].values

    # Load scaler and encoder
    scaler = joblib.load(SCALER_PATH)
    encoder = joblib.load(ENCODER_PATH)

    # Normalize features and encode labels
    X_scaled = scaler.transform(X)
    y_encoded = encoder.transform(y)

    # Train-test split (must be same as used in training)
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_encoded, test_size=0.2, random_state=42
    )

    # Load model
    model = joblib.load(MODEL_PATH)

    # Predict
    y_pred = model.predict(X_test)

    # Visualize confusion matrix
    labels = list(encoder.classes_)
    visualize_confusion_matrix(y_test, y_pred, labels)

    # Visualize classification report
    visualize_classification_report(y_test, y_pred, labels)

if __name__ == "__main__":
    main()
