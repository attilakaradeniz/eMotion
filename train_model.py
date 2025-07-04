import os
import logging
import pandas as pd
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import joblib

DATASET_PATH = "dataset.csv"
MODEL_PATH = "model.pkl"
LOG_DIR = "logs"

# Logging ayarları
os.makedirs(LOG_DIR, exist_ok=True)
logging.basicConfig(
    filename=os.path.join(LOG_DIR, f"train_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def load_dataset():
    if not os.path.exists(DATASET_PATH):
        logging.error(f"Dataset not found at {DATASET_PATH}")
        raise FileNotFoundError("Dataset not found!")
    
    df = pd.read_csv(DATASET_PATH)
    logging.info(f"Loaded dataset with shape {df.shape}")
    return df

def preprocess_data(df):
    X = df.drop("label", axis=1).values
    y = df["label"].values

    # Özellikleri ölçekle (çok önemli)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y, scaler

def train(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    logging.info("Training completed.")
    y_pred = model.predict(X_test)

    print("\n📊 Classification Report:\n")
    print(classification_report(y_test, y_pred))
    print("🧩 Confusion Matrix:\n")
    print(confusion_matrix(y_test, y_pred))

    logging.info("Model evaluation completed.")
    return model

def save_model(model, scaler):
    joblib.dump({"model": model, "scaler": scaler}, MODEL_PATH)
    logging.info(f"Model saved to {MODEL_PATH}")
    print(f"\n✅ Model saved to: {MODEL_PATH}")

def main():
    print("\n🧠 Training Emotion Classifier")
    print("------------------------------")

    df = load_dataset()
    X_scaled, y, scaler = preprocess_data(df)
    model = train(X_scaled, y)
    save_model(model, scaler)

if __name__ == "__main__":
    main()
