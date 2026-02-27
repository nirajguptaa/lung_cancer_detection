"""
Lung Cancer Detection - Model Evaluation Module
Evaluates the trained CNN model on test data
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os

from tensorflow.keras.models import load_model
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

# =========================
# CONFIG
# =========================
CLASS_NAMES = ["Normal", "Benign", "Malignant"]
MODEL_PATH = "../models/custom_cnn.h5"   # change to vgg16.h5 if needed
OUTPUT_DIR = "../evaluation_results"

os.makedirs(OUTPUT_DIR, exist_ok=True)


# =========================
# LOAD DATA
# =========================
def load_test_data():
    X_test = np.load("../data/processed/X_test.npy")
    y_test = np.load("../data/processed/y_test.npy")
    return X_test, y_test


# =========================
# EVALUATION
# =========================
def evaluate_model():
    print("Loading model...")
    model = load_model(MODEL_PATH)

    print("Loading test data...")
    X_test, y_test = load_test_data()

    print("Running predictions...")
    y_pred_probs = model.predict(X_test)
    y_pred = np.argmax(y_pred_probs, axis=1)

    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    recall = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

    print("\n===== MODEL PERFORMANCE =====")
    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1-Score : {f1:.4f}")

    # Classification report
    report = classification_report(
        y_test,
        y_pred,
        target_names=CLASS_NAMES,
        zero_division=0
    )

    print("\nClassification Report:\n")
    print(report)

    # Save report to file
    with open(os.path.join(OUTPUT_DIR, "classification_report.txt"), "w") as f:
        f.write(report)

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES
    )
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "confusion_matrix.png"), dpi=300)
    plt.show()

    # Save metrics summary
    summary_df = pd.DataFrame([{
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1-Score": f1
    }])

    summary_df.to_csv(
        os.path.join(OUTPUT_DIR, "evaluation_summary.csv"),
        index=False
    )

    print("\n✅ Evaluation completed successfully!")
    print(f"Results saved in: {OUTPUT_DIR}")


# =========================
# MAIN
# =========================
if __name__ == "__main__":
    evaluate_model()