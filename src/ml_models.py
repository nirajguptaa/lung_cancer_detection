"""
Lung Cancer Detection - Traditional Machine Learning Models
Models: SVM, Random Forest, KNN, Decision Tree, Naive Bayes, Gradient Boosting
"""

import os
import time
import joblib
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


class TraditionalMLModels:
    def __init__(self):
        self.models = {}
        self.results = {}

    #  Reduce image size to make ML feasible
    def reduce_dimensionality(self, X, new_size=(64, 64)):
        reduced = []
        for img in X:
            img = cv2.resize(img, new_size)
            reduced.append(img)
        return np.array(reduced)

    #  Flatten images
    def flatten_images(self, X):
        return X.reshape(X.shape[0], -1)

    #  Initialize models
    def initialize_models(self):
        self.models = {
            "SVM": SVC(kernel="rbf", probability=True),
            "Random Forest": RandomForestClassifier(n_estimators=100, n_jobs=-1),
            "KNN": KNeighborsClassifier(n_neighbors=5),
            "Decision Tree": DecisionTreeClassifier(),
            "Naive Bayes": GaussianNB(),
            "Gradient Boosting": GradientBoostingClassifier()
        }

    #  Train & evaluate one model
    def train_and_evaluate(self, model_name, model, X_train, y_train, X_test, y_test):
        print(f"\nTraining {model_name}...")

        start = time.time()
        model.fit(X_train, y_train)
        train_time = time.time() - start

        y_pred = model.predict(X_test)

        self.results[model_name] = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, average="weighted"),
            "recall": recall_score(y_test, y_pred, average="weighted"),
            "f1": f1_score(y_test, y_pred, average="weighted"),
            "train_time": train_time,
            "predictions": y_pred
        }

        print(f"{model_name} Accuracy: {self.results[model_name]['accuracy']:.4f}")

        joblib.dump(model, f"../models/{model_name.replace(' ', '_')}.pkl")

    #  Train all models
    def train_all(self, X_train, y_train, X_test, y_test):
        self.initialize_models()

        # Reduce + flatten
        X_train = self.flatten_images(self.reduce_dimensionality(X_train))
        X_test = self.flatten_images(self.reduce_dimensionality(X_test))

        for name, model in self.models.items():
            self.train_and_evaluate(name, model, X_train, y_train, X_test, y_test)

    #  Compare results
    def compare_models(self):
        df = pd.DataFrame([
            {
                "Model": name,
                "Accuracy": res["accuracy"],
                "Precision": res["precision"],
                "Recall": res["recall"],
                "F1-Score": res["f1"],
                "Training Time (s)": res["train_time"]
            }
            for name, res in self.results.items()
        ]).sort_values("F1-Score", ascending=False)

        print("\nMODEL COMPARISON\n")
        print(df)

        df.to_csv("ml_model_comparison.csv", index=False)
        return df

    #  Plot confusion matrices
    def plot_confusion_matrices(self, y_test, class_names):
        plt.figure(figsize=(18, 10))

        for i, (name, res) in enumerate(self.results.items(), 1):
            plt.subplot(2, 3, i)
            cm = confusion_matrix(y_test, res["predictions"])
            sns.heatmap(cm, annot=True, fmt="d",
                        xticklabels=class_names,
                        yticklabels=class_names,
                        cmap="Blues")
            plt.title(name)

        plt.tight_layout()
        plt.savefig("ml_confusion_matrices.png", dpi=300)
        plt.show()


# =========================
# MAIN
# =========================
if __name__ == "__main__":

    # Ensure model directory exists
    os.makedirs("../models", exist_ok=True)

    # Load processed data
    X_train = np.load("../data/processed/X_train.npy")
    y_train = np.load("../data/processed/y_train.npy")
    X_test = np.load("../data/processed/X_test.npy")
    y_test = np.load("../data/processed/y_test.npy")

    # Train models
    ml = TraditionalMLModels()
    ml.train_all(X_train, y_train, X_test, y_test)

    # Compare models
    df = ml.compare_models()

    # Plot confusion matrices
    class_names = ["Normal", "Benign", "Malignant"]
    ml.plot_confusion_matrices(y_test, class_names)

    print("\n Traditional ML models completed successfully!")