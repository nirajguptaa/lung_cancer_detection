"""
Lung Cancer Detection - Deep Learning CNN Models
Includes: Custom CNN and VGG16 (Transfer Learning)
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, Dense, Dropout,
    Flatten, BatchNormalization, GlobalAveragePooling2D
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.applications import VGG16
from sklearn.metrics import classification_report, confusion_matrix


# =====================
# CONFIG
# =====================
IMG_SIZE = 224
NUM_CLASSES = 3
EPOCHS = 25
BATCH_SIZE = 16
LR = 0.0001


# =====================
# LOAD DATA
# =====================
def load_data():
    X_train = np.load("../data/processed/X_train.npy")
    y_train = np.load("../data/processed/y_train.npy")
    X_val = np.load("../data/processed/X_val.npy")
    y_val = np.load("../data/processed/y_val.npy")
    X_test = np.load("../data/processed/X_test.npy")
    y_test = np.load("../data/processed/y_test.npy")

    y_train = to_categorical(y_train, NUM_CLASSES)
    y_val = to_categorical(y_val, NUM_CLASSES)
    y_test = to_categorical(y_test, NUM_CLASSES)

    return X_train, y_train, X_val, y_val, X_test, y_test


# =====================
# CUSTOM CNN
# =====================
def build_custom_cnn():
    model = Sequential([
        Conv2D(32, (3, 3), activation="relu", input_shape=(IMG_SIZE, IMG_SIZE, 3)),
        BatchNormalization(),
        MaxPooling2D(),

        Conv2D(64, (3, 3), activation="relu"),
        BatchNormalization(),
        MaxPooling2D(),

        Conv2D(128, (3, 3), activation="relu"),
        BatchNormalization(),
        MaxPooling2D(),

        Flatten(),
        Dense(256, activation="relu"),
        Dropout(0.5),
        Dense(NUM_CLASSES, activation="softmax")
    ])

    model.compile(
        optimizer=Adam(LR),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model


# =====================
# VGG16 TRANSFER LEARNING
# =====================
def build_vgg16():
    base_model = VGG16(
        weights="imagenet",
        include_top=False,
        input_shape=(IMG_SIZE, IMG_SIZE, 3)
    )
    base_model.trainable = False

    model = Sequential([
        base_model,
        GlobalAveragePooling2D(),
        Dense(256, activation="relu"),
        Dropout(0.5),
        Dense(NUM_CLASSES, activation="softmax")
    ])

    model.compile(
        optimizer=Adam(LR),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model


# =====================
# TRAIN MODEL
# =====================
def train_model(model, name, X_train, y_train, X_val, y_val):
    os.makedirs("../models", exist_ok=True)

    callbacks = [
        EarlyStopping(patience=5, restore_best_weights=True),
        ModelCheckpoint(
            f"../models/{name}.h5",
            save_best_only=True,
            monitor="val_accuracy"
        )
    ]

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=1
    )

    return history


# =====================
# EVALUATION
# =====================
def evaluate_model(model, X_test, y_test, title):
    y_pred = np.argmax(model.predict(X_test), axis=1)
    y_true = np.argmax(y_test, axis=1)

    print(f"\n{title} Classification Report:")
    print(classification_report(
        y_true, y_pred,
        target_names=["Normal", "Benign", "Malignant"]
    ))

    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(
        cm, annot=True, fmt="d",
        xticklabels=["Normal", "Benign", "Malignant"],
        yticklabels=["Normal", "Benign", "Malignant"],
        cmap="Blues"
    )
    plt.title(f"{title} Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.savefig(f"{title}_confusion_matrix.png", dpi=300)
    plt.show()


# =====================
# MAIN
# =====================
if __name__ == "__main__":
    print("Loading data...")
    X_train, y_train, X_val, y_val, X_test, y_test = load_data()

    # -------- Custom CNN --------
    print("\nTraining Custom CNN...")
    cnn = build_custom_cnn()
    train_model(cnn, "custom_cnn", X_train, y_train, X_val, y_val)
    evaluate_model(cnn, X_test, y_test, "Custom_CNN")

    # -------- VGG16 --------
    print("\nTraining VGG16 Transfer Learning...")
    vgg = build_vgg16()
    train_model(vgg, "vgg16", X_train, y_train, X_val, y_val)
    evaluate_model(vgg, X_test, y_test, "VGG16")

    print("\n Deep Learning training completed successfully!")