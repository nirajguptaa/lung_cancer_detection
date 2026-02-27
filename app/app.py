import os
import numpy as np
import cv2
import joblib
from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.vgg16 import preprocess_input

app = Flask(__name__)

MODEL_DIR = "models"
IMG_SIZE = 224
CLASS_NAMES = ["Normal", "Benign", "Malignant"]

# Load CNN models
cnn_models = {
    "Custom CNN": load_model(os.path.join(MODEL_DIR, "custom_cnn.h5")),
    "VGG16": load_model(os.path.join(MODEL_DIR, "vgg16.h5"))
}

# Load ML models
ml_models = {
    "Random Forest": joblib.load(os.path.join(MODEL_DIR, "Random_Forest.pkl")),
    "SVM": joblib.load(os.path.join(MODEL_DIR, "SVM.pkl")),
    "KNN": joblib.load(os.path.join(MODEL_DIR, "KNN.pkl")),
    "Decision Tree": joblib.load(os.path.join(MODEL_DIR, "Decision_Tree.pkl"))
}

def preprocess_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img / 255.0
    return np.expand_dims(img, axis=0)

def preprocess_for_ml(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (64, 64))
    img = img / 255.0
    return img.reshape(1, -1)

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    comparison = {}

    if request.method == "POST":
        file = request.files["image"]
        model_choice = request.form["model"]

        image_path = os.path.join("static", file.filename)
        file.save(image_path)

        if model_choice in cnn_models:
            img = preprocess_image(image_path)
            pred = cnn_models[model_choice].predict(img)
            result = CLASS_NAMES[np.argmax(pred)]

        elif model_choice == "Compare All":
            for name, model in cnn_models.items():
                img = preprocess_image(image_path)
                pred = model.predict(img)
                comparison[name] = CLASS_NAMES[np.argmax(pred)]

            for name, model in ml_models.items():
                img = preprocess_for_ml(image_path)
                pred = model.predict(img)
                comparison[name] = CLASS_NAMES[pred[0]]

        else:
            img = preprocess_for_ml(image_path)
            pred = ml_models[model_choice].predict(img)
            result = CLASS_NAMES[pred[0]]

    return render_template("index.html", result=result, comparison=comparison)

if __name__ == "__main__":
    app.run(debug=True)