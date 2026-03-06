"""
app.py — Flask entry point for the Lung Cancer Detection System.
Includes:
• CT Scan upload
• EfficientNet prediction
• Grad-CAM explainability
• AI chatbot assistant
• PDF medical report generation
"""

import os
import uuid

from dotenv import load_dotenv
load_dotenv()

import cv2
import numpy as np
from flask import Flask, render_template, request, jsonify, session, send_file
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.models import load_model

from gradcam import generate_gradcam, overlay_heatmap
from chatbot import chat, build_system_prompt, clear_history
from report_generator import generate_report


# ──────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

TEMPLATE_DIR = os.path.join(BASE_DIR, "templates")
STATIC_DIR = os.path.join(BASE_DIR, "static")
MODEL_PATH = os.path.join(BASE_DIR, "models", "efficientnet_final.h5")


# ──────────────────────────────────────────────
# Flask app
# ──────────────────────────────────────────────

app = Flask(__name__, template_folder=TEMPLATE_DIR, static_folder=STATIC_DIR)

app.secret_key = os.environ.get(
    "FLASK_SECRET_KEY",
    "lung-cancer-detection-secret-key"
)

IMG_SIZE = 224

CLASS_NAMES = ["Benign", "Malignant", "Normal"]


# ──────────────────────────────────────────────
# Load Model
# ──────────────────────────────────────────────

model = load_model(MODEL_PATH)
model.trainable = False

print("Model loaded successfully")


# ──────────────────────────────────────────────
# Image preprocessing
# ──────────────────────────────────────────────

def preprocess_image(image_path: str) -> np.ndarray:

    img = cv2.imread(image_path)

    if img is None:
        raise ValueError("Invalid image")

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

    img = preprocess_input(img.astype(np.float32))

    return np.expand_dims(img, axis=0)


# ──────────────────────────────────────────────
# GradCAM
# ──────────────────────────────────────────────

def save_gradcam_overlay(image_path, img_array, base_filename):

    try:

        heatmap = generate_gradcam(model, img_array)

        original = cv2.imread(image_path)

        overlay = overlay_heatmap(
            heatmap,
            original,
            alpha=0.45,
            thresh_val=160
        )

        heatmap_filename = "heatmap_" + base_filename

        save_path = os.path.join(STATIC_DIR, heatmap_filename)

        cv2.imwrite(save_path, overlay)

        return heatmap_filename

    except Exception as e:

        print("GradCAM failed:", e)

        return None


# ──────────────────────────────────────────────
# MAIN ROUTE
# ──────────────────────────────────────────────

@app.route("/", methods=["GET", "POST"])
def index():

    ctx = dict(
        result=None,
        confidence=None,
        risk=None,
        probs=None,
        age=None,
        smoking=None,
        family_history=None,
        symptoms=None,
        heatmap_filename=None,
        uploaded_filename=None,
        chat_context=None
    )

    if request.method != "POST":
        return render_template("index.html", **ctx)

    # Patient info
    ctx["age"] = request.form.get("age")
    ctx["smoking"] = request.form.get("smoking")
    ctx["family_history"] = request.form.get("family_history")
    ctx["symptoms"] = request.form.get("symptoms")

    # Validate file
    if "image" not in request.files or request.files["image"].filename == "":
        return render_template("index.html", **ctx, error="Please upload an image")

    file = request.files["image"]

    os.makedirs(STATIC_DIR, exist_ok=True)

    filename = str(uuid.uuid4()) + ".png"

    image_path = os.path.join(STATIC_DIR, filename)

    file.save(image_path)

    ctx["uploaded_filename"] = filename

    try:
        img_array = preprocess_image(image_path)
    except Exception:
        return render_template("index.html", **ctx, error="Invalid image file")

    # Prediction
    preds = model.predict(img_array, verbose=0)

    probs = preds[0].tolist()

    predicted_idx = int(np.argmax(preds))

    result = CLASS_NAMES[predicted_idx]

    confidence = float(np.max(preds)) * 100

    risk_map = {
        "Normal": "Low Risk",
        "Benign": "Moderate Risk",
        "Malignant": "High Risk"
    }

    risk = risk_map[result]

    ctx.update(
        result=result,
        confidence=confidence,
        risk=risk,
        probs=probs
    )

    # GradCAM
    ctx["heatmap_filename"] = save_gradcam_overlay(
        image_path,
        img_array,
        filename
    )

    # Store filenames for report generation
    session["uploaded_filename"] = filename
    session["heatmap_filename"] = ctx["heatmap_filename"]

    # Store scan context for chatbot
    session["scan_context"] = {
        "result": result,
        "confidence": confidence,
        "probs": probs,
        "age": ctx["age"],
        "smoking": ctx["smoking"],
        "family_history": ctx["family_history"],
        "symptoms": ctx["symptoms"]
    }

    # Reset chat history
    if "chat_session_id" in session:
        clear_history(session["chat_session_id"])

    session["chat_session_id"] = str(uuid.uuid4())

    ctx["chat_context"] = session["scan_context"]

    return render_template("index.html", **ctx)


# ──────────────────────────────────────────────
# CHATBOT ROUTE
# ──────────────────────────────────────────────

@app.route("/chat", methods=["POST"])
def chat_route():

    data = request.get_json()

    message = data.get("message", "").strip()

    if not message:
        return jsonify({"error": "Empty message"}), 400

    ctx = session.get("scan_context", {})

    system_prompt = build_system_prompt(
        result=ctx.get("result"),
        confidence=ctx.get("confidence"),
        probs=ctx.get("probs"),
        age=ctx.get("age"),
        smoking=ctx.get("smoking"),
        family_history=ctx.get("family_history"),
        symptoms=ctx.get("symptoms")
    )

    if "chat_session_id" not in session:
        session["chat_session_id"] = str(uuid.uuid4())

    reply = chat(
        session_id=session["chat_session_id"],
        user_message=message,
        system_prompt=system_prompt
    )

    return jsonify({"reply": reply})


# ──────────────────────────────────────────────
# RESET CHAT
# ──────────────────────────────────────────────

@app.route("/chat/reset", methods=["POST"])
def chat_reset():

    if "chat_session_id" in session:
        clear_history(session["chat_session_id"])
        session["chat_session_id"] = str(uuid.uuid4())

    return jsonify({"status": "ok"})


# ──────────────────────────────────────────────
# DOWNLOAD PDF REPORT
# ──────────────────────────────────────────────

@app.route("/download_report")
def download_report():

    ctx = session.get("scan_context")

    if not ctx:
        return "No scan data available"

    result = ctx.get("result")

    risk_map = {
        "Normal": "Low Risk",
        "Benign": "Moderate Risk",
        "Malignant": "High Risk"
    }

    risk = risk_map.get(result, "Unknown")

    uploaded_filename = session.get("uploaded_filename")
    heatmap_filename = session.get("heatmap_filename")

    scan_path = None
    heatmap_path = None

    if uploaded_filename:
        scan_path = os.path.join(STATIC_DIR, uploaded_filename)

    if heatmap_filename:
        heatmap_path = os.path.join(STATIC_DIR, heatmap_filename)

    report_data = {
        "age": ctx.get("age"),
        "smoking": ctx.get("smoking"),
        "family_history": ctx.get("family_history"),
        "symptoms": ctx.get("symptoms"),
        "result": result,
        "confidence": ctx.get("confidence"),
        "risk": risk,
        "probs": ctx.get("probs"),
        "scan_path": scan_path,
        "heatmap_path": heatmap_path
    }

    output_path = os.path.join(STATIC_DIR, "report.pdf")

    generate_report(report_data, output_path)

    return send_file(output_path, as_attachment=True)


# ──────────────────────────────────────────────
# RUN APP
# ──────────────────────────────────────────────

if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=5001,
        debug=True
    )