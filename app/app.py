"""
app.py — Flask entry point for the Lung Cancer Detection System.

Project layout (relative to this file, which lives in /app):
    ../models/efficientnet_final.h5   ← trained model
    ../templates/index.html           ← Jinja2 template
    ../static/                        ← uploaded + generated images
"""

import os
import uuid

import cv2
import numpy as np
from flask import Flask, render_template, request
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.models import load_model

from gradcam import generate_gradcam, overlay_heatmap   # our XAI module

# ──────────────────────────────────────────────
# App & paths
# ──────────────────────────────────────────────

app = Flask(__name__, template_folder="../templates", static_folder="../static")

MODEL_PATH  = os.path.join(os.path.dirname(__file__), "../models/efficientnet_final.h5")
STATIC_DIR  = os.path.join(os.path.dirname(__file__), "../static")
IMG_SIZE    = 224

# Must match train_generator.class_indices exactly
CLASS_NAMES = ["Benign", "Malignant", "Normal"]

# Load once at start-up (expensive)
model = load_model(MODEL_PATH)
model.trainable = False          # inference only — saves memory


# ──────────────────────────────────────────────
# Image helpers
# ──────────────────────────────────────────────

def preprocess_image(image_path: str) -> np.ndarray:
    """
    Read a CT scan from disk and return a batch-of-one float32 tensor
    preprocessed the same way as during EfficientNet training.
    """
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise ValueError(f"Could not read image: {image_path}")

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_rgb = cv2.resize(img_rgb, (IMG_SIZE, IMG_SIZE),
                         interpolation=cv2.INTER_AREA)

    # EfficientNet-specific scaling (maps pixel values to [-1, +1])
    img_pre = preprocess_input(img_rgb.astype(np.float32))

    return np.expand_dims(img_pre, axis=0)   # (1, 224, 224, 3)


def save_gradcam_overlay(image_path: str,
                         img_array:  np.ndarray,
                         base_filename: str) -> str | None:
    """
    Run Grad-CAM, overlay the heatmap on the original image, save to
    /static, and return the *relative* path suitable for url_for().

    Returns None (silently) if anything goes wrong so the rest of the
    page still renders.
    """
    try:
        heatmap = generate_gradcam(model, img_array)

        original_bgr = cv2.imread(image_path)
        if original_bgr is None:
            raise ValueError("Cannot reload original image for overlay.")

        overlay = overlay_heatmap(heatmap, original_bgr,
                                  alpha=0.45, thresh_val=160)

        heatmap_filename = "heatmap_" + base_filename
        heatmap_path_abs = os.path.join(STATIC_DIR, heatmap_filename)
        cv2.imwrite(heatmap_path_abs, overlay)

        # Return path relative to /static so url_for('static', …) works
        return heatmap_filename

    except Exception as exc:
        app.logger.warning("Grad-CAM failed: %s", exc)
        return None


# ──────────────────────────────────────────────
# Routes
# ──────────────────────────────────────────────

@app.route("/", methods=["GET", "POST"])
def index():
    # Template context defaults
    ctx = dict(
        result=None, confidence=None, risk=None, probs=None,
        age=None, smoking=None, family_history=None, symptoms=None,
        heatmap_filename=None,
        uploaded_filename=None,
    )

    if request.method != "POST":
        return render_template("index.html", **ctx)

    # ── Patient form fields ──────────────────────────────────────────────
    ctx["age"]            = request.form.get("age")
    ctx["smoking"]        = request.form.get("smoking")
    ctx["family_history"] = request.form.get("family_history")
    ctx["symptoms"]       = request.form.get("symptoms")

    # ── File validation ──────────────────────────────────────────────────
    if "image" not in request.files or request.files["image"].filename == "":
        return render_template("index.html", **ctx,
                               error="Please upload a CT scan image.")

    file = request.files["image"]

    # Persist original upload
    base_filename       = str(uuid.uuid4()) + ".png"
    image_path_abs      = os.path.join(STATIC_DIR, base_filename)
    os.makedirs(STATIC_DIR, exist_ok=True)
    file.save(image_path_abs)
    ctx["uploaded_filename"] = base_filename

    # ── Preprocessing ────────────────────────────────────────────────────
    try:
        img_array = preprocess_image(image_path_abs)
    except ValueError as exc:
        app.logger.error("Preprocessing error: %s", exc)
        return render_template("index.html", **ctx,
                               error="Invalid image file. Please upload a valid CT scan.")

    # ── Model inference ──────────────────────────────────────────────────
    raw_preds      = model.predict(img_array, verbose=0)          # (1, 3)
    probs          = raw_preds[0].tolist()                        # Python list
    predicted_idx  = int(np.argmax(raw_preds))
    result         = CLASS_NAMES[predicted_idx]
    confidence     = float(np.max(raw_preds)) * 100

    app.logger.info("Prediction: %s | Confidence: %.1f%% | Raw: %s",
                    result, confidence, probs)

    # ── Risk classification ──────────────────────────────────────────────
    risk_map = {"Normal": "Low Risk", "Benign": "Moderate Risk", "Malignant": "High Risk"}
    risk = risk_map[result]

    ctx.update(result=result, confidence=confidence, risk=risk, probs=probs)

    # ── Grad-CAM ─────────────────────────────────────────────────────────
    ctx["heatmap_filename"] = save_gradcam_overlay(image_path_abs,
                                                   img_array,
                                                   base_filename)

    return render_template("index.html", **ctx)


# ──────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────

if __name__ == "__main__":
    # debug=False for any public / shared environment
    app.run(host="0.0.0.0", port=5001, debug=True)