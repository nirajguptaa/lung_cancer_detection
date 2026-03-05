"""
gradcam.py — Robust Grad-CAM implementation compatible with EfficientNet,
ResNet, VGG, MobileNet and custom CNN models.

Functions:
1. find_last_conv_layer() → finds last convolution layer (handles nested blocks)
2. generate_gradcam()     → computes Grad-CAM heatmap
3. overlay_heatmap()      → overlays heatmap and draws bounding box
"""

import numpy as np
import cv2
import tensorflow as tf


# --------------------------------------------------
# 1. Find Last Convolution Layer
# --------------------------------------------------

def find_last_conv_layer(model: tf.keras.Model) -> str | None:
    """
    Return the name of the last convolution layer in the model.
    Works with EfficientNet and nested architectures.
    """

    # EfficientNet specific layer
    try:
        model.get_layer("top_conv")
        return "top_conv"
    except Exception:
        pass

    def search_layers(layers):
        for layer in reversed(layers):

            if isinstance(layer, (tf.keras.layers.Conv2D, tf.keras.layers.DepthwiseConv2D)):
                try:
                    shape = layer.output_shape
                    if isinstance(shape, (list, tuple)) and len(shape) == 4:
                        return layer.name
                except Exception:
                    pass

            # nested blocks (EfficientNet / MobileNet)
            if hasattr(layer, "layers"):
                result = search_layers(layer.layers)
                if result is not None:
                    return result

        return None

    return search_layers(model.layers)


# --------------------------------------------------
# 2. Generate GradCAM Heatmap
# --------------------------------------------------

def generate_gradcam(model: tf.keras.Model, img_array: np.ndarray) -> np.ndarray:

    # build model once
    _ = model(img_array, training=False)

    last_conv_layer_name = find_last_conv_layer(model)

    if last_conv_layer_name is None:
        raise ValueError("No convolution layer found in model.")

    last_conv_layer = model.get_layer(last_conv_layer_name)

    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[last_conv_layer.output, model.output]
    )

    with tf.GradientTape() as tape:

        conv_outputs, predictions = grad_model(img_array, training=False)

        if isinstance(conv_outputs, (list, tuple)):
            conv_outputs = conv_outputs[0]

        if isinstance(predictions, (list, tuple)):
            predictions = predictions[0]

        predictions = tf.convert_to_tensor(predictions)

        class_index = tf.argmax(predictions[0])

        loss = predictions[:, class_index]

    grads = tape.gradient(loss, conv_outputs)

    if isinstance(grads, (list, tuple)):
        grads = grads[0]

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]

    heatmap = tf.reduce_sum(pooled_grads * conv_outputs, axis=-1)

    heatmap = tf.maximum(heatmap, 0)

    max_val = tf.reduce_max(heatmap)

    if max_val > 0:
        heatmap = heatmap / max_val

    return heatmap.numpy()


# --------------------------------------------------
# 3. Overlay Heatmap + Bounding Box
# --------------------------------------------------

def overlay_heatmap(heatmap: np.ndarray,
                    original_bgr: np.ndarray,
                    alpha: float = 0.45,
                    thresh_val: int = 150) -> np.ndarray:

    h, w = original_bgr.shape[:2]

    heatmap = cv2.resize(heatmap, (w, h))

    heatmap_uint8 = np.uint8(255 * heatmap)

    colored_heatmap = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)

    overlay = cv2.addWeighted(original_bgr, 1 - alpha,
                              colored_heatmap, alpha, 0)

    # threshold for tumor region
    _, mask = cv2.threshold(heatmap_uint8, thresh_val, 255, cv2.THRESH_BINARY)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:

        largest = max(contours, key=cv2.contourArea)

        x, y, w_box, h_box = cv2.boundingRect(largest)

        pad = 6

        x = max(0, x - pad)
        y = max(0, y - pad)

        w_box = min(original_bgr.shape[1] - x, w_box + 2 * pad)
        h_box = min(original_bgr.shape[0] - y, h_box + 2 * pad)

        cv2.rectangle(overlay, (x, y), (x + w_box, y + h_box), (0, 255, 0), 2)

        cv2.putText(
            overlay,
            "Suspected Region",
            (x, max(y - 10, 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 0),
            2,
            cv2.LINE_AA
        )

    return overlay