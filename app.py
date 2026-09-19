import io
import os
import threading

import numpy as np
from flask import Flask, jsonify, request
from flask_cors import CORS
from PIL import Image, UnidentifiedImageError

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "CNN.keras")

IMG_SIZE = (150, 150)  # same size used in image_dataset_from_directory

# image_dataset_from_directory assigns labels in alphabetical folder order.
CLASS_NAMES = ["buildings", "forest", "glacier", "mountain", "sea", "street"]

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 8 * 1024 * 1024  # 8 MB upload limit
CORS(app)  # allow the frontend (any origin) to call this API

model = None
model_error = None


def load_model_in_background():
    """Import TensorFlow and load the model without blocking the web server.

    Importing TensorFlow and loading the model is slow on small hosts. Doing it
    in a background thread lets Flask start answering /health immediately, so
    gunicorn doesn't kill the worker for taking too long to boot.
    """
    global model, model_error
    try:
        import tensorflow as tf

        loaded = tf.keras.models.load_model(MODEL_PATH)
        # Warm-up so the first real request isn't slow.
        loaded.predict(np.zeros((1, *IMG_SIZE, 3), dtype=np.float32), verbose=0)
        model = loaded
    except Exception as exc:  # noqa: BLE001
        model_error = f"{type(exc).__name__}: {exc}"


threading.Thread(target=load_model_in_background, daemon=True).start()


def preprocess(file_bytes: bytes) -> np.ndarray:
    """Turn uploaded bytes into a (1, 150, 150, 3) float32 batch.

    The notebook's CNN has no Rescaling layer and was trained on raw
    0-255 pixel values, so the image must NOT be divided by 255 here.
    """
    img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    img = img.resize(IMG_SIZE, Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32)
    return np.expand_dims(arr, axis=0)


@app.get("/")
def index():
    return jsonify(status="ok", service="scene-classification", classes=CLASS_NAMES)


@app.get("/health")
def health():
    # Always answers right away; model_ready tells the page when it can classify.
    return jsonify(status="ok", model_ready=model is not None, model_error=model_error)


@app.post("/predict")
def predict():
    if model is None:
        if model_error:
            return jsonify(error=f"The model failed to load: {model_error}"), 500
        return jsonify(error="The model is still loading. Try again in a moment."), 503

    if "file" not in request.files or request.files["file"].filename == "":
        return jsonify(error="No image uploaded. Send a file in the 'file' field."), 400

    try:
        batch = preprocess(request.files["file"].read())
    except (UnidentifiedImageError, OSError):
        return jsonify(error="That file isn't a readable image. Try a JPG or PNG."), 400

    probs = model.predict(batch, verbose=0)[0]
    top = int(np.argmax(probs))

    return jsonify(
        label=CLASS_NAMES[top],
        confidence=float(probs[top]),
        probabilities={name: float(p) for name, p in zip(CLASS_NAMES, probs)},
    )


@app.errorhandler(413)
def too_large(_):
    return jsonify(error="Image is too large. Keep it under 8 MB."), 413


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
