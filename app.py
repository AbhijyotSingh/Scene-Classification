import io
import os

import numpy as np
from flask import Flask, jsonify, request
from flask_cors import CORS
from PIL import Image, UnidentifiedImageError
import tensorflow as tf

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "CNN.keras")

IMG_SIZE = (150, 150)  # same size used in image_dataset_from_directory

# image_dataset_from_directory assigns labels in alphabetical folder order.
CLASS_NAMES = ["buildings", "forest", "glacier", "mountain", "sea", "street"]

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 8 * 1024 * 1024  # 8 MB upload limit
CORS(app)  # allow the frontend (any origin) to call this API

# Load once at startup, not on every request.
model = tf.keras.models.load_model(MODEL_PATH)


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
    return jsonify(status="ok")


@app.post("/predict")
def predict():
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
