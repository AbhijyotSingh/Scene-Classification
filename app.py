import io
import os
import threading

import numpy as np
from flask import Flask, jsonify, request
from flask_cors import CORS
from PIL import Image, UnidentifiedImageError

try:
    from tflite_runtime.interpreter import Interpreter  # small runtime used on Render
except ImportError:  # local development with full TensorFlow installed
    import tensorflow as tf

    Interpreter = tf.lite.Interpreter

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "CNN.tflite")

IMG_SIZE = (150, 150)  # same size used in image_dataset_from_directory

# image_dataset_from_directory assigns labels in alphabetical folder order.
CLASS_NAMES = ["buildings", "forest", "glacier", "mountain", "sea", "street"]

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 8 * 1024 * 1024  # 8 MB upload limit
CORS(app)  # allow the frontend (any origin) to call this API

interpreter = None
input_index = output_index = None
model_error = None
lock = threading.Lock()  # a TFLite interpreter is not thread-safe

try:
    interpreter = Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()
    input_index = interpreter.get_input_details()[0]["index"]
    output_index = interpreter.get_output_details()[0]["index"]
except Exception as exc:  # noqa: BLE001
    model_error = f"{type(exc).__name__}: {exc}"


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
    return jsonify(status="ok", model_ready=interpreter is not None, model_error=model_error)


@app.post("/predict")
def predict():
    if interpreter is None:
        return jsonify(error=f"The model failed to load: {model_error}"), 500

    if "file" not in request.files or request.files["file"].filename == "":
        return jsonify(error="No image uploaded. Send a file in the 'file' field."), 400

    try:
        batch = preprocess(request.files["file"].read())
    except (UnidentifiedImageError, OSError):
        return jsonify(error="That file isn't a readable image. Try a JPG or PNG."), 400

    with lock:
        interpreter.set_tensor(input_index, batch)
        interpreter.invoke()
        probs = interpreter.get_tensor(output_index)[0].copy()

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
