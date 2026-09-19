"""Run this locally (where TensorFlow is installed), in the folder with CNN.keras.

    python convert_to_tflite.py

It creates CNN.tflite and checks that its predictions match the Keras model.
"""
import numpy as np
import tensorflow as tf

model = tf.keras.models.load_model(r"C:\Users\Abhijyot Singh Roda\Desktop\Coding Stuff\Python\Projects\Not so random projects\ML Projects\Projects\CNN\Scene Classification\CNN.keras")

converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

with open("CNN.tflite", "wb") as f:
    f.write(tflite_model)
print(f"Saved CNN.tflite ({len(tflite_model) / 1024:.0f} KB)")

# Sanity check: same input through both models should give (almost) the same output.
x = np.random.uniform(0, 255, size=(1, 150, 150, 3)).astype(np.float32)
keras_out = model.predict(x, verbose=0)[0]

interp = tf.lite.Interpreter(model_content=tflite_model)
interp.allocate_tensors()
inp = interp.get_input_details()[0]
out = interp.get_output_details()[0]
interp.set_tensor(inp["index"], x)
interp.invoke()
tflite_out = interp.get_tensor(out["index"])[0]

print("Input shape :", inp["shape"], inp["dtype"])
print("Max difference between Keras and TFLite:", float(np.max(np.abs(keras_out - tflite_out))))