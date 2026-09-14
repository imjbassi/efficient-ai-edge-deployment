import os
from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image


def representative_data_gen(image_dir=None, limit=100):
    """Yield real calibration images; refuse synthetic calibration by default."""
    image_dir = image_dir or os.environ.get("EDGE_CALIBRATION_DIR")
    if not image_dir:
        raise ValueError("Set EDGE_CALIBRATION_DIR to a directory of calibration images.")
    paths = sorted(p for p in Path(image_dir).rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"})[:limit]
    if not paths:
        raise ValueError(f"No supported images found in {image_dir}")
    for path in paths:
        image = Image.open(path).convert("RGB").resize((224, 224))
        data = (np.asarray(image, dtype=np.float32) / 127.5) - 1.0
        yield [np.expand_dims(data, axis=0)]


def apply_int8_quantization(keras_model, output_path):
    converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_data_gen
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    tflite_model_quant = converter.convert()
    with open(output_path, "wb") as f:
        f.write(tflite_model_quant)
    print(f"Quantization complete. Saved to {output_path}")
    print(f"Model size: {os.path.getsize(output_path) / (1024 * 1024):.2f} MB")


if __name__ == "__main__":
    from model_factory import get_mobilenet_v2

    os.makedirs("models", exist_ok=True)
    apply_int8_quantization(get_mobilenet_v2(), "models/mobilenet_v2_int8.tflite")
