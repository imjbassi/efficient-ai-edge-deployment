import os
from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image


def representative_data_gen(image_dir=None, limit=100):
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


def quantize_model(keras_model_path, output_tflite_path):
    model = tf.keras.models.load_model(keras_model_path)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_data_gen
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    with open(output_tflite_path, "wb") as f:
        f.write(converter.convert())
    print(f"Quantized model saved to {output_tflite_path}")


if __name__ == "__main__":
    from model_factory import get_mobilenet_v2

    model_path = "mobilenet_v2.h5"
    get_mobilenet_v2().save(model_path)
    quantize_model(model_path, "src/mobilenet_v2_quant.tflite")
