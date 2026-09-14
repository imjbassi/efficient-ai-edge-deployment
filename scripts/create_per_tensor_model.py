"""Create the per-tensor INT8 control used by the latency ablation."""

from __future__ import annotations

import argparse
import random
import sys
import tempfile
from pathlib import Path

import numpy as np
import tensorflow as tf


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from experiment import image_paths, preprocess  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("models/mobilenet_v2_int8_per_tensor.tflite"),
    )
    args = parser.parse_args()
    paths = image_paths(args.data_dir / "train")
    random.Random(20260914).shuffle(paths)
    calibration = paths[:200]
    if len(calibration) != 200:
        parser.error("Expected at least 200 calibration images")

    model = tf.keras.applications.MobileNetV2(weights="imagenet", include_top=True)
    with tempfile.TemporaryDirectory(prefix="edge-per-tensor-") as directory:
        tf.saved_model.save(model, directory)

        def representative_dataset():
            for path in calibration:
                yield [np.expand_dims(preprocess(path), 0)]

        converter = tf.lite.TFLiteConverter.from_saved_model(directory)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = representative_dataset
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8
        converter._experimental_disable_per_channel = True
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_bytes(converter.convert())
    print(args.output)


if __name__ == "__main__":
    main()
