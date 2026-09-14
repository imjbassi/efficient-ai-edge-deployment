"""Diagnose the per-tensor INT8 failure using predictions and weight ranges."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import tensorflow as tf


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from experiment import image_paths, make_interpreter, predict, preprocess  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_inventory(interpreter: tf.lite.Interpreter) -> dict:
    details = interpreter.get_tensor_details()
    scale_counts = [len(detail["quantization_parameters"]["scales"]) for detail in details]
    return {
        "tensor_count": len(details),
        "tensors_with_multiple_scales": sum(count > 1 for count in scale_counts),
        "maximum_scales_per_tensor": max(scale_counts),
        "input_dtype": str(interpreter.get_input_details()[0]["dtype"]),
        "output_dtype": str(interpreter.get_output_details()[0]["dtype"]),
    }


def depthwise_weight_ranges(interpreter: tf.lite.Interpreter) -> list[dict]:
    layers = []
    for detail in interpreter.get_tensor_details():
        parameters = detail["quantization_parameters"]
        scales = np.asarray(parameters["scales"], dtype=np.float64)
        zero_points = np.asarray(parameters["zero_points"], dtype=np.float64)
        if "depthwise" not in detail["name"].lower() or len(scales) <= 1:
            continue
        try:
            quantized = interpreter.get_tensor(detail["index"]).astype(np.float64)
        except ValueError:
            continue
        axis = int(parameters["quantized_dimension"])
        shape = [1] * quantized.ndim
        shape[axis] = len(scales)
        dequantized = (quantized - zero_points.reshape(shape)) * scales.reshape(shape)
        channel_first = np.moveaxis(dequantized, axis, 0).reshape(len(scales), -1)
        channel_ranges = np.ptp(channel_first, axis=1)
        nonzero_ranges = channel_ranges[channel_ranges > 0]
        if not len(nonzero_ranges):
            continue
        layers.append({
            "name": detail["name"],
            "shape": detail["shape"].tolist(),
            "channels": len(scales),
            "scale_min": float(scales.min()),
            "scale_max": float(scales.max()),
            "scale_ratio_max_min": float(scales.max() / scales.min()),
            "channel_range_min": float(nonzero_ranges.min()),
            "channel_range_median": float(np.median(nonzero_ranges)),
            "channel_range_max": float(nonzero_ranges.max()),
            "channel_range_ratio_max_min": float(
                nonzero_ranges.max() / nonzero_ranges.min()
            ),
        })
    return layers


def prediction_distribution(model: Path, data_dir: Path) -> dict:
    interpreter = make_interpreter(model, 1)
    counts: Counter[int] = Counter()
    maximum_scores = []
    paths = image_paths(data_dir / "val")
    for index, path in enumerate(paths, start=1):
        scores = predict(interpreter, preprocess(path))
        counts[int(np.argmax(scores))] += 1
        maximum_scores.append(float(np.max(scores)))
        if index % 500 == 0:
            print(f"Per-tensor prediction audit {index}/{len(paths)}", flush=True)
    probabilities = np.asarray(list(counts.values()), dtype=np.float64) / len(paths)
    entropy_bits = -float(np.sum(probabilities * np.log2(probabilities)))
    return {
        "samples": len(paths),
        "unique_top1_classes": len(counts),
        "entropy_bits": entropy_bits,
        "maximum_possible_entropy_bits": math.log2(1000),
        "dominant_class_fraction": max(counts.values()) / len(paths),
        "mean_maximum_output_score": float(np.mean(maximum_scores)),
        "top1_histogram": [
            {"class_index": class_index, "count": count, "fraction": count / len(paths)}
            for class_index, count in counts.most_common()
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument(
        "--per-channel-model",
        type=Path,
        default=Path("models/mobilenet_v2_int8.tflite"),
    )
    parser.add_argument(
        "--per-tensor-model",
        type=Path,
        default=Path("models/mobilenet_v2_int8_per_tensor.tflite"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/quantization_diagnostics.json"),
    )
    args = parser.parse_args()

    per_channel = make_interpreter(args.per_channel_model, 1)
    per_tensor = make_interpreter(args.per_tensor_model, 1)
    depthwise = depthwise_weight_ranges(per_channel)
    if not depthwise:
        raise RuntimeError("No per-channel depthwise weight tensors were found")
    result = {
        "schema_version": 1,
        "interpretation": (
            "The same preprocessing and evaluator are used for both INT8 models. "
            "The per-tensor control has signed INT8 I/O and no multi-scale tensors; "
            "the per-channel model exposes large within-layer channel-range ratios."
        ),
        "models": {
            "per_channel": {
                "path": args.per_channel_model.as_posix(),
                "sha256": sha256(args.per_channel_model),
                "inventory": tensor_inventory(per_channel),
            },
            "per_tensor": {
                "path": args.per_tensor_model.as_posix(),
                "sha256": sha256(args.per_tensor_model),
                "inventory": tensor_inventory(per_tensor),
            },
        },
        "per_tensor_prediction_distribution": prediction_distribution(
            args.per_tensor_model, args.data_dir
        ),
        "per_channel_depthwise_weight_ranges": {
            "layers": depthwise,
            "layer_count": len(depthwise),
            "maximum_scale_ratio": max(
                layer["scale_ratio_max_min"] for layer in depthwise
            ),
            "maximum_channel_range_ratio": max(
                layer["channel_range_ratio_max_min"] for layer in depthwise
            ),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({
        "prediction_distribution": result["per_tensor_prediction_distribution"],
        "depthwise_summary": {
            key: value
            for key, value in result["per_channel_depthwise_weight_ranges"].items()
            if key != "layers"
        },
    }, indent=2))


if __name__ == "__main__":
    main()
