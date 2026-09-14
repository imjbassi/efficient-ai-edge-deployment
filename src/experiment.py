"""Reproduce the MobileNetV2 FP32-versus-INT8 experiment on Imagenette."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import random
import statistics
import tempfile
import time
from pathlib import Path

import numpy as np
import PIL
import tensorflow
from PIL import Image


IMAGENETTE_LABELS = {
    "n01440764": 0,
    "n02102040": 217,
    "n02979186": 482,
    "n03000684": 491,
    "n03028079": 497,
    "n03394916": 566,
    "n03417042": 569,
    "n03425413": 571,
    "n03445777": 574,
    "n03888257": 701,
}
IMAGENETTE_ARCHIVE_SHA256 = "64d0c4859f35a461889e0147755a999a48b49bf38a7e0f9bd27003f10db02fe5"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def image_paths(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES)


def preprocess(path: Path) -> np.ndarray:
    with Image.open(path) as source:
        image = source.convert("RGB")
        width, height = image.size
        scale = 256 / min(width, height)
        image = image.resize((round(width * scale), round(height * scale)), Image.Resampling.BILINEAR)
        left = (image.width - 224) // 2
        top = (image.height - 224) // 2
        image = image.crop((left, top, left + 224, top + 224))
        array = np.asarray(image, dtype=np.float32)
    return (array / 127.5) - 1.0


def quantize_input(array: np.ndarray, detail: dict) -> np.ndarray:
    dtype = detail["dtype"]
    if dtype not in (np.int8, np.uint8):
        return array.astype(dtype)
    scale, zero_point = detail["quantization"]
    if scale <= 0:
        raise ValueError("Invalid input quantization metadata")
    bounds = np.iinfo(dtype)
    return np.clip(np.rint(array / scale + zero_point), bounds.min, bounds.max).astype(dtype)


def convert_models(calibration_paths: list[Path], model_dir: Path) -> tuple[Path, Path]:
    model_dir.mkdir(parents=True, exist_ok=True)
    model = tensorflow.keras.applications.MobileNetV2(weights="imagenet", include_top=True)
    fp32_path = model_dir / "mobilenet_v2_fp32.tflite"
    int8_path = model_dir / "mobilenet_v2_int8.tflite"
    with tempfile.TemporaryDirectory(prefix="edge-deployment-") as directory:
        tensorflow.saved_model.save(model, directory)
        fp32_path.write_bytes(
            tensorflow.lite.TFLiteConverter.from_saved_model(directory).convert()
        )

        def representative_dataset():
            for path in calibration_paths:
                yield [np.expand_dims(preprocess(path), 0)]

        converter = tensorflow.lite.TFLiteConverter.from_saved_model(directory)
        converter.optimizations = [tensorflow.lite.Optimize.DEFAULT]
        converter.representative_dataset = representative_dataset
        converter.target_spec.supported_ops = [tensorflow.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tensorflow.int8
        converter.inference_output_type = tensorflow.int8
        int8_path.write_bytes(converter.convert())
    return fp32_path, int8_path


def make_interpreter(model_path: Path, threads: int):
    interpreter = tensorflow.lite.Interpreter(model_path=str(model_path), num_threads=threads)
    interpreter.allocate_tensors()
    return interpreter


def predict(interpreter, image: np.ndarray) -> np.ndarray:
    input_detail = interpreter.get_input_details()[0]
    output_detail = interpreter.get_output_details()[0]
    value = quantize_input(np.expand_dims(image, 0), input_detail)
    interpreter.set_tensor(input_detail["index"], value)
    interpreter.invoke()
    output = interpreter.get_tensor(output_detail["index"])[0]
    scale, zero_point = output_detail.get("quantization", (0.0, 0))
    if scale:
        output = (output.astype(np.float32) - zero_point) * scale
    return output


def evaluate(interpreter, paths: list[Path]) -> dict:
    correct_top1 = 0
    correct_top5 = 0
    per_class = {name: {"correct_top1": 0, "count": 0} for name in IMAGENETTE_LABELS}
    for index, path in enumerate(paths, start=1):
        synset = path.parent.name
        label = IMAGENETTE_LABELS[synset]
        scores = predict(interpreter, preprocess(path))
        ranking = np.argsort(scores)[-5:][::-1]
        hit = int(ranking[0] == label)
        correct_top1 += hit
        correct_top5 += int(label in ranking)
        per_class[synset]["correct_top1"] += hit
        per_class[synset]["count"] += 1
        if index % 500 == 0:
            print(f"Evaluated {index}/{len(paths)} images")
    count = len(paths)
    def wilson(successes: int) -> list[float]:
        z = 1.959963984540054
        proportion = successes / count
        denominator = 1 + z**2 / count
        center = (proportion + z**2 / (2 * count)) / denominator
        half_width = z * ((proportion * (1 - proportion) / count + z**2 / (4 * count**2)) ** 0.5) / denominator
        return [center - half_width, center + half_width]
    return {
        "samples": count,
        "top1_accuracy": correct_top1 / count,
        "top5_accuracy": correct_top5 / count,
        "top1_correct": correct_top1,
        "top5_correct": correct_top5,
        "top1_wilson_95": wilson(correct_top1),
        "top5_wilson_95": wilson(correct_top5),
        "per_class": per_class,
    }


def benchmark(interpreter, arrays: list[np.ndarray], warmup: int) -> dict:
    for index in range(warmup):
        predict(interpreter, arrays[index % len(arrays)])
    latencies = []
    for image in arrays:
        start = time.perf_counter_ns()
        predict(interpreter, image)
        latencies.append((time.perf_counter_ns() - start) / 1_000_000)
    return {
        "samples": len(latencies),
        "warmup_runs": warmup,
        "mean_ms": statistics.fmean(latencies),
        "median_ms": statistics.median(latencies),
        "p90_ms": float(np.percentile(latencies, 90)),
        "stdev_ms": statistics.stdev(latencies) if len(latencies) > 1 else 0.0,
        "throughput_fps_from_mean": 1000 / statistics.fmean(latencies),
        "raw_ms": latencies,
    }


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def cpu_name() -> str:
    if platform.system() == "Windows":
        try:
            import winreg

            key_path = r"HARDWARE\DESCRIPTION\System\CentralProcessor\0"
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, key_path) as key:
                return str(winreg.QueryValueEx(key, "ProcessorNameString")[0]).strip()
        except OSError:
            pass
    return platform.processor() or "unknown"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--archive", type=Path)
    parser.add_argument("--output", type=Path, default=Path("results/experiment.json"))
    parser.add_argument("--model-dir", type=Path, default=Path("models"))
    parser.add_argument("--calibration-samples", type=int, default=200)
    parser.add_argument("--latency-samples", type=int, default=250)
    parser.add_argument("--warmup", type=int, default=25)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--reuse-models", action="store_true")
    args = parser.parse_args()

    if args.archive and sha256(args.archive) != IMAGENETTE_ARCHIVE_SHA256:
        parser.error("Imagenette archive checksum does not match the pinned digest")

    train_paths = image_paths(args.data_dir / "train")
    val_paths = image_paths(args.data_dir / "val")
    if not train_paths or not val_paths:
        parser.error("Expected Imagenette train/ and val/ directories")
    random.Random(20260914).shuffle(train_paths)
    calibration_paths = train_paths[: args.calibration_samples]
    fp32_path = args.model_dir / "mobilenet_v2_fp32.tflite"
    int8_path = args.model_dir / "mobilenet_v2_int8.tflite"
    if not args.reuse_models or not fp32_path.is_file() or not int8_path.is_file():
        fp32_path, int8_path = convert_models(calibration_paths, args.model_dir)

    validation = {"fp32": None, "int8": None}
    for name, path in (("fp32", fp32_path), ("int8", int8_path)):
        print(f"Evaluating {name}")
        validation[name] = evaluate(make_interpreter(path, args.threads), val_paths)

    latency_paths = val_paths.copy()
    random.Random(20260914).shuffle(latency_paths)
    latency_arrays = [preprocess(path) for path in latency_paths[: args.latency_samples]]
    latency = {}
    for name, path in (("fp32", fp32_path), ("int8", int8_path)):
        print(f"Benchmarking {name}")
        latency[name] = benchmark(make_interpreter(path, args.threads), latency_arrays, args.warmup)

    result = {
        "schema_version": 1,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "dataset": {
            "name": "Imagenette 160 px",
            "url": "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-160.tgz",
            "archive_sha256": IMAGENETTE_ARCHIVE_SHA256,
            "calibration_samples": len(calibration_paths),
            "validation_samples": len(val_paths),
            "calibration_seed": 20260914,
            "evaluation_preprocessing": "RGB; resize shortest side to 256; center crop 224; scale to [-1,1]",
        },
        "environment": {
            "platform": platform.platform(),
            "processor": cpu_name(),
            "python": platform.python_version(),
            "tensorflow": tensorflow.__version__,
            "pillow": PIL.__version__,
            "threads": args.threads,
            "tensorflow_onednn": os.environ.get("TF_ENABLE_ONEDNN_OPTS", "default"),
        },
        "models": {
            "fp32": {"path": fp32_path.as_posix(), "bytes": fp32_path.stat().st_size, "sha256": sha256(fp32_path)},
            "int8": {"path": int8_path.as_posix(), "bytes": int8_path.stat().st_size, "sha256": sha256(int8_path)},
        },
        "accuracy": validation,
        "latency": latency,
        "scope": "Host CPU experiment on the Imagenette validation subset; preprocessing excluded from latency.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
