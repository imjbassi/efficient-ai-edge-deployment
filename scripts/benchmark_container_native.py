"""Run the paper's native invocation protocol inside the service container."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import random
import statistics
import time
from pathlib import Path

import numpy as np
import psutil
from PIL import Image
from tflite_runtime.interpreter import Interpreter


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def preprocess(path: Path) -> np.ndarray:
    with Image.open(path) as source:
        image = source.convert("RGB")
        width, height = image.size
        scale = 256 / min(width, height)
        image = image.resize(
            (round(width * scale), round(height * scale)),
            Image.Resampling.BILINEAR,
        )
        left = (image.width - 224) // 2
        top = (image.height - 224) // 2
        image = image.crop((left, top, left + 224, top + 224))
        array = np.asarray(image, dtype=np.float32)
    return (array / 127.5) - 1.0


def quantize_input(array: np.ndarray, detail: dict) -> np.ndarray:
    dtype = detail["dtype"]
    scale, zero_point = detail["quantization"]
    bounds = np.iinfo(dtype)
    return np.clip(np.rint(array / scale + zero_point), bounds.min, bounds.max).astype(dtype)


def delegated_op_count(interpreter: Interpreter) -> int:
    return sum(op["op_name"] == "DELEGATE" for op in interpreter._get_ops_details())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--delegate-probe-model", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=500)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260914)
    args = parser.parse_args()

    paths = sorted(
        path for path in (args.data_dir / "val").rglob("*")
        if path.suffix.lower() in IMAGE_SUFFIXES
    )
    random.Random(args.seed).shuffle(paths)
    paths = paths[: args.samples]
    if len(paths) != args.samples:
        parser.error(f"requested {args.samples} images but found {len(paths)}")
    relative_paths = [path.relative_to(args.data_dir).as_posix() for path in paths]
    selection_sha256 = hashlib.sha256("\n".join(relative_paths).encode()).hexdigest()
    arrays = [preprocess(path) for path in paths]

    interpreter = Interpreter(model_path=str(args.model), num_threads=args.threads)
    interpreter.allocate_tensors()
    input_detail = interpreter.get_input_details()[0]
    values = [quantize_input(np.expand_dims(array, 0), input_detail) for array in arrays]
    for index in range(args.warmup):
        interpreter.set_tensor(input_detail["index"], values[index % len(values)])
        interpreter.invoke()

    raw_ms = []
    for value in values:
        interpreter.set_tensor(input_detail["index"], value)
        started = time.perf_counter_ns()
        interpreter.invoke()
        raw_ms.append((time.perf_counter_ns() - started) / 1_000_000)

    delegate_probe = None
    if args.delegate_probe_model:
        probe = Interpreter(model_path=str(args.delegate_probe_model), num_threads=args.threads)
        probe.allocate_tensors()
        delegate_probe = {
            "model": args.delegate_probe_model.as_posix(),
            "model_sha256": sha256(args.delegate_probe_model),
            "delegated_ops": delegated_op_count(probe),
        }

    result = {
        "schema_version": 1,
        "protocol": {
            "samples": args.samples,
            "warmup_invocations": args.warmup,
            "threads": args.threads,
            "seed": args.seed,
            "preprocessing_timed": False,
            "input_selection_sha256": selection_sha256,
        },
        "model": {
            "path": args.model.as_posix(),
            "sha256": sha256(args.model),
        },
        "runtime": {
            "package": "tflite-runtime",
            "version": importlib.metadata.version("tflite-runtime"),
            "platform": platform.platform(),
            "cpu_count_visible": psutil.cpu_count(),
            "delegated_ops": delegated_op_count(interpreter),
            "xnnpack_active_for_benchmarked_int8_graph": delegated_op_count(interpreter) > 0,
            "fp32_delegate_probe": delegate_probe,
        },
        "latency": {
            "mean_ms": statistics.fmean(raw_ms),
            "median_ms": statistics.median(raw_ms),
            "p90_ms": float(np.percentile(raw_ms, 90)),
            "p99_ms": float(np.percentile(raw_ms, 99)),
            "raw_ms": raw_ms,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({**result["protocol"], **result["model"], **result["runtime"], **result["latency"]}, indent=2))


if __name__ == "__main__":
    main()
