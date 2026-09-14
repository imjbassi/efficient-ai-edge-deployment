"""Run one isolated TFLite latency replicate in a fresh process."""

from __future__ import annotations

import argparse
import json
import os
import platform
import random
import statistics
import time
from pathlib import Path

import numpy as np
import psutil
import tensorflow as tf

from experiment import image_paths, preprocess, quantize_input


def percentile(values: list[float], probability: float) -> float:
    return float(np.percentile(values, probability))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=500)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--disable-default-delegates", action="store_true")
    parser.add_argument("--affinity", help="Comma-separated logical CPU identifiers")
    parser.add_argument("--seed", type=int, default=20260914)
    args = parser.parse_args()

    process = psutil.Process()
    requested_affinity = [int(value) for value in args.affinity.split(",")] if args.affinity else []
    if requested_affinity:
        process.cpu_affinity(requested_affinity)
    try:
        process.nice(psutil.ABOVE_NORMAL_PRIORITY_CLASS)
    except (AttributeError, psutil.AccessDenied):
        pass

    paths = image_paths(args.data_dir / "val")
    random.Random(args.seed).shuffle(paths)
    paths = paths[: args.samples]
    arrays = [preprocess(path) for path in paths]
    options = {"model_path": str(args.model), "num_threads": args.threads}
    if args.disable_default_delegates:
        options["experimental_op_resolver_type"] = (
            tf.lite.experimental.OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES
        )
    idle_cpu_percent = psutil.cpu_percent(interval=1.0)
    created = time.perf_counter_ns()
    interpreter = tf.lite.Interpreter(**options)
    input_detail = interpreter.get_input_details()[0]
    if args.batch_size > 1:
        resized_shape = list(input_detail["shape"])
        resized_shape[0] = args.batch_size
        interpreter.resize_tensor_input(input_detail["index"], resized_shape, strict=False)
    interpreter.allocate_tensors()
    initialization_ms = (time.perf_counter_ns() - created) / 1_000_000
    input_detail = interpreter.get_input_details()[0]

    batches = []
    for start in range(0, len(arrays), args.batch_size):
        batch = arrays[start : start + args.batch_size]
        if len(batch) == args.batch_size:
            batches.append(quantize_input(np.stack(batch), input_detail))
    if not batches:
        parser.error("Not enough images for one complete batch")

    def invoke(value: np.ndarray) -> None:
        interpreter.set_tensor(input_detail["index"], value)
        interpreter.invoke()

    for index in range(args.warmup):
        invoke(batches[index % len(batches)])
    raw_invocation_ms = []
    for value in batches:
        started = time.perf_counter_ns()
        invoke(value)
        raw_invocation_ms.append((time.perf_counter_ns() - started) / 1_000_000)
    raw_per_image_ms = [value / args.batch_size for value in raw_invocation_ms]
    result = {
        "model": args.model.as_posix(),
        "threads": args.threads,
        "batch_size": args.batch_size,
        "default_delegates": not args.disable_default_delegates,
        "images": len(batches) * args.batch_size,
        "invocations": len(batches),
        "warmup_invocations": args.warmup,
        "idle_cpu_percent_before_interpreter": idle_cpu_percent,
        "cpu_affinity": process.cpu_affinity(),
        "process_priority": process.nice(),
        "initialization_ms": initialization_ms,
        "mean_ms_per_image": statistics.fmean(raw_per_image_ms),
        "median_ms_per_image": statistics.median(raw_per_image_ms),
        "p90_ms_per_image": percentile(raw_per_image_ms, 90),
        "p99_ms_per_image": percentile(raw_per_image_ms, 99),
        "raw_invocation_ms": raw_invocation_ms,
        "raw_ms_per_image": raw_per_image_ms,
        "platform": platform.platform(),
        "tensorflow": tf.__version__,
        "tensorflow_onednn": os.environ.get("TF_ENABLE_ONEDNN_OPTS", "default"),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if not key.startswith("raw_")}))


if __name__ == "__main__":
    main()
