"""Orchestrate latency replicates with a fresh process for every run."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import statistics
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import psutil
from scipy.stats import t as student_t


def t_critical_95(sample_count: int) -> float:
    if sample_count < 2:
        return 0.0
    return float(student_t.ppf(0.975, df=sample_count - 1))


def summarize(runs: list[dict]) -> dict:
    means = [run["mean_ms_per_image"] for run in runs]
    mean = statistics.fmean(means)
    if len(means) > 1:
        standard_error = statistics.stdev(means) / len(means) ** 0.5
        half_width = t_critical_95(len(means)) * standard_error
        ci = [mean - half_width, mean + half_width]
    else:
        ci = [mean, mean]
    pooled = [value for run in runs for value in run["raw_ms_per_image"]]
    return {
        "replicates": len(runs),
        "mean_of_run_means_ms": mean,
        "run_mean_stdev_ms": statistics.stdev(means) if len(means) > 1 else 0.0,
        "run_mean_t_95": ci,
        "pooled_median_ms": statistics.median(pooled),
        "pooled_p90_ms": float(np.percentile(pooled, 90)),
        "pooled_p99_ms": float(np.percentile(pooled, 99)),
        "run_means_ms": means,
    }


def compare(fp32_runs: list[dict], int8_runs: list[dict]) -> dict:
    if len(fp32_runs) != len(int8_runs):
        raise ValueError("Matched latency configurations require equal replicate counts")
    log_ratios = [
        math.log(fp32["mean_ms_per_image"] / int8["mean_ms_per_image"])
        for fp32, int8 in zip(fp32_runs, int8_runs)
    ]
    center = statistics.fmean(log_ratios)
    standard_error = statistics.stdev(log_ratios) / len(log_ratios) ** 0.5
    half_width = t_critical_95(len(log_ratios)) * standard_error
    return {
        "definition": "FP32 run mean divided by matched INT8 run mean; values above one favor INT8",
        "replicate_speedups": [math.exp(value) for value in log_ratios],
        "geometric_mean_speedup": math.exp(center),
        "paired_log_t_95": [math.exp(center - half_width), math.exp(center + half_width)],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("results/repeated_latency.json"))
    parser.add_argument("--replicates", type=int, default=20)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("results/repeated_latency.tmp"),
        help="Resumable checkpoint updated after every fresh-process run",
    )
    parser.add_argument("--host-note", default="")
    args = parser.parse_args()

    configurations = [
        {"name": "fp32_xnnpack_t1_b1", "model": "models/mobilenet_v2_fp32.tflite", "threads": 1, "batch": 1, "samples": 500, "delegates": True},
        {"name": "int8_perchannel_xnnpack_t1_b1", "model": "models/mobilenet_v2_int8.tflite", "threads": 1, "batch": 1, "samples": 500, "delegates": True},
        {"name": "fp32_xnnpack_t2_b1", "model": "models/mobilenet_v2_fp32.tflite", "threads": 2, "batch": 1, "samples": 500, "delegates": True},
        {"name": "int8_perchannel_xnnpack_t2_b1", "model": "models/mobilenet_v2_int8.tflite", "threads": 2, "batch": 1, "samples": 500, "delegates": True},
        {"name": "fp32_xnnpack_t4_b1", "model": "models/mobilenet_v2_fp32.tflite", "threads": 4, "batch": 1, "samples": 500, "delegates": True},
        {"name": "int8_perchannel_xnnpack_t4_b1", "model": "models/mobilenet_v2_int8.tflite", "threads": 4, "batch": 1, "samples": 500, "delegates": True},
        {"name": "fp32_xnnpack_t6_b1", "model": "models/mobilenet_v2_fp32.tflite", "threads": 6, "batch": 1, "samples": 500, "delegates": True},
        {"name": "int8_perchannel_xnnpack_t6_b1", "model": "models/mobilenet_v2_int8.tflite", "threads": 6, "batch": 1, "samples": 500, "delegates": True},
        {"name": "fp32_builtin_t1_b1", "model": "models/mobilenet_v2_fp32.tflite", "threads": 1, "batch": 1, "samples": 100, "delegates": False},
        {"name": "int8_perchannel_builtin_t1_b1", "model": "models/mobilenet_v2_int8.tflite", "threads": 1, "batch": 1, "samples": 100, "delegates": False},
        {"name": "fp32_xnnpack_t1_b4", "model": "models/mobilenet_v2_fp32.tflite", "threads": 1, "batch": 4, "samples": 500, "delegates": True},
        {"name": "int8_perchannel_xnnpack_t1_b4", "model": "models/mobilenet_v2_int8.tflite", "threads": 1, "batch": 4, "samples": 500, "delegates": True},
    ]
    per_tensor = Path("models/mobilenet_v2_int8_per_tensor.tflite")
    if per_tensor.is_file():
        configurations.append({"name": "int8_pertensor_xnnpack_t1_b1", "model": per_tensor.as_posix(), "threads": 1, "batch": 1, "samples": 500, "delegates": True})

    environment = os.environ.copy()
    environment["TF_ENABLE_ONEDNN_OPTS"] = "0"
    logical_cpus = psutil.cpu_count(logical=True) or 1
    physical_count = psutil.cpu_count(logical=False) or logical_cpus
    stride = max(1, logical_cpus // physical_count)
    physical_cpu_ids = list(range(0, logical_cpus, stride))[:physical_count]
    worker = Path(__file__).with_name("latency_worker.py")
    all_runs = {config["name"]: [] for config in configurations}
    if args.checkpoint.is_file():
        checkpoint = json.loads(args.checkpoint.read_text(encoding="utf-8"))
        if checkpoint.get("replicates") != args.replicates:
            parser.error("Checkpoint replicate count does not match --replicates")
        for name in all_runs:
            all_runs[name] = checkpoint.get("runs", {}).get(name, [])
        print(f"Resuming from {args.checkpoint}", flush=True)
    with tempfile.TemporaryDirectory(prefix="edge-latency-") as directory:
        temporary = Path(directory)
        paired_names = [
            ("fp32_xnnpack_t1_b1", "int8_perchannel_xnnpack_t1_b1"),
            ("fp32_xnnpack_t2_b1", "int8_perchannel_xnnpack_t2_b1"),
            ("fp32_xnnpack_t4_b1", "int8_perchannel_xnnpack_t4_b1"),
            ("fp32_xnnpack_t6_b1", "int8_perchannel_xnnpack_t6_b1"),
            ("fp32_builtin_t1_b1", "int8_perchannel_builtin_t1_b1"),
            ("fp32_xnnpack_t1_b4", "int8_perchannel_xnnpack_t1_b4"),
        ]
        config_by_name = {config["name"]: config for config in configurations}
        standalone_names = [name for name in config_by_name if not any(name in pair for pair in paired_names)]
        for replicate in range(args.replicates):
            pairs = paired_names.copy()
            random.Random(20260914 + replicate).shuffle(pairs)
            execution_names = []
            for pair_index, pair in enumerate(pairs):
                execution_names.extend(pair if (replicate + pair_index) % 2 == 0 else pair[::-1])
            execution_names.extend(standalone_names)
            for name in execution_names:
                if len(all_runs[name]) > replicate:
                    continue
                config = config_by_name[name]
                output = temporary / f"{config['name']}-{replicate}.json"
                command = [
                    sys.executable,
                    str(worker),
                    "--data-dir", str(args.data_dir),
                    "--model", str(config["model"]),
                    "--output", str(output),
                    "--samples", str(config["samples"]),
                    "--warmup", "50",
                    "--threads", str(config["threads"]),
                    "--batch-size", str(config["batch"]),
                    "--seed", str(20260914 + replicate),
                    "--affinity", ",".join(
                        str(value) for value in physical_cpu_ids[: min(config["threads"], physical_count)]
                    ),
                ]
                if not config["delegates"]:
                    command.append("--disable-default-delegates")
                print(f"Replicate {replicate + 1}/{args.replicates}: {config['name']}", flush=True)
                subprocess.run(command, check=True, env=environment)
                all_runs[config["name"]].append(json.loads(output.read_text(encoding="utf-8")))
                args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
                args.checkpoint.write_text(
                    json.dumps({"replicates": args.replicates, "runs": all_runs}, indent=2),
                    encoding="utf-8",
                )

    summaries = {
        name: {"summary": summarize(runs), "runs": runs} for name, runs in all_runs.items()
    }
    comparison_pairs = {
        "xnnpack_t1_b1": ("fp32_xnnpack_t1_b1", "int8_perchannel_xnnpack_t1_b1"),
        "xnnpack_t2_b1": ("fp32_xnnpack_t2_b1", "int8_perchannel_xnnpack_t2_b1"),
        "xnnpack_t4_b1": ("fp32_xnnpack_t4_b1", "int8_perchannel_xnnpack_t4_b1"),
        "xnnpack_t6_b1": ("fp32_xnnpack_t6_b1", "int8_perchannel_xnnpack_t6_b1"),
        "builtin_t1_b1": ("fp32_builtin_t1_b1", "int8_perchannel_builtin_t1_b1"),
        "xnnpack_t1_b4": ("fp32_xnnpack_t1_b4", "int8_perchannel_xnnpack_t1_b4"),
    }
    result = {
        "schema_version": 1,
        "protocol": {
            "independent_process_per_run": True,
            "order": "deterministically shuffled matched blocks; FP32/INT8 order alternates within blocks",
            "preprocessing_timed": False,
            "warmup_invocations_per_run": 50,
            "confidence_interval": (
                "two-sided 95% Student t interval across run means "
                f"(df={args.replicates - 1})"
            ),
            "host_note": args.host_note,
            "logical_cpus": logical_cpus,
            "physical_cpu_ids_used": physical_cpu_ids,
            "power_scheme": subprocess.run(
                ["powercfg", "/getactivescheme"], capture_output=True, text=True, check=False
            ).stdout.strip(),
        },
        "configurations": summaries,
        "matched_speedups": {
            name: compare(all_runs[fp32], all_runs[int8])
            for name, (fp32, int8) in comparison_pairs.items()
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    args.checkpoint.unlink(missing_ok=True)
    print(json.dumps({name: value["summary"] for name, value in result["configurations"].items()}, indent=2))
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
