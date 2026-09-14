"""Verify checked-in models, measurements, and publication assets."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_MODELS = {
    "mobilenet_v2_fp32.tflite": (
        13_986_764,
        "1029ab065d2f8a7225f81b1204f5df9cc947ab6756a0b5f77dd4350dcc2a35b7",
    ),
    "mobilenet_v2_int8.tflite": (
        3_991_688,
        "ea1a077a43691c77c224d473cb30ca637733adc219df994e74973b98e0b66167",
    ),
    "mobilenet_v2_int8_per_tensor.tflite": (
        3_583_560,
        "38ca72c6abce6ad9cef394478a1cb13308e731564ab0ddd8c96591f748e2c7cb",
    ),
}


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def main() -> None:
    for filename, (expected_size, expected_hash) in EXPECTED_MODELS.items():
        model = ROOT / "models" / filename
        require(model.stat().st_size == expected_size, f"Size mismatch for {model}")
        require(sha256(model) == expected_hash, f"SHA-256 mismatch for {model}")

    original = load_json(ROOT / "results" / "windows_cpu_imagenette.json")
    require(original["dataset"]["validation_samples"] == 3925, "Unexpected dataset size")
    for name in ("fp32", "int8"):
        require(original["accuracy"][name]["samples"] == 3925, f"Unexpected {name} accuracy count")
        require(len(original["latency"][name]["raw_ms"]) == 500, f"Unexpected {name} latency count")

    paired = load_json(ROOT / "results" / "paired_accuracy.json")
    require(paired["samples"] == 3925, "Unexpected paired sample count")
    top1 = paired["analysis"]["top1"]
    require(top1["contingency"] == {
        "both_correct": 2494,
        "fp32_only": 125,
        "int8_only": 89,
        "both_wrong": 1217,
    }, "Unexpected top-1 paired contingency")
    require(top1["paired_bootstrap_95"][1] < 0, "Top-1 paired CI no longer excludes zero")

    per_tensor = load_json(ROOT / "results" / "per_tensor_accuracy.json")
    require(per_tensor["samples"] == 3925, "Unexpected per-tensor sample count")
    require(per_tensor["top1_correct"] == 41, "Unexpected per-tensor top-1 result")

    repeated = load_json(ROOT / "results" / "repeated_latency.json")
    require(len(repeated["configurations"]) == 13, "Unexpected latency configuration count")
    for name, config in repeated["configurations"].items():
        require(config["summary"]["replicates"] == 20, f"Unexpected replicate count for {name}")
    primary = repeated["matched_speedups"]["xnnpack_t1_b1"]
    builtin = repeated["matched_speedups"]["builtin_t1_b1"]
    require(primary["paired_log_t_95"][0] > 1, "Primary speedup CI no longer exceeds one")
    require(builtin["paired_log_t_95"][1] < 1, "Built-in-kernel speedup CI no longer falls below one")
    require(
        repeated["matched_speedups"]["xnnpack_t4_b1"]["paired_log_t_95"][0] > 1,
        "Four-thread speedup CI no longer exceeds one",
    )

    diagnostics = load_json(ROOT / "results" / "quantization_diagnostics.json")
    prediction = diagnostics["per_tensor_prediction_distribution"]
    depthwise = diagnostics["per_channel_depthwise_weight_ranges"]
    require(prediction["samples"] == 3925, "Unexpected diagnostics sample count")
    require(prediction["dominant_class_fraction"] > 0.19, "Per-tensor collapse signal changed")
    require(depthwise["layer_count"] == 17, "Unexpected depthwise-layer count")
    require(depthwise["maximum_channel_range_ratio"] > 700, "Depthwise range evidence changed")

    container = load_json(ROOT / "results" / "container_benchmark.json")
    require(container["cold_start"]["samples"] == 20, "Unexpected cold-start count")
    require(container["single_stream_http"]["samples"] == 200, "Unexpected HTTP sample count")
    require(container["runtime"]["read_only_root"] is True, "Container root is not read-only")
    stages = container["server_breakdown"]["stages"]
    for name in (
        "framework_parse",
        "upload_read",
        "decode",
        "resize_crop",
        "normalize_quantize",
        "invoke",
        "serialize",
        "client_transport",
    ):
        require(len(stages[name]["raw_ms"]) == 200, f"Unexpected {name} sample count")

    image_sizes = load_json(ROOT / "results" / "image_size_comparison.json")
    slim = image_sizes["images"]["slim"]
    standard = image_sizes["images"]["standard"]
    require(slim["gzip_docker_save_bytes"] > 0, "Missing compressed slim-image size")
    require(
        slim["uncompressed_bytes"] < standard["uncompressed_bytes"],
        "Slim image is not smaller than standard-base control",
    )

    for asset in (
        ROOT / "paper" / "figures" / "latency_audit.pdf",
        ROOT / "paper" / "figures" / "latency_audit.png",
    ):
        require(asset.stat().st_size > 0, f"Missing or empty figure: {asset}")

    print("Models, measurements, and publication assets verified")


if __name__ == "__main__":
    main()
