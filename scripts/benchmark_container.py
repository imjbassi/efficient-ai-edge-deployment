"""Measure the packaged service through Docker Compose on localhost."""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import statistics
import subprocess
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path

import numpy as np


SERVICE_URL = "http://127.0.0.1:8000"
IMAGE_NAME = "efficient-ai-edge-deployment-edge-inference-service:latest"
ROOT = Path(__file__).resolve().parents[1]


def docker_executable() -> str:
    discovered = shutil.which("docker")
    if discovered:
        return discovered
    local_app_data = os.environ.get("LOCALAPPDATA")
    if local_app_data:
        bundled = Path(local_app_data) / "Programs" / "DockerDesktop" / "resources" / "bin" / "docker.exe"
        if bundled.is_file():
            return str(bundled)
    raise FileNotFoundError("Docker CLI was not found on PATH or in Docker Desktop's default location")


def docker_environment() -> dict[str, str]:
    environment = os.environ.copy()
    docker_directory = str(Path(docker_executable()).parent)
    environment["PATH"] = docker_directory + os.pathsep + environment.get("PATH", "")
    return environment


def docker(*arguments: str, capture: bool = False) -> str:
    result = subprocess.run(
        [docker_executable(), *arguments],
        check=True,
        text=True,
        capture_output=capture,
        env=docker_environment(),
    )
    return result.stdout.strip() if capture else ""


def get_json(path: str) -> dict:
    with urllib.request.urlopen(f"{SERVICE_URL}{path}", timeout=2) as response:
        return json.loads(response.read())


def await_health(timeout: float = 60.0) -> dict:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            return get_json("/health")
        except (OSError, urllib.error.URLError):
            time.sleep(0.05)
    raise TimeoutError("Container did not become healthy")


def multipart_request(image: bytes) -> urllib.request.Request:
    boundary = f"----edge-{uuid.uuid4().hex}"
    prefix = (
        f"--{boundary}\r\n"
        'Content-Disposition: form-data; name="file"; filename="sample.jpg"\r\n'
        "Content-Type: image/jpeg\r\n\r\n"
    ).encode()
    body = prefix + image + f"\r\n--{boundary}--\r\n".encode()
    return urllib.request.Request(
        f"{SERVICE_URL}/predict",
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        method="POST",
    )


def timed_request(image: bytes) -> tuple[float, dict, dict[str, float]]:
    request = multipart_request(image)
    started = time.perf_counter_ns()
    with urllib.request.urlopen(request, timeout=10) as response:
        payload = json.loads(response.read())
        headers = {
            "server_total": float(response.headers["x-edge-server-total-ms"]),
            "serialize": float(response.headers["x-edge-serialize-ms"]),
            "endpoint_pre_serialize": float(
                response.headers["x-edge-endpoint-pre-serialize-ms"]
            ),
        }
    return (time.perf_counter_ns() - started) / 1_000_000, payload, headers


def summarize(values: list[float]) -> dict:
    return {
        "mean_ms": statistics.fmean(values),
        "median_ms": statistics.median(values),
        "p90_ms": float(np.percentile(values, 90)),
        "p99_ms": float(np.percentile(values, 99)),
        "raw_ms": values,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=Path("data/imagenette2-160"))
    parser.add_argument("--output", type=Path, default=Path("results/container_benchmark.json"))
    parser.add_argument("--cold-starts", type=int, default=20)
    parser.add_argument("--requests", type=int, default=200)
    parser.add_argument("--warmup-requests", type=int, default=20)
    args = parser.parse_args()
    image = args.image.read_bytes()

    docker("compose", "config", "--quiet")
    docker("compose", "build")
    docker("compose", "down", "--remove-orphans")
    cold_starts = []
    cold_rss = []
    try:
        for index in range(args.cold_starts):
            started = time.perf_counter_ns()
            docker("compose", "up", "-d", "--force-recreate")
            health = await_health()
            cold_starts.append((time.perf_counter_ns() - started) / 1_000_000)
            cold_rss.append(health["rss_mib"])
            print(f"Cold start {index + 1}/{args.cold_starts}: {cold_starts[-1]:.1f} ms")
            if index + 1 < args.cold_starts:
                docker("compose", "down")

        for _ in range(args.warmup_requests):
            timed_request(image)
        latencies = []
        service_inference = []
        server_stages: dict[str, list[float]] = {}
        prediction = None
        for index in range(args.requests):
            latency, prediction, timing_headers = timed_request(image)
            latencies.append(latency)
            service_inference.append(prediction["inference_ms"])
            request_stages = dict(prediction["timing_ms"])
            request_stages.update(timing_headers)
            accounted = sum(
                request_stages[name]
                for name in (
                    "framework_parse",
                    "upload_read",
                    "decode",
                    "resize_crop",
                    "normalize_quantize",
                    "queue_wait",
                    "invoke",
                    "postprocess",
                    "serialize",
                )
            )
            request_stages["server_unattributed"] = max(
                0.0, request_stages["server_total"] - accounted
            )
            request_stages["client_transport"] = max(
                0.0, latency - request_stages["server_total"]
            )
            for name, value in request_stages.items():
                server_stages.setdefault(name, []).append(value)
            if (index + 1) % 50 == 0:
                print(f"HTTP request {index + 1}/{args.requests}")
        final_health = get_json("/health")
        inspect = json.loads(docker("image", "inspect", IMAGE_NAME, capture=True))[0]
        host_config = json.loads(
            docker("inspect", "edge_inference_service", capture=True)
        )[0]["HostConfig"]
        runtime_user = docker("exec", "edge_inference_service", "id", capture=True)
        result = {
            "schema_version": 2,
            "environment": {
                "host_platform": platform.platform(),
                "docker_client": docker("version", "--format", "{{.Client.Version}}", capture=True),
                "docker_server": docker("version", "--format", "{{.Server.Version}}", capture=True),
                "compose": docker("compose", "version", "--short", capture=True),
            },
            "image": {
                "name": IMAGE_NAME,
                "id": inspect["Id"],
                "uncompressed_bytes": inspect["Size"],
            },
            "cold_start": {
                "definition": "host time from compose up invocation to first successful HTTP health response",
                "samples": args.cold_starts,
                "raw_ms": cold_starts,
                "median_ms": statistics.median(cold_starts),
                "min_ms": min(cold_starts),
                "max_ms": max(cold_starts),
                "rss_mib_after_ready": cold_rss,
            },
            "single_stream_http": {
                "endpoint": "/predict over localhost",
                "warmup_requests": args.warmup_requests,
                "samples": args.requests,
                "mean_ms": statistics.fmean(latencies),
                "median_ms": statistics.median(latencies),
                "p90_ms": float(np.percentile(latencies, 90)),
                "p99_ms": float(np.percentile(latencies, 99)),
                "raw_ms": latencies,
                "service_reported_inference_ms": service_inference,
                "last_prediction": prediction,
            },
            "server_breakdown": {
                "definition": (
                    "server-side stages are instrumented with perf_counter_ns; "
                    "client_transport is client wall time minus server_total"
                ),
                "stages": {
                    name: summarize(values) for name, values in server_stages.items()
                },
            },
            "runtime": {
                "rss_mib_after_requests": final_health["rss_mib"],
                "user": runtime_user,
                "read_only_root": host_config["ReadonlyRootfs"],
                "memory_limit_bytes": host_config["Memory"],
                "nano_cpus": host_config["NanoCpus"],
            },
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(json.dumps({
            "image_mib": result["image"]["uncompressed_bytes"] / 2**20,
            "cold_start_median_ms": result["cold_start"]["median_ms"],
            "http_median_ms": result["single_stream_http"]["median_ms"],
            "http_p99_ms": result["single_stream_http"]["p99_ms"],
            "rss_mib": result["runtime"]["rss_mib_after_requests"],
        }, indent=2))
    finally:
        docker("compose", "down", "--remove-orphans")

    native_output = args.output.parent / "container_native_latency.json"
    docker(
        "run",
        "--rm",
        "--cpus",
        "2",
        "--memory",
        "512m",
        "--read-only",
        "--tmpfs",
        "/tmp:size=64m",
        "-v",
        f"{(ROOT / 'scripts' / 'benchmark_container_native.py').resolve()}:/app/benchmark_container_native.py:ro",
        "-v",
        f"{args.data_dir.resolve()}:/data:ro",
        "-v",
        f"{(ROOT / 'models' / 'mobilenet_v2_fp32.tflite').resolve()}:/models/mobilenet_v2_fp32.tflite:ro",
        "-v",
        f"{native_output.parent.resolve()}:/output",
        IMAGE_NAME,
        "python",
        "/app/benchmark_container_native.py",
        "--data-dir",
        "/data",
        "--model",
        "/app/models/mobilenet_v2_int8.tflite",
        "--delegate-probe-model",
        "/models/mobilenet_v2_fp32.tflite",
        "--threads",
        "1",
        "--samples",
        "500",
        "--warmup",
        "50",
        "--output",
        f"/output/{native_output.name}",
    )


if __name__ == "__main__":
    main()
