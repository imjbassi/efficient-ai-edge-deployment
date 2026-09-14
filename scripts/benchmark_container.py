"""Measure the packaged service through Docker Compose on localhost."""

from __future__ import annotations

import argparse
import json
import os
import platform
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


def docker(*arguments: str, capture: bool = False) -> str:
    result = subprocess.run(
        ["docker", *arguments],
        check=True,
        text=True,
        capture_output=capture,
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


def timed_request(image: bytes) -> tuple[float, dict]:
    request = multipart_request(image)
    started = time.perf_counter_ns()
    with urllib.request.urlopen(request, timeout=10) as response:
        payload = json.loads(response.read())
    return (time.perf_counter_ns() - started) / 1_000_000, payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("results/container_benchmark.json"))
    parser.add_argument("--cold-starts", type=int, default=5)
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
        prediction = None
        for index in range(args.requests):
            latency, prediction = timed_request(image)
            latencies.append(latency)
            service_inference.append(prediction["inference_ms"])
            if (index + 1) % 50 == 0:
                print(f"HTTP request {index + 1}/{args.requests}")
        final_health = get_json("/health")
        inspect = json.loads(docker("image", "inspect", IMAGE_NAME, capture=True))[0]
        host_config = json.loads(
            docker("inspect", "edge_inference_service", capture=True)
        )[0]["HostConfig"]
        runtime_user = docker("exec", "edge_inference_service", "id", capture=True)
        result = {
            "schema_version": 1,
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


if __name__ == "__main__":
    main()
