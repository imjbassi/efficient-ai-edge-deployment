"""Compare compressed and uncompressed production-image sizes across base images."""

from __future__ import annotations

import argparse
import gzip
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SLIM_BASE = (
    "python:3.10-slim-bookworm@sha256:"
    "68d914ec641a0b69267ce65184d000a2bc3a9ee2590ab702b82250ab2385735a"
)
DEFAULT_STANDARD_BASE = (
    "python:3.10-bookworm@sha256:"
    "94c362db08c5b38857943d31b10558ff1856e918605c474d205d72a534929d4e"
)


def docker_executable() -> str:
    discovered = shutil.which("docker")
    if discovered:
        return discovered
    local_app_data = os.environ.get("LOCALAPPDATA")
    if local_app_data:
        bundled = Path(local_app_data) / "Programs" / "DockerDesktop" / "resources" / "bin" / "docker.exe"
        if bundled.is_file():
            return str(bundled)
    raise FileNotFoundError("Docker CLI was not found")


def docker_environment() -> dict[str, str]:
    environment = os.environ.copy()
    docker_directory = str(Path(docker_executable()).parent)
    environment["PATH"] = docker_directory + os.pathsep + environment.get("PATH", "")
    return environment


def docker(*arguments: str, capture: bool = False) -> str:
    result = subprocess.run(
        [docker_executable(), *arguments],
        cwd=ROOT,
        check=True,
        text=True,
        capture_output=capture,
        env=docker_environment(),
    )
    return result.stdout.strip() if capture else ""


def compressed_archive_size(image: str, destination: Path) -> int:
    with destination.open("wb") as raw, gzip.GzipFile(
        filename="", mode="wb", fileobj=raw, compresslevel=9, mtime=0
    ) as compressed:
        process = subprocess.Popen(
            [docker_executable(), "image", "save", image],
            cwd=ROOT,
            stdout=subprocess.PIPE,
            env=docker_environment(),
        )
        assert process.stdout is not None
        shutil.copyfileobj(process.stdout, compressed, length=1024 * 1024)
        process.stdout.close()
        return_code = process.wait()
        if return_code:
            raise subprocess.CalledProcessError(return_code, process.args)
    return destination.stat().st_size


def inspect_image(name: str, base: str, archive: Path) -> dict:
    inspect = json.loads(docker("image", "inspect", name, capture=True))[0]
    return {
        "name": name,
        "id": inspect["Id"],
        "base_image": base,
        "uncompressed_bytes": inspect["Size"],
        "gzip_docker_save_bytes": compressed_archive_size(name, archive),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("results/image_size_comparison.json"))
    parser.add_argument("--slim-base", default=DEFAULT_SLIM_BASE)
    parser.add_argument("--standard-base", default=DEFAULT_STANDARD_BASE)
    args = parser.parse_args()

    variants = (
        ("edge-inference-slim:measurement", args.slim_base),
        ("edge-inference-standard:measurement", args.standard_base),
    )
    for name, base in variants:
        docker("build", "--build-arg", f"BASE_IMAGE={base}", "-t", name, ".")

    with tempfile.TemporaryDirectory(prefix="edge-image-size-") as directory:
        temporary = Path(directory)
        images = {
            "slim": inspect_image(variants[0][0], variants[0][1], temporary / "slim.tar.gz"),
            "standard": inspect_image(
                variants[1][0], variants[1][1], temporary / "standard.tar.gz"
            ),
        }
    images["slim"]["uncompressed_reduction_vs_standard_percent"] = (
        1 - images["slim"]["uncompressed_bytes"] / images["standard"]["uncompressed_bytes"]
    ) * 100
    images["slim"]["compressed_reduction_vs_standard_percent"] = (
        1
        - images["slim"]["gzip_docker_save_bytes"]
        / images["standard"]["gzip_docker_save_bytes"]
    ) * 100
    result = {
        "schema_version": 1,
        "compressed_definition": (
            "byte size of `docker image save` output compressed with deterministic gzip level 9"
        ),
        "images": images,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
