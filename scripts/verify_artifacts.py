"""Verify that checked-in models and headline result fields agree."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULT_PATH = ROOT / "results" / "windows_cpu_imagenette.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    result = json.loads(RESULT_PATH.read_text(encoding="utf-8"))
    for name in ("fp32", "int8"):
        recorded = result["models"][name]
        model = ROOT / recorded["path"]
        if model.stat().st_size != recorded["bytes"]:
            raise RuntimeError(f"Size mismatch for {model}")
        if sha256(model) != recorded["sha256"]:
            raise RuntimeError(f"SHA-256 mismatch for {model}")

    if result["dataset"]["validation_samples"] != 3925:
        raise RuntimeError("Unexpected validation sample count")
    for name in ("fp32", "int8"):
        if result["accuracy"][name]["samples"] != 3925:
            raise RuntimeError(f"Unexpected {name} accuracy sample count")
        if len(result["latency"][name]["raw_ms"]) != 500:
            raise RuntimeError(f"Unexpected {name} latency sample count")
    print("Model hashes and result artifact verified")


if __name__ == "__main__":
    main()
