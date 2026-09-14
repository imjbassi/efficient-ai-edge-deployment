"""Single entry point for regenerating the paper evidence and compiled PDF."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def run(
    *arguments: str,
    environment: dict[str, str] | None = None,
    cwd: Path = ROOT,
) -> None:
    print("+", " ".join(arguments), flush=True)
    subprocess.run(arguments, cwd=cwd, env=environment, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data/imagenette2-160"))
    parser.add_argument("--image", type=Path)
    parser.add_argument("--measure", action="store_true", help="rerun accuracy and 20-process latency")
    parser.add_argument("--container", action="store_true", help="rerun Docker serving and size benchmarks")
    parser.add_argument("--tectonic", default=shutil.which("tectonic") or "tectonic")
    args = parser.parse_args()
    environment = os.environ.copy()
    environment["TF_ENABLE_ONEDNN_OPTS"] = "0"

    if args.measure:
        run(sys.executable, "src/paired_accuracy.py", "--data-dir", str(args.data_dir), environment=environment)
        run(sys.executable, "src/evaluate_model.py", "--model", "models/mobilenet_v2_int8_per_tensor.tflite", "--data-dir", str(args.data_dir), "--output", "results/per_tensor_accuracy.json", environment=environment)
        run(sys.executable, "scripts/analyze_quantization.py", "--data-dir", str(args.data_dir), environment=environment)
        run(sys.executable, "src/repeated_latency.py", "--data-dir", str(args.data_dir), "--replicates", "20", environment=environment)

    if args.container:
        if args.image is None:
            parser.error("--container requires --image")
        run(
            sys.executable,
            "scripts/benchmark_container.py",
            "--image",
            str(args.image),
            "--data-dir",
            str(args.data_dir),
        )
        run(sys.executable, "scripts/benchmark_image_size.py")

    run(sys.executable, "scripts/plot_latency_audit.py", environment=environment)
    run(sys.executable, "scripts/verify_artifacts.py")
    run(
        args.tectonic,
        "main.tex",
        "--outdir",
        str(ROOT / "output" / "pdf"),
        environment=environment,
        cwd=ROOT / "paper",
    )


if __name__ == "__main__":
    main()
