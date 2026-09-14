"""Evaluate one TFLite model on the complete Imagenette validation split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from experiment import evaluate, image_paths, make_interpreter


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = evaluate(make_interpreter(args.model, 1), image_paths(args.data_dir / "val"))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "per_class"}, indent=2))


if __name__ == "__main__":
    main()
