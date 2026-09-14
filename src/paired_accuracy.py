"""Paired FP32/INT8 accuracy analysis on the same Imagenette images."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from experiment import IMAGENETTE_LABELS, image_paths, make_interpreter, predict, preprocess


def exact_mcnemar(fp32_only: int, int8_only: int) -> float:
    disagreements = fp32_only + int8_only
    if disagreements == 0:
        return 1.0
    smaller = min(fp32_only, int8_only)
    tail = sum(math.comb(disagreements, k) for k in range(smaller + 1)) / 2**disagreements
    return min(1.0, 2 * tail)


def paired_bootstrap(
    fp32_only: int,
    int8_only: int,
    ties: int,
    *,
    seed: int,
    resamples: int,
) -> list[float]:
    total = fp32_only + int8_only + ties
    probabilities = np.array([fp32_only, ties, int8_only], dtype=np.float64) / total
    rng = np.random.default_rng(seed)
    counts = rng.multinomial(total, probabilities, size=resamples)
    differences = (counts[:, 2] - counts[:, 0]) / total
    return [float(value) for value in np.percentile(differences, [2.5, 97.5])]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--fp32-model", type=Path, default=Path("models/mobilenet_v2_fp32.tflite"))
    parser.add_argument("--int8-model", type=Path, default=Path("models/mobilenet_v2_int8.tflite"))
    parser.add_argument("--output", type=Path, default=Path("results/paired_accuracy.json"))
    parser.add_argument("--bootstrap-resamples", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=20260914)
    args = parser.parse_args()

    paths = image_paths(args.data_dir / "val")
    fp32 = make_interpreter(args.fp32_model, 1)
    int8 = make_interpreter(args.int8_model, 1)
    rows: list[dict] = []
    counts = {
        "top1": {"both_correct": 0, "fp32_only": 0, "int8_only": 0, "both_wrong": 0},
        "top5": {"both_correct": 0, "fp32_only": 0, "int8_only": 0, "both_wrong": 0},
    }

    for index, path in enumerate(paths, start=1):
        label = IMAGENETTE_LABELS[path.parent.name]
        image = preprocess(path)
        fp32_ranking = np.argsort(predict(fp32, image))[-5:][::-1]
        int8_ranking = np.argsort(predict(int8, image))[-5:][::-1]
        row = {
            "path": path.relative_to(args.data_dir).as_posix(),
            "label": label,
            "fp32_top1": int(fp32_ranking[0]),
            "int8_top1": int(int8_ranking[0]),
            "fp32_top1_correct": bool(fp32_ranking[0] == label),
            "int8_top1_correct": bool(int8_ranking[0] == label),
            "fp32_top5_correct": bool(label in fp32_ranking),
            "int8_top5_correct": bool(label in int8_ranking),
        }
        rows.append(row)
        for metric in ("top1", "top5"):
            fp_correct = row[f"fp32_{metric}_correct"]
            int_correct = row[f"int8_{metric}_correct"]
            if fp_correct and int_correct:
                category = "both_correct"
            elif fp_correct:
                category = "fp32_only"
            elif int_correct:
                category = "int8_only"
            else:
                category = "both_wrong"
            counts[metric][category] += 1
        if index % 500 == 0:
            print(f"Paired evaluation {index}/{len(paths)}")

    analyses = {}
    for offset, metric in enumerate(("top1", "top5")):
        table = counts[metric]
        ties = table["both_correct"] + table["both_wrong"]
        delta = (table["int8_only"] - table["fp32_only"]) / len(paths)
        analyses[metric] = {
            "contingency": table,
            "int8_minus_fp32": delta,
            "paired_bootstrap_95": paired_bootstrap(
                table["fp32_only"],
                table["int8_only"],
                ties,
                seed=args.seed + offset,
                resamples=args.bootstrap_resamples,
            ),
            "mcnemar_exact_two_sided_p": exact_mcnemar(
                table["fp32_only"], table["int8_only"]
            ),
        }

    result = {
        "schema_version": 1,
        "samples": len(paths),
        "bootstrap_seed": args.seed,
        "bootstrap_resamples": args.bootstrap_resamples,
        "analysis": analyses,
        "per_image": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(analyses, indent=2))
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
