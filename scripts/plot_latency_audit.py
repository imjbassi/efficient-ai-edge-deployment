"""Generate the manuscript latency-distribution and speedup figure."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RESULT = ROOT / "results" / "repeated_latency.json"
OUTPUT = ROOT / "paper" / "figures" / "latency_audit.pdf"


def pooled(result: dict, name: str) -> np.ndarray:
    return np.array(
        [
            value
            for run in result["configurations"][name]["runs"]
            for value in run["raw_ms_per_image"]
        ]
    )


def main() -> None:
    result = json.loads(RESULT.read_text(encoding="utf-8"))
    replicates = result["configurations"]["fp32_xnnpack_t1_b1"]["summary"]["replicates"]
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 8,
        "axes.labelsize": 8,
        "legend.fontsize": 7,
    })
    figure, axes = plt.subplots(1, 2, figsize=(7.05, 2.35), constrained_layout=True)

    for name, label, color in (
        ("fp32_xnnpack_t1_b1", "FP32", "#255F85"),
        ("int8_perchannel_xnnpack_t1_b1", "INT8 per-channel", "#D1495B"),
    ):
        values = np.sort(pooled(result, name))
        ecdf = np.arange(1, len(values) + 1) / len(values)
        axes[0].step(values, ecdf, where="post", label=label, color=color, linewidth=1.4)
    axes[0].set_xlabel("Latency per image (ms)")
    axes[0].set_ylabel("Empirical cumulative probability")
    axes[0].set_xlim(0, 20)
    axes[0].set_ylim(0, 1.01)
    axes[0].grid(alpha=0.25, linewidth=0.5)
    axes[0].legend(loc="lower right", frameon=False)
    axes[0].set_title(f"(a) One-thread XNNPACK, {replicates} × 500 calls")

    names = ["xnnpack_t1_b1", "xnnpack_t2_b1", "xnnpack_t4_b1", "xnnpack_t6_b1", "builtin_t1_b1"]
    labels = ["XNN\n1 th.", "XNN\n2 th.", "XNN\n4 th.", "XNN\n6 th.", "Built-in\n1 th."]
    entries = [result["matched_speedups"][name] for name in names]
    centers = np.array([entry["geometric_mean_speedup"] for entry in entries])
    lower = centers - np.array([entry["paired_log_t_95"][0] for entry in entries])
    upper = np.array([entry["paired_log_t_95"][1] for entry in entries]) - centers
    x = np.arange(len(entries))
    colors = ["#D1495B" if value > 1 else "#6C757D" for value in centers]
    for index, color in enumerate(colors):
        axes[1].errorbar(
            x[index],
            centers[index],
            yerr=[[lower[index]], [upper[index]]],
            fmt="none",
            ecolor=color,
            capsize=3,
            linewidth=1.1,
        )
    axes[1].scatter(x, centers, c=colors, s=24, zorder=3)
    axes[1].axhline(1, color="black", linestyle="--", linewidth=0.8)
    axes[1].set_xticks(x, labels)
    axes[1].set_ylabel("FP32 / INT8 latency (×)")
    axes[1].set_ylim(0.35, 1.7)
    axes[1].grid(axis="y", alpha=0.25, linewidth=0.5)
    axes[1].set_title("(b) Paired geometric speedup with 95% CI")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUTPUT, bbox_inches="tight")
    figure.savefig(OUTPUT.with_suffix(".png"), dpi=220, bbox_inches="tight")
    print(OUTPUT)


if __name__ == "__main__":
    main()
