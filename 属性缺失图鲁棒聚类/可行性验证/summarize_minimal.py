"""Aggregate the minimal diagnostic and save tables/figures for review."""
from __future__ import annotations

import csv
import math
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
INPUT = ROOT / "results_minimal.csv"
SUMMARY = ROOT / "summary_minimal.csv"
FIGURE = ROOT / "feasibility_curves.png"

NUMERIC = [
    "acc", "nmi", "ari", "missing_acc", "observed_acc", "rmse",
    "cosine", "cold_start", "mean_degree_missing",
]


def mean_std(values):
    values = [float(v) for v in values if v not in ("", "nan", "NaN")]
    if not values:
        return "", ""
    mean = sum(values) / len(values)
    if len(values) == 1:
        std = 0.0
    else:
        std = math.sqrt(sum((x - mean) ** 2 for x in values) / (len(values) - 1))
    return mean, std


def main():
    with INPUT.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    groups = defaultdict(list)
    for row in rows:
        groups[(float(row["rate"]), row["method"])].append(row)

    out = []
    for (rate, method), items in sorted(groups.items()):
        record = {"rate": rate, "method": method, "n_masks": len(items)}
        for key in NUMERIC:
            avg, std = mean_std([item[key] for item in items])
            record[f"{key}_mean"] = avg
            record[f"{key}_std"] = std
        out.append(record)

    with SUMMARY.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=out[0].keys())
        writer.writeheader()
        writer.writerows(out)

    # Plot metrics with one shared y-axis per metric. Error bars are mask std.
    methods = ["complete", "zero", "mean", "neighbor_mean", "fp"]
    labels = {"complete": "complete", "zero": "zero", "mean": "global mean",
              "neighbor_mean": "neighbor mean", "fp": "FP"}
    colors = {"complete": "black", "zero": "#d95f02", "mean": "#7570b3",
              "neighbor_mean": "#1b9e77", "fp": "#e7298a"}
    rates = sorted({float(row["rate"]) for row in out})
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), constrained_layout=True)
    for ax, metric, title in zip(axes, ["acc", "nmi", "ari"], ["ACC", "NMI", "ARI"]):
        for method in methods:
            values = [next(r for r in out if r["rate"] == rate and r["method"] == method)
                      for rate in rates]
            y = [r[f"{metric}_mean"] for r in values]
            err = [r[f"{metric}_std"] for r in values]
            ax.errorbar(rates, y, yerr=err, marker="o", linewidth=1.8,
                        capsize=3, label=labels[method], color=colors[method])
        ax.set_title(title)
        ax.set_xlabel("missing rate")
        ax.set_xticks(rates)
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("score")
    axes[-1].legend(frameon=False, fontsize=8)
    fig.savefig(FIGURE, dpi=180)
    plt.close(fig)

    print(f"saved {SUMMARY}")
    print(f"saved {FIGURE}")
    print("\nMean metrics by missing rate and method:")
    for row in out:
        print(f"rate={row['rate']:.1f} method={row['method']:<13} "
              f"ACC={row['acc_mean']:.4f} NMI={row['nmi_mean']:.4f} "
              f"ARI={row['ari_mean']:.4f} RMSE={row['rmse_mean']:.4f} "
              f"cold={row['cold_start_mean']:.1f}")


if __name__ == "__main__":
    main()
