"""Benchmark every method variant on every bundled dataset (Task 20: model comparison + ablation).

Run from the repo root:
    python experiments/run_benchmark.py                 # the 16 benchmark datasets
                                                        # (camouflage sets: run_camouflage_eval.py)
    python experiments/run_benchmark.py --quick         # 3 datasets, for a fast check

Writes:
    experiments/results/benchmark.csv          one row per (dataset, method)
    experiments/results/benchmark_summary.csv  mean/std ROC-AUC and AP per method
    outputs/figures/benchmark_auc.png          AUC per dataset and method
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from src.data_loading.load_hsi import list_datasets  # noqa: E402
from src.evaluation.compare import DEFAULT_VARIANTS, benchmark, format_table, save_csv, summarize  # noqa: E402


def plot(rows, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    datasets = list(dict.fromkeys(r["dataset"] for r in rows))
    methods = list(dict.fromkeys(r["method"] for r in rows))
    grid = np.array([[next(r["roc_auc"] for r in rows if r["dataset"] == d and r["method"] == m)
                      for m in methods] for d in datasets], float)
    fig, ax = plt.subplots(figsize=(1.1 * len(methods) + 3, 0.38 * len(datasets) + 1.8))
    im = ax.imshow(grid, cmap="viridis", vmin=np.nanmin(grid), vmax=1, aspect="auto")
    ax.set_xticks(range(len(methods)), methods, rotation=35, ha="right", fontsize=8)
    ax.set_yticks(range(len(datasets)), datasets, fontsize=8)
    for i in range(len(datasets)):
        for j in range(len(methods)):
            v = grid[i, j]
            ax.text(j, i, "-" if np.isnan(v) else f"{v:.3f}", ha="center", va="center", fontsize=7,
                    color="black" if v > np.nanmean(grid) else "white")
    fig.colorbar(im, ax=ax, label="ROC-AUC")
    ax.set_title("ROC-AUC by dataset and method")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="only 3 datasets")
    ap.add_argument("--datasets", nargs="*", help="dataset names (default: all)")
    args = ap.parse_args()
    names = args.datasets or (list_datasets("anomaly") + list_datasets("target"))
    if args.quick:
        names = ["abu-airport-1", "san-diego", "hydice-urban"]

    rows = benchmark(names, DEFAULT_VARIANTS)
    out = ROOT / "experiments" / "results"
    save_csv(rows, out / "benchmark.csv")
    summary = summarize(rows, "roc_auc")
    ap_summary = {s["method"]: s for s in summarize(rows, "ap")}
    for s in summary:
        s["mean_ap"] = ap_summary[s["method"]]["mean_ap"]
    save_csv(summary, out / "benchmark_summary.csv")
    plot(rows, ROOT / "outputs" / "figures" / "benchmark_auc.png")
    print()
    print(format_table(summary, ("method", "mean_roc_auc", "std_roc_auc", "mean_ap", "mean_runtime_s")))
    errs = [r for r in rows if r.get("errors")]
    if errs:
        print(f"\n{len(errs)} runs reported errors, e.g. {errs[0]['dataset']} / {errs[0]['method']}: "
              f"{errs[0]['errors']}")


if __name__ == "__main__":
    main()
