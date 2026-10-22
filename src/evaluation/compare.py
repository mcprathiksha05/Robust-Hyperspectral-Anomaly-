"""Task 20 - Model comparison (and ablation).

    from src.evaluation.compare import compare_methods, benchmark, DEFAULT_VARIANTS, save_csv
    rows = compare_methods(cube, gt)                     # one dataset, every variant
    rows = benchmark(["abu-airport-1", "san-diego"])     # many datasets
    save_csv(rows, "experiments/results/benchmark.csv")

Every variant is a set of PipelineConfig overrides, so all methods share the
same preprocessing, thresholding rule and metrics - a fair comparison.
Variants that switch one component on or off at a time double as the
ablation study (e.g. 'RX + PCA' vs 'RX baseline' isolates PCA).
"""
from __future__ import annotations

import csv
from dataclasses import replace
from pathlib import Path

import numpy as np

from src.pipeline import PipelineConfig, run_pipeline

DEFAULT_VARIANTS: dict[str, dict] = {
    "RX baseline":        dict(detectors=("rx",), pca_components=None, rx_robust=False),
    "RX + PCA":           dict(detectors=("rx",), rx_robust=False),
    "Robust RX + PCA":    dict(detectors=("rx",)),
    "Spatial residual":   dict(detectors=("spatial",)),
    "Local RX":           dict(detectors=("local_rx",)),
    "Autoencoder":        dict(detectors=("autoencoder",)),
    "Fusion RX+spatial":  dict(detectors=("rx", "spatial")),
    "Fusion RX+spatial+AE": dict(detectors=("rx", "spatial", "autoencoder")),
}

SUMMARY_KEYS = ("roc_auc", "ap", "pd_at_far_0.01", "precision", "recall", "f1", "fpr",
                "false_alarms_per_megapixel")


def compare_methods(cube, gt, variants: dict | None = None, base: PipelineConfig | None = None,
                    dataset: str = "", scene_labels=None) -> list[dict]:
    """Run each variant on one image; returns one row per variant (metrics + runtime + memory)."""
    variants = variants or DEFAULT_VARIANTS
    base = base or PipelineConfig()
    rows = []
    for name, overrides in variants.items():
        cfg = replace(base, **overrides)
        row = {"dataset": dataset, "method": name}
        try:
            r = run_pipeline(cube, gt, cfg, scene_labels=scene_labels)
            m = r["metrics"].get("fused", {})
            row.update({k: m.get(k, np.nan) for k in SUMMARY_KEYS})
            row.update(runtime_s=r["timings"]["total"], peak_memory_mb=r.get("peak_memory_mb", np.nan),
                       detected_pixels=r["stats"]["detected_pixels"],
                       errors="; ".join(f"{k}: {v}" for k, v in r["errors"].items()))
        except Exception as exc:  # keep the comparison going if one variant breaks
            row.update({k: np.nan for k in SUMMARY_KEYS}, runtime_s=np.nan, peak_memory_mb=np.nan,
                       detected_pixels=0, errors=str(exc))
        rows.append(row)
    return rows


def benchmark(datasets, variants: dict | None = None, base: PipelineConfig | None = None,
              verbose: bool = True) -> list[dict]:
    """compare_methods over several bundled datasets."""
    from src.data_loading.load_hsi import load
    rows = []
    for name in datasets:
        cube, gt, meta = load(name)
        labels = meta.get("scene_labels")
        names = meta.get("scene_label_names")
        new = compare_methods(cube, gt, variants, base, dataset=name, scene_labels=labels)
        rows += new
        if verbose:
            best = max(new, key=lambda r: np.nan_to_num(r["roc_auc"]))
            print(f"{name:26s} best AUC {best['roc_auc']:.4f} ({best['method']})", flush=True)
    return rows


def summarize(rows: list[dict], key: str = "roc_auc") -> list[dict]:
    """Mean and standard deviation of ``key`` (and runtime) per method across datasets."""
    out = []
    for method in dict.fromkeys(r["method"] for r in rows):
        vals = np.array([r[key] for r in rows if r["method"] == method], float)
        rt = np.array([r["runtime_s"] for r in rows if r["method"] == method], float)
        out.append({"method": method, f"mean_{key}": np.nanmean(vals), f"std_{key}": np.nanstd(vals),
                    "datasets": int(np.isfinite(vals).sum()), "mean_runtime_s": np.nanmean(rt)})
    return out


def save_csv(rows: list[dict], path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.6g}" if isinstance(v, float) else v) for k, v in r.items()})
    return path


def format_table(rows: list[dict], keys=("method", "roc_auc", "ap", "f1", "fpr", "runtime_s")) -> str:
    """Plain-text table for printing in a terminal or notebook."""
    def fmt(v):
        return f"{v:.4f}" if isinstance(v, float) else str(v)
    widths = [max(len(k), *(len(fmt(r.get(k, ""))) for r in rows)) for k in keys]
    lines = ["  ".join(k.ljust(w) for k, w in zip(keys, widths))]
    lines += ["  ".join(fmt(r.get(k, "")).ljust(w) for k, w in zip(keys, widths)) for r in rows]
    return "\n".join(lines)
