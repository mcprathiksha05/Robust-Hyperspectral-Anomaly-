"""Evaluate detectors on the camouflage datasets (build them first with build_camouflage_datasets.py).

    python experiments/run_camouflage_eval.py

Writes experiments/results/camouflage_eval.csv and outputs/figures/camouflage_abundance.png.

Two analyses:
1. Synthetic implants: ROC-AUC vs target abundance (how well camouflaged the target is).
2. MUUFL flights: ROC-AUC for camouflage cloths in the open, in shadow, and under trees.
   Pixels within 1 pixel of a target are left out of the background ('halo'), because
   sub-pixel cloths spill into neighbouring pixels and GPS positions are ~1 m accurate.
Note: with only 20 implanted targets, AUC differences below ~0.07 are within chance.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
from sklearn.metrics import average_precision_score, roc_auc_score  # noqa: E402

from src.data_loading.camouflage import muufl_target_masks  # noqa: E402
from src.data_loading.load_hsi import list_datasets, load  # noqa: E402
from src.evaluation.compare import save_csv  # noqa: E402
from src.pipeline import PipelineConfig, run_pipeline  # noqa: E402

METHODS = {
    "RX baseline": dict(detectors=("rx",), pca_components=None, rx_robust=False),
    "Robust RX + PCA": dict(detectors=("rx",)),
    "Spatial residual": dict(detectors=("spatial",)),
    "Fusion RX+spatial": dict(detectors=("rx", "spatial")),
}
OCCLUSION = {0: "open", 1: "shadow", 2: "under trees"}


def scores_for(cube, gt, overrides):
    cfg = PipelineConfig(profile_memory=False, **overrides)
    return run_pipeline(cube, gt, cfg)["fused"]


def main():
    names = list_datasets("camouflage")
    if not names:
        sys.exit("No camouflage datasets found. Run experiments/build_camouflage_datasets.py first.")
    rows = []
    for name in names:
        cube, gt, meta = load(name)
        valid = meta.get("valid_mask", np.ones(gt.shape, bool)).astype(bool)
        for method, ov in METHODS.items():
            s = scores_for(cube, gt, ov)
            row = {"dataset": name, "method": method}
            if name.startswith("implant"):
                row.update(scene=name.split("-a")[0].removeprefix("implant-"),
                           abundance=float(meta["abundance"]),
                           roc_auc=roc_auc_score(gt.ravel(), s.ravel()),
                           ap=average_precision_score(gt.ravel(), s.ravel()))
                rows.append(row)
            else:
                bg = valid & ~meta["gt_halo"].astype(bool)
                for k, lab in OCCLUSION.items():
                    sel = (meta["target_occlusion"] == k) & (meta["target_size_m"] < 6)
                    tk, _ = muufl_target_masks(meta["target_row"][sel], meta["target_col"][sel],
                                               meta["target_size_m"][sel], cube.shape, one_based=False, halo=0)
                    tk = tk.astype(bool) & valid
                    idx = tk | bg
                    rows.append({**row, "occlusion": lab, "targets": int(sel.sum()),
                                 "roc_auc": roc_auc_score(tk[idx], s[idx]),
                                 "ap": average_precision_score(tk[idx], s[idx])})
        print(f"done {name}", flush=True)
    save_csv(rows, ROOT / "experiments" / "results" / "camouflage_eval.csv")
    plot(rows)
    report(rows)


def plot(rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    imp = [r for r in rows if "abundance" in r]
    if not imp:
        return
    abund = sorted({r["abundance"] for r in imp})
    fig, ax = plt.subplots(figsize=(6.5, 4))
    for method in METHODS:
        mean = [np.mean([r["roc_auc"] for r in imp if r["method"] == method and r["abundance"] == a]) for a in abund]
        ax.plot([100 * a for a in abund], mean, marker="o", label=method)
    ax.axhline(0.5, color="0.6", ls=":", lw=1)
    ax.text(100 * abund[-1], 0.51, "chance", ha="right", va="bottom", fontsize=8, color="0.4")
    ax.set_xlabel("Target abundance in pixel (%)  — lower = better camouflaged")
    ax.set_ylabel("ROC-AUC (mean over 4 scenes)")
    ax.set_title("Detection of implanted camouflaged targets")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    out = ROOT / "outputs" / "figures" / "camouflage_abundance.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130)


def report(rows):
    imp = [r for r in rows if "abundance" in r]
    if imp:
        abund = sorted({r["abundance"] for r in imp})
        print("\nImplants: mean ROC-AUC over scenes")
        print("method".ljust(20) + "".join(f"a={a:<7.2f}" for a in abund))
        for m in METHODS:
            vals = [np.mean([r["roc_auc"] for r in imp if r["method"] == m and r["abundance"] == a]) for a in abund]
            print(m.ljust(20) + "".join(f"{v:<9.3f}" for v in vals))
    mu = [r for r in rows if "occlusion" in r]
    if mu:
        print("\nMUUFL camouflage cloths: mean ROC-AUC over 3 flights")
        print("method".ljust(20) + "".join(f"{o:<13}" for o in OCCLUSION.values()))
        for m in METHODS:
            vals = [np.mean([r["roc_auc"] for r in mu if r["method"] == m and r["occlusion"] == o])
                    for o in OCCLUSION.values()]
            print(m.ljust(20) + "".join(f"{v:<13.3f}" for v in vals))


if __name__ == "__main__":
    main()
