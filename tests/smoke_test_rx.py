"""Smoke test running RX detector on real benchmark datasets via load_hsi.

This test loads real hyperspectral cubes, applies the global RX baseline,
and computes quantitative evaluation metrics without fabricating any numbers.

Usage:
    python tests/smoke_test_rx.py [dataset_name]
    e.g.
    python tests/smoke_test_rx.py abu-airport-1
    python tests/smoke_test_rx.py san-diego
"""

import sys
from pathlib import Path
import numpy as np

# Ensure project root is first in sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loading.load_hsi import load, list_datasets
from src.models.rx import RXDetector
from src.evaluation.metrics import evaluate_anomaly_map


def run_smoke_test(dataset_name: str = "abu-airport-1"):
    print(f"\n=======================================================")
    print(f" SMOKE TEST: {dataset_name}")
    print(f"=======================================================")

    # 1. Load dataset
    print(f"[1/4] Loading dataset via load_hsi.load('{dataset_name}')...")
    cube, gt, meta = load(dataset_name, as_float=True)
    print(f"      Cube shape: {cube.shape} (dtype: {cube.dtype})")
    print(f"      Ground truth shape: {gt.shape} (positive anomaly pixels: {int(np.sum(gt))})")
    print(f"      Metadata keys: {list(meta.keys())}")

    # 2. Run RX Detector
    print(f"[2/4] Fitting and predicting with Global RXDetector...")
    detector = RXDetector(regularization=1e-6, method="cholesky")
    scores = detector.fit_predict(cube)
    print(f"      Score map generated: shape={scores.shape}, min={scores.min():.4f}, max={scores.max():.4f}")

    # 3. Compute Metrics
    print(f"[3/4] Evaluating detector predictions against ground truth...")
    metrics = evaluate_anomaly_map(gt, scores, target_tpr=0.95)

    print("\n--- Empirical Benchmark Results ---")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"  {k:30s}: {v:.6f}")
        else:
            print(f"  {k:30s}: {v}")

    # 4. Verify Validity
    print(f"\n[4/4] Verifying metric bounds...")
    assert 0.0 <= metrics["auc_roc"] <= 1.0, f"Invalid ROC-AUC: {metrics['auc_roc']}"
    assert 0.0 <= metrics["auc_pr"] <= 1.0, f"Invalid PR-AUC: {metrics['auc_pr']}"
    assert 0.0 <= metrics["f1_max"] <= 1.0, f"Invalid F1-max: {metrics['f1_max']}"
    assert 0.0 <= metrics["fpr_at_95_tpr"] <= 1.0, f"Invalid FPR: {metrics['fpr_at_95_tpr']}"
    print("      All metric invariants satisfied successfully.")
    return metrics


if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else "abu-airport-1"
    run_smoke_test(target)
