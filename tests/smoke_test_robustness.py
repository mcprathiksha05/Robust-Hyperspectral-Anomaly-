"""Integration smoke test evaluating RX baseline under Stage 2 robustness corruptions.

Executes controlled stress-testing experiments on a real hyperspectral anomaly dataset
(abu-airport-1) and reports empirical metric deltas without fabricating results.
"""

import sys
from pathlib import Path
import numpy as np

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loading.load_hsi import load
from src.models.rx import RXDetector
from src.evaluation.metrics import evaluate_anomaly_map
from src.robustness.noise import add_gaussian_noise, add_stripe_noise, add_dead_pixels
from src.robustness.clutter import inject_spectral_clutter, apply_shadow_attenuation
from src.robustness.bands import drop_random_bands, add_localized_band_noise


def run_robustness_smoke_test(dataset_name: str = "abu-airport-1", seed: int = 42):
    print(f"\n================================================================================")
    print(f" STAGE 2 INTEGRATION SMOKE TEST: ROBUSTNESS PERTURBATION STRESS-TESTING")
    print(f" Dataset: {dataset_name} | Random Seed: {seed}")
    print(f"================================================================================")

    # 1. Load clean dataset
    cube, gt, meta = load(dataset_name, as_float=True)
    h, w, b = cube.shape
    total_px = h * w
    target_px = int(np.sum(gt))
    print(f"[Loaded] Spatial shape: ({h}, {w}), Bands: {b}, Target Anomaly Pixels: {target_px} ({target_px/total_px*100:.2f}%)")

    # 2. Evaluate Baseline
    print(f"\n[Baseline] Running Global RX on clean unperturbed cube...")
    detector = RXDetector(regularization=1e-6)
    clean_scores = detector.fit_predict(cube)
    baseline_metrics = evaluate_anomaly_map(gt, clean_scores, target_tpr=0.95)

    experiments = [
        (
            "Noise: Gaussian (SNR 20 dB)",
            lambda c: add_gaussian_noise(c, snr_db=20.0, seed=seed),
        ),
        (
            "Noise: Gaussian (SNR 10 dB)",
            lambda c: add_gaussian_noise(c, snr_db=10.0, seed=seed),
        ),
        (
            "Noise: Stripe Drift (8% cols)",
            lambda c: add_stripe_noise(c, stripe_density=0.08, band_fraction=0.3, stripe_intensity=0.3, seed=seed),
        ),
        (
            "Noise: Dead Pixels (1% px)",
            lambda c: add_dead_pixels(c, pixel_fraction=0.01, mode="all_bands", seed=seed),
        ),
        (
            "Clutter: Spectral Mix (a=0.35)",
            lambda c: inject_spectral_clutter(c, abundance=0.35, spatial_fraction=0.08, seed=seed),
        ),
        (
            "Clutter: Shadow (gamma=0.3)",
            lambda c: apply_shadow_attenuation(c, attenuation_factor=0.3, region_fraction=0.15, seed=seed),
        ),
        (
            "Bands: Dropout (15% bands)",
            lambda c: drop_random_bands(c, fraction=0.15, mode="zero", seed=seed),
        ),
        (
            "Bands: Local Noise (SNR 8 dB)",
            lambda c: add_localized_band_noise(c, band_fraction=0.15, snr_db=8.0, seed=seed),
        ),
    ]

    results = [("Clean Baseline", baseline_metrics, 0.0, 0.0)]

    for name, transform_fn in experiments:
        corrupted_cube, t_meta = transform_fn(cube)
        corrupted_scores = detector.fit_predict(corrupted_cube)
        m = evaluate_anomaly_map(gt, corrupted_scores, target_tpr=0.95)
        delta_auc = m["auc_roc"] - baseline_metrics["auc_roc"]
        delta_f1 = m["f1_max"] - baseline_metrics["f1_max"]
        results.append((name, m, delta_auc, delta_f1))

    # Print Formatted Results Table
    print("\n" + "=" * 95)
    print(f"{'Condition':<32} {'ROC-AUC':<10} {'PR-AUC':<10} {'F1-max':<10} {'FPR@95%':<10} {'Delta AUC':<10}")
    print("=" * 95)
    for name, m, d_auc, d_f1 in results:
        d_auc_str = f"{d_auc:+.4f}" if name != "Clean Baseline" else "baseline"
        print(
            f"{name:<32} {m['auc_roc']:<10.4f} {m['auc_pr']:<10.4f} {m['f1_max']:<10.4f} {m['fpr_at_95_tpr']:<10.4f} {d_auc_str:<10}"
        )
    print("=" * 95)
    return results


if __name__ == "__main__":
    ds = sys.argv[1] if len(sys.argv) > 1 else "abu-airport-1"
    run_robustness_smoke_test(ds)
