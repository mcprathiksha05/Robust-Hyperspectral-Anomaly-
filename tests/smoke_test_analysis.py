"""Stage 3 Real-Data Validation Script.

Executes real experiments for Features 4, 5, 6, 7, and 8 on local benchmark datasets
(abu-airport-1, abu-airport-2, abu-airport-3, abu-urban-4, abu-airport-4) using the
reference RXDetector and shared evaluation metrics.
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
from src.analysis.ablation import run_ablation_study, build_default_ablation_suite
from src.analysis.cross_scene import evaluate_cross_scene_transfer
from src.analysis.sensitivity import sweep_noise_snr, sweep_detector_regularization
from src.analysis.stability import evaluate_corruption_stability
from src.analysis.uncertainty import estimate_perturbation_uncertainty
from src.robustness.noise import add_gaussian_noise


def run_stage3_real_validation():
    print("=" * 80)
    print(" STAGE 3 REAL-DATA VALIDATION RUNNER")
    print("=" * 80)

    # 1. Feature 4: Ablation Studies on abu-airport-1
    print("\n--- [Experiment 1: Feature 4 - Ablation Study on abu-airport-1] ---")
    c_air1, gt_air1, _ = load("abu-airport-1", as_float=True)
    ablation_configs = build_default_ablation_suite()
    ablation_res = run_ablation_study(c_air1, gt_air1, ablation_configs)

    print(f"{'Configuration':<42} {'ROC-AUC':<10} {'PR-AUC':<10} {'F1-max':<10} {'Time (ms)':<10}")
    print("-" * 80)
    for c in ablation_res["configurations"]:
        m = c["metrics"]
        print(f"{c['name']:<42} {m['auc_roc']:<10.4f} {m['auc_pr']:<10.4f} {m['f1_max']:<10.4f} {c['elapsed_ms']:<10.1f}")
    print(f"\nBest Config by AUC: {ablation_res['best_configuration_by_auc']['name']} ({ablation_res['best_configuration_by_auc']['metrics']['auc_roc']:.4f})")

    # 2. Feature 5: Cross-Scene Transfer
    print("\n--- [Experiment 2: Feature 5 - Cross-Scene Transfer from abu-airport-1] ---")
    transfer_res = evaluate_cross_scene_transfer(
        source_scene="abu-airport-1",
        target_scenes=["abu-airport-1", "abu-airport-2", "abu-airport-3", "abu-urban-4", "abu-airport-4"],
        band_mismatch_strategy="skip",
    )
    print(f"{'Source Scene':<16} {'Target Scene':<16} {'Transfer Type':<14} {'Status':<12} {'ROC-AUC':<10} {'PR-AUC':<10}")
    print("-" * 80)
    for t in transfer_res["target_results"]:
        status = t["status"]
        if status == "success":
            auc_str = f"{t['metrics']['auc_roc']:.4f}"
            pr_str = f"{t['metrics']['auc_pr']:.4f}"
        else:
            auc_str = "N/A"
            pr_str = "N/A"
        print(f"{t['source_scene']:<16} {t['target_scene']:<16} {t['transfer_type']:<14} {status:<12} {auc_str:<10} {pr_str:<10}")

    print(f"\nMean Cross-Scene ROC-AUC (compatible scenes): {transfer_res['cross_scene_summary']['mean_cross_auc_roc']:.4f}")

    # 3. Feature 6: Sensitivity Analysis
    print("\n--- [Experiment 3: Feature 6 - Noise SNR & Regularization Sensitivity] ---")
    snr_sweep = sweep_noise_snr(c_air1, gt_air1, snr_values=[30.0, 25.0, 20.0, 15.0, 10.0])
    print(f"{'Noise SNR (dB)':<16} {'ROC-AUC':<10} {'PR-AUC':<10} {'F1-max':<10} {'FPR @ 95% TPR':<14}")
    print("-" * 65)
    for row in snr_sweep["table"]:
        print(f"{row['snr_db']:<16.1f} {row['auc_roc']:<10.4f} {row['auc_pr']:<10.4f} {row['f1_max']:<10.4f} {row['fpr_at_95_tpr']:<14.4f}")

    # 4. Feature 7: Stability Testing
    print("\n--- [Experiment 4: Feature 7 - Stability under Stochastic Noise (SNR 20 dB)] ---")
    def noise_20db(c, seed):
        return add_gaussian_noise(c, snr_db=20.0, seed=seed)

    stability_res = evaluate_corruption_stability(c_air1, gt_air1, noise_20db, n_runs=5, base_seed=100)
    s = stability_res["summary"]
    print(f"Total Trials: {s['total_trials']} | Valid: {s['valid_trials']} | Failed: {s['failed_trials']}")
    print(f"ROC-AUC: Mean = {s['mean_auc_roc']:.4f} +/- {s['std_auc_roc']:.4f} [Min: {s['min_auc_roc']:.4f}, Max: {s['max_auc_roc']:.4f}]")
    print(f"PR-AUC:  Mean = {s['mean_auc_pr']:.4f} +/- {s['std_auc_pr']:.4f}")
    print(f"F1-max:  Mean = {s['mean_f1_max']:.4f} +/- {s['std_f1_max']:.4f}")

    # 5. Feature 8: Uncertainty Estimation
    print("\n--- [Experiment 5: Feature 8 - Perturbation Uncertainty Mapping] ---")
    uncert_res = estimate_perturbation_uncertainty(c_air1, RXDetector(), n_perturbations=10, snr_db=30.0, base_seed=42)
    print(f"Spatial Map Shape: {uncert_res['std_map'].shape}")
    print(f"Baseline Score Range: min={uncert_res['baseline_scores'].min():.2f}, max={uncert_res['baseline_scores'].max():.2f}")
    print(f"Global Mean Uncertainty (Std): {uncert_res['global_mean_uncertainty']:.4f}")
    print(f"Global Mean Relative Uncertainty (CV): {uncert_res['global_mean_relative_uncertainty']:.4f}")
    print(f"Disclaimer Verified: '{uncert_res['disclaimer'][:65]}...'")

    print("\n" + "=" * 80)
    print(" STAGE 3 REAL-DATA VALIDATION COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    run_stage3_real_validation()
