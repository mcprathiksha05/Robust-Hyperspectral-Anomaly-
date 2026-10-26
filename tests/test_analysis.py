"""Automated unit tests for Stage 3 Analysis Protocols.

Covers:
    - Feature 4: Ablation studies (ablation.py)
    - Feature 5: Cross-scene testing & band compatibility (cross_scene.py)
    - Feature 6: Sensitivity analysis & parameter sweeps (sensitivity.py)
    - Feature 7: Stability testing & failure auditing (stability.py)
    - Feature 8: Uncertainty estimation & sensitivity maps (uncertainty.py)
"""

import pytest
import numpy as np

from src.models.rx import RXDetector
from src.analysis.ablation import (
    apply_normalization,
    apply_band_subsampling,
    run_ablation_study,
    build_default_ablation_suite,
)
from src.analysis.cross_scene import (
    evaluate_within_scene_benchmark,
    evaluate_cross_scene_transfer,
    evaluate_cross_scene_matrix,
)
from src.analysis.sensitivity import (
    run_parameter_sweep,
    sweep_noise_snr,
    sweep_band_dropout,
    sweep_clutter_abundance,
    sweep_detector_regularization,
    sweep_decision_thresholds,
)
from src.analysis.stability import (
    compute_stability_summary,
    evaluate_corruption_stability,
    evaluate_detector_stability,
)
from src.analysis.uncertainty import (
    estimate_perturbation_uncertainty,
)
from src.robustness.noise import add_gaussian_noise


@pytest.fixture
def synthetic_cube():
    """Synthetic HSI cube of shape (20, 20, 10)."""
    np.random.seed(42)
    return np.random.uniform(50.0, 300.0, size=(20, 20, 10)).astype(np.float32)


@pytest.fixture
def synthetic_gt():
    """Synthetic binary ground truth of shape (20, 20)."""
    gt = np.zeros((20, 20), dtype=np.int32)
    gt[8:12, 8:12] = 1
    return gt


# ============================================================================
# 1. Feature 4: Ablation Studies Tests
# ============================================================================


def test_normalization_methods(synthetic_cube):
    cube_copy = synthetic_cube.copy()

    # Raw / None
    c_raw = apply_normalization(synthetic_cube, "raw")
    assert np.array_equal(c_raw, synthetic_cube)

    # Min-Max: each band in [0.0, 1.0]
    c_mm = apply_normalization(synthetic_cube, "minmax")
    assert c_mm.shape == synthetic_cube.shape
    assert np.all(c_mm >= -1e-6) and np.all(c_mm <= 1.0 + 1e-6)

    # Z-Score: zero mean, unit std
    c_zs = apply_normalization(synthetic_cube, "zscore")
    assert c_zs.shape == synthetic_cube.shape
    band_means = np.mean(c_zs, axis=(0, 1))
    band_stds = np.std(c_zs, axis=(0, 1))
    assert np.all(np.abs(band_means) < 1e-5)
    assert np.all(np.abs(band_stds - 1.0) < 1e-4)

    # L2-Norm: pixel vector unit norm
    c_l2 = apply_normalization(synthetic_cube, "l2")
    pixel_norms = np.linalg.norm(c_l2, axis=2)
    assert np.all(np.abs(pixel_norms - 1.0) < 1e-5)

    # Original cube preserved
    assert np.array_equal(synthetic_cube, cube_copy)

    # Invalid method raises ValueError
    with pytest.raises(ValueError, match="Unsupported normalization method"):
        apply_normalization(synthetic_cube, "invalid_norm")


def test_band_subsampling(synthetic_cube):
    c_sub, indices = apply_band_subsampling(synthetic_cube, retained_fraction=0.5)
    assert c_sub.shape == (20, 20, 5)
    assert len(indices) == 5

    with pytest.raises(ValueError, match="retained_fraction must be in"):
        apply_band_subsampling(synthetic_cube, retained_fraction=1.5)


def test_run_ablation_study(synthetic_cube, synthetic_gt):
    cube_copy = synthetic_cube.copy()
    gt_copy = synthetic_gt.copy()

    configs = [
        {"name": "Raw Baseline", "normalization": "raw", "retained_band_fraction": 1.0},
        {"name": "MinMax Normalized", "normalization": "minmax", "retained_band_fraction": 1.0},
        {"name": "Half Bands Retained", "normalization": "raw", "retained_band_fraction": 0.5},
    ]

    results = run_ablation_study(synthetic_cube, synthetic_gt, configs, detector_cls=RXDetector)

    assert results["total_configurations_tested"] == 3
    assert len(results["configurations"]) == 3
    assert "best_configuration_by_auc" in results
    assert "best_configuration_by_f1" in results

    # Data preservation
    assert np.array_equal(synthetic_cube, cube_copy)
    assert np.array_equal(synthetic_gt, gt_copy)

    # Empty configs guard
    with pytest.raises(ValueError, match="must not be empty"):
        run_ablation_study(synthetic_cube, synthetic_gt, [])


# ============================================================================
# 2. Feature 5: Cross-Scene Testing Tests
# ============================================================================


def test_within_scene_benchmark_real_dataset():
    # Run on one real dataset
    report = evaluate_within_scene_benchmark(dataset_names=["abu-airport-1"])
    assert report["total_scenes_evaluated"] == 1
    assert "mean_auc_roc" in report["aggregate_summary"]
    assert report["per_scene_results"][0]["scene_name"] == "abu-airport-1"
    assert report["per_scene_results"][0]["transfer_type"] == "within_scene"


def test_cross_scene_transfer_compatible_and_incompatible():
    # abu-airport-1 and abu-airport-2 both have 205 bands
    # abu-airport-4 has 191 bands (incompatible)
    report_skip = evaluate_cross_scene_transfer(
        source_scene="abu-airport-1",
        target_scenes=["abu-airport-1", "abu-airport-2", "abu-airport-4"],
        band_mismatch_strategy="skip",
    )

    targets = report_skip["target_results"]
    assert len(targets) == 3

    # abu-airport-1 (same scene -> within_scene)
    assert targets[0]["transfer_type"] == "within_scene"
    assert targets[0]["status"] == "success"

    # abu-airport-2 (compatible 205 bands -> cross_scene success)
    assert targets[1]["transfer_type"] == "cross_scene"
    assert targets[1]["status"] == "success"

    # abu-airport-4 (191 bands != 205 -> skipped_band_mismatch)
    assert targets[2]["transfer_type"] == "cross_scene"
    assert targets[2]["status"] == "skipped_band_mismatch"

    # Test truncate_min strategy
    report_trunc = evaluate_cross_scene_transfer(
        source_scene="abu-airport-1",
        target_scenes=["abu-airport-4"],
        band_mismatch_strategy="truncate_min",
    )
    assert report_trunc["target_results"][0]["status"] == "success"


def test_cross_scene_matrix():
    # abu-airport-1, abu-airport-2, abu-airport-3 all share 205 bands
    matrix_report = evaluate_cross_scene_matrix(["abu-airport-1", "abu-airport-2"])
    assert len(matrix_report["auc_matrix"]) == 2
    assert len(matrix_report["auc_matrix"][0]) == 2
    assert matrix_report["bands"] == 205

    # Incompatible bands in matrix raises ValueError
    with pytest.raises(ValueError, match="identical band counts"):
        evaluate_cross_scene_matrix(["abu-airport-1", "abu-airport-4"])


# ============================================================================
# 3. Feature 6: Sensitivity Analysis Tests
# ============================================================================


def test_sweep_noise_snr(synthetic_cube, synthetic_gt):
    report = sweep_noise_snr(
        synthetic_cube, synthetic_gt, snr_values=[30.0, 15.0], seed=42
    )
    assert report["param_name"] == "snr_db"
    assert report["param_type"] == "corruption"
    assert len(report["results"]) == 2
    assert len(report["table"]) == 2


def test_sweep_decision_thresholds(synthetic_gt):
    np.random.seed(42)
    scores = np.random.uniform(0.0, 100.0, size=(20, 20))
    report = sweep_decision_thresholds(scores, synthetic_gt, n_points=10)

    assert report["param_name"] == "threshold"
    assert len(report["tradeoff_curve"]) > 0
    row0 = report["tradeoff_curve"][0]
    assert "tpr" in row0
    assert "fpr" in row0
    assert "precision" in row0
    assert "f1" in row0
    assert "false_alarms_per_mp" in row0


# ============================================================================
# 4. Feature 7: Stability Testing Tests
# ============================================================================


def test_compute_stability_summary():
    # 3 success, 1 failure
    records = [
        {"status": "success", "metrics": {"auc_roc": 0.85, "f1_max": 0.40, "fpr_at_95_tpr": 0.10, "false_alarms_per_mp": 1000.0, "auc_pr": 0.30}},
        {"status": "success", "metrics": {"auc_roc": 0.87, "f1_max": 0.42, "fpr_at_95_tpr": 0.12, "false_alarms_per_mp": 1200.0, "auc_pr": 0.32}},
        {"status": "success", "metrics": {"auc_roc": 0.86, "f1_max": 0.41, "fpr_at_95_tpr": 0.11, "false_alarms_per_mp": 1100.0, "auc_pr": 0.31}},
        {"status": "failed", "metrics": None, "error": "Simulated error"},
    ]

    summary = compute_stability_summary(records)
    assert summary["total_trials"] == 4
    assert summary["valid_trials"] == 3
    assert summary["failed_trials"] == 1
    assert summary["failure_rate"] == 0.25
    assert summary["mean_auc_roc"] == pytest.approx(0.86)
    assert summary["std_auc_roc"] == pytest.approx(np.std([0.85, 0.87, 0.86]))


def test_evaluate_corruption_stability(synthetic_cube, synthetic_gt):
    cube_copy = synthetic_cube.copy()

    def noisy_transform(c, seed):
        return add_gaussian_noise(c, snr_db=30.0, seed=seed)

    report = evaluate_corruption_stability(
        synthetic_cube,
        synthetic_gt,
        transform_fn=noisy_transform,
        n_runs=4,
        base_seed=10,
    )

    assert report["n_runs"] == 4
    assert report["summary"]["valid_trials"] == 4
    assert report["summary"]["failed_trials"] == 0
    assert np.isfinite(report["summary"]["mean_auc_roc"])
    assert np.array_equal(synthetic_cube, cube_copy)

    with pytest.raises(ValueError, match="n_runs must be positive"):
        evaluate_corruption_stability(synthetic_cube, synthetic_gt, noisy_transform, n_runs=0)


# ============================================================================
# 5. Feature 8: Uncertainty Estimation Tests
# ============================================================================


def test_uncertainty_estimation_shapes_and_invariants(synthetic_cube):
    cube_copy = synthetic_cube.copy()
    h, w, b = synthetic_cube.shape

    report = estimate_perturbation_uncertainty(
        synthetic_cube, detector=RXDetector(), n_perturbations=4, snr_db=30.0, base_seed=42
    )

    # Check spatial shapes
    assert report["baseline_scores"].shape == (h, w)
    assert report["mean_scores"].shape == (h, w)
    assert report["variance_map"].shape == (h, w)
    assert report["std_map"].shape == (h, w)
    assert report["relative_uncertainty_map"].shape == (h, w)
    assert report["spread_90_map"].shape == (h, w)

    # Non-negativity and finite checks
    assert np.all(np.isfinite(report["std_map"]))
    assert np.all(report["std_map"] >= 0.0)
    assert np.all(np.isfinite(report["relative_uncertainty_map"]))
    assert np.all(report["relative_uncertainty_map"] >= 0.0)

    # Disclaimer check
    assert "Disclaimer" in report["disclaimer"]

    # Original cube preserved
    assert np.array_equal(synthetic_cube, cube_copy)

    # Input validation: n < 2 raises ValueError
    with pytest.raises(ValueError, match="n_perturbations must be at least 2"):
        estimate_perturbation_uncertainty(synthetic_cube, n_perturbations=1)
