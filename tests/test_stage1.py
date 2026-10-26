"""Automated unit tests for Stage 1: Base detector, RX baseline, and evaluation metrics."""

import pytest
import numpy as np

from src.models.base import BaseDetector
from src.models.rx import RXDetector
from src.evaluation.metrics import (
    sanitize_inputs,
    roc_auc,
    average_precision,
    max_f1_score,
    fpr_at_tpr,
    false_alarms_per_megapixel,
    evaluate_anomaly_map,
)


class DummyDetector(BaseDetector):
    """Minimal concrete detector for testing BaseDetector validation."""

    def fit(self, cube: np.ndarray) -> "DummyDetector":
        self.validate_cube(cube)
        self.is_fitted = True
        return self

    def predict(self, cube: np.ndarray) -> np.ndarray:
        cube_val = self.validate_cube(cube)
        h, w, _ = cube_val.shape
        scores = np.ones((h, w), dtype=np.float64)
        return self.validate_score_map(scores, expected_shape=(h, w))


# ============================================================================
# 1. Base Detector Interface Tests
# ============================================================================


def test_base_detector_input_validation():
    det = DummyDetector()

    # Valid 3D input
    valid_cube = np.random.rand(10, 10, 5).astype(np.float32)
    sanitized = det.validate_cube(valid_cube)
    assert sanitized.shape == (10, 10, 5)

    # Invalid dimensions (2D)
    with pytest.raises(ValueError, match="Cube must be 3-dimensional"):
        det.validate_cube(np.random.rand(10, 10))

    # Invalid dimensions (4D)
    with pytest.raises(ValueError, match="Cube must be 3-dimensional"):
        det.validate_cube(np.random.rand(10, 10, 5, 2))

    # Non-finite values (NaN)
    nan_cube = valid_cube.copy()
    nan_cube[0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="non-finite values"):
        det.validate_cube(nan_cube)

    # Non-finite values (Inf)
    inf_cube = valid_cube.copy()
    inf_cube[0, 0, 0] = np.inf
    with pytest.raises(ValueError, match="non-finite values"):
        det.validate_cube(inf_cube)

    # Non-array input
    with pytest.raises(TypeError):
        det.validate_cube([[1, 2], [3, 4]])


def test_base_detector_output_validation():
    det = DummyDetector()
    expected_shape = (10, 10)

    # Valid 2D scores
    valid_scores = np.zeros(expected_shape, dtype=np.float64)
    out = det.validate_score_map(valid_scores, expected_shape)
    assert out.shape == expected_shape

    # Shape mismatch
    with pytest.raises(ValueError, match="does not match expected spatial dimensions"):
        det.validate_score_map(np.zeros((8, 8)), expected_shape)

    # Non-2D scores
    with pytest.raises(ValueError, match="Score map must be 2-dimensional"):
        det.validate_score_map(np.zeros((10, 10, 1)), expected_shape)

    # Non-finite scores
    nan_scores = valid_scores.copy()
    nan_scores[1, 1] = np.nan
    with pytest.raises(ValueError, match="non-finite values"):
        det.validate_score_map(nan_scores, expected_shape)


# ============================================================================
# 2. Reference RX Detector Tests
# ============================================================================


def test_rx_detector_output_and_ranking():
    np.random.seed(42)
    h, w, b = 25, 25, 12
    # Create background Gaussian data
    cube = np.random.normal(loc=100.0, scale=10.0, size=(h, w, b)).astype(np.float32)

    # Implant a stark spectral anomaly at (12, 12)
    cube[12, 12, :] += 150.0

    rx = RXDetector(regularization=1e-3, method="cholesky")
    scores = rx.fit_predict(cube)

    assert scores.shape == (h, w)
    assert np.all(np.isfinite(scores))
    assert np.all(scores >= 0.0)

    # The implanted anomaly must rank as the highest anomaly score in the scene
    max_pos = np.unravel_index(np.argmax(scores), scores.shape)
    assert max_pos == (12, 12), f"Expected peak at (12, 12), got {max_pos}"


def test_rx_degenerate_covariance():
    np.random.seed(42)
    h, w, b = 20, 20, 8
    cube = np.random.rand(h, w, b).astype(np.float32)

    # Make bands 1, 2, 3 exact duplicates of band 0 (severely collinear / singular covariance)
    cube[:, :, 1] = cube[:, :, 0]
    cube[:, :, 2] = cube[:, :, 0]
    cube[:, :, 3] = cube[:, :, 0]

    # Test all three solver strategies
    for method in ("cholesky", "solve", "pinv"):
        rx = RXDetector(regularization=1e-3, method=method)
        scores = rx.fit_predict(cube)
        assert scores.shape == (h, w)
        assert np.all(np.isfinite(scores))
        assert np.all(scores >= 0.0)


def test_rx_unfitted_and_band_mismatch():
    rx = RXDetector()
    cube = np.ones((10, 10, 5), dtype=np.float32)

    # Calling predict before fit
    with pytest.raises(RuntimeError, match="must be fitted before"):
        rx.predict(cube)

    rx.fit(cube)

    # Predicting on a cube with different number of bands
    with pytest.raises(ValueError, match="band count mismatch"):
        rx.predict(np.ones((10, 10, 6), dtype=np.float32))


# ============================================================================
# 3. Evaluation Metrics Tests
# ============================================================================


def test_metrics_perfect_detector():
    y_true = np.array([0, 0, 0, 1, 1], dtype=np.int32)
    y_score = np.array([0.1, 0.2, 0.3, 0.8, 0.9], dtype=np.float64)

    assert roc_auc(y_true, y_score) == pytest.approx(1.0)
    assert average_precision(y_true, y_score) == pytest.approx(1.0)

    f1_val, thresh = max_f1_score(y_true, y_score)
    assert f1_val == pytest.approx(1.0)
    assert thresh <= 0.8

    assert fpr_at_tpr(y_true, y_score, target_tpr=0.95) == pytest.approx(0.0)
    assert false_alarms_per_megapixel(y_true, y_score, target_tpr=0.95) == pytest.approx(0.0)


def test_metrics_inverted_detector():
    y_true = np.array([0, 0, 0, 1, 1], dtype=np.int32)
    # Inverted scores
    y_score = np.array([0.9, 0.8, 0.7, 0.2, 0.1], dtype=np.float64)

    assert roc_auc(y_true, y_score) == pytest.approx(0.0)
    assert fpr_at_tpr(y_true, y_score, target_tpr=0.95) == pytest.approx(1.0)


def test_metrics_constant_scores():
    y_true = np.array([0, 0, 1, 1], dtype=np.int32)
    y_score = np.array([0.5, 0.5, 0.5, 0.5], dtype=np.float64)

    # Constant scores give random guess (0.5 AUC) and prevalence AP (2/4 = 0.5)
    assert roc_auc(y_true, y_score) == pytest.approx(0.5)
    assert average_precision(y_true, y_score) == pytest.approx(0.5)


def test_false_alarms_per_megapixel_calculation():
    # 10 pixels total: 2 anomalies, 8 background
    y_true = np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 1])
    # Predicted 1 false alarm (background index 0 with high score)
    y_score = np.array([0.9, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.95, 0.99])

    # At threshold 0.85: TP=2, FP=1, TN=7, FN=0 -> TPR = 1.0 (>= 0.95)
    far = false_alarms_per_megapixel(y_true, y_score, threshold=0.85)
    # 1 false alarm / 10 pixels * 1,000,000 = 100,000
    assert far == pytest.approx(100_000.0)


def test_evaluate_anomaly_map_dictionary():
    y_true = np.array([[0, 1], [0, 0]])
    y_score = np.array([[0.2, 0.9], [0.1, 0.3]])

    results = evaluate_anomaly_map(y_true, y_score, target_tpr=0.95)

    assert "auc_roc" in results
    assert "auc_pr" in results
    assert "f1_max" in results
    assert "optimal_threshold" in results
    assert "fpr_at_95_tpr" in results
    assert "false_alarms_per_mp" in results
    assert results["total_pixels"] == 4
    assert results["anomaly_pixels"] == 1
    assert results["anomaly_pct"] == pytest.approx(25.0)


def test_metrics_edge_case_single_class():
    # Single class mask (no anomalies)
    y_all_zero = np.zeros(10, dtype=np.int32)
    y_score = np.linspace(0, 1, 10)

    # Undefined mathematically: returns nan
    assert np.isnan(roc_auc(y_all_zero, y_score))
    assert np.isnan(average_precision(y_all_zero, y_score))
    f1_val, _ = max_f1_score(y_all_zero, y_score)
    assert f1_val == 0.0


def test_metrics_invalid_inputs():
    with pytest.raises(TypeError):
        roc_auc([0, 1], [0.1, 0.9])

    with pytest.raises(ValueError, match="Shape mismatch"):
        roc_auc(np.array([0, 1]), np.array([0.1, 0.5, 0.9]))

    with pytest.raises(ValueError, match="NaN or Inf"):
        roc_auc(np.array([0, 1]), np.array([np.nan, 0.5]))

    with pytest.raises(ValueError, match="binary values"):
        roc_auc(np.array([0, 2]), np.array([0.1, 0.5]))
