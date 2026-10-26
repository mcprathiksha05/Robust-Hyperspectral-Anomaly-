"""Feature 6: Sensitivity analysis and parameter sweep framework.

Evaluates how performance metrics vary in response to continuous or discrete sweeps
over:
    - Corruption parameters: Gaussian noise SNR, band corruption fraction, clutter abundance.
    - Detector parameters: Covariance regularization lambda, solver choice.
    - Operational decision thresholds: Precision, Recall, FPR, and False Alarm trade-offs.

Threshold Reporting Protocol:
    Post-hoc threshold optimization (e.g. F1-max) requires ground truth and cannot
    be reported as an independent deployment result. This module clearly separates:
    - Unsupervised threshold sweeps (trade-off curves without prior labels).
    - Labeled validation operating points (fixed target TPR or post-hoc max F1).
"""

from typing import Any, Callable, Dict, List, Literal, Optional, Sequence, Tuple, Type
import numpy as np
from sklearn import metrics as sk_metrics

from src.models.base import BaseDetector
from src.models.rx import RXDetector
from src.evaluation.metrics import evaluate_anomaly_map
from src.robustness.noise import add_gaussian_noise
from src.robustness.bands import drop_random_bands
from src.robustness.clutter import inject_spectral_clutter


def run_parameter_sweep(
    param_name: str,
    param_values: Sequence[Any],
    param_type: Literal["corruption", "detector", "threshold"],
    evaluate_fn: Callable[[Any], Dict[str, Any]],
) -> Dict[str, Any]:
    """Execute a generic 1D parameter sweep and return tabular results.

    Args:
        param_name: Name of parameter being varied.
        param_values: Sequence of values to evaluate.
        param_type: Category: 'corruption' (environmental/sensor), 'detector' (hyperparameter),
                    or 'threshold' (operational decision point).
        evaluate_fn: Callable mapping param_value -> dictionary of evaluation metrics.

    Returns:
        dict: Sweep report containing 'param_name', 'param_type', 'param_values',
              'results', and 'table'.
    """
    results: List[Dict[str, Any]] = []

    for val in param_values:
        metrics = evaluate_fn(val)
        record = {
            param_name: val,
            "metrics": metrics,
        }
        results.append(record)

    # Build clean tabular summary
    table = []
    for r in results:
        val = r[param_name]
        m = r["metrics"]
        row = {
            param_name: val,
            "auc_roc": m.get("auc_roc"),
            "auc_pr": m.get("auc_pr"),
            "f1_max": m.get("f1_max"),
            "fpr_at_95_tpr": m.get("fpr_at_95_tpr"),
            "false_alarms_per_mp": m.get("false_alarms_per_mp"),
        }
        table.append(row)

    return {
        "param_name": param_name,
        "param_type": param_type,
        "param_values": list(param_values),
        "results": results,
        "table": table,
    }


def sweep_noise_snr(
    cube: np.ndarray,
    gt: np.ndarray,
    snr_values: Optional[Sequence[float]] = None,
    detector_cls: Type[BaseDetector] = RXDetector,
    seed: int = 42,
    target_tpr: float = 0.95,
    snr_db_levels: Optional[Sequence[float]] = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Evaluate detector sensitivity to varying Gaussian noise SNR (dB).

    Args:
        cube: Hyperspectral cube of shape (H, W, B).
        gt: Binary ground truth mask (H, W).
        snr_values: Sequence of SNR values in decibels to evaluate.
        detector_cls: Detector class adhering to BaseDetector.
        seed: Random seed for reproducible noise generation.
        target_tpr: Target TPR for false alarm calculation.
        snr_db_levels: Optional alias for snr_values.

    Returns:
        dict: Sweep report categorized as param_type='corruption'.
    """
    if snr_values is None:
        snr_values = snr_db_levels if snr_db_levels is not None else (35.0, 30.0, 25.0, 20.0, 15.0, 10.0, 5.0)

    detector = detector_cls()

    def evaluate_snr(snr: float) -> Dict[str, Any]:
        corrupted, _ = add_gaussian_noise(cube, snr_db=float(snr), seed=seed)
        scores = detector.fit_predict(corrupted)
        return evaluate_anomaly_map(gt, scores, target_tpr=target_tpr)

    report = run_parameter_sweep(
        param_name="snr_db",
        param_values=snr_values,
        param_type="corruption",
        evaluate_fn=evaluate_snr,
    )
    report["points"] = report["results"]
    return report


def sweep_band_dropout(
    cube: np.ndarray,
    gt: np.ndarray,
    fraction_values: Sequence[float] = (0.05, 0.10, 0.20, 0.30, 0.40, 0.50),
    detector_cls: Type[BaseDetector] = RXDetector,
    seed: int = 42,
    target_tpr: float = 0.95,
) -> Dict[str, Any]:
    """Evaluate sensitivity to varying fractions of corrupted/dropped spectral channels.

    Args:
        cube: Hyperspectral cube.
        gt: Ground truth mask.
        fraction_values: Fractions of bands to drop, in (0, 1].
        detector_cls: Detector class.
        seed: Random seed.
        target_tpr: Operating point TPR.

    Returns:
        dict: Sweep report categorized as param_type='corruption'.
    """
    detector = detector_cls()

    def evaluate_fraction(frac: float) -> Dict[str, Any]:
        corrupted, _ = drop_random_bands(cube, fraction=float(frac), mode="zero", seed=seed)
        scores = detector.fit_predict(corrupted)
        return evaluate_anomaly_map(gt, scores, target_tpr=target_tpr)

    return run_parameter_sweep(
        param_name="dropped_band_fraction",
        param_values=fraction_values,
        param_type="corruption",
        evaluate_fn=evaluate_fraction,
    )


def sweep_clutter_abundance(
    cube: np.ndarray,
    gt: np.ndarray,
    abundance_values: Sequence[float] = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7),
    spatial_fraction: float = 0.05,
    detector_cls: Type[BaseDetector] = RXDetector,
    seed: int = 42,
    target_tpr: float = 0.95,
) -> Dict[str, Any]:
    """Evaluate sensitivity to varying background clutter mixing abundance.

    Args:
        cube: Hyperspectral cube.
        gt: Ground truth mask.
        abundance_values: Sequence of clutter mixing abundances in (0, 1].
        spatial_fraction: Spatial proportion of scene perturbed.
        detector_cls: Detector class.
        seed: Random seed.
        target_tpr: Operating point TPR.

    Returns:
        dict: Sweep report categorized as param_type='corruption'.
    """
    detector = detector_cls()

    def evaluate_abundance(alpha: float) -> Dict[str, Any]:
        corrupted, _ = inject_spectral_clutter(
            cube, abundance=float(alpha), spatial_fraction=spatial_fraction, seed=seed
        )
        scores = detector.fit_predict(corrupted)
        return evaluate_anomaly_map(gt, scores, target_tpr=target_tpr)

    return run_parameter_sweep(
        param_name="clutter_abundance",
        param_values=abundance_values,
        param_type="corruption",
        evaluate_fn=evaluate_abundance,
    )


def sweep_detector_regularization(
    cube: np.ndarray,
    gt: np.ndarray,
    reg_values: Sequence[float] = (1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1),
    detector_cls: Type[BaseDetector] = RXDetector,
    target_tpr: float = 0.95,
) -> Dict[str, Any]:
    """Evaluate sensitivity to the detector covariance regularization hyperparameter lambda.

    Args:
        cube: Hyperspectral cube.
        gt: Ground truth mask.
        reg_values: Regularization multiplier values.
        detector_cls: Detector class.
        target_tpr: Operating point TPR.

    Returns:
        dict: Sweep report categorized as param_type='detector'.
    """
    def evaluate_reg(reg: float) -> Dict[str, Any]:
        det = detector_cls(regularization=float(reg))
        scores = det.fit_predict(cube)
        return evaluate_anomaly_map(gt, scores, target_tpr=target_tpr)

    return run_parameter_sweep(
        param_name="regularization",
        param_values=reg_values,
        param_type="detector",
        evaluate_fn=evaluate_reg,
    )


def sweep_decision_thresholds(
    scores: np.ndarray,
    gt: np.ndarray,
    n_points: int = 50,
) -> Dict[str, Any]:
    """Analyze the operational detection trade-offs across the score threshold range.

    Computes True Positive Rate (TPR), False Positive Rate (FPR), Precision, Recall,
    F1-score, and False Alarms per Megapixel at evenly spaced threshold percentiles.

    Important Protocol Distinction:
        This function maps the operational trade-off curve across all thresholds.
        Thresholds must NOT be selected post-hoc to claim high F1 without independent
        validation.

    Args:
        scores: 2D or 1D anomaly score map.
        gt: Binary ground truth mask.
        n_points: Number of threshold evaluation points.

    Returns:
        dict: Operational trade-off curve table categorized as param_type='threshold'.
    """
    y_true = np.asarray(gt).ravel().astype(np.int32)
    y_score = np.asarray(scores).ravel().astype(np.float64)

    total_pixels = len(y_true)
    n_pos = int(np.sum(y_true == 1))
    n_neg = int(np.sum(y_true == 0))

    if n_pos == 0 or n_neg == 0:
        raise ValueError("Ground truth must contain both positive and negative samples.")

    # Select candidate thresholds evenly spaced between min and 99.9th percentile
    thresh_candidates = np.percentile(y_score, np.linspace(5.0, 99.9, n_points))
    thresh_candidates = np.unique(thresh_candidates)

    tradeoff_table = []
    for th in thresh_candidates:
        pred = (y_score >= th).astype(np.int32)
        tp = int(np.sum((y_true == 1) & (pred == 1)))
        fp = int(np.sum((y_true == 0) & (pred == 1)))
        fn = int(np.sum((y_true == 1) & (pred == 0)))
        tn = int(np.sum((y_true == 0) & (pred == 0)))

        tpr = tp / n_pos if n_pos > 0 else 0.0
        fpr = fp / n_neg if n_neg > 0 else 0.0
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tpr
        f1 = (2.0 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
        far_mp = (fp / total_pixels) * 1_000_000.0

        tradeoff_table.append({
            "threshold": float(th),
            "tpr": float(tpr),
            "fpr": float(fpr),
            "precision": float(prec),
            "recall": float(rec),
            "f1": float(f1),
            "false_alarms_per_mp": float(far_mp),
        })

    return {
        "param_name": "threshold",
        "param_type": "threshold",
        "tradeoff_curve": tradeoff_table,
        "n_evaluated_thresholds": len(tradeoff_table),
    }
