"""Feature 8: Uncertainty estimation and score sensitivity mapping.

Quantifies pixel-level prediction variability via Monte Carlo input perturbations.

Scientific Definition & Calibration Disclaimer:
    The uncertainty maps produced by this module measure empirical score dispersion
    (variance, standard deviation, and coefficient of variation) across controlled,
    reproducible stochastic perturbations (e.g. mild Gaussian noise or spectral jitter).
    They reflect DETECTOR FRAGILITY AND SENSITIVITY to sensor perturbations.
    They do NOT represent Bayesian posterior probabilities or calibrated confidence
    estimates unless explicit probability calibration (e.g. isotonic regression
    or Platt scaling) is applied.
"""

from typing import Any, Callable, Dict, List, Optional, Tuple, Type
import numpy as np

from src.models.base import BaseDetector
from src.models.rx import RXDetector
from src.robustness.noise import add_gaussian_noise


def estimate_perturbation_uncertainty(
    cube: np.ndarray,
    detector: Optional[BaseDetector] = None,
    n_perturbations: int = 10,
    snr_db: float = 30.0,
    base_seed: int = 42,
    perturbation_fn: Optional[Callable[[np.ndarray, int], Tuple[np.ndarray, Dict[str, Any]]]] = None,
    num_perturbations: Optional[int] = None,
    perturbation_type: str = "gaussian",
    **kwargs: Any,
) -> Dict[str, Any]:
    """Estimate spatial uncertainty maps via Monte Carlo perturbation score sampling.

    Procedure:
        1. Fit the detector on the original clean cube (if not already fitted) and
           record the baseline score map.
        2. Across T independent trials with seeds base_seed + t:
           Apply a mild perturbation (default: 30 dB SNR Gaussian noise) to obtain X_t.
           Compute anomaly score map S_t = detector.predict(X_t).
        3. Compute pixel-wise summary maps:
           - Mean score: E[S]
           - Score variance: Var(S)
           - Score standard deviation: Std(S)
           - Relative uncertainty (Coefficient of Variation): Std(S) / (E[S] + eps)
           - Inter-percentile spread (90% range): Q_95(S) - Q_05(S)

    Args:
        cube: Hyperspectral cube of shape (H, W, B). Original is unchanged.
        detector: Detector instance adhering to BaseDetector (default: RXDetector()).
        n_perturbations: Number of perturbation iterations T (default 10, minimum 2).
        snr_db: Signal-to-noise ratio in dB for Gaussian perturbation (default 30.0 dB).
        base_seed: Base integer seed for reproducibility.
        perturbation_fn: Optional custom perturbation function mapping (cube, seed) -> (corrupted_cube, meta).
                         If None, uses add_gaussian_noise(cube, snr_db=snr_db, seed=seed).
        num_perturbations: Optional alias for n_perturbations.
        perturbation_type: Perturbation type string (default "gaussian").

    Returns:
        dict: Uncertainty estimation report containing:
            - 'baseline_scores': 2D array of shape (H, W)
            - 'mean_scores': 2D array of shape (H, W)
            - 'variance_map': 2D array of shape (H, W)
            - 'std_map': 2D array of shape (H, W)
            - 'relative_uncertainty_map': 2D array of shape (H, W) [Std / (Mean + 1e-12)]
            - 'spread_90_map': 2D array of shape (H, W) [P95 - P05]
            - 'global_mean_uncertainty': Mean of the std_map
            - 'disclaimer': Scientific calibration disclaimer string
    """
    if num_perturbations is not None:
        n_perturbations = int(num_perturbations)

    if n_perturbations < 2:
        raise ValueError(f"n_perturbations must be at least 2, got {n_perturbations}")

    h, w, b = cube.shape
    det = detector or RXDetector()

    # 1. Base prediction
    baseline_scores = det.fit_predict(cube)

    # 2. Perturbation loop
    score_stack = np.zeros((n_perturbations, h, w), dtype=np.float64)

    for t in range(n_perturbations):
        seed = base_seed + t
        if perturbation_fn is not None:
            perturbed_cube, _ = perturbation_fn(cube, seed)
        else:
            perturbed_cube, _ = add_gaussian_noise(cube, snr_db=float(snr_db), seed=seed)

        # Predict on perturbed cube using fitted detector
        scores_t = det.predict(perturbed_cube)
        score_stack[t] = scores_t

    # 3. Compute spatial statistical moments
    mean_map = np.mean(score_stack, axis=0)
    var_map = np.var(score_stack, axis=0, ddof=1)
    std_map = np.sqrt(var_map)

    # Relative uncertainty (Coefficient of Variation)
    # Avoid zero division with eps
    eps = 1e-12
    rel_map = std_map / (mean_map + eps)

    # Percentiles
    p05_map = np.percentile(score_stack, 5.0, axis=0)
    p95_map = np.percentile(score_stack, 95.0, axis=0)
    spread_map = p95_map - p05_map

    disclaimer = (
        "Scientific Disclaimer: This uncertainty map represents empirical score sensitivity "
        "and variance under input perturbation. It quantifies detector score dispersion and fragility "
        "to noise; it does NOT represent calibrated Bayesian posterior probabilities."
    )

    return {
        "baseline_scores": baseline_scores,
        "mean_scores": mean_map,
        "mean_score_map": mean_map,
        "variance_map": var_map,
        "std_map": std_map,
        "std_score_map": std_map,
        "relative_uncertainty_map": rel_map,
        "spread_90_map": spread_map,
        "global_mean_uncertainty": float(np.mean(std_map)),
        "global_mean_relative_uncertainty": float(np.mean(rel_map)),
        "n_perturbations": n_perturbations,
        "num_perturbations": n_perturbations,
        "disclaimer": disclaimer,
        "scientific_disclaimer": disclaimer,
    }
