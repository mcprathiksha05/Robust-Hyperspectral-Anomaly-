"""Feature 4: Ablation study framework for hyperspectral anomaly detection.

Provides modular, reproducible evaluation of alternative processing configurations:
    - Input normalization ablations: raw vs. min-max vs. z-score vs. L2-norm
    - Spectral band retention ablations: full bands vs. subsampled/retained fractions
    - Model regularization ablations: covariance shrinkage levels
    - Matrix solver ablations: Cholesky vs. solve vs. pseudo-inverse

All configurations are strictly documented and evaluated using the project's
shared evaluation metrics.
"""

import time
from typing import Any, Callable, Dict, List, Optional, Tuple, Type
import numpy as np

from src.models.base import BaseDetector
from src.models.rx import RXDetector
from src.evaluation.metrics import evaluate_anomaly_map


def apply_normalization(cube: np.ndarray, method: str) -> np.ndarray:
    """Apply a documented, non-destructive normalization scheme to a 3D cube.

    Args:
        cube: Hyperspectral cube of shape (H, W, B). Original array is not modified.
        method: One of:
            - 'raw' / 'none': Returns float64 copy of input without scaling.
            - 'minmax': Rescales each spectral band independently to [0.0, 1.0].
            - 'zscore': Standardizes each spectral band to zero mean and unit variance.
            - 'l2': Normalizes each spatial pixel vector to unit L2 norm.

    Returns:
        np.ndarray: Normalized float64 cube of shape (H, W, B).
    """
    c = cube.astype(np.float64, copy=True)
    h, w, b = c.shape

    if method in ("raw", "none"):
        return c
    elif method == "minmax":
        min_vals = np.min(c, axis=(0, 1), keepdims=True)
        max_vals = np.max(c, axis=(0, 1), keepdims=True)
        ranges = max_vals - min_vals
        ranges = np.where(ranges > 1e-12, ranges, 1.0)
        return (c - min_vals) / ranges
    elif method == "zscore":
        means = np.mean(c, axis=(0, 1), keepdims=True)
        stds = np.std(c, axis=(0, 1), keepdims=True)
        stds = np.where(stds > 1e-12, stds, 1.0)
        return (c - means) / stds
    elif method == "l2":
        norms = np.linalg.norm(c, axis=2, keepdims=True)
        norms = np.where(norms > 1e-12, norms, 1.0)
        return c / norms
    else:
        raise ValueError(
            f"Unsupported normalization method '{method}'. Supported: 'raw', 'minmax', 'zscore', 'l2'"
        )


def apply_band_subsampling(
    cube: np.ndarray, retained_fraction: float
) -> Tuple[np.ndarray, List[int]]:
    """Subsample spectral channels uniformly across the spectrum.

    Args:
        cube: Hyperspectral cube of shape (H, W, B).
        retained_fraction: Fraction of bands to retain, in (0.0, 1.0].

    Returns:
        Tuple[np.ndarray, List[int]]:
            - subsampled_cube: Cube of shape (H, W, B_retained).
            - retained_indices: 0-based band indices retained.
    """
    if not (0.0 < retained_fraction <= 1.0):
        raise ValueError(f"retained_fraction must be in (0, 1], got {retained_fraction}")

    b = cube.shape[2]
    n_retained = max(1, int(round(b * retained_fraction)))
    indices = np.linspace(0, b - 1, n_retained, dtype=int).tolist()
    # Remove duplicates if any
    indices = sorted(list(dict.fromkeys(indices)))
    return cube[:, :, indices].astype(np.float64, copy=True), indices


def run_ablation_study(
    cube: np.ndarray,
    gt: np.ndarray,
    configurations: List[Dict[str, Any]],
    detector_cls: Type[BaseDetector] = RXDetector,
    target_tpr: float = 0.95,
) -> Dict[str, Any]:
    """Execute a controlled ablation study over a list of processing configurations.

    Args:
        cube: Hyperspectral cube of shape (H, W, B). Original is unchanged.
        gt: Binary ground truth mask of shape (H, W). Original is unchanged.
        configurations: List of configuration dictionaries, each specifying:
            - 'name': Descriptive configuration name (str)
            - 'normalization': 'raw', 'minmax', 'zscore', or 'l2' (default: 'raw')
            - 'retained_band_fraction': float in (0, 1] (default: 1.0)
            - 'regularization': float (default: 1e-4)
            - 'solver_method': 'cholesky', 'solve', or 'pinv' (default: 'cholesky')
        detector_cls: Detector class adhering to BaseDetector (default: RXDetector).
        target_tpr: Operating point TPR for false alarm calculation (default: 0.95).

    Returns:
        dict: Structured study results containing:
            - 'configurations': List of individual configuration result dictionaries
            - 'best_configuration_by_auc': Configuration dictionary achieving max ROC-AUC
            - 'best_configuration_by_f1': Configuration dictionary achieving max F1
            - 'total_configurations_tested': Integer count
    """
    if len(configurations) == 0:
        raise ValueError("configurations list must not be empty")

    results = []

    for cfg in configurations:
        cfg_name = cfg.get("name", "Unnamed_Config")
        norm_method = cfg.get("normalization", "raw")
        band_fraction = float(cfg.get("retained_band_fraction", 1.0))
        reg = float(cfg.get("regularization", 1e-4))
        solver = cfg.get("solver_method", "cholesky")

        t0 = time.perf_counter()

        # 1. Normalization step
        processed_cube = apply_normalization(cube, norm_method)

        # 2. Band retention step
        retained_indices = None
        if band_fraction < 1.0:
            processed_cube, retained_indices = apply_band_subsampling(
                processed_cube, band_fraction
            )

        # 3. Fit and predict detector
        detector = detector_cls(regularization=reg, method=solver)
        scores = detector.fit_predict(processed_cube)

        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        # 4. Shared metrics evaluation
        metrics = evaluate_anomaly_map(gt, scores, target_tpr=target_tpr)

        record: Dict[str, Any] = {
            "name": cfg_name,
            "params": {
                "normalization": norm_method,
                "retained_band_fraction": band_fraction,
                "bands_used": processed_cube.shape[2],
                "regularization": reg,
                "solver_method": solver,
            },
            "metrics": metrics,
            "elapsed_ms": float(elapsed_ms),
        }
        results.append(record)

    # Sort results
    best_auc = max(results, key=lambda x: x["metrics"]["auc_roc"])
    best_f1 = max(results, key=lambda x: x["metrics"]["f1_max"])

    return {
        "configurations": results,
        "best_configuration_by_auc": best_auc,
        "best_configuration_by_f1": best_f1,
        "total_configurations_tested": len(results),
    }


def build_default_ablation_suite() -> List[Dict[str, Any]]:
    """Build a standard benchmark ablation suite comparing normalization, band fractions, and solvers.

    Returns:
        List[Dict[str, Any]]: List of standard configuration dictionaries.
    """
    return [
        {"name": "Baseline (Raw, 100% bands, Reg 1e-4)", "normalization": "raw", "retained_band_fraction": 1.0, "regularization": 1e-4},
        {"name": "Norm: MinMax [0, 1]", "normalization": "minmax", "retained_band_fraction": 1.0, "regularization": 1e-4},
        {"name": "Norm: Z-Score", "normalization": "zscore", "retained_band_fraction": 1.0, "regularization": 1e-4},
        {"name": "Norm: Pixel L2-Norm", "normalization": "l2", "retained_band_fraction": 1.0, "regularization": 1e-4},
        {"name": "Bands: 75% Retained", "normalization": "raw", "retained_band_fraction": 0.75, "regularization": 1e-4},
        {"name": "Bands: 50% Retained", "normalization": "raw", "retained_band_fraction": 0.50, "regularization": 1e-4},
        {"name": "Reg: High (1e-2)", "normalization": "raw", "retained_band_fraction": 1.0, "regularization": 1e-2},
        {"name": "Reg: Minimal (1e-6)", "normalization": "raw", "retained_band_fraction": 1.0, "regularization": 1e-6},
        {"name": "Solver: Pseudo-Inverse (Pinv)", "normalization": "raw", "retained_band_fraction": 1.0, "regularization": 1e-4, "solver_method": "pinv"},
    ]
