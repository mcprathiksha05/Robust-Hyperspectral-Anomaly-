"""Non-destructive hyperspectral cube diagnostics and PCA visualization helpers."""

from typing import Any, Dict, Optional

import numpy as np
from sklearn.decomposition import PCA


def inspect_cube_quality(
    cube: np.ndarray,
    ground_truth: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Summarize dimensions, invalid values, band variability, labels, and memory."""
    if not isinstance(cube, np.ndarray):
        raise TypeError("cube must be a numpy.ndarray")
    if cube.ndim != 3 or any(size <= 0 for size in cube.shape):
        raise ValueError(f"cube must have positive (H, W, B) dimensions, got {cube.shape}")
    if not np.issubdtype(cube.dtype, np.number):
        raise TypeError("cube must contain numeric values")
    if ground_truth is not None:
        if not isinstance(ground_truth, np.ndarray):
            raise TypeError("ground_truth must be a numpy.ndarray")
        if ground_truth.shape != cube.shape[:2]:
            raise ValueError(
                f"ground_truth shape {ground_truth.shape} does not match cube spatial shape {cube.shape[:2]}"
            )

    h, w, bands = cube.shape
    finite = np.isfinite(cube)
    nan_count = int(np.count_nonzero(np.isnan(cube)))
    infinite_count = int(np.count_nonzero(np.isinf(cube)))
    band_std = np.full(bands, np.nan, dtype=np.float64)
    band_mean = np.full(bands, np.nan, dtype=np.float64)
    for index in range(bands):
        values = cube[:, :, index][finite[:, :, index]].astype(np.float64, copy=False)
        if values.size:
            band_mean[index] = float(np.mean(values))
            band_std[index] = float(np.std(values))

    constant_bands = np.flatnonzero(np.isfinite(band_std) & (band_std <= 1e-12)).tolist()
    complete_pixels = np.all(finite, axis=2).reshape(-1)
    complete_count = int(np.count_nonzero(complete_pixels))
    max_abs_correlation = None
    if complete_count >= 2:
        pixel_values = cube.reshape(-1, bands)[complete_pixels]
        if len(pixel_values) > 5000:
            sample_indices = np.linspace(0, len(pixel_values) - 1, 5000, dtype=int)
            pixel_values = pixel_values[sample_indices]
        variable_bands = np.isfinite(band_std) & (band_std > 1e-12)
        if np.count_nonzero(variable_bands) > 1:
            correlations = np.corrcoef(pixel_values[:, variable_bands], rowvar=False)
            upper = np.triu_indices_from(correlations, k=1)
            finite_correlations = np.abs(correlations[upper])
            finite_correlations = finite_correlations[np.isfinite(finite_correlations)]
            if finite_correlations.size:
                max_abs_correlation = float(np.max(finite_correlations))

    label_values = None
    positive_label_count = None
    binary_labels = None
    if ground_truth is not None:
        unique_labels = np.unique(ground_truth)
        label_values = unique_labels.tolist()
        binary_labels = bool(np.all(np.isin(unique_labels, [0, 1])))
        positive_label_count = int(np.count_nonzero(ground_truth > 0))

    return {
        "shape": (h, w, bands),
        "dtype": str(cube.dtype),
        "memory_bytes": int(cube.nbytes),
        "covariance_bytes_float64": int(bands * bands * np.dtype(np.float64).itemsize),
        "total_values": int(cube.size),
        "finite_values": int(np.count_nonzero(finite)),
        "nan_values": nan_count,
        "infinite_values": infinite_count,
        "complete_pixels": complete_count,
        "constant_band_indices": constant_bands,
        "band_mean": band_mean,
        "band_std": band_std,
        "max_abs_band_correlation": max_abs_correlation,
        "label_values": label_values,
        "positive_label_count": positive_label_count,
        "binary_labels": binary_labels,
    }


def compute_pca_component_views(
    cube: np.ndarray,
    n_components: int = 3,
    max_fit_samples: int = 10_000,
    random_state: int = 42,
) -> Dict[str, Any]:
    """Fit a reproducible PCA model on a bounded pixel sample and project the full cube."""
    if not isinstance(cube, np.ndarray):
        raise TypeError("cube must be a numpy.ndarray")
    if cube.ndim != 3 or any(size <= 0 for size in cube.shape):
        raise ValueError(f"cube must have positive (H, W, B) dimensions, got {cube.shape}")
    if not np.all(np.isfinite(cube)):
        raise ValueError("PCA visualization requires a cube without NaN or infinite values")
    if not isinstance(n_components, int) or n_components < 1:
        raise ValueError("n_components must be a positive integer")
    if not isinstance(max_fit_samples, int) or max_fit_samples < 2:
        raise ValueError("max_fit_samples must be an integer of at least 2")

    h, w, bands = cube.shape
    flat = cube.reshape(h * w, bands)
    n_fit = min(len(flat), max_fit_samples)
    if n_components > min(bands, n_fit - 1):
        raise ValueError(
            "n_components must not exceed the smaller of the band count "
            f"({bands}) and fit sample rank ({n_fit - 1})"
        )
    if n_fit < len(flat):
        indices = np.random.default_rng(random_state).choice(
            len(flat), size=n_fit, replace=False
        )
        fit_pixels = flat[indices]
    else:
        fit_pixels = flat

    if not np.any(np.var(fit_pixels.astype(np.float64), axis=0) > 0.0):
        raise ValueError("PCA visualization requires nonzero spectral variance")

    solver = "randomized" if n_components < min(fit_pixels.shape) else "full"
    pca = PCA(
        n_components=n_components,
        svd_solver=solver,
        random_state=random_state,
    )
    pca.fit(fit_pixels)

    component_maps = np.empty((n_components, h * w), dtype=np.float32)
    for start in range(0, len(flat), 8192):
        stop = min(start + 8192, len(flat))
        component_maps[:, start:stop] = pca.transform(flat[start:stop]).T

    return {
        "component_maps": component_maps.reshape(n_components, h, w),
        "components": pca.components_.copy(),
        "explained_variance_ratio": pca.explained_variance_ratio_.copy(),
        "fit_sample_count": int(n_fit),
        "random_state": int(random_state),
    }
