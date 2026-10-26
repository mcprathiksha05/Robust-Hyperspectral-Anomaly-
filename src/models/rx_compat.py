"""Compatibility helpers for the legacy RX API."""

import numpy as np
from scipy.ndimage import uniform_filter

from src.utils.validation import (
    HSIValidationError,
    check_windows,
    validate_cube,
)


def rx_scores(cube, mask=None, **kwargs):
    """Fit the current RX detector and return its anomaly score map."""
    from src.models.rx import RXDetector

    detector = RXDetector(**kwargs)
    detector.fit(cube, mask=mask)
    return detector.predict(cube)


def ring_sums(arr, inner: int, outer: int):
    """Calculate sums over the outer window minus the inner window."""
    size_outer = (outer, outer) + (1,) * (arr.ndim - 2)
    size_inner = (inner, inner) + (1,) * (arr.ndim - 2)

    sum_outer = uniform_filter(
        arr, size=size_outer, mode="reflect"
    ) * (outer * outer)
    sum_inner = uniform_filter(
        arr, size=size_inner, mode="reflect"
    ) * (inner * inner)

    return sum_outer - sum_inner, outer * outer - inner * inner


def local_rx(
    cube,
    inner: int = 5,
    outer: int = 15,
    reg: float = 1e-3,
    max_memory_mb: int = 1500,
):
    """Compute dual-window local RX scores."""
    cube = validate_cube(cube)
    inner, outer = check_windows(inner, outer, cube.shape)

    h, w, bands = cube.shape
    ring_count = outer * outer - inner * inner

    if ring_count <= bands:
        raise HSIValidationError(
            f"The ring has {ring_count} pixels but there are {bands} bands. "
            "Enlarge the outer window or reduce bands with PCA."
        )

    need_mb = h * w * bands * bands * 8 * 3 / 1e6
    if need_mb > max_memory_mb:
        raise HSIValidationError(
            f"Local RX would need about {need_mb:.0f} MB. "
            f"Reduce bands with PCA (currently {bands}) or crop the image."
        )

    x = cube.astype(np.float64)
    first_sum, count = ring_sums(x, inner, outer)

    outer_product = x[..., :, None] * x[..., None, :]
    second_sum, _ = ring_sums(outer_product, inner, outer)
    del outer_product

    mean = first_sum / count
    covariance = (
        second_sum / count
        - mean[..., :, None] * mean[..., None, :]
    )

    trace = (
        np.trace(covariance, axis1=-2, axis2=-1)[..., None, None]
        / bands
    )
    covariance += (
        reg * np.maximum(trace, 1e-12)
    ) * np.eye(bands)

    difference = x - mean
    solution = np.linalg.solve(
        covariance.reshape(-1, bands, bands),
        difference.reshape(-1, bands, 1),
    ).reshape(h, w, bands)

    return np.maximum((difference * solution).sum(axis=-1), 0)
