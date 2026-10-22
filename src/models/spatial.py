"""Task 5 - Spatial residual detector.

    from src.models.spatial import spatial_residual
    scores, residual = spatial_residual(cube, inner=3, outer=9)

Each pixel is compared with the average spectrum of the ring of pixels around
it (outer window minus a guard window that keeps the target out of its own
background). Pixels that differ from their neighbourhood score high, even if
the same material is common elsewhere in the image - which global RX misses.

metric:
  'euclidean' - length of the residual vector
  'zscore'    - residual divided by the ring's per-band standard deviation (local contrast)
  'sam'       - spectral angle between pixel and ring mean (ignores brightness, good under shadow)
"""
from __future__ import annotations

import numpy as np

from src.models.rx import ring_sums
from src.utils.validation import check_choice, check_windows, validate_cube

METRICS = {"euclidean", "zscore", "sam"}


def spatial_residual(cube, inner: int = 3, outer: int = 9, metric: str = "zscore"):
    """Return (scores (H, W), residual (H, W, B)). ``residual`` = pixel minus ring mean."""
    cube = validate_cube(cube)
    inner, outer = check_windows(inner, outer, cube.shape)
    check_choice(metric, "metric", METRICS)
    x = cube.astype(np.float64)
    s1, n = ring_sums(x, inner, outer)
    mean = s1 / n
    residual = x - mean
    if metric == "euclidean":
        scores = np.linalg.norm(residual, axis=-1)
    elif metric == "zscore":
        s2, _ = ring_sums(x * x, inner, outer)
        var = np.maximum(s2 / n - mean * mean, 0)
        floor = 1e-3 * np.median(var) + 1e-12  # avoid dividing by ~0 in flat areas
        scores = np.sqrt(((residual ** 2) / (var + floor)).mean(-1))
    else:
        num = (x * mean).sum(-1)
        den = np.linalg.norm(x, axis=-1) * np.linalg.norm(mean, axis=-1)
        scores = np.arccos(np.clip(num / np.maximum(den, 1e-12), -1, 1))
    return scores, residual.astype(np.float32)
