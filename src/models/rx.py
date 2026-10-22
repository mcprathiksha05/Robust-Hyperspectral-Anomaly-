"""Task 4 - RX (Reed-Xiaoli) anomaly detector.

    from src.models.rx import RXDetector, local_rx
    rx = RXDetector(robust=True).fit(cube)
    scores = rx.score(cube)                      # (H, W) Mahalanobis distance
    contrib = rx.contributions(cube[r, c])       # per-band terms that sum to the score

Global RX models the background as one Gaussian (mean mu, covariance Sigma) and
scores each pixel x by  (x - mu)^T Sigma^-1 (x - mu).

``robust=True`` refits mu and Sigma after excluding the most anomalous pixels,
so targets do not contaminate the background model.

``local_rx`` estimates mu and Sigma separately for every pixel from a ring of
neighbours (outer window minus a guard window), using integral-image sums so it
runs in one vectorised pass instead of a Python loop per pixel.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import uniform_filter

from src.utils.validation import (HSIValidationError, check_float, check_int, check_windows,
                                  validate_cube)


class RXDetector:
    def __init__(self, reg: float = 1e-6, robust: bool = True, robust_fraction: float = 0.01,
                 robust_iterations: int = 1, max_fit_pixels: int | None = 300_000, random_state: int = 0):
        self.reg = check_float(reg, "reg", 0)
        self.robust = bool(robust)
        self.robust_fraction = check_float(robust_fraction, "robust_fraction", 0, 0.5)
        self.robust_iterations = check_int(robust_iterations, "robust_iterations", 0, 10)
        self.max_fit_pixels = max_fit_pixels
        self.random_state = random_state

    def fit(self, cube, mask=None):
        """Estimate background mean/covariance. ``mask`` = pixels allowed in the fit (e.g. valid_mask)."""
        cube = validate_cube(cube)
        x = cube.reshape(-1, cube.shape[2]).astype(np.float64)
        if mask is not None:
            x = x[np.asarray(mask, bool).ravel()]
        if not np.isfinite(x).all():
            raise HSIValidationError("RX input contains NaN/inf. Run preprocess() first.")
        if self.max_fit_pixels and x.shape[0] > self.max_fit_pixels:
            rng = np.random.default_rng(self.random_state)
            x = x[rng.choice(x.shape[0], self.max_fit_pixels, replace=False)]
        if x.shape[0] <= x.shape[1]:
            raise HSIValidationError(
                f"RX needs more pixels ({x.shape[0]}) than bands ({x.shape[1]}). Apply PCA first.")
        self._fit_stats(x)
        iters = self.robust_iterations if self.robust else 0
        for _ in range(iters):
            s = self._mahalanobis(x)
            keep = s <= np.quantile(s, 1 - self.robust_fraction)
            if keep.sum() > x.shape[1] + 1:
                self._fit_stats(x[keep])
        return self

    def _fit_stats(self, x):
        self.mean_ = x.mean(0)
        cov = np.cov(x - self.mean_, rowvar=False).reshape(x.shape[1], x.shape[1])
        lam = self.reg * max(np.trace(cov) / cov.shape[0], 1e-12)
        cov = cov + lam * np.eye(cov.shape[0])
        self.cov_ = cov
        evals, evecs = np.linalg.eigh(cov)
        evals = np.maximum(evals, lam if lam > 0 else 1e-12)
        self.inv_cov_ = (evecs / evals) @ evecs.T
        self.condition_number_ = float(evals.max() / evals.min())

    def _mahalanobis(self, x):
        d = x - self.mean_
        return np.einsum("ij,jk,ik->i", d, self.inv_cov_, d)

    def score(self, cube):
        """RX score per pixel, shape (H, W). Higher = more anomalous."""
        self._check()
        cube = validate_cube(cube, min_pixels=1)  # fitted model can score any number of pixels
        h, w, b = cube.shape
        if b != self.mean_.size:
            raise HSIValidationError(f"RX was fitted on {self.mean_.size} bands but got {b}.")
        out = np.empty(h * w)
        flat = cube.reshape(-1, b)
        for i in range(0, h * w, 100_000):  # chunked to bound memory
            out[i:i + 100_000] = self._mahalanobis(flat[i:i + 100_000].astype(np.float64))
        return np.maximum(out, 0).reshape(h, w)

    def fit_score(self, cube, mask=None):
        return self.fit(cube, mask).score(cube)

    def contributions(self, spectrum):
        """Additive per-band decomposition: d_b * (Sigma^-1 d)_b, which sums exactly to the RX score.

        Large positive terms are the bands that make this pixel unusual.
        """
        self._check()
        d = np.asarray(spectrum, dtype=np.float64).ravel() - self.mean_
        return d * (self.inv_cov_ @ d)

    def _check(self):
        if not hasattr(self, "inv_cov_"):
            raise HSIValidationError("RXDetector must be fitted first.")


def rx_scores(cube, mask=None, **kwargs):
    """Convenience: fit global RX on ``cube`` and return the (H, W) score map."""
    return RXDetector(**kwargs).fit_score(cube, mask)


def ring_sums(arr, inner: int, outer: int):
    """Sum and pixel count over the ring (outer window minus inner window) around each pixel.

    ``arr`` has spatial axes first: (H, W, ...). Borders use reflection.
    """
    size_o = (outer, outer) + (1,) * (arr.ndim - 2)
    size_i = (inner, inner) + (1,) * (arr.ndim - 2)
    so = uniform_filter(arr, size=size_o, mode="reflect") * (outer * outer)
    si = uniform_filter(arr, size=size_i, mode="reflect") * (inner * inner)
    return so - si, outer * outer - inner * inner


def local_rx(cube, inner: int = 5, outer: int = 15, reg: float = 1e-3, max_memory_mb: int = 1500):
    """Dual-window local RX. Works best on PCA-reduced data (few bands)."""
    cube = validate_cube(cube)
    inner, outer = check_windows(inner, outer, cube.shape)
    h, w, d = cube.shape
    n = outer * outer - inner * inner
    if n <= d:
        raise HSIValidationError(
            f"The ring has {n} pixels but there are {d} bands; enlarge the outer window or use PCA.")
    need_mb = h * w * d * d * 8 * 3 / 1e6
    if need_mb > max_memory_mb:
        raise HSIValidationError(
            f"Local RX would need about {need_mb:.0f} MB. Reduce bands with PCA "
            f"(currently {d}) or crop the image.")
    x = cube.astype(np.float64)
    s1, n = ring_sums(x, inner, outer)
    outer_prod = x[..., :, None] * x[..., None, :]
    s2, _ = ring_sums(outer_prod, inner, outer)
    del outer_prod
    mu = s1 / n
    cov = s2 / n - mu[..., :, None] * mu[..., None, :]
    tr = np.trace(cov, axis1=-2, axis2=-1)[..., None, None] / d
    cov += (reg * np.maximum(tr, 1e-12)) * np.eye(d)
    diff = x - mu
    sol = np.linalg.solve(cov.reshape(-1, d, d), diff.reshape(-1, d, 1)).reshape(h, w, d)
    return np.maximum((diff * sol).sum(-1), 0)
