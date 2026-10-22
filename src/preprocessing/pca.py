"""Task 3 - PCA dimensionality reduction.

    from src.preprocessing.pca import PCA
    pca = PCA(n_components=0.999)        # int = number of components, float = variance to keep
    reduced = pca.fit_transform(cube)    # (H, W, k)
    pca.explained_variance_ratio_, pca.n_components_

Implemented directly with NumPy (eigen-decomposition of the band covariance)
so the frontend can show explained variance and the explanation module can map
component contributions back to original bands via ``components_``.
"""
from __future__ import annotations

import numpy as np

from src.utils.validation import HSIValidationError, check_float, check_int, validate_cube


class PCA:
    def __init__(self, n_components: int | float = 0.999, whiten: bool = False,
                 max_fit_pixels: int | None = 200_000, random_state: int = 0):
        if isinstance(n_components, float) and not float(n_components).is_integer():
            check_float(n_components, "n_components (variance fraction)", 0, 1, lo_inclusive=False)
        else:
            n_components = check_int(n_components, "n_components", lo=1)
        self.n_components = n_components
        self.whiten = whiten
        self.max_fit_pixels = max_fit_pixels
        self.random_state = random_state

    # ------------------------------------------------------------------ fit
    def fit(self, cube, mask=None):
        """Fit on the cube's pixels (optionally only where ``mask`` is True)."""
        cube = validate_cube(cube)
        x = cube.reshape(-1, cube.shape[2]).astype(np.float64)
        if mask is not None:
            x = x[np.asarray(mask, bool).ravel()]
        if not np.isfinite(x).all():
            raise HSIValidationError("PCA input contains NaN/inf. Run preprocess() first.")
        if self.max_fit_pixels and x.shape[0] > self.max_fit_pixels:
            rng = np.random.default_rng(self.random_state)
            x = x[rng.choice(x.shape[0], self.max_fit_pixels, replace=False)]
        b = x.shape[1]
        if isinstance(self.n_components, int) and self.n_components > b:
            raise HSIValidationError(f"n_components={self.n_components} exceeds the number of bands ({b}).")

        self.mean_ = x.mean(0)
        cov = np.cov(x - self.mean_, rowvar=False)
        evals, evecs = np.linalg.eigh(cov)                 # ascending
        order = np.argsort(evals)[::-1]
        evals, evecs = np.clip(evals[order], 0, None), evecs[:, order]
        total = evals.sum()
        if total <= 0:
            raise HSIValidationError("All bands are constant; PCA has nothing to keep.")
        ratio = evals / total
        if isinstance(self.n_components, int):
            k = self.n_components
        else:
            k = int(np.searchsorted(np.cumsum(ratio), self.n_components) + 1)
        k = max(1, min(k, b))
        # Sign convention: largest loading positive, so results are deterministic.
        signs = np.sign(evecs[np.abs(evecs).argmax(0), np.arange(b)])
        evecs = evecs * np.where(signs == 0, 1, signs)

        self.n_components_ = k
        self.components_ = evecs[:, :k].T                   # (k, B)
        self.explained_variance_ = evals[:k]
        self.explained_variance_ratio_ = ratio[:k]
        self.cumulative_variance_ = float(ratio[:k].sum())
        self.all_variance_ratio_ = ratio                      # for scree plots
        return self

    # ------------------------------------------------------------ transform
    def transform(self, cube):
        self._check_fitted()
        cube = validate_cube(cube, min_pixels=1)
        h, w, b = cube.shape
        if b != self.mean_.size:
            raise HSIValidationError(f"PCA was fitted on {self.mean_.size} bands but got {b}.")
        z = (cube.reshape(-1, b).astype(np.float64) - self.mean_) @ self.components_.T
        if self.whiten:
            z /= np.sqrt(np.maximum(self.explained_variance_, 1e-12))
        return z.reshape(h, w, self.n_components_).astype(np.float32)

    def fit_transform(self, cube, mask=None):
        return self.fit(cube, mask).transform(cube)

    def inverse_transform(self, reduced):
        """Map (H, W, k) back to (H, W, B) - useful for reconstruction-error checks."""
        self._check_fitted()
        reduced = np.asarray(reduced, dtype=np.float64)
        z = reduced.reshape(-1, reduced.shape[-1])
        if self.whiten:
            z = z * np.sqrt(np.maximum(self.explained_variance_, 1e-12))
        x = z @ self.components_ + self.mean_
        return x.reshape(*reduced.shape[:-1], -1).astype(np.float32)

    def summary(self) -> dict:
        self._check_fitted()
        return {"n_components": self.n_components_, "input_bands": int(self.mean_.size),
                "variance_kept": round(self.cumulative_variance_, 6),
                "explained_variance_ratio": self.explained_variance_ratio_.round(6).tolist()}

    def _check_fitted(self):
        if not hasattr(self, "components_"):
            raise HSIValidationError("PCA must be fitted before use (call fit or fit_transform).")
