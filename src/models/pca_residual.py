"""PCA reconstruction-residual anomaly detector."""

from typing import Optional

import numpy as np
from sklearn.decomposition import PCA

from src.models.base import BaseDetector


class PCAResidualDetector(BaseDetector):
    """Score pixels by spectral reconstruction error from a global PCA subspace.

    PCA is fitted without labels. A deterministic pixel sample bounds fitting
    memory for large scenes; inference is evaluated in batches.
    """

    def __init__(
        self,
        n_components: int = 8,
        random_state: int = 42,
        max_fit_samples: int = 50_000,
        name: Optional[str] = "PCA-Residual",
    ):
        super().__init__(name=name)
        if not isinstance(n_components, int) or n_components < 1:
            raise ValueError("n_components must be a positive integer")
        if not isinstance(max_fit_samples, int) or max_fit_samples < 2:
            raise ValueError("max_fit_samples must be an integer of at least 2")
        self.n_components = n_components
        self.random_state = int(random_state)
        self.max_fit_samples = max_fit_samples
        self.pca_: Optional[PCA] = None
        self.n_bands_: Optional[int] = None

    def fit(self, cube: np.ndarray) -> "PCAResidualDetector":
        cube_val = self.validate_cube(cube)
        n_pixels = cube_val.shape[0] * cube_val.shape[1]
        n_bands = cube_val.shape[2]
        n_fit = min(n_pixels, self.max_fit_samples)

        if self.n_components > min(n_fit - 1, n_bands):
            raise ValueError(
                "n_components must not exceed the smaller of the spectral band "
                f"count ({n_bands}) and fit sample rank ({n_fit - 1})"
            )

        pixels = cube_val.reshape(n_pixels, n_bands)
        if n_fit < n_pixels:
            indices = np.random.default_rng(self.random_state).choice(
                n_pixels, size=n_fit, replace=False
            )
            fit_pixels = pixels[indices]
        else:
            fit_pixels = pixels

        if not np.any(np.var(fit_pixels.astype(np.float64), axis=0) > 0.0):
            raise ValueError("PCA fit data must contain nonzero spectral variance")

        solver = (
            "randomized"
            if self.n_components < min(fit_pixels.shape)
            else "full"
        )
        self.pca_ = PCA(
            n_components=self.n_components,
            svd_solver=solver,
            random_state=self.random_state,
        )
        self.pca_.fit(fit_pixels)
        self.n_bands_ = n_bands
        self.is_fitted = True
        return self

    def predict(self, cube: np.ndarray) -> np.ndarray:
        if not self.is_fitted or self.pca_ is None or self.n_bands_ is None:
            raise RuntimeError("Detector must be fitted before calling predict.")

        cube_val = self.validate_cube(cube)
        h, w, bands = cube_val.shape
        if bands != self.n_bands_:
            raise ValueError(
                f"Spectral band count mismatch: fitted on {self.n_bands_} bands, got {bands} bands."
            )

        pixels = cube_val.reshape(h * w, bands)
        scores = np.empty(h * w, dtype=np.float64)
        batch_size = 8192
        for start in range(0, len(pixels), batch_size):
            stop = min(start + batch_size, len(pixels))
            batch = pixels[start:stop]
            projected = self.pca_.transform(batch)
            reconstructed = self.pca_.inverse_transform(projected)
            residual = batch - reconstructed
            scores[start:stop] = np.einsum("ij,ij->i", residual, residual)

        return self.validate_score_map(scores.reshape(h, w), expected_shape=(h, w))

    def reconstruction_residual(self, spectrum: np.ndarray) -> np.ndarray:
        """Return per-band reconstruction residuals for one fitted spectrum."""
        if not self.is_fitted or self.pca_ is None or self.n_bands_ is None:
            raise RuntimeError("Detector must be fitted before explaining a spectrum.")
        if not isinstance(spectrum, np.ndarray):
            raise TypeError("spectrum must be a numpy.ndarray")
        if spectrum.ndim != 1 or spectrum.shape[0] != self.n_bands_:
            raise ValueError(
                f"spectrum must have shape ({self.n_bands_},), got {spectrum.shape}"
            )
        if not np.all(np.isfinite(spectrum)):
            raise ValueError("spectrum must contain only finite values")
        values = spectrum.reshape(1, -1)
        return (values - self.pca_.inverse_transform(self.pca_.transform(values)))[0]
