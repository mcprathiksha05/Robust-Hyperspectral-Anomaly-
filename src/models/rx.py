"""Global Reed-Xiaoli (RX) anomaly detector.

Implements standard Global RX and an optional robust variant that iteratively
reduces the influence of pixels with high Mahalanobis distances during
background estimation.

The robust variant does not use ground-truth target labels.
"""

from typing import Literal, Optional

import numpy as np
from scipy import linalg

from src.models.base import BaseDetector


class RXDetector(BaseDetector):
    """Global Reed-Xiaoli anomaly detector.

    Parameters
    ----------
    regularization:
        Relative diagonal covariance regularization.
    method:
        Solver: "cholesky", "solve", or "pinv".
    name:
        Detector name.
    reg:
        Compatibility alias for regularization.
    robust:
        If True, iteratively trims high-distance pixels when estimating the
        background mean and covariance.
    random_state:
        Optional seed used for deterministic initialization if subsampling
        is needed in future implementations. The current implementation is
        deterministic and does not require randomness.
    """

    def __init__(
        self,
        regularization: float = 1e-6,
        method: Literal["cholesky", "solve", "pinv"] = "cholesky",
        name: Optional[str] = "Global-RX",
        reg: Optional[float] = None,
        robust: bool = False,
        random_state: Optional[int] = None,
    ):
        super().__init__(name=name)

        if reg is not None:
            regularization = reg

        if regularization < 0:
            raise ValueError(
                "Regularization factor must be non-negative, "
                f"got {regularization}"
            )

        if method not in ("cholesky", "solve", "pinv"):
            raise ValueError(
                "Method must be 'cholesky', 'solve', or 'pinv', "
                f"got {method}"
            )

        self.regularization = float(regularization)
        self.method = method
        self.robust = bool(robust)
        self.random_state = random_state

        self.mean_: Optional[np.ndarray] = None
        self.cov_: Optional[np.ndarray] = None
        self.chol_lower_: Optional[np.ndarray] = None
        self.cov_inv_: Optional[np.ndarray] = None

    @staticmethod
    def _estimate_mean_covariance(
        x: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Estimate the mean and sample covariance of an (N, B) array."""
        mean = np.mean(x, axis=0)
        centered = x - mean

        if x.shape[0] > 1:
            cov = np.atleast_2d(np.cov(centered, rowvar=False, ddof=1))
        else:
            cov = np.zeros((x.shape[1], x.shape[1]), dtype=np.float64)

        return mean, cov

    def _regularize_covariance(self, cov: np.ndarray) -> np.ndarray:
        """Apply scale-aware diagonal regularization."""
        bands = cov.shape[0]
        trace = float(np.trace(cov))
        scale = trace / bands if bands > 0 and trace > 0 else 1.0

        cov_reg = cov + (
            np.eye(bands, dtype=np.float64)
            * self.regularization
            * scale
        )

        # Symmetrize to suppress tiny floating-point asymmetries.
        return 0.5 * (cov_reg + cov_reg.T)

    @staticmethod
    def _mahalanobis_scores(
        x: np.ndarray,
        mean: np.ndarray,
        cov: np.ndarray,
    ) -> np.ndarray:
        """Compute squared Mahalanobis distances with stable fallbacks."""
        centered = x - mean
        bands = cov.shape[0]

        try:
            chol = linalg.cholesky(cov, lower=True, check_finite=False)
            whitened = linalg.solve_triangular(
                chol,
                centered.T,
                lower=True,
                check_finite=False,
            )
            scores = np.sum(whitened * whitened, axis=0)
        except linalg.LinAlgError:
            try:
                solved = linalg.solve(
                    cov,
                    centered.T,
                    assume_a="sym",
                    check_finite=False,
                )
                scores = np.sum(centered.T * solved, axis=0)
            except linalg.LinAlgError:
                inverse = linalg.pinvh(cov)
                scores = np.sum((centered @ inverse) * centered, axis=1)

        return np.maximum(np.asarray(scores, dtype=np.float64), 0.0)

    def _fit_standard(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Fit the conventional global RX background model."""
        mean, cov = self._estimate_mean_covariance(x)
        return mean, self._regularize_covariance(cov)

    def _fit_robust(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Iteratively estimate background statistics with upper-tail trimming.

        This is a practical trimmed-covariance estimator, not an exact
        minimum-covariance-determinant estimator. It starts from standard
        statistics and repeatedly retains the lowest-distance 80% of pixels.
        The minimum retained sample count is chosen to keep covariance
        estimation meaningful for the number of spectral bands.
        """
        n_samples, n_bands = x.shape

        if n_samples < 3:
            return self._fit_standard(x)

        mean, cov = self._fit_standard(x)

        # Keep enough observations to estimate covariance, while trimming
        # high-distance pixels that can inflate global RX statistics.
        min_keep = min(n_samples, max(n_bands + 2, int(np.ceil(0.5 * n_samples))))
        keep_count = max(min_keep, int(np.floor(0.8 * n_samples)))
        keep_count = min(keep_count, n_samples)

        previous_indices = None

        for _ in range(10):
            distances = self._mahalanobis_scores(x, mean, cov)

            # Stable sorting makes the result deterministic when scores tie.
            indices = np.argsort(distances, kind="stable")[:keep_count]

            if (
                previous_indices is not None
                and np.array_equal(indices, previous_indices)
            ):
                break

            previous_indices = indices
            candidate_mean, candidate_cov = self._estimate_mean_covariance(
                x[indices]
            )
            mean = candidate_mean
            cov = self._regularize_covariance(candidate_cov)

        return mean, cov

    def fit(self, cube: np.ndarray) -> "RXDetector":
        """Fit the detector on a hyperspectral cube of shape (H, W, B)."""
        cube_val = self.validate_cube(cube)
        _, _, bands = cube_val.shape

        x = cube_val.reshape(-1, bands).astype(np.float64)

        if not np.all(np.isfinite(x)):
            raise ValueError("Input cube must contain only finite values.")

        # Clear all cached factors before refitting. This prevents factors
        # from a previous fit being accidentally reused.
        self.mean_ = None
        self.cov_ = None
        self.chol_lower_ = None
        self.cov_inv_ = None
        self.is_fitted = False

        if self.robust:
            mean, cov_reg = self._fit_robust(x)
        else:
            mean, cov_reg = self._fit_standard(x)

        self.mean_ = mean
        self.cov_ = cov_reg

        if self.method == "cholesky":
            try:
                self.chol_lower_ = linalg.cholesky(
                    cov_reg,
                    lower=True,
                    check_finite=False,
                )
            except linalg.LinAlgError:
                # Add scale-aware jitter, with a positive absolute floor.
                scale = max(
                    float(np.trace(cov_reg)) / bands
                    if bands > 0
                    else 1.0,
                    1.0,
                )
                jitter = max(
                    self.regularization * scale * 10.0,
                    np.finfo(np.float64).eps * scale * 100.0,
                )

                try:
                    cov_reg = cov_reg + np.eye(bands) * jitter
                    self.cov_ = cov_reg
                    self.chol_lower_ = linalg.cholesky(
                        cov_reg,
                        lower=True,
                        check_finite=False,
                    )
                except linalg.LinAlgError:
                    self.cov_inv_ = linalg.pinvh(cov_reg)
                    self.chol_lower_ = None

        elif self.method == "pinv":
            self.cov_inv_ = linalg.pinvh(cov_reg)

        # "solve" intentionally stores no factor; predict() solves directly.
        self.is_fitted = True
        return self

    def predict(self, cube: np.ndarray) -> np.ndarray:
        """Return the squared Mahalanobis anomaly score map."""
        if not self.is_fitted or self.mean_ is None or self.cov_ is None:
            raise RuntimeError("Detector must be fitted before calling predict.")

        cube_val = self.validate_cube(cube)
        height, width, bands = cube_val.shape

        if bands != self.mean_.size:
            raise ValueError(
                f"Spectral band count mismatch: fitted on "
                f"{self.mean_.size} bands, got {bands} bands."
            )

        x = cube_val.reshape(-1, bands).astype(np.float64)
        centered = x - self.mean_

        if self.chol_lower_ is not None:
            whitened = linalg.solve_triangular(
                self.chol_lower_,
                centered.T,
                lower=True,
                check_finite=False,
            )
            scores = np.sum(whitened * whitened, axis=0)

        elif self.cov_inv_ is not None:
            scores = np.sum(
                (centered @ self.cov_inv_) * centered,
                axis=1,
            )

        else:
            try:
                solved = linalg.solve(
                    self.cov_,
                    centered.T,
                    assume_a="sym",
                    check_finite=False,
                )
                scores = np.sum(centered.T * solved, axis=0)
            except linalg.LinAlgError:
                inverse = linalg.pinvh(self.cov_)
                scores = np.sum((centered @ inverse) * centered, axis=1)

        scores = np.maximum(np.asarray(scores, dtype=np.float64), 0.0)
        score_map = scores.reshape(height, width)

        return self.validate_score_map(
            score_map,
            expected_shape=(height, width),
        )

    def score(self, cube: np.ndarray) -> np.ndarray:
        """Compatibility alias for predict()."""
        return self.predict(cube)

    def fit_score(
        self,
        cube: np.ndarray,
        mask: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Fit and return scores.

        The mask parameter is accepted for API compatibility. Ground-truth
        labels are deliberately not used to estimate the background.
        """
        self.fit(cube)
        return self.predict(cube)

    def contributions(self, pixel: np.ndarray) -> np.ndarray:
        """Return per-band contributions to the RX quadratic-form score."""
        if not self.is_fitted or self.mean_ is None or self.cov_ is None:
            raise RuntimeError(
                "Detector must be fitted before calling contributions."
            )

        x = np.asarray(pixel, dtype=np.float64).reshape(-1)

        if x.size != self.mean_.size:
            raise ValueError(
                "Pixel band count does not match fitted detector."
            )

        delta = x - self.mean_

        if self.cov_inv_ is not None:
            inverse_delta = self.cov_inv_ @ delta
        elif self.chol_lower_ is not None:
            inverse_delta = linalg.cho_solve(
                (self.chol_lower_, True),
                delta,
                check_finite=False,
            )
        else:
            try:
                inverse_delta = linalg.solve(
                    self.cov_,
                    delta,
                    assume_a="sym",
                    check_finite=False,
                )
            except linalg.LinAlgError:
                inverse_delta = linalg.pinvh(self.cov_) @ delta

        return delta * inverse_delta