"""Base detector interface for hyperspectral anomaly and target detection.

This module defines the abstract base contract for all detectors in the
project. All models implemented by Member 1 or Member 2 must adhere to this
interface to ensure seamless interoperability with robustness testing,
profiling, and dashboard components.
"""

from abc import ABC, abstractmethod
from typing import Optional, Tuple, Union
import numpy as np


class BaseDetector(ABC):
    """Abstract Base Class for Hyperspectral Anomaly Detectors.

    Contract:
    - Input: Hyperspectral cube of shape (H, W, B) as numpy.ndarray.
    - Output: Anomaly score map of shape (H, W) as numpy.ndarray with dtype float.
      Higher scores strictly indicate higher likelihood of being anomalous.
    """

    def __init__(self, name: Optional[str] = None):
        self.name = name or self.__class__.__name__
        self.is_fitted = False

    def validate_cube(self, cube: np.ndarray) -> np.ndarray:
        """Validate and sanitize an input hyperspectral cube.

        Args:
            cube: Array representing the hyperspectral image.

        Returns:
            np.ndarray: Validated 3D float32 or float64 cube.

        Raises:
            TypeError: If input is not a numpy array.
            ValueError: If cube is not 3-dimensional, has non-positive dimensions,
                        or contains non-finite values (NaN / Inf).
        """
        if not isinstance(cube, np.ndarray):
            raise TypeError(f"Expected numpy.ndarray, got {type(cube).__name__}")

        if cube.ndim != 3:
            raise ValueError(f"Cube must be 3-dimensional (H, W, B), got shape {cube.shape}")

        h, w, b = cube.shape
        if h <= 0 or w <= 0 or b <= 0:
            raise ValueError(f"All dimensions must be positive, got shape ({h}, {w}, {b})")

        if not np.all(np.isfinite(cube)):
            raise ValueError("Input cube contains non-finite values (NaN or Inf)")

        # Convert integers or float16 to float32 for stable computation
        if not np.issubdtype(cube.dtype, np.floating):
            cube = cube.astype(np.float32)

        return cube

    def validate_score_map(self, scores: np.ndarray, expected_shape: Tuple[int, int]) -> np.ndarray:
        """Validate output anomaly score map.

        Args:
            scores: Computed 2D score map.
            expected_shape: Expected (H, W) spatial dimensions.

        Returns:
            np.ndarray: Validated 2D float score map.

        Raises:
            ValueError: If scores is not 2D, shape does not match (H, W),
                        or contains non-finite values.
        """
        if not isinstance(scores, np.ndarray):
            raise TypeError(f"Expected numpy.ndarray for scores, got {type(scores).__name__}")

        if scores.ndim != 2:
            raise ValueError(f"Score map must be 2-dimensional (H, W), got shape {scores.shape}")

        if scores.shape != expected_shape:
            raise ValueError(
                f"Score map shape {scores.shape} does not match expected spatial dimensions {expected_shape}"
            )

        if not np.all(np.isfinite(scores)):
            raise ValueError("Anomaly score map contains non-finite values (NaN or Inf)")

        return scores.astype(np.float64)

    @abstractmethod
    def fit(self, cube: np.ndarray) -> "BaseDetector":
        """Fit background model or learn representations from hyperspectral cube.

        Args:
            cube: Hyperspectral cube of shape (H, W, B).

        Returns:
            self: The fitted detector instance.
        """
        pass

    @abstractmethod
    def predict(self, cube: np.ndarray) -> np.ndarray:
        """Compute anomaly score map for a hyperspectral cube.

        Args:
            cube: Hyperspectral cube of shape (H, W, B).

        Returns:
            np.ndarray: Anomaly score map of shape (H, W).
                        Higher values indicate stronger anomaly evidence.
        """
        pass

    def fit_predict(self, cube: np.ndarray) -> np.ndarray:
        """Fit model and return anomaly score map for the same cube.

        Args:
            cube: Hyperspectral cube of shape (H, W, B).

        Returns:
            np.ndarray: Anomaly score map of shape (H, W).
        """
        return self.fit(cube).predict(cube)

    def __repr__(self) -> str:
        return f"{self.name}(fitted={self.is_fitted})"
