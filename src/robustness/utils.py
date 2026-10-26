"""Shared utility and validation helpers for robustness transformations."""

from typing import Optional, Tuple
import numpy as np


def check_cube(cube: np.ndarray) -> np.ndarray:
    """Validate that input is a 3D numpy array with finite values.

    Does not modify the original cube.

    Args:
        cube: Input hyperspectral array.

    Returns:
        np.ndarray: Validated cube.

    Raises:
        TypeError: If cube is not a numpy array.
        ValueError: If cube is not 3D, has non-positive dimensions, or contains NaNs/Infs.
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

    return cube


def get_rng(seed: Optional[int] = None) -> np.random.Generator:
    """Construct a reproducible numpy random Generator.

    Args:
        seed: Optional integer seed for reproducibility.

    Returns:
        np.random.Generator: Initialized random generator.
    """
    if seed is not None:
        if not isinstance(seed, (int, np.integer)):
            raise TypeError(f"Seed must be an integer, got {type(seed).__name__}")
        if seed < 0 or seed > 2**32 - 1:
            raise ValueError(f"Seed must be in [0, 2^32 - 1], got {seed}")
    return np.random.default_rng(seed)
