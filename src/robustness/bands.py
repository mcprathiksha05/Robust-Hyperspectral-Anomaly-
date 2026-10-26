"""Feature 3: Spectral band corruption transforms for hyperspectral imagery.

This module provides modular, non-destructive spectral band perturbations to
evaluate detector resilience against faulty detector channels, atmospheric
water absorption bands, and optical transmission losses.

All transforms adhere to the signature:
    transform(cube, ..., seed=None) -> (corrupted_cube, metadata)

Important Guarantees:
    - The original input cube is never modified in-place.
    - Spectral and spatial dimensions (H, W, B) are strictly preserved;
      bands are masked/zeroed/perturbed rather than silently sliced or deleted.
    - Wavelength metadata is never assumed to be present for every dataset.
"""

from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple, Union
import numpy as np

from src.robustness.utils import check_cube, get_rng


def drop_random_bands(
    cube: np.ndarray,
    num_bands: Optional[int] = None,
    fraction: float = 0.1,
    mode: Literal["zero", "mean", "noise"] = "zero",
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Corrupt randomly selected spectral bands while preserving (H, W, B) shape.

    Args:
        cube: Hyperspectral cube of shape (H, W, B). Original array is not modified.
        num_bands: Explicit number of bands to corrupt. If None, uses round(B * fraction).
        fraction: Proportion of bands to corrupt in (0.0, 1.0], used when num_bands is None.
        mode:
            - 'zero': Corrupted bands are filled with 0.0.
            - 'mean': Corrupted bands are filled with the cube's global mean.
            - 'noise': Corrupted bands are replaced with zero-mean Gaussian noise.
        seed: Optional random seed for reproducibility.

    Returns:
        Tuple[np.ndarray, Dict[str, Any]]:
            - corrupted_cube: Float64 cube with corrupted spectral bands.
            - metadata: Dictionary detailing corrupted_bands boolean mask, indices, and mode.
    """
    check_cube(cube)
    rng = get_rng(seed)
    h, w, b = cube.shape
    corrupted = cube.astype(np.float64, copy=True)

    if num_bands is not None:
        if not (1 <= num_bands <= b):
            raise ValueError(f"num_bands must be in [1, {b}], got {num_bands}")
        k = num_bands
    else:
        if not (0.0 < fraction <= 1.0):
            raise ValueError(f"fraction must be in (0, 1], got {fraction}")
        k = max(1, int(round(b * fraction)))

    if mode not in ("zero", "mean", "noise"):
        raise ValueError(f"mode must be 'zero', 'mean', or 'noise', got {mode}")

    selected_bands = sorted(rng.choice(b, size=k, replace=False).tolist())
    corrupted_mask = np.zeros(b, dtype=bool)
    corrupted_mask[selected_bands] = True

    if mode == "zero":
        corrupted[:, :, selected_bands] = 0.0
    elif mode == "mean":
        mean_val = float(np.mean(cube))
        corrupted[:, :, selected_bands] = mean_val
    elif mode == "noise":
        band_std = float(np.std(cube))
        noise = rng.normal(loc=0.0, scale=band_std, size=(h, w, k))
        corrupted[:, :, selected_bands] = noise

    metadata: Dict[str, Any] = {
        "transform": "random_band_dropout",
        "params": {"num_bands": k, "fraction": float(k / b), "mode": mode},
        "seed": seed,
        "corrupted_bands": corrupted_mask,
        "corrupted_band_indices": selected_bands,
        "num_corrupted_bands": k,
        "total_bands": b,
    }
    return corrupted, metadata


def mask_selected_bands(
    cube: np.ndarray,
    band_indices: Sequence[int],
    mode: Literal["zero", "mean"] = "zero",
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Mask explicitly specified spectral band indices while preserving (H, W, B) shape.

    Args:
        cube: Hyperspectral cube of shape (H, W, B). Original array is not modified.
        band_indices: List or sequence of 0-based band indices to mask.
        mode: 'zero' or 'mean'.

    Returns:
        Tuple[np.ndarray, Dict[str, Any]]:
            - corrupted_cube: Float64 cube with specified bands masked.
            - metadata: Dictionary detailing affected band indices and boolean mask.
    """
    check_cube(cube)
    h, w, b = cube.shape
    corrupted = cube.astype(np.float64, copy=True)

    if len(band_indices) == 0:
        raise ValueError("band_indices must not be empty")

    indices = list(set(int(idx) for idx in band_indices))
    for idx in indices:
        if not (0 <= idx < b):
            raise IndexError(f"Band index {idx} out of bounds for cube with {b} bands")

    corrupted_mask = np.zeros(b, dtype=bool)
    corrupted_mask[indices] = True

    if mode == "zero":
        corrupted[:, :, indices] = 0.0
    elif mode == "mean":
        corrupted[:, :, indices] = float(np.mean(cube))
    else:
        raise ValueError(f"mode must be 'zero' or 'mean', got {mode}")

    metadata: Dict[str, Any] = {
        "transform": "selected_band_masking",
        "params": {"mode": mode},
        "seed": None,
        "corrupted_bands": corrupted_mask,
        "corrupted_band_indices": sorted(indices),
        "num_corrupted_bands": len(indices),
        "total_bands": b,
    }
    return corrupted, metadata


def add_localized_band_noise(
    cube: np.ndarray,
    band_indices: Optional[Sequence[int]] = None,
    band_fraction: float = 0.15,
    snr_db: float = 10.0,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Inject severe Gaussian noise into a localized subset of spectral channels.

    Simulates poor detector quantum efficiency, channel-specific thermal drift,
    or low optical throughput on specific sensor elements.

    Args:
        cube: Hyperspectral cube of shape (H, W, B). Original array is not modified.
        band_indices: Optional explicit list of bands to corrupt. If None, random bands are picked.
        band_fraction: Fraction of bands to corrupt when band_indices is None.
        snr_db: Signal-to-noise ratio in dB for the noisy bands (lower = more severe).
        seed: Optional random seed for reproducibility.

    Returns:
        Tuple[np.ndarray, Dict[str, Any]]:
            - corrupted_cube: Float64 cube with localized noisy bands.
            - metadata: Dictionary detailing affected bands and noise power.
    """
    check_cube(cube)
    if snr_db <= 0:
        raise ValueError(f"snr_db must be positive, got {snr_db}")

    rng = get_rng(seed)
    h, w, b = cube.shape
    corrupted = cube.astype(np.float64, copy=True)

    if band_indices is not None:
        indices = sorted(set(int(idx) for idx in band_indices))
        for idx in indices:
            if not (0 <= idx < b):
                raise IndexError(f"Band index {idx} out of bounds for {b} bands")
    else:
        if not (0.0 < band_fraction <= 1.0):
            raise ValueError(f"band_fraction must be in (0, 1], got {band_fraction}")
        k = max(1, int(round(b * band_fraction)))
        indices = sorted(rng.choice(b, size=k, replace=False).tolist())

    corrupted_mask = np.zeros(b, dtype=bool)
    corrupted_mask[indices] = True

    # Compute signal power per affected band and inject Gaussian noise
    for idx in indices:
        band_data = corrupted[:, :, idx]
        p_signal = max(float(np.mean(band_data**2)), 1e-12)
        p_noise = p_signal / (10.0 ** (snr_db / 10.0))
        noise = rng.normal(loc=0.0, scale=np.sqrt(p_noise), size=(h, w))
        corrupted[:, :, idx] += noise

    metadata: Dict[str, Any] = {
        "transform": "localized_band_noise",
        "params": {"snr_db": float(snr_db), "band_fraction": float(len(indices) / b)},
        "seed": seed,
        "corrupted_bands": corrupted_mask,
        "corrupted_band_indices": indices,
        "num_corrupted_bands": len(indices),
        "total_bands": b,
    }
    return corrupted, metadata


def mask_wavelength_ranges(
    cube: np.ndarray,
    wavelengths: Optional[Sequence[float]] = None,
    ranges: Sequence[Tuple[float, float]] = ((1350.0, 1420.0), (1800.0, 1950.0)),
    mode: Literal["zero", "mean"] = "zero",
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Mask spectral bands falling within designated physical wavelength intervals.

    Typically used to mask atmospheric absorption windows (e.g. water vapor H2O
    absorption near 1400 nm and 1900 nm).

    Requirement:
        Wavelength metadata must be provided explicitly. If None, raises a clear
        informative ValueError rather than silently guessing or failing.

    Args:
        cube: Hyperspectral cube of shape (H, W, B). Original array is not modified.
        wavelengths: 1D array-like of band center wavelengths of length B.
        ranges: Sequence of (min_wavelength, max_wavelength) intervals.
        mode: 'zero' or 'mean'.

    Returns:
        Tuple[np.ndarray, Dict[str, Any]]:
            - corrupted_cube: Float64 cube with absorption bands masked.
            - metadata: Dictionary detailing matched bands, wavelengths, and ranges.

    Raises:
        ValueError: If wavelengths is None or length mismatches cube bands.
    """
    check_cube(cube)
    h, w, b = cube.shape

    if wavelengths is None:
        raise ValueError(
            "Wavelength metadata is required for wavelength-based masking but was not provided. "
            "Please supply the wavelengths array (e.g. from MUUFL metadata) or use "
            "mask_selected_bands with explicit band indices."
        )

    w_arr = np.asarray(wavelengths, dtype=np.float64).ravel()
    if len(w_arr) != b:
        raise ValueError(
            f"Wavelength array length {len(w_arr)} does not match cube band count {b}"
        )

    matched_indices: List[int] = []
    for low, high in ranges:
        if low >= high:
            raise ValueError(f"Invalid wavelength range ({low}, {high}): low must be < high")
        in_range = np.where((w_arr >= low) & (w_arr <= high))[0]
        matched_indices.extend(in_range.tolist())

    matched_indices = sorted(set(matched_indices))

    if len(matched_indices) == 0:
        # None of the bands fell within the specified intervals; cube unchanged
        corrupted = cube.astype(np.float64, copy=True)
        corrupted_mask = np.zeros(b, dtype=bool)
    else:
        corrupted, _ = mask_selected_bands(cube, band_indices=matched_indices, mode=mode)
        corrupted_mask = np.zeros(b, dtype=bool)
        corrupted_mask[matched_indices] = True

    metadata: Dict[str, Any] = {
        "transform": "wavelength_range_masking",
        "params": {"ranges": list(ranges), "mode": mode},
        "seed": None,
        "corrupted_bands": corrupted_mask,
        "corrupted_band_indices": matched_indices,
        "num_corrupted_bands": len(matched_indices),
        "total_bands": b,
    }
    return corrupted, metadata
