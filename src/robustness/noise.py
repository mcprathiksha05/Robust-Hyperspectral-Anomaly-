"""Feature 1: Noise robustness testing transforms for hyperspectral imagery.

This module provides modular, non-destructive noise corruptions to stress-test
anomaly and target detection algorithms under realistic sensor and environmental
degradations.

All transforms adhere to the signature:
    transform(cube, ..., seed=None) -> (corrupted_cube, metadata)

Supported transforms:
    - Additive Gaussian noise (SNR in dB, global or per-band)
    - Poisson / shot noise (quantum photon arrival statistics)
    - Stripe noise (pushbroom calibration drift across columns)
    - Dead pixels (zeroed detector response)
    - Saturated pixels (sensor bloom / stuck hot pixels)
    - Salt-and-pepper noise (impulse transmission noise)
"""

from typing import Any, Dict, List, Literal, Optional, Tuple, Union
import numpy as np

from src.robustness.utils import check_cube, get_rng


def add_gaussian_noise(
    cube: np.ndarray,
    snr_db: float,
    per_band: bool = False,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Add additive zero-mean Gaussian noise at a designated Signal-to-Noise Ratio (SNR).

    Definition:
        SNR_dB = 10 * log10(P_signal / P_noise)
        where P_signal is the mean square signal intensity:
            P_signal = (1 / (H * W * B)) * sum(x^2)
        and P_noise = P_signal / (10^(SNR_dB / 10)).
        Additive noise is drawn from N(0, P_noise).

    Args:
        cube: Hyperspectral cube of shape (H, W, B). Original array is not modified.
        snr_db: Desired SNR in decibels (e.g. 30 dB for moderate noise, 15 dB for severe).
        per_band: If True, computes P_signal and injects noise independently per spectral band.
                  If False, computes global P_signal across the entire cube.
        seed: Optional random seed for reproducible noise generation.

    Returns:
        Tuple[np.ndarray, Dict[str, Any]]:
            - corrupted_cube: Float64 copy of cube with added noise, shape (H, W, B).
            - metadata: Dictionary containing transform parameters, effective SNR, and noise statistics.
    """
    check_cube(cube)
    if snr_db <= 0:
        raise ValueError(f"SNR in dB must be positive, got {snr_db}")

    rng = get_rng(seed)
    h, w, b = cube.shape
    corrupted = cube.astype(np.float64, copy=True)

    if per_band:
        # Calculate signal power per band: shape (B,)
        p_signal = np.mean(corrupted**2, axis=(0, 1))
        # Guard against zero power bands
        p_signal = np.maximum(p_signal, 1e-12)
        p_noise = p_signal / (10.0 ** (snr_db / 10.0))
        noise_std = np.sqrt(p_noise)
        # Generate noise with shape (H, W, B)
        noise = rng.normal(loc=0.0, scale=1.0, size=(h, w, b)) * noise_std
    else:
        p_signal = float(np.mean(corrupted**2))
        p_signal = max(p_signal, 1e-12)
        p_noise = p_signal / (10.0 ** (snr_db / 10.0))
        noise_std = float(np.sqrt(p_noise))
        noise = rng.normal(loc=0.0, scale=noise_std, size=(h, w, b))

    corrupted += noise

    # Measure actual generated noise power
    mean_p_signal = float(np.mean(p_signal))
    actual_noise_power = float(np.mean(noise**2))
    measured_snr_db = (
        float(10.0 * np.log10(mean_p_signal / actual_noise_power))
        if actual_noise_power > 0
        else float("inf")
    )

    metadata: Dict[str, Any] = {
        "transform": "additive_gaussian_noise",
        "params": {"snr_db": float(snr_db), "per_band": bool(per_band)},
        "seed": seed,
        "signal_power": float(np.mean(p_signal)),
        "target_snr_db": float(snr_db),
        "measured_snr_db": measured_snr_db,
        "noise_std_mean": float(np.mean(noise_std)),
    }
    return corrupted, metadata


def add_poisson_noise(
    cube: np.ndarray,
    photon_scale: float = 1000.0,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Simulate Poisson / photon shot noise on a hyperspectral cube.

    Physical Scaling Assumption & Convention:
        In photodetectors, photon arrival follows a Poisson distribution:
            k ~ Poisson(lambda), with Var(k) = lambda.
        Hyperspectral cubes stored as reflectance [0, 1] or radiance in arbitrary
        units must be mapped to an effective photon count to exhibit realistic
        shot-noise variance.
        The `photon_scale` parameter defines how many photons correspond to unit intensity:
            lambda = max(cube, 0) * photon_scale
            k ~ Poisson(lambda)
            corrupted = k / photon_scale

    Args:
        cube: Hyperspectral cube of shape (H, W, B). Original array is not modified.
        photon_scale: Scaling factor mapping cube values to expected photon counts.
                      Higher scale = lower relative noise (higher SNR). Default is 1000.0.
        seed: Optional random seed for reproducibility.

    Returns:
        Tuple[np.ndarray, Dict[str, Any]]:
            - corrupted_cube: Float64 cube corrupted with Poisson shot noise.
            - metadata: Dictionary detailing photon_scale, seed, and noise statistics.
    """
    check_cube(cube)
    if photon_scale <= 0:
        raise ValueError(f"photon_scale must be strictly positive, got {photon_scale}")

    rng = get_rng(seed)
    corrupted = cube.astype(np.float64, copy=True)

    # Floor at zero to ensure non-negative Poisson intensity parameter
    lam = np.maximum(corrupted, 0.0) * float(photon_scale)

    # Generate Poisson counts
    counts = rng.poisson(lam=lam)
    corrupted = counts.astype(np.float64) / float(photon_scale)

    metadata: Dict[str, Any] = {
        "transform": "poisson_noise",
        "params": {"photon_scale": float(photon_scale)},
        "seed": seed,
        "intensity_min": float(np.min(corrupted)),
        "intensity_max": float(np.max(corrupted)),
    }
    return corrupted, metadata


def add_stripe_noise(
    cube: np.ndarray,
    stripe_density: float = 0.05,
    band_fraction: float = 0.3,
    stripe_intensity: float = 0.2,
    mode: Literal["multiplicative", "additive"] = "multiplicative",
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Simulate vertical stripe noise caused by pushbroom sensor calibration drift.

    Args:
        cube: Hyperspectral cube of shape (H, W, B). Original array is not modified.
        stripe_density: Fraction of image columns affected by striping, in (0.0, 1.0].
        band_fraction: Fraction of spectral bands affected by striping, in (0.0, 1.0].
        stripe_intensity: Relative gain or offset magnitude.
                          In multiplicative mode: column is scaled by (1 + delta), delta in [-intensity, +intensity].
                          In additive mode: column is shifted by delta * band_std.
        mode: Either 'multiplicative' or 'additive'.
        seed: Optional random seed for reproducibility.

    Returns:
        Tuple[np.ndarray, Dict[str, Any]]:
            - corrupted_cube: Float64 cube with vertical stripe artifacts.
            - metadata: Dictionary detailing affected columns, affected bands, and parameters.
    """
    check_cube(cube)
    if not (0.0 < stripe_density <= 1.0):
        raise ValueError(f"stripe_density must be in (0, 1], got {stripe_density}")
    if not (0.0 < band_fraction <= 1.0):
        raise ValueError(f"band_fraction must be in (0, 1], got {band_fraction}")
    if stripe_intensity <= 0:
        raise ValueError(f"stripe_intensity must be positive, got {stripe_intensity}")
    if mode not in ("multiplicative", "additive"):
        raise ValueError(f"mode must be 'multiplicative' or 'additive', got {mode}")

    rng = get_rng(seed)
    h, w, b = cube.shape
    corrupted = cube.astype(np.float64, copy=True)

    n_cols = max(1, int(round(w * stripe_density)))
    cols = sorted(rng.choice(w, size=n_cols, replace=False).tolist())

    n_bands = max(1, int(round(b * band_fraction)))
    bands = sorted(rng.choice(b, size=n_bands, replace=False).tolist())

    band_stds = np.std(corrupted, axis=(0, 1))

    for c in cols:
        for b_idx in bands:
            delta = rng.uniform(-stripe_intensity, stripe_intensity)
            if mode == "multiplicative":
                corrupted[:, c, b_idx] *= (1.0 + delta)
            else:
                corrupted[:, c, b_idx] += delta * (band_stds[b_idx] if band_stds[b_idx] > 0 else 1.0)

    metadata: Dict[str, Any] = {
        "transform": "stripe_noise",
        "params": {
            "stripe_density": float(stripe_density),
            "band_fraction": float(band_fraction),
            "stripe_intensity": float(stripe_intensity),
            "mode": mode,
        },
        "seed": seed,
        "affected_columns": cols,
        "affected_bands": bands,
        "num_affected_columns": len(cols),
        "num_affected_bands": len(bands),
        "num_striped_columns": len(cols),
        "num_striped_bands": len(bands),
    }
    return corrupted, metadata


def add_dead_pixels(
    cube: np.ndarray,
    pixel_fraction: float = 0.005,
    mode: Literal["all_bands", "random_bands"] = "all_bands",
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Simulate dead photodetector sensor pixels (zero output).

    Args:
        cube: Hyperspectral cube of shape (H, W, B). Original array is not modified.
        pixel_fraction: Fraction of spatial pixels corrupted, in (0.0, 1.0].
        mode:
            - 'all_bands': Selected spatial pixels are zeroed across all B bands.
            - 'random_bands': For each selected pixel, a random subset of bands is zeroed.
        seed: Optional random seed for reproducibility.

    Returns:
        Tuple[np.ndarray, Dict[str, Any]]:
            - corrupted_cube: Float64 cube with dead pixels set to 0.0.
            - metadata: Dictionary detailing affected pixel coordinates and parameters.
    """
    check_cube(cube)
    if not (0.0 < pixel_fraction <= 1.0):
        raise ValueError(f"pixel_fraction must be in (0, 1], got {pixel_fraction}")
    if mode not in ("all_bands", "random_bands"):
        raise ValueError(f"mode must be 'all_bands' or 'random_bands', got {mode}")

    rng = get_rng(seed)
    h, w, b = cube.shape
    corrupted = cube.astype(np.float64, copy=True)

    n_pixels = max(1, int(round(h * w * pixel_fraction)))
    flat_indices = rng.choice(h * w, size=n_pixels, replace=False)
    rows, cols = np.unravel_index(flat_indices, (h, w))

    if mode == "all_bands":
        corrupted[rows, cols, :] = 0.0
    else:
        for r, c in zip(rows, cols):
            # Zero out 20% to 80% of bands at this pixel
            k = rng.integers(1, b + 1)
            b_sel = rng.choice(b, size=k, replace=False)
            corrupted[r, c, b_sel] = 0.0

    metadata: Dict[str, Any] = {
        "transform": "dead_pixels",
        "params": {"pixel_fraction": float(pixel_fraction), "mode": mode},
        "seed": seed,
        "num_dead_pixels": int(n_pixels),
        "pixel_coordinates": list(zip(rows.tolist(), cols.tolist())),
    }
    return corrupted, metadata


def add_saturated_pixels(
    cube: np.ndarray,
    pixel_fraction: float = 0.005,
    saturation_value: Optional[float] = None,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Simulate sensor pixel saturation / hot-pixel blooming at max dynamic range.

    Args:
        cube: Hyperspectral cube of shape (H, W, B). Original array is not modified.
        pixel_fraction: Fraction of spatial pixels saturated, in (0.0, 1.0].
        saturation_value: Explicit saturation value. If None, defaults to 1.5 * max(cube).
        seed: Optional random seed for reproducibility.

    Returns:
        Tuple[np.ndarray, Dict[str, Any]]:
            - corrupted_cube: Float64 cube with saturated pixels.
            - metadata: Dictionary detailing saturated coordinates and saturation value.
    """
    check_cube(cube)
    if not (0.0 < pixel_fraction <= 1.0):
        raise ValueError(f"pixel_fraction must be in (0, 1], got {pixel_fraction}")

    rng = get_rng(seed)
    h, w, b = cube.shape
    corrupted = cube.astype(np.float64, copy=True)

    sat_val = (
        float(saturation_value)
        if saturation_value is not None
        else float(np.max(cube) * 1.5)
    )

    n_pixels = max(1, int(round(h * w * pixel_fraction)))
    flat_indices = rng.choice(h * w, size=n_pixels, replace=False)
    rows, cols = np.unravel_index(flat_indices, (h, w))

    corrupted[rows, cols, :] = sat_val

    metadata: Dict[str, Any] = {
        "transform": "saturated_pixels",
        "params": {
            "pixel_fraction": float(pixel_fraction),
            "saturation_value": sat_val,
        },
        "seed": seed,
        "num_saturated_pixels": int(n_pixels),
        "pixel_coordinates": list(zip(rows.tolist(), cols.tolist())),
    }
    return corrupted, metadata


def add_salt_and_pepper_noise(
    cube: np.ndarray,
    amount: float = 0.01,
    salt_vs_pepper: float = 0.5,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Simulate impulse transmission / salt-and-pepper noise.

    Args:
        cube: Hyperspectral cube of shape (H, W, B). Original array is not modified.
        amount: Proportion of the image pixels to corrupt, in (0.0, 1.0].
        salt_vs_pepper: Ratio of salt (max value) to pepper (min value), in [0.0, 1.0].
        seed: Optional random seed for reproducibility.

    Returns:
        Tuple[np.ndarray, Dict[str, Any]]:
            - corrupted_cube: Float64 cube with impulse noise.
            - metadata: Dictionary detailing affected count, bounds, and parameters.
    """
    check_cube(cube)
    if not (0.0 < amount <= 1.0):
        raise ValueError(f"amount must be in (0, 1], got {amount}")
    if not (0.0 <= salt_vs_pepper <= 1.0):
        raise ValueError(f"salt_vs_pepper must be in [0, 1], got {salt_vs_pepper}")

    rng = get_rng(seed)
    h, w, b = cube.shape
    corrupted = cube.astype(np.float64, copy=True)

    min_val = float(np.min(cube))
    max_val = float(np.max(cube))

    n_corrupt = max(1, int(round(h * w * amount)))
    flat_indices = rng.choice(h * w, size=n_corrupt, replace=False)
    rows, cols = np.unravel_index(flat_indices, (h, w))

    n_salt = int(round(n_corrupt * salt_vs_pepper))
    salt_r, salt_c = rows[:n_salt], cols[:n_salt]
    pepper_r, pepper_c = rows[n_salt:], cols[n_salt:]

    if len(salt_r) > 0:
        corrupted[salt_r, salt_c, :] = max_val
    if len(pepper_r) > 0:
        corrupted[pepper_r, pepper_c, :] = min_val

    metadata: Dict[str, Any] = {
        "transform": "salt_and_pepper",
        "params": {"amount": float(amount), "salt_vs_pepper": float(salt_vs_pepper)},
        "seed": seed,
        "num_corrupted_pixels": int(n_corrupt),
        "num_salt_pixels": int(n_salt),
        "num_pepper_pixels": int(n_corrupt - n_salt),
        "min_value": min_val,
        "max_value": max_val,
    }
    return corrupted, metadata
