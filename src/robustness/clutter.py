"""Feature 2: Clutter robustness testing transforms for hyperspectral imagery.

This module provides modular, reproducible synthetic clutter and occlusion
perturbations designed to evaluate detector robustness against non-target spectral
interference, background camouflage, shadows, and canopy obstruction.

Disclaimer:
    Synthetic spectral mixing and spatial attenuation provide controlled,
    parametric stress tests for detector robustness; they do NOT claim to model
    full nonlinear radiative transfer physics, bidirectional reflectance (BRDF),
    or physical 3D camouflage fabrics.

All transforms adhere to the signature:
    transform(cube, ..., seed=None) -> (corrupted_cube, metadata)

Supported transforms:
    - Background spectral mixing (linear unmixing stress-testing)
    - Spatially localized clutter patch injection
    - Shadow-like illumination attenuation
    - Localized occlusion simulation
    - MUUFL-specific target occlusion metadata filtering
"""

from typing import Any, Dict, List, Literal, Optional, Tuple, Union
import numpy as np

from src.robustness.utils import check_cube, get_rng


def inject_spectral_clutter(
    cube: np.ndarray,
    clutter_spectrum: Optional[np.ndarray] = None,
    abundance: float = 0.3,
    spatial_fraction: float = 0.05,
    patch_size: Optional[Tuple[int, int]] = None,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Inject background spectral clutter via linear mixing.

    Formulation:
        x_mixed = (1 - alpha) * x + alpha * c
        where alpha in [0, 1] is clutter abundance, and c in R^B is the clutter spectrum.

    Args:
        cube: Hyperspectral cube of shape (H, W, B). Original array is not modified.
        clutter_spectrum: 1D array of shape (B,) representing the clutter endmember.
                          If None, a synthetic clutter spectrum is estimated from the
                          high-variance components of the background.
        abundance: Clutter mixing abundance alpha in (0.0, 1.0].
        spatial_fraction: Proportion of spatial pixels to perturb, in (0.0, 1.0].
        patch_size: Optional (patch_h, patch_w) tuple to inject contiguous clutter patches
                    instead of scattered pixels.
        seed: Optional random seed for reproducibility.

    Returns:
        Tuple[np.ndarray, Dict[str, Any]]:
            - corrupted_cube: Float64 cube with injected clutter.
            - metadata: Dictionary detailing clutter mask, abundance, and parameters.
    """
    check_cube(cube)
    if not (0.0 < abundance <= 1.0):
        raise ValueError(f"abundance must be in (0, 1], got {abundance}")
    if not (0.0 < spatial_fraction <= 1.0):
        raise ValueError(f"spatial_fraction must be in (0, 1], got {spatial_fraction}")

    rng = get_rng(seed)
    h, w, b = cube.shape
    corrupted = cube.astype(np.float64, copy=True)

    # 1. Determine clutter endmember spectrum c of shape (B,)
    if clutter_spectrum is not None:
        c_spec = np.asarray(clutter_spectrum, dtype=np.float64).ravel()
        if len(c_spec) != b:
            raise ValueError(
                f"clutter_spectrum dimension mismatch: expected {b} bands, got {len(c_spec)}"
            )
    else:
        # Synthesize a realistic background clutter endmember:
        # Use mean spectrum of random spatial patch plus moderate spectral perturbation
        r_rand = rng.integers(0, max(1, h - 5))
        c_rand = rng.integers(0, max(1, w - 5))
        local_sample = cube[r_rand : r_rand + 5, c_rand : c_rand + 5, :]
        c_spec = np.mean(local_sample, axis=(0, 1)).astype(np.float64)

    # 2. Build 2D clutter mask
    clutter_mask = np.zeros((h, w), dtype=bool)

    if patch_size is not None:
        ph, pw = patch_size
        if ph <= 0 or pw <= 0:
            raise ValueError(f"patch_size dimensions must be positive, got {patch_size}")
        ph = min(ph, h)
        pw = min(pw, w)
        pixels_per_patch = ph * pw
        total_target_pixels = int(round(h * w * spatial_fraction))
        n_patches = max(1, total_target_pixels // pixels_per_patch)

        for _ in range(n_patches):
            pr = rng.integers(0, h - ph + 1)
            pc = rng.integers(0, w - pw + 1)
            clutter_mask[pr : pr + ph, pc : pc + pw] = True
    else:
        n_pixels = max(1, int(round(h * w * spatial_fraction)))
        flat_idx = rng.choice(h * w, size=n_pixels, replace=False)
        rows, cols = np.unravel_index(flat_idx, (h, w))
        clutter_mask[rows, cols] = True

    # 3. Apply linear mixing on selected pixels
    # x_new = (1 - alpha) * x + alpha * c
    mask_3d = clutter_mask[:, :, np.newaxis]
    corrupted = np.where(mask_3d, (1.0 - abundance) * corrupted + abundance * c_spec, corrupted)

    metadata: Dict[str, Any] = {
        "transform": "spectral_clutter_mixing",
        "params": {
            "abundance": float(abundance),
            "spatial_fraction": float(spatial_fraction),
            "patch_size": patch_size,
        },
        "seed": seed,
        "clutter_mask": clutter_mask,
        "num_clutter_pixels": int(np.sum(clutter_mask)),
        "clutter_spectrum_mean": float(np.mean(c_spec)),
    }
    return corrupted, metadata


def inject_localized_clutter_patch(
    cube: np.ndarray,
    bounds: Optional[Tuple[int, int, int, int]] = None,
    clutter_spectrum: Optional[np.ndarray] = None,
    abundance: float = 0.5,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Inject a contiguous rectangular clutter patch into an explicit or random bounding box.

    Args:
        cube: Hyperspectral cube of shape (H, W, B). Original array is not modified.
        bounds: Optional (row_min, row_max, col_min, col_max) integer bounds.
                If None, selects a random (H/4, W/4) bounding box.
        clutter_spectrum: Optional (B,) endmember spectrum.
        abundance: Mixing abundance in (0.0, 1.0].
        seed: Optional random seed for reproducibility.

    Returns:
        Tuple[np.ndarray, Dict[str, Any]]:
            - corrupted_cube: Float64 cube with localized clutter patch.
            - metadata: Dictionary detailing patch coordinates and abundance.
    """
    check_cube(cube)
    if not (0.0 < abundance <= 1.0):
        raise ValueError(f"abundance must be in (0, 1], got {abundance}")

    rng = get_rng(seed)
    h, w, b = cube.shape
    corrupted = cube.astype(np.float64, copy=True)

    if bounds is None:
        ph, pw = max(2, h // 4), max(2, w // 4)
        r0 = rng.integers(0, h - ph + 1)
        c0 = rng.integers(0, w - pw + 1)
        r1, c1 = r0 + ph, c0 + pw
    else:
        r0, r1, c0, c1 = bounds
        if not (0 <= r0 < r1 <= h and 0 <= c0 < c1 <= w):
            raise ValueError(f"Invalid bounding box {bounds} for cube of shape ({h}, {w})")

    if clutter_spectrum is not None:
        c_spec = np.asarray(clutter_spectrum, dtype=np.float64).ravel()
        if len(c_spec) != b:
            raise ValueError(f"Expected clutter spectrum of length {b}, got {len(c_spec)}")
    else:
        # Use background mean spectrum from outside the patch
        c_spec = np.mean(cube, axis=(0, 1)).astype(np.float64) * 0.8

    patch = corrupted[r0:r1, c0:c1, :]
    corrupted[r0:r1, c0:c1, :] = (1.0 - abundance) * patch + abundance * c_spec

    clutter_mask = np.zeros((h, w), dtype=bool)
    clutter_mask[r0:r1, c0:c1] = True

    metadata: Dict[str, Any] = {
        "transform": "localized_clutter_patch",
        "params": {"bounds": (int(r0), int(r1), int(c0), int(c1)), "abundance": float(abundance)},
        "seed": seed,
        "clutter_mask": clutter_mask,
        "num_clutter_pixels": int((r1 - r0) * (c1 - c0)),
    }
    return corrupted, metadata


def apply_shadow_attenuation(
    cube: np.ndarray,
    attenuation_factor: float = 0.35,
    region_fraction: float = 0.15,
    blur_boundary: bool = True,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Simulate cloud, terrain, or canopy shadow illumination attenuation.

    In shadowed regions, direct sunlight is occluded, leaving diffuse skylight
    and resulting in an intensity reduction by attenuation_factor gamma in (0, 1].
    Formula:
        x_shadow = gamma * x

    Args:
        cube: Hyperspectral cube of shape (H, W, B). Original array is not modified.
        attenuation_factor: Residual intensity ratio gamma in (0.0, 1.0]. Lower = darker shadow.
        region_fraction: Approximate fraction of scene covered by shadow, in (0.0, 1.0].
        blur_boundary: If True, applies penumbra gradient along shadow edge.
        seed: Optional random seed for reproducibility.

    Returns:
        Tuple[np.ndarray, Dict[str, Any]]:
            - corrupted_cube: Float64 cube with shadowed region.
            - metadata: Dictionary detailing shadow mask, attenuation, and affected pixel count.
    """
    check_cube(cube)
    if not (0.0 < attenuation_factor <= 1.0):
        raise ValueError(f"attenuation_factor must be in (0, 1], got {attenuation_factor}")
    if not (0.0 < region_fraction <= 1.0):
        raise ValueError(f"region_fraction must be in (0, 1], got {region_fraction}")

    rng = get_rng(seed)
    h, w, b = cube.shape
    corrupted = cube.astype(np.float64, copy=True)

    # Generate an elliptical / blob shadow region
    center_r = rng.integers(h // 4, 3 * h // 4)
    center_c = rng.integers(w // 4, 3 * w // 4)
    radius_r = max(2, int(np.sqrt(h * w * region_fraction / np.pi) * 0.8))
    radius_c = max(2, int(np.sqrt(h * w * region_fraction / np.pi) * 1.2))

    y_grid, x_grid = np.ogrid[:h, :w]
    dist_sq = ((y_grid - center_r) / radius_r) ** 2 + ((x_grid - center_c) / radius_c) ** 2

    if blur_boundary:
        # Penumbra transition between dist_sq in [0.8, 1.2]
        # Weight 1.0 inside shadow (dark), 0.0 outside
        shadow_weight = np.clip((1.2 - dist_sq) / 0.4, 0.0, 1.0)
    else:
        shadow_weight = (dist_sq <= 1.0).astype(np.float64)

    # Multiplicative attenuation: x * (1 - shadow_weight * (1 - gamma))
    multiplier = 1.0 - shadow_weight * (1.0 - attenuation_factor)
    corrupted *= multiplier[:, :, np.newaxis]

    shadow_mask = shadow_weight > 0.5

    metadata: Dict[str, Any] = {
        "transform": "shadow_attenuation",
        "params": {
            "attenuation_factor": float(attenuation_factor),
            "region_fraction": float(region_fraction),
            "blur_boundary": bool(blur_boundary),
        },
        "seed": seed,
        "shadow_mask": shadow_mask,
        "num_shadow_pixels": int(np.sum(shadow_mask)),
        "center": (int(center_r), int(center_c)),
    }
    return corrupted, metadata


def apply_occlusion_simulation(
    cube: np.ndarray,
    occlusion_type: Literal["zero", "mean", "clutter"] = "zero",
    spatial_fraction: float = 0.05,
    patch_size: Tuple[int, int] = (5, 5),
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Simulate localized physical obstructions (tree branches, overhead netting, foliage).

    Args:
        cube: Hyperspectral cube of shape (H, W, B). Original array is not modified.
        occlusion_type:
            - 'zero': Occluded pixels are blocked to zero.
            - 'mean': Occluded pixels are replaced with scene mean spectrum.
            - 'clutter': Occluded pixels are replaced with high-frequency noise clutter.
        spatial_fraction: Fraction of scene occluded, in (0.0, 1.0].
        patch_size: (H_p, W_p) size of individual occlusion patches.
        seed: Optional random seed for reproducibility.

    Returns:
        Tuple[np.ndarray, Dict[str, Any]]:
            - corrupted_cube: Float64 cube with localized occlusions.
            - metadata: Dictionary detailing occlusion mask and affected counts.
    """
    check_cube(cube)
    if not (0.0 < spatial_fraction <= 1.0):
        raise ValueError(f"spatial_fraction must be in (0, 1], got {spatial_fraction}")
    if occlusion_type not in ("zero", "mean", "clutter"):
        raise ValueError(
            f"occlusion_type must be 'zero', 'mean', or 'clutter', got {occlusion_type}"
        )

    rng = get_rng(seed)
    h, w, b = cube.shape
    corrupted = cube.astype(np.float64, copy=True)

    ph, pw = patch_size
    ph = min(max(1, ph), h)
    pw = min(max(1, pw), w)

    pixels_per_patch = ph * pw
    target_pixels = int(round(h * w * spatial_fraction))
    n_patches = max(1, target_pixels // pixels_per_patch)

    occlusion_mask = np.zeros((h, w), dtype=bool)
    for _ in range(n_patches):
        r0 = rng.integers(0, h - ph + 1)
        c0 = rng.integers(0, w - pw + 1)
        occlusion_mask[r0 : r0 + ph, c0 : c0 + pw] = True

    mask_3d = occlusion_mask[:, :, np.newaxis]

    if occlusion_type == "zero":
        corrupted = np.where(mask_3d, 0.0, corrupted)
    elif occlusion_type == "mean":
        mean_spec = np.mean(cube, axis=(0, 1))
        corrupted = np.where(mask_3d, mean_spec, corrupted)
    elif occlusion_type == "clutter":
        # Random high-variance clutter replacement
        clutter_noise = rng.uniform(
            float(np.min(cube)), float(np.max(cube)), size=(h, w, b)
        )
        corrupted = np.where(mask_3d, clutter_noise, corrupted)

    metadata: Dict[str, Any] = {
        "transform": "occlusion_simulation",
        "params": {
            "occlusion_type": occlusion_type,
            "spatial_fraction": float(spatial_fraction),
            "patch_size": patch_size,
        },
        "seed": seed,
        "occlusion_mask": occlusion_mask,
        "num_occluded_pixels": int(np.sum(occlusion_mask)),
    }
    return corrupted, metadata


def get_muufl_occlusion_subsets(
    meta: Dict[str, Any]
) -> Dict[str, np.ndarray]:
    """Extract real physical occlusion category masks from MUUFL Gulfport metadata.

    MUUFL Target Occlusion Legend:
        0 = Unoccluded cloth panels
        1 = Targets in shadow
        2 = Targets occluded under tree canopies

    Args:
        meta: Metadata dictionary returned by load_hsi.load('muufl-gulfport-campus1').

    Returns:
        dict: Mapping occlusion category names to integer index arrays into the 50 targets.

    Raises:
        ValueError: If required MUUFL metadata keys are missing.
    """
    required_keys = ["target_row", "target_col", "target_occlusion"]
    for k in required_keys:
        if k not in meta:
            raise ValueError(
                f"Missing required MUUFL key '{k}'. Available keys: {list(meta.keys())}"
            )

    occ = np.asarray(meta["target_occlusion"])
    return {
        "unoccluded_target_indices": np.where(occ == 0)[0],
        "shadow_target_indices": np.where(occ == 1)[0],
        "under_tree_target_indices": np.where(occ == 2)[0],
    }
