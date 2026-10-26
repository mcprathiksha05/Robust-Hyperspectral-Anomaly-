"""Automated unit tests for Stage 2 Robustness Perturbation Engine.

Covers:
    - Feature 1: Noise robustness transforms (Gaussian, Poisson, stripe, dead, saturated, salt-and-pepper)
    - Feature 2: Clutter robustness transforms (Spectral mixing, localized patch, shadow, occlusion, MUUFL)
    - Feature 3: Spectral band corruption transforms (Dropout, selection, localized noise, wavelength)
    - Contract invariants: non-destructive inputs, reproducibility, shapes, dtypes, bounds, and edge cases.
"""

import pytest
import numpy as np

from src.robustness.noise import (
    add_gaussian_noise,
    add_poisson_noise,
    add_stripe_noise,
    add_dead_pixels,
    add_saturated_pixels,
    add_salt_and_pepper_noise,
)
from src.robustness.clutter import (
    inject_spectral_clutter,
    inject_localized_clutter_patch,
    apply_shadow_attenuation,
    apply_occlusion_simulation,
    get_muufl_occlusion_subsets,
)
from src.robustness.bands import (
    drop_random_bands,
    mask_selected_bands,
    add_localized_band_noise,
    mask_wavelength_ranges,
)


@pytest.fixture
def sample_cube():
    """Synthetic float32 hyperspectral cube of shape (24, 24, 16)."""
    np.random.seed(42)
    return np.random.uniform(50.0, 500.0, size=(24, 24, 16)).astype(np.float32)


@pytest.fixture
def sample_gt():
    """Synthetic binary ground truth mask of shape (24, 24)."""
    gt = np.zeros((24, 24), dtype=np.int32)
    gt[10:14, 10:14] = 1
    return gt


# ============================================================================
# 1. Feature 1: Noise Robustness Tests
# ============================================================================


def test_gaussian_noise_shape_and_snr(sample_cube):
    cube_copy = sample_cube.copy()

    # Moderate noise (30 dB)
    out_30, meta_30 = add_gaussian_noise(sample_cube, snr_db=30.0, seed=123)
    assert out_30.shape == sample_cube.shape
    assert np.all(np.isfinite(out_30))
    assert np.array_equal(sample_cube, cube_copy)  # Original unchanged
    assert meta_30["target_snr_db"] == 30.0
    assert abs(meta_30["measured_snr_db"] - 30.0) < 1.5

    # Severe noise (10 dB)
    out_10, meta_10 = add_gaussian_noise(sample_cube, snr_db=10.0, seed=123)
    assert meta_10["noise_std_mean"] > meta_30["noise_std_mean"]

    # Per-band mode
    out_band, meta_band = add_gaussian_noise(sample_cube, snr_db=20.0, per_band=True, seed=123)
    assert out_band.shape == sample_cube.shape
    assert meta_band["params"]["per_band"] is True


def test_poisson_noise(sample_cube):
    cube_copy = sample_cube.copy()
    out, meta = add_poisson_noise(sample_cube, photon_scale=500.0, seed=42)

    assert out.shape == sample_cube.shape
    assert np.all(np.isfinite(out))
    assert np.all(out >= 0.0)  # Poisson counts / scale are non-negative
    assert np.array_equal(sample_cube, cube_copy)
    assert meta["params"]["photon_scale"] == 500.0


def test_stripe_noise(sample_cube):
    out, meta = add_stripe_noise(
        sample_cube,
        stripe_density=0.1,
        band_fraction=0.25,
        stripe_intensity=0.3,
        mode="multiplicative",
        seed=42,
    )
    assert out.shape == sample_cube.shape
    assert np.all(np.isfinite(out))
    assert len(meta["affected_columns"]) > 0
    assert len(meta["affected_bands"]) > 0

    # Test additive mode
    out_add, meta_add = add_stripe_noise(
        sample_cube, stripe_density=0.1, stripe_intensity=0.2, mode="additive", seed=42
    )
    assert out_add.shape == sample_cube.shape


def test_dead_pixels(sample_cube):
    out_all, meta_all = add_dead_pixels(sample_cube, pixel_fraction=0.05, mode="all_bands", seed=42)
    assert out_all.shape == sample_cube.shape
    assert meta_all["num_dead_pixels"] > 0

    # Dead pixels must be exactly 0 across all bands
    for r, c in meta_all["pixel_coordinates"]:
        assert np.all(out_all[r, c, :] == 0.0)

    # Random bands mode
    out_rand, meta_rand = add_dead_pixels(
        sample_cube, pixel_fraction=0.05, mode="random_bands", seed=42
    )
    assert out_rand.shape == sample_cube.shape


def test_saturated_pixels(sample_cube):
    sat_val = 9999.0
    out, meta = add_saturated_pixels(
        sample_cube, pixel_fraction=0.05, saturation_value=sat_val, seed=42
    )
    assert out.shape == sample_cube.shape
    for r, c in meta["pixel_coordinates"]:
        assert np.all(out[r, c, :] == sat_val)


def test_salt_and_pepper_noise(sample_cube):
    out, meta = add_salt_and_pepper_noise(
        sample_cube, amount=0.05, salt_vs_pepper=0.5, seed=42
    )
    assert out.shape == sample_cube.shape
    assert meta["num_corrupted_pixels"] > 0
    assert meta["num_salt_pixels"] + meta["num_pepper_pixels"] == meta["num_corrupted_pixels"]


# ============================================================================
# 2. Feature 2: Clutter Robustness Tests
# ============================================================================


def test_spectral_clutter_mixing(sample_cube, sample_gt):
    gt_copy = sample_gt.copy()
    out, meta = inject_spectral_clutter(
        sample_cube, abundance=0.4, spatial_fraction=0.1, seed=42
    )
    assert out.shape == sample_cube.shape
    assert np.all(np.isfinite(out))
    assert np.array_equal(sample_gt, gt_copy)  # Ground truth unchanged!
    assert meta["clutter_mask"].shape == sample_cube.shape[:2]
    assert meta["num_clutter_pixels"] > 0

    # Patch mode
    out_patch, meta_patch = inject_spectral_clutter(
        sample_cube, abundance=0.5, spatial_fraction=0.1, patch_size=(4, 4), seed=42
    )
    assert out_patch.shape == sample_cube.shape


def test_localized_clutter_patch(sample_cube):
    bounds = (4, 12, 4, 12)
    out, meta = inject_localized_clutter_patch(sample_cube, bounds=bounds, abundance=0.6, seed=42)

    assert out.shape == sample_cube.shape
    # Outside the bounding box, pixels must remain exactly equal to original
    mask = meta["clutter_mask"]
    assert np.all(out[~mask] == sample_cube[~mask])
    # Inside the bounding box, pixels must have changed
    assert not np.all(out[mask] == sample_cube[mask])


def test_shadow_attenuation(sample_cube):
    out, meta = apply_shadow_attenuation(
        sample_cube, attenuation_factor=0.2, region_fraction=0.2, seed=42
    )
    assert out.shape == sample_cube.shape
    assert np.all(np.isfinite(out))
    assert meta["num_shadow_pixels"] > 0
    # Shadowed pixels must have lower or equal intensity
    shadow_mask = meta["shadow_mask"]
    assert np.mean(out[shadow_mask]) < np.mean(sample_cube[shadow_mask])


def test_occlusion_simulation(sample_cube):
    for occ_type in ("zero", "mean", "clutter"):
        out, meta = apply_occlusion_simulation(
            sample_cube, occlusion_type=occ_type, spatial_fraction=0.1, patch_size=(3, 3), seed=42
        )
        assert out.shape == sample_cube.shape
        assert meta["num_occluded_pixels"] > 0
        if occ_type == "zero":
            assert np.all(out[meta["occlusion_mask"]] == 0.0)


def test_muufl_occlusion_metadata():
    dummy_meta = {
        "target_row": np.array([10, 20, 30]),
        "target_col": np.array([15, 25, 35]),
        "target_occlusion": np.array([0, 1, 2]),
    }
    subsets = get_muufl_occlusion_subsets(dummy_meta)
    assert np.array_equal(subsets["unoccluded_target_indices"], [0])
    assert np.array_equal(subsets["shadow_target_indices"], [1])
    assert np.array_equal(subsets["under_tree_target_indices"], [2])


# ============================================================================
# 3. Feature 3: Spectral Band Corruption Tests
# ============================================================================


def test_drop_random_bands(sample_cube):
    cube_copy = sample_cube.copy()
    out, meta = drop_random_bands(sample_cube, num_bands=4, mode="zero", seed=42)

    # Dimensionality invariant: shape is STRICTLY preserved
    assert out.shape == sample_cube.shape
    assert np.array_equal(sample_cube, cube_copy)
    assert meta["num_corrupted_bands"] == 4
    assert np.sum(meta["corrupted_bands"]) == 4

    # Dropped bands are zeroed
    dropped_idx = meta["corrupted_band_indices"]
    assert np.all(out[:, :, dropped_idx] == 0.0)

    # Clean bands are completely untouched
    clean_idx = [i for i in range(sample_cube.shape[2]) if i not in dropped_idx]
    assert np.array_equal(out[:, :, clean_idx], sample_cube[:, :, clean_idx])


def test_mask_selected_bands(sample_cube):
    selected = [1, 3, 7]
    out, meta = mask_selected_bands(sample_cube, band_indices=selected, mode="zero")

    assert out.shape == sample_cube.shape
    assert meta["corrupted_band_indices"] == selected
    assert np.all(out[:, :, selected] == 0.0)


def test_localized_band_noise(sample_cube):
    selected = [2, 4]
    out, meta = add_localized_band_noise(
        sample_cube, band_indices=selected, snr_db=5.0, seed=42
    )
    assert out.shape == sample_cube.shape
    assert meta["corrupted_band_indices"] == selected

    # Unselected bands must remain identical
    unselected = [i for i in range(sample_cube.shape[2]) if i not in selected]
    assert np.array_equal(out[:, :, unselected], sample_cube[:, :, unselected])
    # Selected bands must differ
    assert not np.array_equal(out[:, :, selected], sample_cube[:, :, selected])


def test_wavelength_range_masking(sample_cube):
    # 16 bands with synthetic center wavelengths (in nm)
    wavelengths = np.linspace(400.0, 2000.0, sample_cube.shape[2])

    # Water absorption around 1400 nm and 1900 nm
    ranges = [(1350.0, 1450.0), (1850.0, 1950.0)]
    out, meta = mask_wavelength_ranges(sample_cube, wavelengths=wavelengths, ranges=ranges)

    assert out.shape == sample_cube.shape
    assert meta["num_corrupted_bands"] > 0
    for idx in meta["corrupted_band_indices"]:
        wl = wavelengths[idx]
        assert (1350.0 <= wl <= 1450.0) or (1850.0 <= wl <= 1950.0)
        assert np.all(out[:, :, idx] == 0.0)

    # Missing wavelengths must raise an informative ValueError
    with pytest.raises(ValueError, match="Wavelength metadata is required"):
        mask_wavelength_ranges(sample_cube, wavelengths=None)


# ============================================================================
# 4. Reproducibility & Stochastic Divergence Tests
# ============================================================================


def test_reproducibility(sample_cube):
    # Same seed -> identical outputs
    out_a, _ = add_gaussian_noise(sample_cube, snr_db=25.0, seed=123)
    out_b, _ = add_gaussian_noise(sample_cube, snr_db=25.0, seed=123)
    assert np.array_equal(out_a, out_b)

    # Different seed -> distinct outputs
    out_c, _ = add_gaussian_noise(sample_cube, snr_db=25.0, seed=999)
    assert not np.array_equal(out_a, out_c)

    # Clutter reproducibility
    clutter_a, _ = inject_spectral_clutter(sample_cube, abundance=0.3, seed=55)
    clutter_b, _ = inject_spectral_clutter(sample_cube, abundance=0.3, seed=55)
    assert np.array_equal(clutter_a, clutter_b)


# ============================================================================
# 5. Dtype Flexibility & Input Validation Tests
# ============================================================================


def test_dtype_flexibility():
    # uint16 (like ABU Airport)
    cube_u16 = np.random.randint(100, 5000, size=(10, 10, 5), dtype=np.uint16)
    out_u16, _ = add_gaussian_noise(cube_u16, snr_db=20.0, seed=42)
    assert out_u16.dtype == np.float64
    assert np.all(np.isfinite(out_u16))

    # int16 (like ABU Beach / Urban)
    cube_i16 = np.random.randint(0, 1000, size=(10, 10, 5), dtype=np.int16)
    out_i16, _ = add_poisson_noise(cube_i16, photon_scale=10.0, seed=42)
    assert out_i16.dtype == np.float64
    assert np.all(np.isfinite(out_i16))

    # float64
    cube_f64 = np.random.rand(10, 10, 5).astype(np.float64)
    out_f64, _ = drop_random_bands(cube_f64, num_bands=2, seed=42)
    assert out_f64.dtype == np.float64
    assert np.all(np.isfinite(out_f64))


def test_invalid_parameters(sample_cube):
    # 2D cube
    with pytest.raises(ValueError, match="Cube must be 3-dimensional"):
        add_gaussian_noise(np.random.rand(10, 10), snr_db=20.0)

    # Negative SNR
    with pytest.raises(ValueError, match="SNR in dB must be positive"):
        add_gaussian_noise(sample_cube, snr_db=-5.0)

    # Negative photon scale
    with pytest.raises(ValueError, match="photon_scale must be strictly positive"):
        add_poisson_noise(sample_cube, photon_scale=-10.0)

    # Out of range fraction
    with pytest.raises(ValueError, match="stripe_density must be in"):
        add_stripe_noise(sample_cube, stripe_density=1.5)

    with pytest.raises(ValueError, match="abundance must be in"):
        inject_spectral_clutter(sample_cube, abundance=0.0)

    with pytest.raises(ValueError, match="num_bands must be in"):
        drop_random_bands(sample_cube, num_bands=999)
