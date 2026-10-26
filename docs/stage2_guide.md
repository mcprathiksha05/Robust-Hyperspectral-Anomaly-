# Stage 2 Guide: Robustness Perturbation Engine

This document provides complete documentation for the Stage 2 Robustness Perturbation Engine of the project:
*"Robust Hyperspectral Anomaly and Target Detection for Defense Applications Under Camouflage, Clutter, and Computational Constraints"*.

---

## 1. Architectural Design & Contract Invariants

All robustness transforms are located in `src/robustness/` and adhere to a unified, non-destructive signature:

```python
corrupted_cube, metadata = transform(cube: np.ndarray, ..., seed: Optional[int] = None)
```

### Core Invariants:
1. **Non-Destructive:** The input `cube` is never modified in-place; all transformations operate on a deep float64 copy.
2. **Shape Preservation:** The spatial and spectral dimensions $(H, W, B)$ are strictly preserved across all corruptions. Bands are never silently sliced away or reduced.
3. **Ground Truth Integrity:** The ground truth mask `gt` remains completely untouched. Any synthetic clutter or occlusion spatial masks are returned within `metadata` under dedicated keys (`clutter_mask`, `occlusion_mask`, `shadow_mask`).
4. **Reproducibility:** Every stochastic transformation accepts a `seed` parameter and utilizes modern, isolated `np.random.default_rng(seed)` generators.
5. **Dtype Neutrality:** Safely accepts `uint16`, `int16`, `float32`, or `float64` cubes.

---

## 2. Feature 1: Noise Robustness Transforms (`src/robustness/noise.py`)

### `add_gaussian_noise(cube, snr_db, per_band=False, seed=None)`
- **Physical Model:** Zero-mean additive Gaussian thermal sensor noise.
- **Formulation:**
  $$\text{SNR}_{\text{dB}} = 10 \log_{10}\left(\frac{P_{\text{signal}}}{P_{\text{noise}}}\right), \quad P_{\text{noise}} = \frac{P_{\text{signal}}}{10^{\text{SNR}_{\text{dB}}/10}}$$
  Noise is drawn from $\mathcal{N}(0, P_{\text{noise}})$.
- **Parameters:**
  - `snr_db` (float > 0): Signal-to-noise ratio in decibels (e.g. 30 dB = moderate, 10 dB = severe).
  - `per_band` (bool): Compute noise power independently per spectral band if True, or globally if False.
  - `seed` (Optional[int]): Random seed.
- **Metadata Returned:** `transform`, `target_snr_db`, `measured_snr_db`, `signal_power`, `noise_std_mean`.

### `add_poisson_noise(cube, photon_scale=1000.0, seed=None)`
- **Physical Model:** Photon arrival statistics and shot noise: $k \sim \text{Poisson}(\lambda)$.
- **Scaling Convention:** Because reflectance or calibrated radiance cubes vary in numerical scale ($[0, 1]$ or $[0, 10000]$), `photon_scale` specifies the effective quantum efficiency / photon count multiplier:
  $$\lambda = \max(x, 0) \times \text{photon\_scale}, \quad k \sim \text{Poisson}(\lambda), \quad x_{\text{noisy}} = \frac{k}{\text{photon\_scale}}$$
- **Parameters:** `photon_scale` (float > 0, default 1000.0), `seed`.
- **Metadata Returned:** `transform`, `photon_scale`, `intensity_min`, `intensity_max`.

### `add_stripe_noise(cube, stripe_density=0.05, band_fraction=0.3, stripe_intensity=0.2, mode="multiplicative", seed=None)`
- **Physical Model:** Pushbroom linear array detector calibration drift across flight track columns.
- **Parameters:**
  - `stripe_density` (float in (0, 1]): Proportion of columns corrupted.
  - `band_fraction` (float in (0, 1]): Proportion of spectral channels affected.
  - `stripe_intensity` (float > 0): Gain perturbation magnitude.
  - `mode`: `'multiplicative'` ($x \cdot (1 + \delta)$) or `'additive'` ($x + \delta \sigma$).
- **Metadata Returned:** `affected_columns`, `affected_bands`, `num_affected_columns`, `num_affected_bands`.

### `add_dead_pixels(cube, pixel_fraction=0.005, mode="all_bands", seed=None)`
- **Physical Model:** Sensor photodiode cell failure (zero response).
- **Parameters:** `pixel_fraction` (fraction of spatial pixels), `mode` (`'all_bands'` or `'random_bands'`).
- **Metadata Returned:** `num_dead_pixels`, `pixel_coordinates`.

### `add_saturated_pixels(cube, pixel_fraction=0.005, saturation_value=None, seed=None)`
- **Physical Model:** Hot pixels / sensor well saturation / optical bloom.
- **Parameters:** `pixel_fraction`, `saturation_value` (if None, defaults to $1.5 \times \max(x)$).
- **Metadata Returned:** `num_saturated_pixels`, `saturation_value`, `pixel_coordinates`.

### `add_salt_and_pepper_noise(cube, amount=0.01, salt_vs_pepper=0.5, seed=None)`
- **Physical Model:** Impulsive transmission bit-errors and sensor spikes.
- **Parameters:** `amount` (fraction of pixels), `salt_vs_pepper` (ratio of max to min saturation).
- **Metadata Returned:** `num_corrupted_pixels`, `num_salt_pixels`, `num_pepper_pixels`.

---

## 3. Feature 2: Clutter Robustness Transforms (`src/robustness/clutter.py`)

### `inject_spectral_clutter(cube, clutter_spectrum=None, abundance=0.3, spatial_fraction=0.05, patch_size=None, seed=None)`
- **Physical Model:** Sub-pixel spectral mixing and camouflage interference.
- **Formulation:** $x_{\text{mixed}} = (1 - \alpha) x + \alpha c$, where $\alpha$ is abundance and $c$ is the clutter endmember.
- **Parameters:** `clutter_spectrum` (shape $(B,)$ or None for background high-variance synthesis), `abundance` in $(0, 1]$, `spatial_fraction`, `patch_size`.
- **Metadata Returned:** `clutter_mask` (2D boolean mask), `num_clutter_pixels`, `clutter_spectrum_mean`.
- **Disclaimer:** Synthetic linear unmixing provides a controlled stress-test; it does not claim to model nonlinear 3D weave scattering or anisotropic BRDF.

### `inject_localized_clutter_patch(cube, bounds=None, clutter_spectrum=None, abundance=0.5, seed=None)`
- **Physical Model:** Contiguous spatial camouflage net / emplaced clutter patch.
- **Parameters:** `bounds` ($(r_0, r_1, c_0, c_1)$ bounding box), `abundance`, `seed`.
- **Metadata Returned:** `clutter_mask`, `num_clutter_pixels`, `params`.

### `apply_shadow_attenuation(cube, attenuation_factor=0.35, region_fraction=0.15, blur_boundary=True, seed=None)`
- **Physical Model:** Cloud or terrain shadow occlusion where direct solar irradiance is blocked.
- **Formulation:** $x_{\text{shadow}} = \gamma \cdot x$, with smooth penumbra gradient if `blur_boundary=True`.
- **Metadata Returned:** `shadow_mask`, `num_shadow_pixels`, `center`.

### `apply_occlusion_simulation(cube, occlusion_type="zero", spatial_fraction=0.05, patch_size=(5, 5), seed=None)`
- **Physical Model:** Tree canopies, camouflage netting, or structural obstacles occluding ground targets.
- **Types:** `'zero'`, `'mean'`, or `'clutter'`.
- **Metadata Returned:** `occlusion_mask`, `num_occluded_pixels`.

### `get_muufl_occlusion_subsets(meta)`
- **Real MUUFL Gulfport Metadata Integration:** Extracts target indices categorized by actual airborne ground truth:
  - `unoccluded_target_indices`: Unoccluded cloth panels (flag 0).
  - `shadow_target_indices`: Cloth panels in natural shadow (flag 1).
  - `under_tree_target_indices`: Cloth panels occluded under tree canopy (flag 2).

---

## 4. Feature 3: Spectral Band Corruption (`src/robustness/bands.py`)

### `drop_random_bands(cube, num_bands=None, fraction=0.1, mode="zero", seed=None)`
- **Contract Guarantee:** Corrupts selected bands while strictly preserving the $(H, W, B)$ cube shape.
- **Modes:** `'zero'` (0.0 fill), `'mean'` (scene mean fill), `'noise'` (Gaussian noise fill).
- **Metadata Returned:** `corrupted_bands` (boolean mask of length $B$), `corrupted_band_indices`.

### `mask_selected_bands(cube, band_indices, mode="zero")`
- **Use Case:** Masking specific faulty sensor channels.
- **Parameters:** `band_indices` (sequence of 0-based indices), `mode`.
- **Metadata Returned:** `corrupted_bands`, `corrupted_band_indices`.

### `add_localized_band_noise(cube, band_indices=None, band_fraction=0.15, snr_db=10.0, seed=None)`
- **Physical Model:** Wavelength-specific sensor degradation or low optical throughput at detector edges.
- **Parameters:** `band_indices`, `band_fraction`, `snr_db`, `seed`.
- **Metadata Returned:** `corrupted_bands`, `corrupted_band_indices`.

### `mask_wavelength_ranges(cube, wavelengths=None, ranges=[(1350, 1420), (1800, 1950)], mode="zero")`
- **Atmospheric Absorption Modeling:** Masks atmospheric $\text{H}_2\text{O}$ and $\text{CO}_2$ absorption windows.
- **Validation Rule:** Requires `wavelengths` sequence (e.g. from MUUFL); raises an informative `ValueError` if `wavelengths` is missing rather than guessing.

---

## 5. Verification Commands

Run unit tests:
```bash
python -m pytest tests/test_robustness.py -v
```

Run regression tests across the entire codebase:
```bash
python -m pytest tests/ -v
```

Run integration smoke test on real benchmark data:
```bash
python tests/smoke_test_robustness.py abu-airport-1
```
