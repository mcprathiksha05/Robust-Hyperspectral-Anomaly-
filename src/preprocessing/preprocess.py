"""Task 2 - Preprocessing.

    from src.preprocessing.preprocess import preprocess
    clean, info = preprocess(cube, normalize="robust")

Steps, in order (each can be switched off):
1. Mark invalid pixels (any NaN/inf band, or optionally all-zero "no data" pixels).
2. Repair invalid values with the band median of valid pixels, so shapes stay intact.
3. Drop constant bands (zero variance), which make covariance matrices singular.
4. Clip extreme values per band to percentiles (limits sensor spikes).
5. Normalise: 'zscore', 'robust' (median/IQR), 'minmax', 'l2' (per-pixel unit length), or 'none'.

``info`` records everything that was done (valid_mask, dropped bands, statistics)
so results are reproducible and the frontend can report it.
"""
from __future__ import annotations

import numpy as np

from src.utils.validation import HSIValidationError, check_choice, check_float, validate_cube

NORMALIZERS = {"none", "zscore", "robust", "minmax", "l2"}


def preprocess(cube, normalize: str = "robust", drop_constant_bands: bool = True,
               clip_percentiles: tuple[float, float] | None = (0.1, 99.9),
               zero_is_invalid: bool = False, bands_to_drop=None):
    """Clean and normalise a cube. Returns (clean_cube float32, info dict)."""
    cube = validate_cube(cube)
    check_choice(normalize, "normalize", NORMALIZERS)
    h, w, b = cube.shape
    x = cube.reshape(-1, b).astype(np.float64)  # copy; never modify the caller's array
    info: dict = {"input_shape": (h, w, b), "normalize": normalize, "warnings": []}

    # 1-2. invalid values ----------------------------------------------------
    finite = np.isfinite(x)
    invalid_pixel = ~finite.all(1)
    if zero_is_invalid:
        invalid_pixel |= np.all(x == 0, axis=1)
    n_bad_vals = int((~finite).sum())
    if invalid_pixel.all():
        raise HSIValidationError("Every pixel has invalid values; nothing to process.")
    if n_bad_vals:
        good = np.where(finite & ~invalid_pixel[:, None], x, np.nan)
        med = np.nanmedian(good, axis=0)
        med = np.where(np.isfinite(med), med, 0.0)  # band entirely invalid -> 0 (dropped below)
        rows, cols = np.where(~finite)
        x[rows, cols] = med[cols]
    info["nan_inf_values_repaired"] = n_bad_vals
    info["invalid_pixels"] = int(invalid_pixel.sum())
    info["valid_mask"] = (~invalid_pixel).reshape(h, w)
    if invalid_pixel.mean() > 0.5:
        info["warnings"].append(f"{invalid_pixel.mean():.0%} of pixels were invalid; results may be unreliable.")

    # 3. band selection ------------------------------------------------------
    keep = np.ones(b, bool)
    if bands_to_drop is not None:
        idx = np.asarray(bands_to_drop, dtype=int)
        if idx.size and (idx.min() < 0 or idx.max() >= b):
            raise HSIValidationError(f"bands_to_drop has indices outside 0..{b - 1}.")
        keep[idx] = False
    if drop_constant_bands:
        std = x[~invalid_pixel].std(0)
        const = std <= 1e-12 * max(1.0, np.abs(x).max())
        keep &= ~const
        info["constant_bands_dropped"] = [int(i) for i in np.where(const)[0]]
    if keep.sum() < 2:
        raise HSIValidationError("Fewer than 2 usable bands remain after removing constant/dropped bands.")
    info["kept_bands"] = np.where(keep)[0]
    x = x[:, keep]

    # 4. clipping --------------------------------------------------------------
    if clip_percentiles is not None:
        lo_p = check_float(clip_percentiles[0], "lower clip percentile", 0, 50)
        hi_p = check_float(clip_percentiles[1], "upper clip percentile", 50, 100)
        lo, hi = np.percentile(x[~invalid_pixel], [lo_p, hi_p], axis=0)
        x = np.clip(x, lo, hi)

    # 5. normalisation ---------------------------------------------------------
    v = x[~invalid_pixel]
    if normalize == "zscore":
        c, s = v.mean(0), v.std(0)
    elif normalize == "robust":
        c = np.median(v, 0)
        q75, q25 = np.percentile(v, [75, 25], axis=0)
        s = q75 - q25
        s = np.where(s > 0, s, v.std(0))
    elif normalize == "minmax":
        c, s = v.min(0), v.max(0) - v.min(0)
    else:
        c, s = np.zeros(x.shape[1]), np.ones(x.shape[1])
    s = np.where(s > 0, s, 1.0)
    x = (x - c) / s
    if normalize == "l2":
        x = x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)
    info["center"], info["scale"] = c, s

    x[invalid_pixel] = 0.0 if normalize != "none" else np.median(v, 0)
    return x.reshape(h, w, -1).astype(np.float32), info
