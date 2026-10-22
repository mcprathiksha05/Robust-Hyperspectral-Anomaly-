"""Task 21 - Spectral signature inspector.

    from src.explain.spectral import compare_spectra, plot_spectra
    comp = compare_spectra(cube, pixels=[(42, 17), (10, 80)], wavelengths=meta.get("wavelengths"))
    fig = plot_spectra(comp)          # matplotlib figure; in Streamlit: st.pyplot(fig)

Shows each selected pixel's spectrum next to the background (mean with a
5-95 % band), and measures how different they are:
  spectral angle (SAM, degrees) - shape difference, insensitive to brightness/shadow
  euclidean distance            - overall magnitude difference
  max |z| band                  - the single band where the pixel deviates most
Use the raw cube (physical units) for the most readable plots.
"""
from __future__ import annotations

import numpy as np

from src.utils.validation import HSIValidationError, check_int, validate_cube


def _check_pixel(cube, row, col):
    h, w = cube.shape[:2]
    row, col = check_int(row, "row"), check_int(col, "column")
    if not (0 <= row < h and 0 <= col < w):
        raise HSIValidationError(f"Pixel ({row}, {col}) is outside the image (rows 0-{h - 1}, columns 0-{w - 1}).")
    return row, col


def get_spectrum(cube, row: int, col: int, window: int = 1) -> np.ndarray:
    """Spectrum at (row, col); ``window`` > 1 averages a window x window neighbourhood (less noise)."""
    cube = validate_cube(cube)
    row, col = _check_pixel(cube, row, col)
    window = check_int(window, "window", 1, 51)
    r = window // 2
    patch = cube[max(row - r, 0):row + r + 1, max(col - r, 0):col + r + 1].astype(np.float64)
    return np.nanmean(patch.reshape(-1, cube.shape[2]), axis=0)


def region_spectrum(cube, mask) -> dict:
    """Mean and standard deviation of all pixels in ``mask`` (e.g. a detected region)."""
    cube = validate_cube(cube)
    mask = np.asarray(mask, bool)
    if mask.shape != cube.shape[:2]:
        raise HSIValidationError("Region mask must match the image height and width.")
    if not mask.any():
        raise HSIValidationError("The selected region is empty.")
    x = cube[mask].astype(np.float64)
    return {"mean": np.nanmean(x, 0), "std": np.nanstd(x, 0), "pixels": int(mask.sum())}


def background_spectrum(cube, exclude_mask=None, valid_mask=None) -> dict:
    """Background statistics, leaving out ``exclude_mask`` pixels (e.g. detections or targets)."""
    cube = validate_cube(cube)
    keep = np.ones(cube.shape[:2], bool)
    if exclude_mask is not None:
        keep &= ~np.asarray(exclude_mask, bool)
    if valid_mask is not None:
        keep &= np.asarray(valid_mask, bool)
    if not keep.any():
        raise HSIValidationError("No background pixels left after exclusions.")
    x = cube[keep].astype(np.float64)
    return {"mean": np.nanmean(x, 0), "std": np.nanstd(x, 0),
            "p5": np.nanpercentile(x, 5, 0), "p95": np.nanpercentile(x, 95, 0), "pixels": int(keep.sum())}


def spectral_angle(a, b) -> float:
    """Spectral angle in degrees (0 = same shape)."""
    a, b = np.asarray(a, float).ravel(), np.asarray(b, float).ravel()
    den = np.linalg.norm(a) * np.linalg.norm(b)
    if den == 0:
        return float("nan")
    return float(np.degrees(np.arccos(np.clip(a @ b / den, -1, 1))))


def compare_spectra(cube, pixels=(), wavelengths=None, labels=None, window: int = 1,
                    regions: dict | None = None, exclude_mask=None, valid_mask=None) -> dict:
    """Collect spectra for pixels/regions plus background stats and similarity measures."""
    cube = validate_cube(cube)
    b = cube.shape[2]
    x = np.arange(b) if wavelengths is None else np.asarray(wavelengths, float).ravel()
    if x.size != b:
        raise HSIValidationError(f"Got {x.size} wavelengths for {b} bands.")
    bg = background_spectrum(cube, exclude_mask, valid_mask)
    std = np.where(bg["std"] > 0, bg["std"], 1.0)
    items = []
    labels = list(labels) if labels is not None else [f"pixel ({r}, {c})" for r, c in pixels]
    if len(labels) != len(pixels):
        raise HSIValidationError("Give one label per pixel.")
    for (r, c), lab in zip(pixels, labels):
        items.append((lab, get_spectrum(cube, r, c, window)))
    for lab, m in (regions or {}).items():
        items.append((lab, region_spectrum(cube, m)["mean"]))
    if not items:
        raise HSIValidationError("Select at least one pixel or region to inspect.")
    spectra = []
    for lab, s in items:
        z = (s - bg["mean"]) / std
        k = int(np.nanargmax(np.abs(z)))
        spectra.append({"label": lab, "values": s,
                        "sam_to_background_deg": spectral_angle(s, bg["mean"]),
                        "euclidean_to_background": float(np.linalg.norm(s - bg["mean"])),
                        "z_scores": z, "max_abs_z": float(abs(z[k])), "max_z_band": k,
                        "max_z_wavelength": float(x[k])})
    n = len(spectra)
    sam = np.array([[spectral_angle(spectra[i]["values"], spectra[j]["values"]) for j in range(n)]
                    for i in range(n)])
    return {"x": x, "x_label": "Wavelength (nm)" if wavelengths is not None else "Band index",
            "background": bg, "spectra": spectra, "pairwise_sam_deg": sam}


def plot_spectra(comparison: dict, ax=None, show_band: bool = True):
    """Plot the output of ``compare_spectra``. Returns the matplotlib Figure."""
    import matplotlib.pyplot as plt
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4))
    else:
        fig = ax.figure
    x, bg = comparison["x"], comparison["background"]
    if show_band:
        ax.fill_between(x, bg["p5"], bg["p95"], color="0.85", label="background 5-95%")
    ax.plot(x, bg["mean"], color="0.4", lw=1.5, ls="--", label="background mean")
    for s in comparison["spectra"]:
        ax.plot(x, s["values"], lw=1.6, label=f"{s['label']} (SAM {s['sam_to_background_deg']:.1f}°)")
    ax.set_xlabel(comparison["x_label"])
    ax.set_ylabel("Value")
    ax.legend(fontsize=8, frameon=False)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig
