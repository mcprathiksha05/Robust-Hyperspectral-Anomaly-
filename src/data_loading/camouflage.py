"""Camouflage datasets: real (MUUFL Gulfport) and controlled synthetic implants.

Real camouflage - MUUFL Gulfport flights
    muufl_target_masks(rows, cols, sizes, shape) builds pixel masks for the emplaced
    cloth targets. Calibration cloths (6 x 10 m, red/black/blue/green) are excluded
    from the camouflage mask because they are bright and trivial to detect.

Synthetic camouflage - target implants
    implant_targets(cube, gt, abundance=0.1) removes the scene's original anomalies
    (fills them from their neighbourhood) and plants new square targets whose
    spectrum is a mix of background and a real man-made material from the same
    image:   pixel = (1 - a) * background + a * material
    Small ``abundance`` a = target barely differs from its surroundings
    (camouflage / sub-pixel target); large a = obvious target. Because the
    material and the background come from the same sensor and scene, the result
    is physically consistent, and the difficulty is controlled exactly.

    The mixing is applied to the noise-free part of each pixel and the pixel's
    own sensor noise is kept:  pixel = background + a * (material - local_mean).
    Blending whole pixels would shrink their noise by (1 - a) and make implants
    look *quieter* than real pixels, which detectors like RX would then miss.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import binary_dilation, label

from src.models.rx_compat import ring_sums
from src.utils.validation import HSIValidationError, check_float, check_int, validate_cube, validate_gt

CALIBRATION_SIZE_M = 6.0


# ------------------------------------------------------------------- MUUFL
def muufl_target_masks(rows, cols, sizes, shape, gsd_m: float = 1.0, one_based: bool = True,
                       include_calibration: bool = False, halo: int = 1):
    """Pixel masks for MUUFL targets.

    Square targets of side ``size`` metres cover ceil(size / gsd) pixels (at least 1);
    calibration cloths (size 6) cover 6 x 10 m and are only included on request.
    Returns (mask, halo_mask): ``halo_mask`` adds ``halo`` pixels around each target,
    useful as a 'do not count as false alarm' zone because sub-pixel targets
    spill into neighbouring pixels and the GPS positions are accurate to ~1 pixel.
    """
    h, w = shape[:2]
    mask = np.zeros((h, w), bool)
    for r, c, s in zip(np.asarray(rows, int), np.asarray(cols, int), np.asarray(sizes, float)):
        if one_based:
            r, c = r - 1, c - 1
        if s >= CALIBRATION_SIZE_M:
            if not include_calibration:
                continue
            hh, ww = int(np.ceil(6 / gsd_m)), int(np.ceil(10 / gsd_m))
        else:
            hh = ww = max(1, int(np.ceil(s / gsd_m)))
        r0, c0 = r - (hh - 1) // 2, c - (ww - 1) // 2
        mask[max(r0, 0):min(r0 + hh, h), max(c0, 0):min(c0 + ww, w)] = True
    halo_mask = binary_dilation(mask, iterations=halo) if halo > 0 else mask.copy()
    return mask.astype(np.uint8), halo_mask.astype(np.uint8)


# --------------------------------------------------------------- implants
def local_mean(cube, size: int = 5):
    """Mean of the size x size neighbourhood excluding the pixel itself (noise-free estimate)."""
    s, n = ring_sums(np.asarray(cube, np.float64), 1, size)
    return s / n


def remove_anomalies(cube, gt, inner: int = 1, outer: int = 9, dilate: int = 2, random_state: int = 0):
    """Replace labelled anomaly pixels (plus a small margin) with background.

    Each hole pixel gets its neighbourhood mean plus a realistic noise vector
    borrowed from a random untouched pixel, so filled areas are neither
    smoother nor noisier than the rest of the image.
    """
    cube = validate_cube(cube).astype(np.float64)
    gt = validate_gt(gt, cube.shape).astype(bool)
    hole = binary_dilation(gt, iterations=dilate) if dilate else gt
    hole0 = hole.copy()
    x = cube.copy()
    known = (~hole).astype(np.float64)
    filled = x * known[..., None]
    # Repeatedly average known neighbours into the hole until it is filled.
    for _ in range(50):
        if not hole.any():
            break
        s, _ = ring_sums(filled, inner, outer) if inner < outer else (filled, 1)
        k, _ = ring_sums(known, inner, outer)
        can = hole & (k > 0.5)
        if not can.any():
            outer += 4
            continue
        x[can] = s[can] / k[can][:, None]
        filled[can], known[can], hole = x[can], 1.0, hole & ~can
    # add back noise: residuals (pixel - local mean) sampled from untouched pixels
    rng = np.random.default_rng(random_state)
    resid = (cube - local_mean(cube))[~binary_dilation(hole0, iterations=3)]
    x[hole0] += resid[rng.integers(0, len(resid), int(hole0.sum()))]
    return x.astype(np.float32)


def implant_targets(cube, gt=None, abundance: float = 0.1, n_targets: int = 20, sizes=(1, 2, 3),
                    material=None, noise_std: float = 0.0, min_spacing: int = 8, border: int = 6,
                    avoid_mask=None, random_state: int = 0):
    """Plant ``n_targets`` square targets with fractional abundance ``abundance``.

    material : spectrum to implant. Default: mean spectrum of the largest original
               anomaly object in ``gt`` (one real material from the same image;
               averaging *different* objects would give a meaningless blend).
    noise_std: extra Gaussian noise as a fraction of each band's std (sensor noise).
    Returns (new_cube float32, new_gt uint8, info dict).
    """
    cube = validate_cube(cube).astype(np.float64)
    a = check_float(abundance, "abundance", 0, 1, lo_inclusive=False)
    n_targets = check_int(n_targets, "n_targets", 1, 500)
    h, w, b = cube.shape
    rng = np.random.default_rng(random_state)

    if gt is not None:
        gt = validate_gt(gt, cube.shape).astype(bool)
        if material is None:
            if not gt.any():
                raise HSIValidationError("gt has no target pixels to take a material spectrum from.")
            lab, n = label(gt, structure=np.ones((3, 3)))
            biggest = np.argmax(np.bincount(lab.ravel())[1:]) + 1
            material = cube[lab == biggest].mean(0)
        background = remove_anomalies(cube, gt).astype(np.float64)
    else:
        background = cube
        if material is None:
            raise HSIValidationError("Give either gt (to take the material from) or material.")
    material = np.asarray(material, float).ravel()
    if material.size != b:
        raise HSIValidationError(f"material has {material.size} values but the cube has {b} bands.")

    blocked = np.zeros((h, w), bool) if avoid_mask is None else np.asarray(avoid_mask, bool).copy()
    smooth = local_mean(background)
    out = background.copy()
    new_gt = np.zeros((h, w), np.uint8)
    placed = []
    tries = 0
    while len(placed) < n_targets and tries < 20_000:
        tries += 1
        s = int(rng.choice(sizes))
        r = int(rng.integers(border, h - border - s))
        c = int(rng.integers(border, w - border - s))
        if blocked[max(r - min_spacing, 0):r + s + min_spacing, max(c - min_spacing, 0):c + s + min_spacing].any():
            continue
        out[r:r + s, c:c + s] = background[r:r + s, c:c + s] + a * (material - smooth[r:r + s, c:c + s])
        new_gt[r:r + s, c:c + s] = 1
        blocked[r:r + s, c:c + s] = True
        placed.append({"row": r, "col": c, "size": s})
    if len(placed) < n_targets:
        raise HSIValidationError(f"Only room for {len(placed)} targets; lower n_targets or min_spacing.")
    if noise_std > 0:
        out += rng.normal(0, 1, out.shape) * (noise_std * background.reshape(-1, b).std(0))

    # How different do the implanted pixels end up from their original background?
    t = new_gt.astype(bool)
    num = (out[t] * background[t]).sum(1)
    den = np.linalg.norm(out[t], axis=1) * np.linalg.norm(background[t], axis=1)
    sam = np.degrees(np.arccos(np.clip(num / np.maximum(den, 1e-12), -1, 1)))
    info = {"abundance": a, "targets": placed, "n_target_pixels": int(t.sum()),
            "material": material, "noise_std": noise_std,
            "mean_sam_to_background_deg": float(sam.mean()),
            "mean_contrast_rel": float(np.mean(np.linalg.norm(out[t] - background[t], axis=1)
                                               / np.maximum(np.linalg.norm(background[t], axis=1), 1e-12)))}
    return out.astype(np.float32), new_gt, info
