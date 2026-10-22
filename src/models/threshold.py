"""Task 8 - Thresholding: turn a continuous score map into a binary detection map.

    from src.models.threshold import apply_threshold, calibrate_threshold
    mask, info = apply_threshold(scores, method="percentile", percent=1.0)

Methods (no ground truth needed):
  'percentile' - flag the top ``percent`` % of pixels
  'sigma'      - flag scores above median + k * robust std (1.4826 * MAD); adapts to the image
  'otsu'       - split the score histogram into two classes automatically
  'fixed'      - a value you choose (``value``), e.g. one returned by calibrate_threshold

Calibrated thresholds (needs a *separate* labelled calibration image):
  calibrate_threshold(cal_scores, cal_gt, target="fpr", value=0.01)
  -> returns the score value to pass as method='fixed' on other images.
  Use robust-normalised scores (fusion.normalize_scores(..., 'robust')) for
  both calibration and test, so the value transfers between images. Never
  calibrate on the same image you report results for - that is optimistic.
"""
from __future__ import annotations

import numpy as np

from src.utils.validation import (HSIValidationError, check_choice, check_float, validate_gt,
                                  validate_score_map)

METHODS = {"percentile", "sigma", "otsu", "fixed"}
TARGETS = {"fpr", "recall", "f1"}


def apply_threshold(scores, method: str = "percentile", percent: float = 1.0, k: float = 3.0,
                    value: float | None = None, valid_mask=None, min_region_size: int = 1):
    """Return (mask uint8 (H, W), info). ``min_region_size`` > 1 removes tiny isolated detections."""
    s = validate_score_map(scores)
    check_choice(method, "threshold method", METHODS)
    valid = np.isfinite(s) if valid_mask is None else (np.asarray(valid_mask, bool) & np.isfinite(s))
    v = s[valid]
    if v.size == 0:
        raise HSIValidationError("No valid pixels to threshold.")

    if method == "percentile":
        percent = check_float(percent, "percent", 0, 100, lo_inclusive=False)
        thr = float(np.percentile(v, 100 - percent))
    elif method == "sigma":
        k = check_float(k, "k", 0)
        med = np.median(v)
        mad = 1.4826 * np.median(np.abs(v - med)) or v.std()
        thr = float(med + k * mad)
    elif method == "otsu":
        thr = otsu_threshold(v)
    else:
        if value is None:
            raise HSIValidationError("method='fixed' needs a threshold value.")
        thr = check_float(value, "threshold value")

    mask = (s > thr) & valid
    removed = 0
    if min_region_size > 1 and mask.any():
        from scipy.ndimage import label
        lab, _ = label(mask, structure=np.ones((3, 3)))
        sizes = np.bincount(lab.ravel())
        small = sizes < min_region_size
        small[0] = False
        removed = int(small[lab].sum())
        mask &= ~small[lab]

    n = int(mask.sum())
    info = {"method": method, "threshold": thr, "detected_pixels": n,
            "detected_percent": round(100 * n / s.size, 4), "removed_small_region_pixels": removed}
    if n == 0:
        info["warning"] = "No pixels exceed the threshold; try a lower threshold or a different method."
    elif n > 0.25 * s.size:
        info["warning"] = "Over 25% of the image is flagged; the threshold is probably too low."
    return mask.astype(np.uint8), info


def otsu_threshold(values, bins: int = 256) -> float:
    """Otsu's method: threshold that best separates the histogram into two groups."""
    v = np.asarray(values, dtype=np.float64).ravel()
    v = v[np.isfinite(v)]
    if v.min() == v.max():
        return float(v.max())
    hist, edges = np.histogram(v, bins=bins)
    centers = (edges[:-1] + edges[1:]) / 2
    w0 = np.cumsum(hist)
    w1 = w0[-1] - w0
    m0 = np.cumsum(hist * centers) / np.maximum(w0, 1)
    m1 = (np.sum(hist * centers) - np.cumsum(hist * centers)) / np.maximum(w1, 1)
    between = w0 * w1 * (m0 - m1) ** 2
    return float(centers[np.argmax(between[:-1])])


def calibrate_threshold(scores, gt, target: str = "fpr", value: float = 0.01) -> dict:
    """Choose a threshold on a labelled calibration image.

    target='fpr', value=0.01  -> highest sensitivity while keeping false-positive rate <= 1%
    target='recall', value=0.9 -> smallest number of false alarms while detecting >= 90% of targets
    target='f1'                -> threshold with the best F1 on the calibration image
    """
    s = validate_score_map(scores)
    gt = validate_gt(gt, s.shape + (1,)).astype(bool)
    check_choice(target, "target", TARGETS)
    if gt.all() or not gt.any():
        raise HSIValidationError("Calibration needs both target and background pixels in the ground truth.")
    order = np.argsort(-s.ravel(), kind="stable")
    ss, yy = s.ravel()[order], gt.ravel()[order]
    tp = np.cumsum(yy)
    fp = np.cumsum(~yy)
    # Only evaluate at the last index of each run of tied scores.
    last = np.r_[ss[1:] != ss[:-1], True]
    tp, fp, ss = tp[last], fp[last], ss[last]
    recall = tp / gt.sum()
    fpr = fp / (~gt).sum()
    precision = tp / (tp + fp)
    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-12)
    if target == "fpr":
        value = check_float(value, "target FPR", 0, 1)
        ok = np.where(fpr <= value)[0]
        i = ok[-1] if ok.size else 0
    elif target == "recall":
        value = check_float(value, "target recall", 0, 1)
        i = int(np.argmax(recall >= value))
    else:
        i = int(np.argmax(f1))
    # threshold strictly below the i-th score so that pixel is included with '>'
    nxt = ss[i + 1] if i + 1 < ss.size else ss[i] - 1
    thr = float((ss[i] + nxt) / 2)
    return {"threshold": thr, "target": target, "value": value, "precision": float(precision[i]),
            "recall": float(recall[i]), "fpr": float(fpr[i]), "f1": float(f1[i])}
