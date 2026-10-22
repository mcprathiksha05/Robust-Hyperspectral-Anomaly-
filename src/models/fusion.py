"""Task 6 - Score fusion.

    from src.models.fusion import fuse_scores, normalize_scores
    fused, info = fuse_scores({"rx": rx_map, "spatial": sp_map}, method="mean",
                              weights={"rx": 0.5, "spatial": 0.5})

Detectors produce scores on very different scales (RX can be in the thousands,
spectral angles are below 3), so each map is normalised before combining.

normalisation ('normalize'):
  'robust' - (s - median) / (1.4826 * MAD), then clipped at 0.   Default; outlier-safe.
  'zscore' - (s - mean) / std
  'minmax' - scaled to [0, 1]
  'rank'   - percentile rank in [0, 1]; ignores magnitude entirely

fusion ('method'):
  'mean'      - weighted average of normalised scores (default)
  'max'       - a pixel is as anomalous as its most suspicious detector (high recall)
  'geometric' - weighted geometric mean; high only if all detectors agree (fewer false alarms)
  'rank_mean' - average percentile rank (ignores 'normalize')

The result is rescaled to [0, 1]. ``info['agreement']`` is the fraction of
detectors that put the pixel in their own top ``agreement_top_percent`` - a
relative confidence indicator, not a calibrated probability.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import rankdata

from src.utils.validation import HSIValidationError, check_choice, check_float, validate_score_map

NORMALIZATIONS = {"robust", "zscore", "minmax", "rank"}
FUSIONS = {"mean", "max", "geometric", "rank_mean"}


def normalize_scores(scores, method: str = "robust", valid_mask=None):
    """Put a score map on a comparable scale. Invalid pixels (if a mask is given) get the minimum."""
    s = validate_score_map(scores).copy()
    check_choice(method, "normalize", NORMALIZATIONS)
    finite = np.isfinite(s)
    if valid_mask is not None:
        finite &= np.asarray(valid_mask, bool)
    v = s[finite]
    if method == "robust":
        med = np.median(v)
        mad = 1.4826 * np.median(np.abs(v - med))
        if mad <= 0:
            mad = v.std() or 1.0
        out = np.clip((s - med) / mad, 0, None)
    elif method == "zscore":
        out = (s - v.mean()) / (v.std() or 1.0)
    elif method == "minmax":
        lo, hi = v.min(), v.max()
        out = (s - lo) / ((hi - lo) or 1.0)
    else:
        out = np.zeros_like(s)
        out[finite] = (rankdata(v) - 1) / max(v.size - 1, 1)
    out[~finite] = np.nanmin(out[finite]) if finite.any() else 0.0
    return out


def fuse_scores(score_maps: dict, method: str = "mean", weights: dict | None = None,
                normalize: str = "robust", valid_mask=None, agreement_top_percent: float = 1.0):
    """Combine several (H, W) score maps into one. Returns (fused in [0, 1], info)."""
    if not score_maps:
        raise HSIValidationError("No score maps to fuse.")
    check_choice(method, "fusion method", FUSIONS)
    check_float(agreement_top_percent, "agreement_top_percent", 0, 100, lo_inclusive=False)
    names = list(score_maps)
    maps = {k: validate_score_map(v, f"score map '{k}'") for k, v in score_maps.items()}
    shapes = {v.shape for v in maps.values()}
    if len(shapes) != 1:
        raise HSIValidationError(f"Score maps have different shapes: { {k: v.shape for k, v in maps.items()} }")

    w = np.array([1.0 if weights is None else float(weights.get(k, 0.0)) for k in names])
    if (w < 0).any() or not np.isfinite(w).all():
        raise HSIValidationError("Fusion weights must be non-negative numbers.")
    if w.sum() == 0:
        raise HSIValidationError("At least one fusion weight must be positive.")
    w = w / w.sum()

    norm_method = "rank" if method == "rank_mean" else normalize
    norm = np.stack([normalize_scores(maps[k], norm_method, valid_mask) for k in names])

    if method in ("mean", "rank_mean"):
        fused = np.tensordot(w, norm, axes=1)
    elif method == "max":
        fused = (norm * (w[:, None, None] > 0)).max(0)
    else:  # geometric: work on [0,1]-scaled maps with a small floor
        unit = np.stack([normalize_scores(m, "minmax", valid_mask) for m in norm])
        fused = np.exp(np.tensordot(w, np.log(unit + 1e-3), axes=1))
    fused = normalize_scores(fused, "minmax", valid_mask)

    # agreement: share of detectors (with weight > 0) flagging the pixel in their own top-p%
    active = [i for i in range(len(names)) if w[i] > 0]
    flags = np.stack([maps[names[i]] >= np.percentile(maps[names[i]], 100 - agreement_top_percent)
                      for i in active])
    info = {
        "detectors": names, "weights": dict(zip(names, w.round(4).tolist())), "method": method,
        "normalize": norm_method, "normalized": dict(zip(names, norm)),
        "agreement": flags.mean(0), "agreement_top_percent": agreement_top_percent,
    }
    return fused, info
