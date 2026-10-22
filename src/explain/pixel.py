"""Task 22 - Pixel explanation: why did this pixel get a high anomaly score?

    from src.explain.pixel import explain_pixel
    result = run_pipeline(cube, gt)
    exp = explain_pixel(result, row=42, col=17, wavelengths=meta.get("wavelengths"))
    print(exp["summary"])

It combines several views of the same pixel:
- rank: each detector's score and its percentile in the image
- fusion: how much each detector contributed to the fused score
- RX bands: exact additive split of the RX score, mapped from PCA components back
  to the original bands (sums to the RX score)
- spectral deviation: bands where the pixel is furthest from the image background (z-scores)
- spatial: bands where the pixel differs most from its immediate neighbours
- autoencoder: bands the network could not reconstruct
These are relative explanations of the detectors' behaviour, not proof of what
material the pixel is.
"""
from __future__ import annotations

import numpy as np

from src.utils.validation import HSIValidationError, check_int


def _top(values, k, band_ids, wavelengths, absolute=False):
    v = np.asarray(values, float)
    key = np.abs(v) if absolute else v
    idx = np.argsort(-key)[:k]
    return [{"band": int(band_ids[i]),
             "wavelength": None if wavelengths is None else float(wavelengths[band_ids[i]]),
             "value": float(v[i])} for i in idx]


def _band_text(items):
    parts = []
    for it in items:
        parts.append(f"{it['wavelength']:.0f} nm" if it["wavelength"] is not None else f"band {it['band']}")
    return ", ".join(parts)


def explain_pixel(result: dict, row: int, col: int, top_k: int = 5, wavelengths=None) -> dict:
    if "data" not in result or "scores" not in result:
        raise HSIValidationError("Pass the dictionary returned by run_pipeline().")
    clean = result["data"]["clean"]
    reduced = result["data"]["reduced"]
    h, w = clean.shape[:2]
    row, col = check_int(row, "row"), check_int(col, "column")
    if not (0 <= row < h and 0 <= col < w):
        raise HSIValidationError(f"Pixel ({row}, {col}) is outside the image ({h} x {w}).")
    top_k = check_int(top_k, "top_k", 1, 100)
    band_ids = result.get("preprocess_info", {}).get("kept_bands", np.arange(clean.shape[2]))
    if wavelengths is not None:
        wavelengths = np.asarray(wavelengths, float).ravel()
    pca = result["models"].get("pca")
    valid = result.get("valid_mask", np.ones((h, w), bool))
    out: dict = {"pixel": (row, col), "valid": bool(valid[row, col])}

    # --- scores and ranks -------------------------------------------------------
    ranks = {}
    for name, s in {**result["scores"], "fused": result["fused"]}.items():
        ranks[name] = {"score": float(s[row, col]),
                       "percentile": float(100 * (s < s[row, col]).mean()),
                       "rank": int((s > s[row, col]).sum()) + 1, "of": int(s.size)}
    out["detectors"] = ranks
    out["flagged"] = bool(result["mask"][row, col])
    if result.get("confidence") is not None:
        out["agreement"] = float(result["confidence"][row, col])

    # --- fusion contributions ---------------------------------------------------
    finfo = result.get("fusion_info")
    if finfo:
        contrib = {k: finfo["weights"][k] * float(finfo["normalized"][k][row, col]) for k in finfo["detectors"]}
        tot = sum(contrib.values()) or 1.0
        out["fusion_share"] = {k: v / tot for k, v in contrib.items()}

    # --- RX band attribution ------------------------------------------------------
    rx = result["models"].get("rx")
    if rx is not None:
        c = rx.contributions(reduced[row, col])
        if pca is not None:  # component k's contribution spread over bands by squared loadings
            band_c = c @ (pca.components_ ** 2)
            out["rx_components"] = [{"component": int(i), "value": float(c[i])} for i in np.argsort(-c)[:top_k]]
        else:
            band_c = c
        out["rx_bands"] = _top(band_c, top_k, band_ids, wavelengths)

    # --- deviation from global background ----------------------------------------
    bg = clean[valid & ~result["mask"].astype(bool)]
    mu, sd = bg.mean(0), bg.std(0)
    z = (clean[row, col] - mu) / np.where(sd > 0, sd, 1)
    out["deviation_bands"] = _top(z, top_k, band_ids, wavelengths, absolute=True)
    out["max_abs_z"] = float(np.abs(z).max())

    # --- spatial: difference from neighbours --------------------------------------
    resid = result["data"].get("spatial_residual")
    if resid is not None:
        r = resid[row, col].astype(float)
        band_r = r @ pca.components_ if (pca is not None and r.size == pca.n_components_) else r
        out["spatial_bands"] = _top(band_r, top_k, band_ids, wavelengths, absolute=True)

    # --- autoencoder reconstruction error -----------------------------------------
    ae = result["models"].get("autoencoder")
    if ae is not None:
        out["autoencoder_bands"] = _top(ae.band_errors(clean[row, col]), top_k, band_ids, wavelengths)

    out["summary"] = _summary(out)
    return out


def _summary(e) -> str:
    f = e["detectors"]["fused"]
    s = [f"Pixel {e['pixel']} is {'FLAGGED' if e['flagged'] else 'not flagged'}; its fused score ranks "
         f"#{f['rank']:,} of {f['of']:,} pixels (higher than {f['percentile']:.1f}%)."]
    others = {k: v for k, v in e["detectors"].items() if k != "fused"}
    if others:
        best = min(others, key=lambda k: others[k]["rank"])
        s.append(f"The most suspicious view is '{best}' (rank #{others[best]['rank']:,}).")
    if "fusion_share" in e:
        lead = max(e["fusion_share"], key=e["fusion_share"].get)
        s.append(f"'{lead}' contributes {100 * e['fusion_share'][lead]:.0f}% of the fused score.")
    if "agreement" in e:
        s.append(f"{100 * e['agreement']:.0f}% of detectors place it in their top results "
                 "(relative confidence, not a probability).")
    if "rx_bands" in e:
        s.append(f"RX is driven mostly by {_band_text(e['rx_bands'][:3])}.")
    s.append(f"It deviates most from the background at {_band_text(e['deviation_bands'][:3])} "
             f"(up to {e['max_abs_z']:.1f} standard deviations).")
    if "spatial_bands" in e:
        s.append(f"Compared with its neighbours it differs most at {_band_text(e['spatial_bands'][:3])}.")
    if not e["valid"]:
        s.append("Warning: this pixel had invalid (NaN/inf) values that were repaired.")
    return " ".join(s)
