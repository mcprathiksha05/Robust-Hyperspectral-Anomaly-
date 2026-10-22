"""Tasks 17, 18, 19 - Ground-truth comparison, evaluation metrics, false-alarm analysis.

    from src.evaluation.metrics import compare_to_gt, mask_metrics, score_metrics, false_alarm_analysis

    cmp = compare_to_gt(mask, gt)            # task 17: TP/FP/FN/TN map + pixel and object counts
    m   = mask_metrics(mask, gt)             # task 18: precision, recall, F1, FPR, IoU ...
    s   = score_metrics(scores, gt)          # task 18: ROC-AUC, average precision, Pd at fixed FAR
    fa  = false_alarm_analysis(mask, gt, scores=scores)   # task 19

Pixel counts follow the usual convention: positive = target/anomaly pixel.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import label
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve

from src.utils.validation import HSIValidationError, validate_gt, validate_score_map

TN, TP, FP, FN = 0, 1, 2, 3
CATEGORY_NAMES = {TN: "true negative", TP: "true positive", FP: "false alarm", FN: "missed"}
_EIGHT = np.ones((3, 3), int)


def _prep(mask, gt):
    mask = np.asarray(mask)
    if mask.ndim != 2:
        raise HSIValidationError(f"Detection mask must be 2-D, got shape {mask.shape}.")
    gt = validate_gt(gt, mask.shape + (1,)).astype(bool)
    return mask.astype(bool), gt


# ------------------------------------------------------------------- task 17
def compare_to_gt(mask, gt, valid_mask=None) -> dict:
    """Pixel-by-pixel and object-by-object comparison of a detection mask with ground truth.

    Returns a ``category_map`` (0 TN, 1 TP, 2 FP, 3 FN) for colour-coded display,
    pixel counts, and object counts: a ground-truth object (connected region)
    counts as detected if any of its pixels is detected.
    """
    m, g = _prep(mask, gt)
    if valid_mask is not None:
        v = np.asarray(valid_mask, bool)
        m, g = m & v, g & v
    cat = np.zeros(m.shape, np.uint8)
    cat[m & g], cat[m & ~g], cat[~m & g] = TP, FP, FN
    gt_lab, n_obj = label(g, structure=_EIGHT)
    hit = np.unique(gt_lab[m & g])
    hit = hit[hit > 0]
    det_lab, n_det = label(m, structure=_EIGHT)
    fa_regions = len(set(np.unique(det_lab[m & ~g])) - set(np.unique(det_lab[m & g])) - {0})
    return {
        "category_map": cat, "category_names": CATEGORY_NAMES,
        "tp": int((cat == TP).sum()), "fp": int((cat == FP).sum()),
        "fn": int((cat == FN).sum()), "tn": int((cat == TN).sum()),
        "gt_objects": int(n_obj), "objects_detected": int(hit.size),
        "objects_missed": int(n_obj - hit.size),
        "object_detection_rate": float(hit.size / n_obj) if n_obj else float("nan"),
        "detected_regions": int(n_det), "false_alarm_regions": int(fa_regions),
    }


# ------------------------------------------------------------------- task 18
def mask_metrics(mask, gt) -> dict:
    """Threshold-dependent metrics for a binary detection mask."""
    m, g = _prep(mask, gt)
    tp, fp = int((m & g).sum()), int((m & ~g).sum())
    fn, tn = int((~m & g).sum()), int((~m & ~g).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else float("nan")
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    fpr = fp / (fp + tn) if fp + tn else float("nan")
    return {
        "precision": precision, "recall": recall, "f1": f1, "fpr": fpr,
        "accuracy": (tp + tn) / m.size, "iou": tp / (tp + fp + fn) if tp + fp + fn else float("nan"),
        "false_alarms_per_megapixel": fp / m.size * 1e6,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
    }


def score_metrics(scores, gt, fars=(1e-3, 1e-2, 5e-2), curves: bool = False) -> dict:
    """Threshold-free metrics for a continuous score map.

    roc_auc  - probability a random target pixel outranks a random background pixel
    ap       - average precision (area under precision-recall); sensitive to false alarms
    pd_at_far_X - detection rate when the false-alarm rate is limited to X
    """
    s = validate_score_map(scores)
    g = validate_gt(gt, s.shape + (1,)).ravel().astype(bool)
    if g.all() or not g.any():
        raise HSIValidationError("ROC-AUC needs both target and background pixels in the ground truth.")
    v = s.ravel()
    fpr, tpr, thr = roc_curve(g, v)
    out = {"roc_auc": float(roc_auc_score(g, v)), "ap": float(average_precision_score(g, v)),
           "positive_fraction": float(g.mean())}
    for far in fars:
        out[f"pd_at_far_{far:g}"] = float(np.interp(far, fpr, tpr))
    if curves:
        p, r, _ = precision_recall_curve(g, v)
        out["roc_curve"] = {"fpr": fpr, "tpr": tpr, "thresholds": thr}
        out["pr_curve"] = {"precision": p, "recall": r}
    return out


# ------------------------------------------------------------------- task 19
def false_alarm_analysis(mask, gt, scores=None, scene_labels=None, label_names=None,
                         percents=(0.1, 0.5, 1, 2, 5, 10)) -> dict:
    """How often, where, and in what form background pixels are wrongly flagged.

    - counts and rates: false alarms, FPR, false alarms per megapixel
    - shape: number of separate false-alarm regions and their sizes (speckle vs blobs)
    - by material (if ``scene_labels`` given, e.g. MUUFL's land-cover map)
    - trade-off table (if ``scores`` given): FPR and recall when flagging the top p% of pixels,
      to choose a threshold that balances missed targets against false alarms
    """
    m, g = _prep(mask, gt)
    fa = m & ~g
    n_fa = int(fa.sum())
    lab, n_reg = label(fa, structure=_EIGHT)
    sizes = np.bincount(lab.ravel())[1:] if n_reg else np.array([], int)
    out = {
        "false_alarms": n_fa, "background_pixels": int((~g).sum()),
        "fpr": n_fa / max(int((~g).sum()), 1),
        "false_alarms_per_megapixel": n_fa / m.size * 1e6,
        "false_alarm_share_of_detections": n_fa / max(int(m.sum()), 1),
        "false_alarm_regions": int(n_reg),
        "single_pixel_false_alarms": int((sizes == 1).sum()),
        "largest_false_alarm_region": int(sizes.max()) if n_reg else 0,
        "mean_false_alarm_region_size": float(sizes.mean()) if n_reg else 0.0,
        "false_alarm_map": fa.astype(np.uint8),
    }
    if scene_labels is not None:
        sl = np.asarray(scene_labels)
        if sl.shape != m.shape:
            raise HSIValidationError("scene_labels must have the same height and width as the mask.")
        by = []
        for v in np.unique(sl[~g]):
            region = (sl == v) & ~g
            if label_names is not None and 1 <= v <= len(label_names):
                name = label_names[int(v) - 1]
            else:
                name = "unlabelled" if v < 0 else str(v)
            by.append({"label": int(v), "name": str(name), "background_pixels": int(region.sum()),
                       "false_alarms": int((fa & region).sum()),
                       "fpr": float((fa & region).sum() / max(region.sum(), 1))})
        out["by_label"] = sorted(by, key=lambda r: -r["false_alarms"])
    if scores is not None:
        s = validate_score_map(scores)
        rows = []
        for p in percents:
            thr = np.percentile(s, 100 - p)
            d = s > thr
            rows.append({"top_percent": p, "threshold": float(thr),
                         "fpr": float((d & ~g).sum() / max((~g).sum(), 1)),
                         "recall": float((d & g).sum() / max(g.sum(), 1)),
                         "false_alarms": int((d & ~g).sum())})
        out["tradeoff"] = rows
    return out


def evaluate(scores, mask, gt, valid_mask=None) -> dict:
    """All metrics in one flat dict (handy for tables and the dashboard)."""
    out = {}
    out.update(score_metrics(scores, gt))
    out.update(mask_metrics(mask, gt))
    c = compare_to_gt(mask, gt, valid_mask)
    out.update({k: c[k] for k in ("gt_objects", "objects_detected", "object_detection_rate",
                                  "false_alarm_regions")})
    return out
