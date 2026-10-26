"""Feature 5: Cross-scene testing and generalization framework.

Evaluates detector performance across multiple benchmark hyperspectral scenes.
Enforces strict distinction between:
    - Within-scene benchmarking: Fitting on scene S and evaluating on scene S.
    - Cross-scene transfer: Fitting on source scene S and evaluating on target scene T (S != T).

Spectral Band Compatibility Protocol:
    Hyperspectral sensors and scene crops feature differing band counts (e.g., AVIRIS
    Airport-1 has 205 bands, San Diego has 189 bands, and ROSIS Beach-4 has 102 bands).
    Detectors fitted on B_source spectral channels cannot predict on B_target channels
    without a documented alignment strategy.
    By default, incompatible band counts are safely flagged and skipped rather than
    silently distorted.
"""

from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple, Type
import numpy as np

from src.data_loading.load_hsi import load, list_datasets
from src.models.base import BaseDetector
from src.models.rx import RXDetector
from src.evaluation.metrics import evaluate_anomaly_map


def evaluate_within_scene_benchmark(
    dataset_names: Optional[Sequence[str]] = None,
    detector_cls: Type[BaseDetector] = RXDetector,
    detector_kwargs: Optional[Dict[str, Any]] = None,
    target_tpr: float = 0.95,
) -> Dict[str, Any]:
    """Benchmark a detector across multiple scenes, fitting and evaluating within each scene.

    Args:
        dataset_names: Sequence of scene names. If None, uses all available anomaly datasets.
        detector_cls: Model class adhering to BaseDetector.
        detector_kwargs: Keyword arguments passed to detector constructor.
        target_tpr: Operating point TPR for false alarm calculation (default 0.95).

    Returns:
        dict: Structured report containing:
            - 'per_scene_results': List of per-scene evaluation dictionaries
            - 'aggregate_summary': Mean, std, min, and max for each metric across scenes
            - 'total_scenes_evaluated': Integer count
    """
    if dataset_names is None:
        dataset_names = [d for d in list_datasets() if not d.startswith("muufl")]

    kwargs = detector_kwargs or {}
    results = []

    for name in dataset_names:
        cube, gt, meta = load(name, as_float=True)
        h, w, b = cube.shape

        detector = detector_cls(**kwargs)
        scores = detector.fit_predict(cube)

        metrics = evaluate_anomaly_map(gt, scores, target_tpr=target_tpr) if gt is not None else None

        record: Dict[str, Any] = {
            "scene_name": name,
            "transfer_type": "within_scene",
            "spatial_shape": (h, w),
            "bands": b,
            "anomaly_pixels": int(np.sum(gt)) if gt is not None else None,
            "anomaly_pct": float(np.mean(gt) * 100.0) if gt is not None else None,
            "metrics": metrics,
        }
        results.append(record)

    # Compute aggregate metrics
    evaluated_metrics = [r["metrics"] for r in results if r["metrics"] is not None]
    auc_list = [metrics["auc_roc"] for metrics in evaluated_metrics]
    pr_list = [metrics["auc_pr"] for metrics in evaluated_metrics]
    f1_list = [metrics["f1_max"] for metrics in evaluated_metrics]
    far_list = [metrics["false_alarms_per_mp"] for metrics in evaluated_metrics]

    summary = {
        "mean_auc_roc": float(np.mean(auc_list)) if auc_list else None,
        "std_auc_roc": float(np.std(auc_list)) if auc_list else None,
        "min_auc_roc": float(np.min(auc_list)) if auc_list else None,
        "max_auc_roc": float(np.max(auc_list)) if auc_list else None,
        "mean_auc_pr": float(np.mean(pr_list)) if pr_list else None,
        "std_auc_pr": float(np.std(pr_list)) if pr_list else None,
        "mean_f1_max": float(np.mean(f1_list)) if f1_list else None,
        "std_f1_max": float(np.std(f1_list)) if f1_list else None,
        "mean_false_alarms_per_mp": float(np.mean(far_list)) if far_list else None,
    }

    return {
        "per_scene_results": results,
        "aggregate_summary": summary,
        "total_scenes_evaluated": len(results),
    }


def evaluate_cross_scene_transfer(
    source_scene: str,
    target_scenes: Sequence[str],
    detector_cls: Type[BaseDetector] = RXDetector,
    detector_kwargs: Optional[Dict[str, Any]] = None,
    band_mismatch_strategy: Literal["skip", "truncate_min"] = "skip",
    target_tpr: float = 0.95,
) -> Dict[str, Any]:
    """Evaluate genuine cross-scene transfer from a single source scene to multiple target scenes.

    The model background parameters are learned exclusively on source_scene, and then
    evaluated without re-fitting on each target scene.

    Args:
        source_scene: Dataset name used to fit the background model.
        target_scenes: Sequence of dataset names to evaluate on.
        detector_cls: Model class adhering to BaseDetector.
        detector_kwargs: Keyword arguments for detector constructor.
        band_mismatch_strategy:
            - 'skip': Flag and skip scenes where B_target != B_source.
            - 'truncate_min': Truncate both source and target to min(B_src, B_tgt) bands.
        target_tpr: Operating point TPR for false alarm calculation.

    Returns:
        dict: Detailed transfer report containing per-target records and cross-scene summaries.
    """
    kwargs = detector_kwargs or {}

    # Load source scene and fit detector
    src_cube, src_gt, _ = load(source_scene, as_float=True)
    b_src = src_cube.shape[2]

    detector = detector_cls(**kwargs)

    transfer_results = []

    for tgt_name in target_scenes:
        tgt_cube, tgt_gt, _ = load(tgt_name, as_float=True)
        b_tgt = tgt_cube.shape[2]

        is_same_scene = (tgt_name == source_scene)
        transfer_type = "within_scene" if is_same_scene else "cross_scene"

        if b_src != b_tgt:
            if band_mismatch_strategy == "skip":
                record = {
                    "source_scene": source_scene,
                    "target_scene": tgt_name,
                    "transfer_type": transfer_type,
                    "source_bands": b_src,
                    "target_bands": b_tgt,
                    "status": "skipped_band_mismatch",
                    "reason": f"Band count mismatch: source has {b_src} bands, target has {b_tgt} bands.",
                    "metrics": None,
                }
                transfer_results.append(record)
                continue
            elif band_mismatch_strategy == "truncate_min":
                b_min = min(b_src, b_tgt)
                fit_cube = src_cube[:, :, :b_min]
                eval_cube = tgt_cube[:, :, :b_min]
                detector.fit(fit_cube)
                scores = detector.predict(eval_cube)
            else:
                raise ValueError(f"Unknown band_mismatch_strategy: {band_mismatch_strategy}")
        else:
            # Bands match: fit on source, predict on target
            detector.fit(src_cube)
            scores = detector.predict(tgt_cube)

        metrics = (
            evaluate_anomaly_map(tgt_gt, scores, target_tpr=target_tpr)
            if tgt_gt is not None
            else None
        )

        record = {
            "source_scene": source_scene,
            "target_scene": tgt_name,
            "transfer_type": transfer_type,
            "source_bands": b_src,
            "target_bands": b_tgt,
            "status": "success",
            "metrics": metrics,
        }
        transfer_results.append(record)

    # Compute summary strictly over genuine cross-scene successful runs
    cross_success = [
        r for r in transfer_results
        if r["status"] == "success"
        and r["transfer_type"] == "cross_scene"
        and r["metrics"] is not None
    ]

    if cross_success:
        cross_auc = [r["metrics"]["auc_roc"] for r in cross_success]
        cross_summary = {
            "mean_cross_auc_roc": float(np.mean(cross_auc)),
            "std_cross_auc_roc": float(np.std(cross_auc)),
            "min_cross_auc_roc": float(np.min(cross_auc)),
            "max_cross_auc_roc": float(np.max(cross_auc)),
            "successful_cross_transfers": len(cross_success),
        }
    else:
        cross_summary = {
            "mean_cross_auc_roc": float(np.nan),
            "successful_cross_transfers": 0,
        }

    return {
        "source_scene": source_scene,
        "target_results": transfer_results,
        "cross_scene_summary": cross_summary,
        "total_targets_evaluated": len(transfer_results),
    }


def evaluate_cross_scene_matrix(
    scene_names: Sequence[str],
    detector_cls: Type[BaseDetector] = RXDetector,
    detector_kwargs: Optional[Dict[str, Any]] = None,
    target_tpr: float = 0.95,
) -> Dict[str, Any]:
    """Compute pairwise cross-scene transfer matrix M[i, j] across band-compatible scenes.

    M[i, j] represents performance when fitting on scene i and predicting on scene j.
    The diagonal M[i, i] corresponds to within-scene baseline performance.

    Args:
        scene_names: Sequence of scene names that share identical spectral band counts.
        detector_cls: Model class adhering to BaseDetector.
        detector_kwargs: Model parameters.
        target_tpr: Operating point TPR.

    Returns:
        dict: Matrix dictionary containing 'scenes', 'auc_matrix', 'f1_matrix'.
    """
    n = len(scene_names)
    auc_matrix = np.full((n, n), np.nan, dtype=np.float64)
    f1_matrix = np.full((n, n), np.nan, dtype=np.float64)

    cubes = []
    gts = []
    for s in scene_names:
        c, g, _ = load(s, as_float=True)
        cubes.append(c)
        gts.append(g)

    # Verify all chosen scenes share identical band count
    band_counts = [c.shape[2] for c in cubes]
    if len(set(band_counts)) > 1:
        raise ValueError(
            f"All scenes in matrix must share identical band counts. Got: {dict(zip(scene_names, band_counts))}"
        )

    kwargs = detector_kwargs or {}
    detector = detector_cls(**kwargs)

    for i in range(n):
        detector.fit(cubes[i])
        for j in range(n):
            scores = detector.predict(cubes[j])
            if gts[j] is None:
                continue
            m = evaluate_anomaly_map(gts[j], scores, target_tpr=target_tpr)
            auc_matrix[i, j] = m["auc_roc"]
            f1_matrix[i, j] = m["f1_max"]

    return {
        "scenes": list(scene_names),
        "auc_matrix": auc_matrix.tolist(),
        "f1_matrix": f1_matrix.tolist(),
        "bands": band_counts[0],
    }
