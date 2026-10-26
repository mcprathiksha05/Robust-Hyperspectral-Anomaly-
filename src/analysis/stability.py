"""Feature 7: Stability testing and multi-seed variability analysis.

Evaluates metric variance across repeated stochastic trials and random seeds.
Distinguishes between:
    - Corruption-induced variability: Deterministic detector evaluated on stochastically
      perturbed inputs across varying seeds.
    - Detector-induced variability: Stochastic detector (e.g. randomized initialization
      or random projection) evaluated on fixed input.

Failure & Non-Finite Handling:
    Failed or non-finite runs are NEVER silently discarded; they are explicitly
    recorded, categorized, and counted in the final stability audit.
"""

from typing import Any, Callable, Dict, List, Literal, Optional, Sequence, Tuple, Type
import numpy as np

from src.models.base import BaseDetector
from src.models.rx import RXDetector
from src.evaluation.metrics import evaluate_anomaly_map


def compute_stability_summary(
    records: List[Dict[str, Any]], metric_keys: Optional[Sequence[str]] = None
) -> Dict[str, Any]:
    """Aggregate statistics over a list of trial records, safely handling failed/non-finite runs.

    Args:
        records: List of per-run dictionaries with 'status' and 'metrics'.
        metric_keys: Sequence of metric keys to summarize.

    Returns:
        dict: Summary statistics including mean, std, min, max, valid count, and failed count.
    """
    if metric_keys is None:
        metric_keys = ["auc_roc", "auc_pr", "f1_max", "fpr_at_95_tpr", "false_alarms_per_mp"]

    valid_runs = [r for r in records if r.get("status") == "success" and r.get("metrics") is not None]
    failed_runs = [r for r in records if r.get("status") != "success"]

    summary: Dict[str, Any] = {
        "total_trials": len(records),
        "valid_trials": len(valid_runs),
        "failed_trials": len(failed_runs),
        "failure_rate": float(len(failed_runs) / len(records)) if len(records) > 0 else 0.0,
    }

    if len(valid_runs) == 0:
        for k in metric_keys:
            summary[f"mean_{k}"] = float(np.nan)
            summary[f"std_{k}"] = float(np.nan)
        return summary

    for k in metric_keys:
        vals = [r["metrics"][k] for r in valid_runs if k in r["metrics"] and np.isfinite(r["metrics"][k])]
        if len(vals) > 0:
            summary[f"mean_{k}"] = float(np.mean(vals))
            summary[f"std_{k}"] = float(np.std(vals))
            summary[f"min_{k}"] = float(np.min(vals))
            summary[f"max_{k}"] = float(np.max(vals))
        else:
            summary[f"mean_{k}"] = float(np.nan)
            summary[f"std_{k}"] = float(np.nan)

    return summary


def evaluate_corruption_stability(
    cube: np.ndarray,
    gt: np.ndarray,
    transform_fn: Callable[[np.ndarray, Optional[int]], Tuple[np.ndarray, Dict[str, Any]]],
    n_runs: int = 10,
    base_seed: int = 100,
    detector_cls: Type[BaseDetector] = RXDetector,
    detector_kwargs: Optional[Dict[str, Any]] = None,
    target_tpr: float = 0.95,
) -> Dict[str, Any]:
    """Evaluate stability under stochastic environmental/sensor corruption across varying seeds.

    Args:
        cube: Hyperspectral cube of shape (H, W, B).
        gt: Ground truth mask (H, W).
        transform_fn: Callable taking (cube, seed=...) and returning (corrupted_cube, metadata).
        n_runs: Number of independent stochastic runs.
        base_seed: Base integer seed (run i uses base_seed + i).
        detector_cls: Detector class adhering to BaseDetector.
        detector_kwargs: Model parameters.
        target_tpr: Operating point TPR.

    Returns:
        dict: Stability report containing per-run records, summary statistics, and variability source.
    """
    if n_runs <= 0:
        raise ValueError(f"n_runs must be positive, got {n_runs}")

    kwargs = detector_kwargs or {}
    detector = detector_cls(**kwargs)

    records = []

    for i in range(n_runs):
        seed = base_seed + i
        run_record: Dict[str, Any] = {
            "run_index": i,
            "seed": seed,
            "status": "pending",
            "metrics": None,
            "error": None,
        }

        try:
            corrupted, t_meta = transform_fn(cube, seed=seed)
            scores = detector.fit_predict(corrupted)

            if not np.all(np.isfinite(scores)):
                run_record["status"] = "non_finite_scores"
                run_record["error"] = "Detector output contains NaN or Inf values."
            else:
                metrics = evaluate_anomaly_map(gt, scores, target_tpr=target_tpr)
                run_record["status"] = "success"
                run_record["metrics"] = metrics
        except Exception as e:
            run_record["status"] = "failed"
            run_record["error"] = str(e)

        records.append(run_record)

    summary = compute_stability_summary(records)

    return {
        "variability_source": "stochastic_corruption",
        "n_runs": n_runs,
        "base_seed": base_seed,
        "records": records,
        "summary": summary,
    }


def evaluate_detector_stability(
    cube: np.ndarray,
    gt: np.ndarray,
    detector_factory: Callable[[int], BaseDetector],
    n_runs: int = 10,
    base_seed: int = 100,
    target_tpr: float = 0.95,
) -> Dict[str, Any]:
    """Evaluate stability of a stochastic detector on fixed, clean input across random seeds.

    Args:
        cube: Hyperspectral cube of shape (H, W, B).
        gt: Ground truth mask (H, W).
        detector_factory: Callable mapping seed -> instantiated BaseDetector.
        n_runs: Number of independent runs.
        base_seed: Base seed.
        target_tpr: Operating point TPR.

    Returns:
        dict: Stability report categorized as variability_source='stochastic_detector'.
    """
    if n_runs <= 0:
        raise ValueError(f"n_runs must be positive, got {n_runs}")

    records = []

    for i in range(n_runs):
        seed = base_seed + i
        run_record: Dict[str, Any] = {
            "run_index": i,
            "seed": seed,
            "status": "pending",
            "metrics": None,
            "error": None,
        }

        try:
            det = detector_factory(seed)
            scores = det.fit_predict(cube)

            if not np.all(np.isfinite(scores)):
                run_record["status"] = "non_finite_scores"
                run_record["error"] = "Detector output contains NaN or Inf values."
            else:
                metrics = evaluate_anomaly_map(gt, scores, target_tpr=target_tpr)
                run_record["status"] = "success"
                run_record["metrics"] = metrics
        except Exception as e:
            run_record["status"] = "failed"
            run_record["error"] = str(e)

        records.append(run_record)

    summary = compute_stability_summary(records)

    return {
        "variability_source": "stochastic_detector",
        "n_runs": n_runs,
        "base_seed": base_seed,
        "records": records,
        "summary": summary,
    }
