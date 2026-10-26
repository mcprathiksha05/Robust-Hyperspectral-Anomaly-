"""Local, evidence-grounded answers for the research dashboard."""

import math
from typing import Any, Dict, Optional, Sequence, Tuple


def answer_research_question(
    question: str,
    dataset_name: str,
    cube_shape: Tuple[int, int, int],
    positive_labels: Optional[int],
    selected_pixel: Tuple[int, int],
    selected_score: Optional[float],
    detector_name: str,
    metrics: Optional[Dict[str, Any]],
    runtime: Optional[Dict[str, Any]],
    runs: Sequence[Dict[str, Any]],
) -> str:
    """Answer a supported question using only active scene and recorded run data."""
    query = question.strip().lower()
    if not query:
        return "Enter a question to query the active scene or recorded experiment history."

    if any(word in query for word in ("dataset", "scene", "bands", "dimensions", "shape")):
        label_summary = (
            f"{positive_labels:,} positive reference pixels."
            if positive_labels is not None
            else "Reference labels are unavailable."
        )
        return (
            f"Active dataset `{dataset_name}` has shape {cube_shape} "
            f"(rows, columns, bands). {label_summary}"
        )

    if "pixel" in query or "score" in query or "spectrum" in query:
        if selected_score is None:
            return (
                f"Selected pixel {selected_pixel} in `{dataset_name}` has not been scored yet. "
                "Run a detector to obtain its anomaly score."
            )
        score_description = (
            "squared spectral reconstruction error outside its fitted PCA subspace"
            if detector_name == "PCA-Residual"
            else "regularized Mahalanobis distance from a scene-wide spectral background model"
        )
        return (
            f"Selected pixel {selected_pixel} in `{dataset_name}` has score "
            f"{selected_score:.6g} from {detector_name}; this is {score_description}. "
            "It is not a calibrated probability or proof of a target."
        )

    if (
        ("algorithm" in query or "model" in query)
        and "memory" not in query
        and "least" not in query
    ):
        return (
            "Implemented unsupervised anomaly models are Global RX (regularized spectral "
            "Mahalanobis distance) and PCA residual (squared reconstruction error outside a "
            "fitted spectral subspace). No supervised CNN or trained weights are available."
        )

    if "memory" in query or "least" in query:
        measured = []
        for run in runs:
            if run.get("status") != "success" or not run.get("memory"):
                continue
            value = run["memory"].get("peak_traced_mb")
            if isinstance(value, (int, float)) and math.isfinite(value):
                measured.append((float(value), run))
        if not measured:
            return "No successful run in the selected dataset history contains a measured memory value."
        value, run = min(measured, key=lambda item: item[0])
        return (
            f"The lowest recorded peak traced heap is {value:.3f} MiB for "
            f"{run.get('detector_name', 'unknown model')} (run {run.get('experiment_id', 'unknown')}). "
            "This compares traced Python allocations, not total process or GPU memory."
        )

    if "runtime" in query or "fast" in query:
        if runtime and isinstance(runtime.get("total_time_mean_sec"), (int, float)):
            return (
                f"The active {detector_name} run took "
                f"{runtime['total_time_mean_sec']:.6g} seconds total on `{dataset_name}`."
            )
        for run in runs:
            run_runtime = run.get("runtime") or {}
            total = run_runtime.get("total_time_mean_sec")
            if run.get("status") == "success" and isinstance(total, (int, float)):
                return (
                    f"The latest successful recorded run took {total:.6g} seconds: "
                    f"{run.get('detector_name', 'unknown model')} "
                    f"(run {run.get('experiment_id', 'unknown')})."
                )
        return "No measured runtime is available for the active scene's successful runs."

    if "compare" in query or "latest experiment" in query or "history" in query:
        successful = [run for run in runs if run.get("status") == "success"]
        if len(successful) < 2:
            return (
                f"History for `{dataset_name}` contains {len(successful)} successful runs; "
                "at least two are needed for a recorded comparison."
            )
        latest, previous = successful[:2]
        latest_metrics = latest.get("metrics") or {}
        previous_metrics = previous.get("metrics") or {}
        shared = [
            key for key in ("auc_roc", "auc_pr", "f1_max")
            if key in latest_metrics and key in previous_metrics
        ]
        if not shared:
            return "The two latest successful runs do not share comparable recorded metrics."
        deltas = ", ".join(
            f"{key}: {float(latest_metrics[key]) - float(previous_metrics[key]):+.4g}"
            for key in shared
            if _finite_number(latest_metrics[key]) and _finite_number(previous_metrics[key])
        )
        if not deltas:
            return "The two latest successful runs have no finite comparable metric values."
        return (
            f"Comparing the latest and previous successful runs for `{dataset_name}` "
            f"({latest.get('detector_name')} vs {previous.get('detector_name')}), "
            f"latest minus previous: {deltas}."
        )

    if "metric" in query or "auc" in query or "result" in query:
        if not metrics:
            return "No active evaluation metrics are available. Run a detector first."
        values = ", ".join(
            f"{key}={value:.4g}" if _finite_number(value) else f"{key}=unavailable"
            for key, value in metrics.items()
            if isinstance(value, (int, float))
        )
        return f"Active {detector_name} metrics for `{dataset_name}`: {values}."

    if "export" in query or "report" in query:
        return (
            "The Export Results section supports active-run CSV metrics, JSON provenance, "
            "compressed NPZ score maps, PNG maps, Markdown reports, and history CSV/JSON/XLSX."
        )

    return (
        "I can answer from the active dataset, selected pixel, implemented models, measured "
        "runtime/memory, recorded metrics, and export options. This local assistant has no "
        "language-model provider, so it cannot support that question from available records."
    )


def _finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(value)
