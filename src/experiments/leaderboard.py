"""Feature 10: Model Leaderboard & Comparative Analytics.

Builds dynamic, transparent model leaderboards directly from genuine experiment records:
    - Multi-metric ranking across ROC-AUC, PR-AUC, F1-max, FPR@95% TPR, and FA/Mpx.
    - Transparent configuration and regularization display.
    - Phase runtime and memory footprint reporting.
    - Filtering by dataset and detector family.
    - Clear visibility of metric definitions and ground-truth dependencies.
    - Strict avoidance of fabricated rows or ungrounded comparisons.
"""

from typing import Any, Dict, List, Optional, Sequence
import pandas as pd
import numpy as np


METRIC_DEFINITIONS: Dict[str, Dict[str, str]] = {
    "auc_roc": {
        "name": "ROC-AUC",
        "description": "Area Under Receiver Operating Characteristic Curve",
        "formula": "Integral of TPR vs. FPR across all decision thresholds",
        "range": "[0.0, 1.0] (1.0 = perfect ranking; 0.5 = random chance)",
        "ground_truth_required": "Yes (Requires binary target mask)",
    },
    "auc_pr": {
        "name": "PR-AUC (Average Precision)",
        "description": "Area Under Precision-Recall Curve",
        "formula": "Weighted mean of precisions achieved at each threshold",
        "range": "[0.0, 1.0] (Reflects target rarity and precision trade-off)",
        "ground_truth_required": "Yes (Requires binary target mask)",
    },
    "f1_max": {
        "name": "Maximum F1 Score",
        "description": "Peak harmonic mean of Precision and Recall across all operational thresholds",
        "formula": "max_t 2 * (P(t) * R(t)) / (P(t) + R(t))",
        "range": "[0.0, 1.0]",
        "ground_truth_required": "Yes (Discovered post-hoc with ground truth)",
    },
    "fpr_at_95_tpr": {
        "name": "FPR @ 95% TPR",
        "description": "False Positive Rate required to detect at least 95% of target pixels",
        "formula": "FP / (FP + TN) at threshold where TP / (TP + FN) >= 0.95",
        "range": "[0.0, 1.0] (Lower is better)",
        "ground_truth_required": "Yes",
    },
    "false_alarms_per_megapixel": {
        "name": "False Alarms per Megapixel",
        "description": "Count of false alarm background pixels normalized per 1,000,000 spatial pixels",
        "formula": "FP_count * (1,000,000 / total_pixels)",
        "range": ">= 0.0 (Lower is better)",
        "ground_truth_required": "Yes",
    },
    "total_time_ms": {
        "name": "Total Latency (ms)",
        "description": "Total execution latency (fit + predict phases)",
        "formula": "time.perf_counter() wall-clock measurement in milliseconds",
        "range": "> 0.0 ms (Lower is better)",
        "ground_truth_required": "No (Unsupervised measurement)",
    },
    "peak_traced_mb": {
        "name": "Peak Traced Heap (MB)",
        "description": "Peak RAM allocated through Python's allocator during execution",
        "formula": "tracemalloc.get_traced_memory()[1] / (1024 * 1024)",
        "range": ">= 0.0 MB",
        "ground_truth_required": "No (Unsupervised measurement)",
    },
    "rss_delta_mb": {
        "name": "Process RSS Delta (MB)",
        "description": "Net change in OS Resident Set Size physical process memory",
        "formula": "psutil.Process().memory_info().rss change",
        "range": "Real value (Subject to OS page retention & CRT caches)",
        "ground_truth_required": "No (Unsupervised measurement)",
    },
}


LEADERBOARD_COLUMNS = [
    "Run ID",
    "Dataset",
    "Detector",
    "Regularization",
    "Solver",
    "Status",
    "ROC-AUC",
    "PR-AUC",
    "Max F1",
    "FPR@95% TPR",
    "FA / Megapixel",
    "Total Time (ms)",
    "Peak Traced (MB)",
    "RSS Delta (MB)",
    "Timestamp",
]


def get_metric_definitions() -> Dict[str, Dict[str, str]]:
    """Return comprehensive metric definitions and scientific context."""
    return METRIC_DEFINITIONS


def build_leaderboard(
    runs: Sequence[Dict[str, Any]],
    dataset_filter: Optional[str] = None,
    detector_filter: Optional[str] = None,
    status_filter: Optional[str] = "success",
    sort_by: str = "ROC-AUC",
    ascending: bool = False,
) -> pd.DataFrame:
    """Construct a clean, formatted Pandas DataFrame leaderboard from actual run records.

    Args:
        runs: Sequence of run dictionaries retrieved from ExperimentHistoryDB.
        dataset_filter: Optional dataset name to filter on.
        detector_filter: Optional detector name to filter on.
        status_filter: 'success', 'failed', 'all', or None.
        sort_by: Column name to sort by (e.g. 'ROC-AUC', 'Total Time (ms)').
        ascending: Whether sort is ascending.

    Returns:
        pd.DataFrame: Formatted leaderboard.
    """
    if len(runs) == 0:
        return pd.DataFrame(columns=LEADERBOARD_COLUMNS)

    rows = []
    for r in runs:
        # Filtering
        ds = r.get("dataset_name", "Unknown")
        det = r.get("detector_name", "Unknown")
        st = r.get("status", "unknown")

        if dataset_filter and ds != dataset_filter:
            continue
        if detector_filter and det != detector_filter:
            continue
        if status_filter and status_filter != "all" and st != status_filter:
            continue

        params = r.get("parameters") or {}
        reg = params.get("regularization")
        reg_str = f"{reg:1.0e}" if isinstance(reg, (int, float)) else str(reg or "default")
        solver = str(params.get("method") or params.get("solver_method") or "cholesky")

        metrics = r.get("metrics") or {}
        runtime = r.get("runtime") or {}
        memory = r.get("memory") or {}

        auc = metrics.get("auc_roc")
        pr = metrics.get("auc_pr")
        f1 = metrics.get("f1_max")
        fpr = metrics.get("fpr_at_95_tpr")
        fa_mpx = metrics.get("false_alarms_per_mp")
        if fa_mpx is None:
            fa_mpx = metrics.get("false_alarms_per_megapixel")

        # Runtime
        total_time_ms = runtime.get("total_time_mean_sec")
        if total_time_ms is not None:
            total_time_ms = total_time_ms * 1000.0
        elif runtime.get("total_elapsed_ms") is not None:
            total_time_ms = runtime.get("total_elapsed_ms")

        # Memory
        peak_mb = memory.get("peak_traced_mb")
        rss_mb = memory.get("rss_delta_mb")

        ts_str = str(r.get("timestamp", ""))[:19].replace("T", " ")

        row = {
            "Run ID": r.get("experiment_id", "N/A"),
            "Dataset": ds,
            "Detector": det,
            "Regularization": reg_str,
            "Solver": solver,
            "Status": st,
            "ROC-AUC": round(float(auc), 4) if auc is not None else np.nan,
            "PR-AUC": round(float(pr), 4) if pr is not None else np.nan,
            "Max F1": round(float(f1), 4) if f1 is not None else np.nan,
            "FPR@95% TPR": round(float(fpr), 4) if fpr is not None else np.nan,
            "FA / Megapixel": round(float(fa_mpx), 1) if fa_mpx is not None else np.nan,
            "Total Time (ms)": round(float(total_time_ms), 1) if total_time_ms is not None else np.nan,
            "Peak Traced (MB)": round(float(peak_mb), 2) if peak_mb is not None else np.nan,
            "RSS Delta (MB)": round(float(rss_mb), 2) if rss_mb is not None else np.nan,
            "Timestamp": ts_str,
        }
        rows.append(row)

    if not rows:
        return pd.DataFrame(columns=LEADERBOARD_COLUMNS)

    df = pd.DataFrame(rows)

    if sort_by in df.columns:
        df = df.sort_values(by=sort_by, ascending=ascending, na_position="last")

    return df.reset_index(drop=True)
