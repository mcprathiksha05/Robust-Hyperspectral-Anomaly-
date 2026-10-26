"""Feature 14: Multi-Format Experiment Result Exporter.

Supports comprehensive, reproducible export of completed hyperspectral experiments:
    - CSV: Tabular metrics, leaderboard tables, and run histories.
    - JSON: Full experiment provenance, hyperparameters, and environment metadata.
    - NPZ: Raw numerical anomaly score maps with spatial/spectral metadata.
    - PNG: High-resolution visual anomaly maps with colorbars.
    - Excel XLSX: Formatted multi-tab or single-tab metric workbooks.
    - Markdown: Downloadable, publication-ready experiment summary reports.

Rules & Protections:
    - Path and filename validation.
    - Overwrite prevention (raises FileExistsError unless overwrite=True).
    - Full provenance retention (dataset, detector, regularization, metrics, seeds).
    - Scientific caveats included in reports and metadata.
"""

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Non-interactive headless backend
import matplotlib.pyplot as plt

from src.experiments.history import _json_serializable
from src.experiments.leaderboard import LEADERBOARD_COLUMNS, build_leaderboard


DEFAULT_EXPORTS_DIR = Path(__file__).resolve().parents[2] / "exports"


def _validate_output_path(
    filepath: Union[str, Path],
    expected_extension: Optional[str] = None,
    overwrite: bool = False,
) -> Path:
    """Validate output filepath, create parent directory, and enforce overwrite policy."""
    path = Path(filepath)
    if not path.name.strip():
        raise ValueError("Filepath cannot have an empty filename")

    if expected_extension:
        ext = expected_extension if expected_extension.startswith(".") else f".{expected_extension}"
        if path.suffix.lower() != ext.lower():
            path = path.with_suffix(ext)

    if path.exists() and not overwrite:
        raise FileExistsError(
            f"File already exists at '{path}'. Set overwrite=True to replace it."
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def export_metrics_csv(
    metrics: Dict[str, Any],
    filepath: Union[str, Path],
    metadata: Optional[Dict[str, Any]] = None,
    overwrite: bool = False,
) -> Path:
    """Export a metrics dictionary with provenance metadata to CSV.

    Args:
        metrics: Dictionary of numerical evaluation metrics.
        filepath: Target CSV file path.
        metadata: Optional experiment metadata dictionary.
        overwrite: Whether to overwrite existing file.

    Returns:
        Path: Path to written file.
    """
    if not isinstance(metrics, dict) or len(metrics) == 0:
        raise ValueError("Metrics dictionary must be a non-empty dict")

    path = _validate_output_path(filepath, expected_extension=".csv", overwrite=overwrite)

    rows = []
    # Provenance metadata headers
    if metadata:
        for k, v in metadata.items():
            rows.append({"Category": "Metadata", "Field": str(k), "Value": str(v)})

    # Metrics
    for k, v in metrics.items():
        rows.append({"Category": "Metric", "Field": str(k), "Value": str(v)})

    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)
    return path


def export_runs_csv(
    runs_or_df: Union[Sequence[Dict[str, Any]], pd.DataFrame],
    filepath: Union[str, Path],
    overwrite: bool = False,
) -> Path:
    """Export a collection of runs or a leaderboard DataFrame to CSV."""
    path = _validate_output_path(filepath, expected_extension=".csv", overwrite=overwrite)

    if isinstance(runs_or_df, pd.DataFrame):
        df = runs_or_df
    else:
        df = build_leaderboard(runs_or_df, status_filter="all")

    df.to_csv(path, index=False)
    return path


def export_experiment_json(
    experiment_record: Dict[str, Any],
    filepath: Union[str, Path],
    overwrite: bool = False,
) -> Path:
    """Export full experiment provenance and results to structured JSON."""
    if not isinstance(experiment_record, dict):
        raise ValueError("experiment_record must be a dictionary")

    path = _validate_output_path(filepath, expected_extension=".json", overwrite=overwrite)

    serializable = _json_serializable(experiment_record)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(serializable, f, indent=2)

    return path


def export_anomaly_map_npz(
    score_map: np.ndarray,
    metadata: Dict[str, Any],
    filepath: Union[str, Path],
    overwrite: bool = False,
) -> Path:
    """Export 2D anomaly score map and associated parameters to a compressed NPZ archive."""
    if not isinstance(score_map, np.ndarray) or score_map.ndim != 2:
        raise ValueError(f"score_map must be a 2D numpy array, got {getattr(score_map, 'shape', type(score_map))}")

    path = _validate_output_path(filepath, expected_extension=".npz", overwrite=overwrite)

    # Save score map as float64 and metadata fields as strings/arrays
    meta_clean = _json_serializable(metadata or {})
    np.savez_compressed(
        path,
        score_map=score_map.astype(np.float64),
        metadata_json=json.dumps(meta_clean),
    )
    return path


def export_anomaly_map_png(
    score_map: np.ndarray,
    filepath: Union[str, Path],
    title: Optional[str] = "Hyperspectral Anomaly Score Map",
    colormap: str = "viridis",
    dpi: int = 200,
    overwrite: bool = False,
) -> Path:
    """Export 2D anomaly score map as a high-resolution PNG image with colorbar."""
    if not isinstance(score_map, np.ndarray) or score_map.ndim != 2:
        raise ValueError(f"score_map must be a 2D numpy array, got {getattr(score_map, 'shape', type(score_map))}")

    path = _validate_output_path(filepath, expected_extension=".png", overwrite=overwrite)

    fig, ax = plt.subplots(figsize=(6, 5), dpi=dpi)
    im = ax.imshow(score_map, cmap=colormap, origin="upper")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Anomaly Score (Statistical Distance)", rotation=270, labelpad=15)
    ax.set_title(title or "Anomaly Map", fontsize=11)
    ax.set_xlabel("Pixel Column")
    ax.set_ylabel("Pixel Row")
    plt.tight_layout()
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    return path


def export_runs_excel(
    runs_or_df: Union[Sequence[Dict[str, Any]], pd.DataFrame],
    filepath: Union[str, Path],
    overwrite: bool = False,
) -> Path:
    """Export experiment runs to Microsoft Excel (.xlsx) workbook."""
    path = _validate_output_path(filepath, expected_extension=".xlsx", overwrite=overwrite)

    if isinstance(runs_or_df, pd.DataFrame):
        df = runs_or_df
    else:
        df = build_leaderboard(runs_or_df, status_filter="all")

    df.to_excel(path, index=False, engine="openpyxl")
    return path


def generate_experiment_report(
    experiment: Dict[str, Any],
    filepath: Optional[Union[str, Path]] = None,
    overwrite: bool = False,
) -> str:
    """Generate a comprehensive Markdown experiment report.

    Args:
        experiment: Complete experiment record dict.
        filepath: Optional path to save the report to disk.
        overwrite: Whether to overwrite existing file.

    Returns:
        str: Markdown report content.
    """
    exp_id = experiment.get("experiment_id", "N/A")
    ts = experiment.get("timestamp", datetime.now(timezone.utc).isoformat())
    ds = experiment.get("dataset_name", "Unknown")
    det = experiment.get("detector_name", "Unknown")
    status = experiment.get("status", "unknown")
    params = experiment.get("parameters") or {}
    metrics = experiment.get("metrics") or {}
    runtime = experiment.get("runtime") or {}
    memory = experiment.get("memory") or {}
    sw = experiment.get("software_info") or {}

    lines = [
        f"# Experiment Report: `{exp_id}`",
        "",
        f"- **Project:** Robust Hyperspectral Anomaly and Target Detection",
        f"- **Timestamp:** {ts}",
        f"- **Status:** `{status.upper()}`",
        f"- **Dataset:** `{ds}`",
        f"- **Detector:** `{det}`",
        "",
        "## 1. Experimental Configuration",
        "```json",
        json.dumps(params, indent=2),
        "```",
        "",
        "## 2. Evaluation Metrics (Ground-Truth Supervised)",
    ]

    if metrics:
        lines.append("| Metric | Value | Description |")
        lines.append("| :--- | :--- | :--- |")
        lines.append(f"| **ROC-AUC** | `{metrics.get('auc_roc', 'N/A')}` | Area Under ROC Curve |")
        lines.append(f"| **PR-AUC** | `{metrics.get('auc_pr', 'N/A')}` | Area Under Precision-Recall Curve |")
        lines.append(f"| **Max F1** | `{metrics.get('f1_max', 'N/A')}` | Peak F1 Across Thresholds |")
        lines.append(f"| **FPR @ 95% TPR** | `{metrics.get('fpr_at_95_tpr', 'N/A')}` | False Alarm Rate at 95% Recall |")
        lines.append(f"| **FA / Megapixel** | `{metrics.get('false_alarms_per_megapixel', 'N/A')}` | Normalized False Alarms |")
    else:
        lines.append("*No ground-truth metrics available for this experiment.*")

    lines.extend([
        "",
        "## 3. Computational Profiling",
        f"- **Fit Latency:** {runtime.get('fit_time_mean_sec', 0.0) * 1000.0:.2f} ms" if 'fit_time_mean_sec' in runtime else f"- **Fit Latency:** N/A",
        f"- **Predict Latency:** {runtime.get('predict_time_mean_sec', 0.0) * 1000.0:.2f} ms" if 'predict_time_mean_sec' in runtime else f"- **Predict Latency:** N/A",
        f"- **Total Latency:** {runtime.get('total_time_mean_sec', 0.0) * 1000.0:.2f} ms" if 'total_time_mean_sec' in runtime else f"- **Total Latency:** N/A",
        f"- **Peak Traced Heap:** {memory.get('peak_traced_mb', 'N/A')} MB",
        f"- **Process RSS Delta:** {memory.get('rss_delta_mb', 'N/A')} MB",
        "",
        "## 4. Software Provenance",
        f"- **Python:** {sw.get('python_version', 'N/A')}",
        f"- **NumPy:** {sw.get('numpy_version', 'N/A')}",
        f"- **Platform:** {sw.get('platform', 'N/A')} ({sw.get('machine', 'N/A')})",
        "",
        "## 5. Scientific Disclaimers & Notes",
        "> 1. **Anomaly Scores:** Raw detector outputs represent statistical Mahalanobis distances from estimated background covariance; they do not represent Bayesian probabilities or verified tactical targets.",
        "> 2. **Threshold Optimization:** F1-max and operating point metrics are determined post-hoc using available ground-truth masks for evaluation and cannot be assumed in unsupervised deployments without an empirical threshold selection method.",
        "> 3. **Cross-Scene Qualification:** Metric comparisons across different scenes reflect different sensor payloads, ground sampling distances, and camouflage conditions, and must be qualified accordingly.",
    ])

    report_text = "\n".join(lines)

    if filepath is not None:
        path = _validate_output_path(filepath, expected_extension=".md", overwrite=overwrite)
        with open(path, "w", encoding="utf-8") as f:
            f.write(report_text)

    return report_text
