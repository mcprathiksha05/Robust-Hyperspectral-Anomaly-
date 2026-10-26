"""Evaluation metrics and benchmark suite for hyperspectral anomaly detection."""

from src.evaluation.metrics import (
    sanitize_inputs,
    roc_auc,
    average_precision,
    max_f1_score,
    fpr_at_tpr,
    false_alarms_per_megapixel,
    evaluate_anomaly_map,
)

__all__ = [
    "sanitize_inputs",
    "roc_auc",
    "average_precision",
    "max_f1_score",
    "fpr_at_tpr",
    "false_alarms_per_megapixel",
    "evaluate_anomaly_map",
]
