"""Stage 5 Automated Unit & Integration Tests.

Validates:
    - Application helper functions & dataset metadata loaders
    - Anomaly map generation, shapes, and invariants
    - Handling of missing or non-target ground truth
    - Metric definitions and scientific metadata
    - Leaderboard construction, sorting, and filtering from genuine records
    - Empty experiment history handling
    - Saving, querying, comparing, and deleting persistent history in SQLite
    - Explicit recording of failed experiments
    - Result exports to CSV, JSON, NPZ, PNG, Excel XLSX, and Markdown reports
    - Protection against accidental overwrite and invalid export paths
    - Stale result invalidation upon configuration changes
    - Headless Streamlit AppTest execution
"""

import json
from pathlib import Path
import tempfile
import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from src.dashboard.components import (
    create_anomaly_heatmap_figure,
    create_rgb_composite_figure,
    create_roc_pr_curves_figure,
    create_spectral_signature_figure,
    create_spectral_slice_figure,
    synthesize_rgb_composite,
)
from src.dashboard.state import (
    get_available_datasets,
    invalidate_results_if_stale,
    load_dataset_cached,
    result_config_hash,
)
from src.evaluation.metrics import evaluate_anomaly_map
from src.experiments.history import ExperimentHistoryDB
from src.experiments.leaderboard import (
    LEADERBOARD_COLUMNS,
    build_leaderboard,
    get_metric_definitions,
)
from src.exports.results import (
    export_anomaly_map_npz,
    export_anomaly_map_png,
    export_experiment_json,
    export_metrics_csv,
    export_runs_csv,
    export_runs_excel,
    generate_experiment_report,
)
from src.models.rx import RXDetector


@pytest.fixture
def synthetic_scene():
    """Create a synthetic (30, 30, 16) cube and ground truth for fast unit testing."""
    np.random.seed(42)
    h, w, b = 30, 30, 16
    cube = np.random.normal(loc=100.0, scale=15.0, size=(h, w, b)).astype(np.float32)
    gt = np.zeros((h, w), dtype=np.uint8)
    gt[5:8, 5:8] = 1  # 9 anomaly pixels
    cube[5:8, 5:8, :] += 50.0
    return cube, gt


@pytest.fixture
def memory_db():
    """In-memory SQLite database for test isolation."""
    return ExperimentHistoryDB(":memory:")


# -----------------------------------------------------------------------------
# 1. Dataset Selection & Helpers
# -----------------------------------------------------------------------------
def test_dataset_selection_and_cached_loading():
    datasets = get_available_datasets()
    assert isinstance(datasets, list)
    assert len(datasets) > 0
    assert "abu-airport-1" in datasets

    cube, gt, meta = load_dataset_cached("abu-airport-1")
    assert cube.ndim == 3
    assert gt.ndim == 2
    assert cube.shape[:2] == gt.shape
    assert cube.dtype == np.uint16
    assert int(np.sum(gt)) == 144


def test_synthesize_rgb_composite(synthetic_scene):
    cube, gt = synthetic_scene
    rgb = synthesize_rgb_composite(cube)
    assert rgb.shape == (30, 30, 3)
    assert rgb.dtype == np.float32
    assert 0.0 <= rgb.min() and rgb.max() <= 1.0


def test_visualization_component_creators(synthetic_scene):
    cube, gt = synthetic_scene
    det = RXDetector(regularization=1e-4)
    scores = det.fit_predict(cube)

    # 1. Anomaly heatmap figure
    fig_map = create_anomaly_heatmap_figure(scores, gt=gt, show_ground_truth=True)
    assert fig_map is not None
    assert len(fig_map.data) == 2  # Heatmap + scatter targets

    # 2. RGB figure
    fig_rgb = create_rgb_composite_figure(cube, gt=gt, show_ground_truth=True)
    assert fig_rgb is not None

    # 3. Spectral slice figure
    fig_slice = create_spectral_slice_figure(cube, band_index=0)
    assert fig_slice is not None

    # 4. Spectral signature figure
    fig_spec = create_spectral_signature_figure(cube, row=5, col=5, target_mask=gt)
    assert fig_spec is not None

    # 5. ROC/PR curves figure
    fig_curves = create_roc_pr_curves_figure(gt, scores)
    assert fig_curves is not None


# -----------------------------------------------------------------------------
# 2. Metric Definitions & Edge Cases
# -----------------------------------------------------------------------------
def test_metric_definitions_completeness():
    defs = get_metric_definitions()
    required = ["auc_roc", "auc_pr", "f1_max", "fpr_at_95_tpr", "false_alarms_per_megapixel"]
    for k in required:
        assert k in defs
        assert "name" in defs[k]
        assert "description" in defs[k]
        assert "formula" in defs[k]
        assert "ground_truth_required" in defs[k]


def test_handling_all_background_ground_truth(synthetic_scene):
    cube, _ = synthetic_scene
    gt_all_zero = np.zeros(cube.shape[:2], dtype=np.uint8)
    det = RXDetector(regularization=1e-4)
    scores = det.fit_predict(cube)

    # evaluate_anomaly_map should handle single-class gracefully
    metrics = evaluate_anomaly_map(gt_all_zero, scores)
    assert np.isnan(metrics["auc_roc"])
    assert np.isnan(metrics["auc_pr"])
    assert metrics["f1_max"] == 0.0


# -----------------------------------------------------------------------------
# 3. Experiment History (Persistent SQLite)
# -----------------------------------------------------------------------------
def test_experiment_history_empty_state(memory_db):
    runs = memory_db.list_runs()
    assert runs == []
    stats = memory_db.get_statistics()
    assert stats["total_runs"] == 0
    assert stats["successful_runs"] == 0
    assert stats["failed_runs"] == 0


def test_experiment_history_save_and_retrieve(memory_db, synthetic_scene):
    cube, gt = synthetic_scene
    det = RXDetector(regularization=1e-4)
    scores = det.fit_predict(cube)
    metrics = evaluate_anomaly_map(gt, scores)

    exp_id = memory_db.save_run(
        dataset_name="synthetic-test",
        detector_name="Global-RX",
        parameters={"regularization": 1e-4, "method": "cholesky"},
        status="success",
        metrics=metrics,
        runtime={"total_time_mean_sec": 0.045},
    )

    rec = memory_db.get_run(exp_id)
    assert rec is not None
    assert rec["experiment_id"] == exp_id
    assert rec["dataset_name"] == "synthetic-test"
    assert rec["detector_name"] == "Global-RX"
    assert rec["parameters"]["regularization"] == 1e-4
    assert rec["status"] == "success"
    assert 0.0 <= rec["metrics"]["auc_roc"] <= 1.0

    # Delete run
    deleted = memory_db.delete_run(exp_id)
    assert deleted is True
    assert memory_db.get_run(exp_id) is None


def test_experiment_history_record_failed_run(memory_db):
    exp_id = memory_db.save_run(
        dataset_name="corrupt-scene",
        detector_name="Global-RX",
        parameters={"regularization": 1e-4},
        status="failed",
        error_message="Matrix singularity encountered during Cholesky factor",
    )

    rec = memory_db.get_run(exp_id)
    assert rec is not None
    assert rec["status"] == "failed"
    assert "singularity" in rec["error_message"]
    assert rec["metrics"] is None

    stats = memory_db.get_statistics()
    assert stats["failed_runs"] == 1
    assert stats["successful_runs"] == 0


def test_experiment_history_compare_runs(memory_db):
    id1 = memory_db.save_run(
        "abu-airport-1",
        "Global-RX",
        {"regularization": 1e-6},
        metrics={"auc_roc": 0.8236, "auc_pr": 0.1011, "f1_max": 0.1933, "fpr_at_95_tpr": 0.25},
    )
    id2 = memory_db.save_run(
        "abu-airport-1",
        "Global-RX",
        {"regularization": 1e-4},
        metrics={"auc_roc": 0.8830, "auc_pr": 0.1075, "f1_max": 0.1947, "fpr_at_95_tpr": 0.20},
    )

    comp = memory_db.compare_runs(id1, id2)
    assert comp["same_dataset"] is True
    assert comp["same_detector"] is True
    assert pytest.approx(comp["metrics_delta"]["auc_roc"], abs=1e-4) == (0.8236 - 0.8830)


# -----------------------------------------------------------------------------
# 4. Model Leaderboard
# -----------------------------------------------------------------------------
def test_build_leaderboard_empty():
    df = build_leaderboard([])
    assert isinstance(df, pd.DataFrame)
    assert list(df.columns) == LEADERBOARD_COLUMNS
    assert len(df) == 0


def test_build_leaderboard_with_records():
    runs = [
        {
            "experiment_id": "exp_1",
            "dataset_name": "abu-airport-1",
            "detector_name": "Global-RX",
            "parameters": {"regularization": 1e-6, "method": "cholesky"},
            "status": "success",
            "metrics": {"auc_roc": 0.8236, "auc_pr": 0.1011, "f1_max": 0.1933},
            "runtime": {"total_time_mean_sec": 0.050},
            "timestamp": "2026-10-20T10:00:00",
        },
        {
            "experiment_id": "exp_2",
            "dataset_name": "abu-airport-1",
            "detector_name": "Global-RX",
            "parameters": {"regularization": 1e-4, "method": "cholesky"},
            "status": "success",
            "metrics": {"auc_roc": 0.8830, "auc_pr": 0.1075, "f1_max": 0.1947},
            "runtime": {"total_time_mean_sec": 0.048},
            "timestamp": "2026-10-20T11:00:00",
        },
    ]

    df = build_leaderboard(runs, sort_by="ROC-AUC", ascending=False)
    assert len(df) == 2
    assert df.iloc[0]["Run ID"] == "exp_2"
    assert df.iloc[0]["ROC-AUC"] == 0.8830
    assert df.iloc[1]["ROC-AUC"] == 0.8236

    # Test filtering
    df_filtered = build_leaderboard(runs, dataset_filter="non-existent")
    assert len(df_filtered) == 0


# -----------------------------------------------------------------------------
# 5. Multi-Format Result Exports
# -----------------------------------------------------------------------------
def test_exports_csv_json_npz_png_excel(synthetic_scene):
    cube, gt = synthetic_scene
    det = RXDetector(regularization=1e-4)
    scores = det.fit_predict(cube)
    metrics = evaluate_anomaly_map(gt, scores)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)

        # 1. Metrics CSV
        p_csv = export_metrics_csv(metrics, tmp / "test_metrics.csv", metadata={"dataset": "synth"})
        assert p_csv.exists()
        df_csv = pd.read_csv(p_csv)
        assert len(df_csv) > 0
        assert "Metric" in df_csv["Category"].values

        # 2. Experiment JSON
        exp_record = {
            "experiment_id": "exp_test_01",
            "dataset": "synth",
            "metrics": metrics,
        }
        p_json = export_experiment_json(exp_record, tmp / "test_exp.json")
        assert p_json.exists()
        with open(p_json, "r") as f:
            data_json = json.load(f)
        assert data_json["experiment_id"] == "exp_test_01"

        # 3. Anomaly Map NPZ
        p_npz = export_anomaly_map_npz(scores, {"dataset": "synth"}, tmp / "test_map.npz")
        assert p_npz.exists()
        with np.load(p_npz) as d:
            assert "score_map" in d.files
            assert d["score_map"].shape == (30, 30)

        # 4. Anomaly Map PNG
        p_png = export_anomaly_map_png(scores, tmp / "test_map.png")
        assert p_png.exists()
        assert p_png.stat().st_size > 500  # valid image file

        # 5. Excel XLSX
        runs = [{"experiment_id": "exp_1", "dataset_name": "synth", "detector_name": "RX", "status": "success", "metrics": metrics}]
        p_xlsx = export_runs_excel(runs, tmp / "test_runs.xlsx")
        assert p_xlsx.exists()
        df_xl = pd.read_excel(p_xlsx)
        assert len(df_xl) == 1

        # 6. Markdown Report
        report_text = generate_experiment_report(exp_record, filepath=tmp / "report.md")
        assert "Experiment Report" in report_text
        assert (tmp / "report.md").exists()

        # 7. Overwrite Protection
        with pytest.raises(FileExistsError):
            export_metrics_csv(metrics, p_csv, overwrite=False)

        # Allow overwrite when True
        p_csv_overwritten = export_metrics_csv(metrics, p_csv, overwrite=True)
        assert p_csv_overwritten.exists()


# -----------------------------------------------------------------------------
# 6. Stale State Invalidation
# -----------------------------------------------------------------------------
def test_stale_results_invalidation(monkeypatch):
    import streamlit as st

    # Simulate session state
    st.session_state["scores"] = np.ones((10, 10))
    st.session_state["metrics"] = {"auc_roc": 0.85}
    st.session_state["last_run_config_hash"] = result_config_hash(
        "abu-airport-1", 0.0001, "cholesky", "Global-RX", None
    )

    # Same config -> not stale
    was_stale = invalidate_results_if_stale("abu-airport-1", 0.0001, "cholesky")
    assert was_stale is False
    assert st.session_state["scores"] is not None

    # Changed dataset -> stale, invalidated!
    was_stale = invalidate_results_if_stale("abu-airport-2", 0.0001, "cholesky")
    assert was_stale is True
    assert st.session_state["scores"] is None
    assert st.session_state["metrics"] is None


# -----------------------------------------------------------------------------
# 7. Streamlit Headless App Execution
# -----------------------------------------------------------------------------
def test_streamlit_app_headless_run():
    app_path = Path(__file__).resolve().parents[1] / "app.py"
    at = AppTest.from_file(str(app_path), default_timeout=15)
    at.run()
    assert len(at.exception) == 0, f"App execution raised exceptions: {at.exception}"
    assert len(at.title) >= 1
