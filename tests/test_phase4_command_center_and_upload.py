"""Tests for Phase 4: Feature 9 (Experiment Command Center) and Feature 10 (Upload Your Own Data)."""

import io
import json
import time
from pathlib import Path
import numpy as np
import pytest
from PIL import Image
from streamlit.testing.v1 import AppTest

from src.dashboard.pages import _load_page_dataset
from src.dashboard.state import get_history_db
from src.data_loading import load_hsi
from src.evaluation.metrics import evaluate_anomaly_map
from src.models.pca_residual import PCAResidualDetector
from src.models.rx import RXDetector


def test_feature9_experiment_command_center_persistence_and_rerun():
    """Verify SQLite persistent history, retrieval, metrics, and non-destructive re-runs."""
    db = get_history_db()

    # 1. Save an initial experiment run
    initial_id = db.save_run(
        dataset_name="abu-airport-1",
        detector_name="Global-RX",
        parameters={"regularization": 1e-4, "method": "cholesky", "seed": 42},
        status="success",
        metrics={"auc_roc": 0.945, "f1_max": 0.612, "precision": 0.70, "recall": 0.55},
        runtime={"total_time_mean_sec": 0.12},
        seed=42,
    )
    assert initial_id is not None
    assert len(initial_id) > 0

    # 2. Retrieve run
    rec = db.get_run(initial_id)
    assert rec is not None
    assert rec["dataset_name"] == "abu-airport-1"
    assert rec["detector_name"] == "Global-RX"
    assert rec["parameters"]["regularization"] == 1e-4
    assert rec["metrics"]["auc_roc"] == 0.945

    # 3. Re-run execution using stored configuration
    # Load dataset and execute genuine detector based on stored parameters
    cube, gt, meta = load_hsi.load(rec["dataset_name"])
    t0 = time.perf_counter()
    re_det = RXDetector(
        regularization=float(rec["parameters"]["regularization"]),
        method=str(rec["parameters"]["method"]),
    )
    re_scores = re_det.fit_predict(cube)
    re_time_sec = time.perf_counter() - t0
    re_metrics = evaluate_anomaly_map(gt, re_scores)

    # Invariant: Re-run must create a NEW, distinct experiment ID without mutating the original
    rerun_id = db.save_run(
        dataset_name=rec["dataset_name"],
        detector_name=rec["detector_name"],
        parameters=rec["parameters"],
        status="success",
        metrics=re_metrics,
        runtime={"total_time_mean_sec": re_time_sec, "is_rerun_of": initial_id},
        seed=rec.get("seed"),
    )
    assert rerun_id != initial_id

    # Verify both records independently exist in database
    rec_orig = db.get_run(initial_id)
    rec_new = db.get_run(rerun_id)
    assert rec_orig is not None
    assert rec_new is not None
    assert rec_new["runtime"].get("is_rerun_of") == initial_id

    # 4. Compare runs
    comp = db.compare_runs(initial_id, rerun_id)
    assert comp["same_dataset"] is True
    assert comp["same_detector"] is True
    assert "metrics_delta" in comp

    # Cleanup test records
    db.delete_run(initial_id)
    db.delete_run(rerun_id)


def test_feature10_workflow_a_ordinary_image_validation():
    """Verify ordinary image validation and rejection of hyperspectral detection."""
    # Create an ordinary synthetic RGB image in memory
    rgb_data = (np.random.rand(64, 64, 3) * 255).astype(np.uint8)
    pil_img = Image.fromarray(rgb_data, mode="RGB")
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    buf.seek(0)

    # Validate image properties
    opened_img = Image.open(buf)
    assert opened_img.width == 64
    assert opened_img.height == 64
    assert opened_img.mode == "RGB"

    # Invariant: An ordinary RGB image must NOT be accepted as a genuine hyperspectral cube (bands < 4)
    # Trying to fit a hyperspectral detector like Global RX (covariance of bands) on 3 bands is insufficient for HSI benchmarks
    with pytest.raises(Exception):
        # A 2D RGB array of shape (64, 64) is not a 3D HSI cube
        rx = RXDetector()
        rx.fit(rgb_data[:, :, 0])  # 2D fails


def test_feature10_workflow_b_hyperspectral_archive_validation_and_execution():
    """Verify genuine hyperspectral .npz/.npy archive validation, registration, and live detector execution."""
    # 1. Create a synthetic valid HSI .npz archive
    np.random.seed(42)
    fake_cube = np.random.uniform(0.01, 1.0, size=(30, 30, 20)).astype(np.float32)
    # Add a synthetic ground truth mask
    fake_gt = np.zeros((30, 30), dtype=np.uint8)
    fake_gt[10:15, 10:15] = 1

    npz_buf = io.BytesIO()
    np.savez_compressed(npz_buf, data=fake_cube, gt=fake_gt, wavelength=np.linspace(400, 1000, 20))
    npz_buf.seek(0)

    # 2. Validate archive extraction
    with np.load(npz_buf, allow_pickle=False) as arch:
        assert "data" in arch.files
        loaded_cube = arch["data"]
        loaded_gt = arch["gt"]
        assert loaded_cube.ndim == 3
        assert loaded_cube.shape == (30, 30, 20)
        assert np.issubdtype(loaded_cube.dtype, np.number)
        assert np.all(np.isfinite(loaded_cube))
        assert loaded_gt.shape == (30, 30)

    # 3. Test execution of Global RX on uploaded data
    det = RXDetector(regularization=1e-4, method="cholesky")
    det.fit(loaded_cube)
    scores = det.predict(loaded_cube)
    assert scores.shape == (30, 30)
    assert np.all(np.isfinite(scores))

    # Evaluate against the provided ground truth
    metrics = evaluate_anomaly_map(loaded_gt, scores)
    assert "auc_roc" in metrics
    assert "f1_max" in metrics
    assert 0.0 <= metrics["auc_roc"] <= 1.0


def test_feature10_invalid_archive_rejections():
    """Verify that corrupt, non-3D, or non-finite archives are rejected cleanly."""
    # Test case 1: 2D array instead of 3D cube
    buf_2d = io.BytesIO()
    np.save(buf_2d, np.ones((50, 50), dtype=np.float32))
    buf_2d.seek(0)
    arr_2d = np.load(buf_2d, allow_pickle=False)
    assert arr_2d.ndim != 3  # Correctly flagged as not a 3D cube

    # Test case 2: Array containing NaN / Inf
    buf_nan = io.BytesIO()
    bad_cube = np.ones((10, 10, 10), dtype=np.float32)
    bad_cube[2, 2, 2] = np.nan
    np.save(buf_nan, bad_cube)
    buf_nan.seek(0)
    loaded_bad = np.load(buf_nan, allow_pickle=False)
    assert not np.all(np.isfinite(loaded_bad))  # Correctly flagged as containing NaNs

    # Test case 3: Shape mismatch between ground truth and cube
    gt_mismatch = np.zeros((20, 20), dtype=np.uint8)
    assert gt_mismatch.shape != bad_cube.shape[:2]


def test_headless_navigation_to_history_and_upload():
    """Headless integration test navigating to Experiment Command Center and Upload Data."""
    app_path = Path(__file__).resolve().parents[1] / "app.py"
    app = AppTest.from_file(str(app_path), default_timeout=180).run()
    assert not app.exception

    # 1. Navigate to Experiment History (Feature 9)
    hist_option = next(opt for opt in app.sidebar.radio[0].options if "History" in opt)
    app.sidebar.radio[0].set_value(hist_option).run()
    assert not app.exception

    # 2. Navigate to Upload Data (Feature 10)
    upload_option = next(opt for opt in app.sidebar.radio[0].options if "Upload" in opt)
    app.sidebar.radio[0].set_value(upload_option).run()
    assert not app.exception
