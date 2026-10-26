"""Tests for Phase 2: Feature 4 (Detection Results Viewer) and Feature 7 (Animated Detection Pipeline)."""

import io
import json
import time
from pathlib import Path
import numpy as np
import pytest
from streamlit.testing.v1 import AppTest

from src.dashboard.components import (
    create_anomaly_heatmap_figure,
    create_ground_truth_mask_figure,
    create_single_band_figure,
)
from src.data_loading import load_hsi
from src.evaluation.metrics import evaluate_anomaly_map
from src.models.rx import RXDetector


def test_feature4_figure_components_generation():
    """Verify single band, ground truth mask, and anomaly heatmap with percentile limits."""
    h, w, b = 25, 25, 10
    cube = np.random.RandomState(42).rand(h, w, b).astype(np.float32)
    gt = np.zeros((h, w), dtype=np.int32)
    gt[5:8, 5:8] = 1

    # Single band figure
    fig_band = create_single_band_figure(cube[:, :, 0], title="Band 0 Test")
    assert fig_band is not None
    assert len(fig_band.data) >= 1
    assert fig_band.data[0].z.shape == (h, w)

    # Ground truth mask figure
    fig_gt = create_ground_truth_mask_figure(gt, title="GT Mask Test")
    assert fig_gt is not None
    assert len(fig_gt.data) >= 2  # Heatmap + Scatter targets

    # Anomaly heatmap with percentile clipping
    scores = np.random.RandomState(42).exponential(scale=2.0, size=(h, w)).astype(np.float64)
    zmin = float(np.min(scores))
    zmax = float(np.percentile(scores, 95.0))
    fig_map = create_anomaly_heatmap_figure(
        scores, gt=gt, colormap="Plasma", show_ground_truth=True, zmin=zmin, zmax=zmax
    )
    assert fig_map is not None
    assert fig_map.data[0].zmin == zmin
    assert fig_map.data[0].zmax == zmax


def test_feature4_real_detector_evaluation_and_exports():
    """Verify RX detector execution on real dataset, metrics computation, and export bundles."""
    cube, gt, meta = load_hsi.load("abu-airport-1")
    h, w, b = cube.shape

    detector = RXDetector(regularization=1e-4, method="cholesky")
    detector.fit(cube)
    scores = detector.predict(cube)

    assert scores.shape == (h, w)
    assert np.all(np.isfinite(scores))

    # Metric evaluation against real ground truth
    metrics = evaluate_anomaly_map(gt, scores, target_tpr=0.95)
    assert "auc_roc" in metrics
    assert "auc_pr" in metrics
    assert "optimal_threshold" in metrics
    assert 0.0 <= metrics["auc_roc"] <= 1.0

    # Test NPZ scores export
    buf_npz = io.BytesIO()
    np.savez_compressed(buf_npz, scores=scores, dataset="abu-airport-1", detector="Global-RX")
    buf_npz.seek(0)
    loaded_npz = np.load(buf_npz)
    assert "scores" in loaded_npz
    assert loaded_npz["scores"].shape == (h, w)

    # Test Binary mask export
    binary_mask = (scores >= metrics["optimal_threshold"]).astype(np.uint8)
    buf_npy = io.BytesIO()
    np.save(buf_npy, binary_mask)
    buf_npy.seek(0)
    loaded_mask = np.load(buf_npy)
    assert loaded_mask.shape == (h, w)
    assert set(np.unique(loaded_mask)).issubset({0, 1})

    # Test JSON metrics export
    json_export = json.dumps({"dataset": "abu-airport-1", "metrics": metrics})
    parsed = json.loads(json_export)
    assert parsed["dataset"] == "abu-airport-1"
    assert parsed["metrics"]["auc_roc"] == metrics["auc_roc"]


def test_feature7_detection_pipeline_sequential_stages():
    """Verify that all 6 stages of the detection pipeline execute accurately with real data."""
    dataset_name = "abu-airport-1"

    # Stage 1: Dataset Loading
    t0 = time.perf_counter()
    cube, gt, meta = load_hsi.load(dataset_name)
    h, w, b = cube.shape
    t_stage1 = (time.perf_counter() - t0) * 1000.0
    assert h == 100 and w == 100 and b == 205
    assert t_stage1 > 0

    # Stage 2: Data Validation
    t0 = time.perf_counter()
    finite_ok = bool(np.all(np.isfinite(cube)))
    nan_count = int(np.isnan(cube).sum())
    inf_count = int(np.isinf(cube).sum())
    t_stage2 = (time.perf_counter() - t0) * 1000.0
    assert finite_ok is True
    assert nan_count == 0 and inf_count == 0

    # Stage 3: Spectral Preprocessing
    t0 = time.perf_counter()
    mean_spec = np.mean(cube, axis=(0, 1), dtype=np.float64)
    t_stage3 = (time.perf_counter() - t0) * 1000.0
    assert mean_spec.shape == (b,)

    # Stage 4: RX Detection
    t0 = time.perf_counter()
    det = RXDetector(regularization=1e-4, method="cholesky")
    det.fit(cube)
    scores = det.predict(cube)
    t_stage4 = (time.perf_counter() - t0) * 1000.0
    assert scores.shape == (h, w)
    assert t_stage4 > 0

    # Stage 5: Quantitative Evaluation
    t0 = time.perf_counter()
    metrics = evaluate_anomaly_map(gt, scores, target_tpr=0.95)
    t_stage5 = (time.perf_counter() - t0) * 1000.0
    assert metrics["auc_roc"] > 0.8
    assert t_stage5 > 0

    # Stage 6: Total pipeline measurement
    total_ms = t_stage1 + t_stage2 + t_stage3 + t_stage4 + t_stage5
    assert total_ms > 0


def test_feature4_and_7_app_tabs_integration():
    """Headless integration test navigating to Detection Results Viewer and Pipeline."""
    app_path = Path(__file__).resolve().parents[1] / "app.py"
    app = AppTest.from_file(str(app_path), default_timeout=180).run()
    assert not app.exception

    # Navigate to Anomaly Detection section
    nav_option = next(opt for opt in app.sidebar.radio[0].options if "Detection Pipeline" in opt or "Anomaly Detection" in opt)
    app.sidebar.radio[0].set_value(nav_option).run()
    assert not app.exception

    # Verify Run Detector button is available and clickable
    run_btn = next(btn for btn in app.button if "Run Detector" in btn.label)
    assert run_btn is not None

    # Run detection
    run_btn.click().run()
    assert not app.exception
    assert app.session_state["scores"] is not None
    assert app.session_state["metrics"] is not None

    # Verify download buttons for Feature 4 export
    download_labels = [btn.label for btn in app.download_button]
    assert any("Download Scores (.npz)" in lbl for lbl in download_labels)
    assert any("Download Binary Mask (.npy)" in lbl for lbl in download_labels)
    assert any("Download Metrics (.json)" in lbl for lbl in download_labels)
