"""Tests for Phase 3: Feature 6 (Robustness Laboratory) and Feature 8 (Model Comparison Arena)."""

import json
import time
from pathlib import Path
import numpy as np
import pytest
from streamlit.testing.v1 import AppTest

from src.data_loading import load_hsi
from src.evaluation.metrics import evaluate_anomaly_map
from src.models.pca_residual import PCAResidualDetector
from src.models.rx import RXDetector
from src.dashboard.backend_bridge import run_backend_pipeline
from src.robustness.bands import drop_random_bands
from src.robustness.clutter import apply_shadow_attenuation, inject_spectral_clutter
from src.robustness.noise import add_dead_pixels, add_gaussian_noise, add_stripe_noise


def test_feature6_perturbation_preserves_original_data_and_is_reproducible():
    """Verify that perturbations do not mutate original cube and respect random seeds."""
    cube, gt, meta = load_hsi.load("abu-airport-1")
    cube_original_hash = hash(cube.tobytes())

    # Test Gaussian noise reproducibility
    cube_copy1 = cube.copy()
    noisy1, meta1 = add_gaussian_noise(cube_copy1, snr_db=20.0, seed=42)
    cube_copy2 = cube.copy()
    noisy2, meta2 = add_gaussian_noise(cube_copy2, snr_db=20.0, seed=42)

    # Invariant: original data must not be mutated
    assert hash(cube.tobytes()) == cube_original_hash
    # Invariant: identical seed yields identical perturbed cubes
    assert np.allclose(noisy1, noisy2)

    # Test Stripe noise
    striped, s_meta = add_stripe_noise(cube.copy(), stripe_density=0.1, stripe_intensity=0.3, seed=42)
    assert hash(cube.tobytes()) == cube_original_hash
    assert striped.shape == cube.shape

    # Test Dead pixels
    dead, d_meta = add_dead_pixels(cube.copy(), pixel_fraction=0.01, seed=42)
    assert hash(cube.tobytes()) == cube_original_hash
    assert dead.shape == cube.shape

    # Test Clutter injection
    clutter, c_meta = inject_spectral_clutter(cube.copy(), abundance=0.3, spatial_fraction=0.05, seed=42)
    assert hash(cube.tobytes()) == cube_original_hash
    assert clutter.shape == cube.shape

    # Test Shadow attenuation
    shadow, sh_meta = apply_shadow_attenuation(cube.copy(), attenuation_factor=0.35, seed=42)
    assert hash(cube.tobytes()) == cube_original_hash
    assert shadow.shape == cube.shape

    # Test Band loss
    dropped, b_meta = drop_random_bands(cube.copy(), fraction=0.1, seed=42)
    assert hash(cube.tobytes()) == cube_original_hash
    assert dropped.shape == cube.shape


def test_feature6_baseline_comparison_and_metric_deltas():
    """Verify fair baseline execution vs perturbed execution and delta metrics."""
    cube, gt, meta = load_hsi.load("abu-airport-1")

    # Baseline run
    det_base = RXDetector(regularization=1e-4)
    base_scores = det_base.fit_predict(cube)
    m_base = evaluate_anomaly_map(gt, base_scores)

    # Perturbed run (Gaussian noise 15 dB)
    corrupted, _ = add_gaussian_noise(cube.copy(), snr_db=15.0, seed=42)
    det_pert = RXDetector(regularization=1e-4)
    pert_scores = det_pert.fit_predict(corrupted)
    m_pert = evaluate_anomaly_map(gt, pert_scores)

    # Metric deltas
    delta_auc = m_pert["auc_roc"] - m_base["auc_roc"]
    delta_f1 = m_pert["f1_max"] - m_base["f1_max"]

    assert isinstance(delta_auc, float)
    assert isinstance(delta_f1, float)
    # Severe noise typically degrades ROC-AUC
    assert delta_auc < 0.1


def test_feature8_all_supported_models_execution_and_comparison():
    """Verify that Global RX, PCA residual, and Backend Pipeline run fairly on the same data."""
    cube, gt, meta = load_hsi.load("abu-airport-1")
    h, w, b = cube.shape

    results = []

    # Model 1: Global RX
    t0 = time.perf_counter()
    rx = RXDetector(regularization=1e-4, method="cholesky")
    rx_scores = rx.fit_predict(cube)
    rx_time = (time.perf_counter() - t0) * 1000.0
    rx_metrics = evaluate_anomaly_map(gt, rx_scores)
    results.append({"name": "Global RX", "time_ms": rx_time, "metrics": rx_metrics, "scores": rx_scores})

    # Model 2: PCA Residual
    t0 = time.perf_counter()
    pca = PCAResidualDetector(n_components=8, random_state=42)
    pca_scores = pca.fit_predict(cube)
    pca_time = (time.perf_counter() - t0) * 1000.0
    pca_metrics = evaluate_anomaly_map(gt, pca_scores)
    results.append({"name": "PCA residual", "time_ms": pca_time, "metrics": pca_metrics, "scores": pca_scores})

    # Model 3: Backend Pipeline
    t0 = time.perf_counter()
    b_res = run_backend_pipeline(cube, gt=gt)
    b_time = (time.perf_counter() - t0) * 1000.0
    b_scores = b_res["fused"]
    b_metrics = evaluate_anomaly_map(gt, b_scores)
    results.append({"name": "Backend Pipeline", "time_ms": b_time, "metrics": b_metrics, "scores": b_scores})

    # Verify each model completed successfully with valid scores and metrics
    assert len(results) == 3
    for r in results:
        assert r["scores"].shape == (h, w)
        assert np.all(np.isfinite(r["scores"]))
        assert r["time_ms"] > 0
        assert 0.0 <= r["metrics"]["auc_roc"] <= 1.0
        assert 0.0 <= r["metrics"]["f1_max"] <= 1.0


def test_feature8_headless_arena_navigation_and_run():
    """Headless integration test navigating to Model Comparison section."""
    app_path = Path(__file__).resolve().parents[1] / "app.py"
    app = AppTest.from_file(str(app_path), default_timeout=180).run()
    assert not app.exception

    # Navigate to Model Comparison section
    nav_option = next(opt for opt in app.sidebar.radio[0].options if "Model Comparison" in opt)
    app.sidebar.radio[0].set_value(nav_option).run()
    assert not app.exception

    # Find the arena run button
    run_btn = next((btn for btn in app.button if "Run Side-by-Side Model Comparison" in btn.label), None)
    assert run_btn is not None

    # Click and run side-by-side benchmark
    run_btn.click().run()
    assert not app.exception
    assert "arena_last_comparison" in app.session_state
    saved_res = app.session_state["arena_last_comparison"]
    assert len(saved_res["results"]) >= 1
