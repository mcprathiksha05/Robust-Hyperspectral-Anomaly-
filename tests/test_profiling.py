"""Automated unit tests for Stage 4: Profiling & Tiled Processing.

Covers:
    - Feature 11: Runtime and memory profiling (runtime.py)
    - Feature 12: Spatial tiled processing & memory-bounded chunking (tiling.py)
    - Invariants: boundary tiles, non-square scenes, coordinate preservation,
      full-scene vs. tiled equivalence (< 1e-8), independent tile mode differences,
      and failure handling.
"""

import pytest
import numpy as np

from src.models.base import BaseDetector
from src.models.rx import RXDetector
from src.evaluation.metrics import evaluate_anomaly_map
from src.profiling.runtime import (
    profile_detector_run,
    benchmark_multiple_detectors,
    get_current_rss_bytes,
)
from src.profiling.tiling import (
    get_tile_slices,
    process_tiled,
    TiledDetectorAdapter,
    compare_tile_sizes,
)


class FailingDetector(BaseDetector):
    """Detector that deliberately raises an exception during fit or predict."""

    def fit(self, cube: np.ndarray) -> "FailingDetector":
        raise RuntimeError("Simulated detector fitting failure")

    def predict(self, cube: np.ndarray) -> np.ndarray:
        raise RuntimeError("Simulated detector prediction failure")


@pytest.fixture
def synthetic_cube():
    """Synthetic HSI cube of shape (40, 50, 12)."""
    np.random.seed(42)
    return np.random.uniform(50.0, 300.0, size=(40, 50, 12)).astype(np.float32)


# ============================================================================
# 1. Feature 12: Tile Slicing & Boundary Tests
# ============================================================================


def test_get_tile_slices_exact_and_partial():
    # Exact division: 100x100 with tile 50x50 -> 4 tiles
    tiles_exact = get_tile_slices(h=100, w=100, tile_h=50, tile_w=50)
    assert len(tiles_exact) == 4
    assert tiles_exact[0] == (0, 50, 0, 50)
    assert tiles_exact[3] == (50, 100, 50, 100)

    # Partial boundary: 100x100 with tile 35x35 -> 3x3 = 9 tiles
    tiles_partial = get_tile_slices(h=100, w=100, tile_h=35, tile_w=35)
    assert len(tiles_partial) == 9
    # Last tile covers [70:100, 70:100] (height 30, width 30)
    assert tiles_partial[-1] == (70, 100, 70, 100)

    # Scene smaller than tile: 20x30 with tile 50x50 -> 1 tile
    tiles_small = get_tile_slices(h=20, w=30, tile_h=50, tile_w=50)
    assert len(tiles_small) == 1
    assert tiles_small[0] == (0, 20, 0, 30)

    # Non-square scene: 60x80 with tile 25x30
    tiles_nonsquare = get_tile_slices(h=60, w=80, tile_h=25, tile_w=30)
    assert len(tiles_nonsquare) == 3 * 3  # 3 rows, 3 cols
    # Verify all coordinates cover full scene without gaps
    covered_pixels = sum((r1 - r0) * (c1 - c0) for r0, r1, c0, c1 in tiles_nonsquare)
    assert covered_pixels == 60 * 80

    # Invalid tile dimensions
    with pytest.raises(ValueError, match="must be positive"):
        get_tile_slices(h=100, w=100, tile_h=0, tile_w=50)


# ============================================================================
# 2. Feature 12: Tiled Processing & Equivalence
# ============================================================================


def test_tiled_global_statistics_reproduces_full_scene(synthetic_cube):
    """Critical Requirement: Global-statistics tiled inference must reproduce

    full-scene RX within floating-point tolerance (< 1e-8).
    """
    h, w, b = synthetic_cube.shape
    det = RXDetector(regularization=1e-4)

    # Full scene prediction
    full_scores = det.fit_predict(synthetic_cube)

    # Tiled prediction with tile (15, 20) (creates partial boundary tiles)
    tiled_scores, meta = process_tiled(
        synthetic_cube, detector=det, tile_size=(15, 20), mode="global_statistics"
    )

    assert tiled_scores.shape == (h, w)
    assert np.all(np.isfinite(tiled_scores))
    assert meta["output_shape"] == (h, w)

    # Verify numerical reproduction down to machine tolerance
    max_diff = np.max(np.abs(full_scores - tiled_scores))
    assert max_diff < 1e-8, f"Tiled inference diverged from full-scene: max_diff={max_diff}"


def test_tiled_independent_tile_mode_differs_from_global(synthetic_cube):
    """Independent tile-wise detection estimates local tile statistics,

    producing scores that naturally differ from global scene statistics.
    """
    det = RXDetector(regularization=1e-4)
    global_scores, _ = process_tiled(
        synthetic_cube, detector=det, tile_size=(20, 25), mode="global_statistics"
    )

    local_scores, meta_local = process_tiled(
        synthetic_cube, detector=det, tile_size=(20, 25), mode="independent_tile"
    )

    assert local_scores.shape == synthetic_cube.shape[:2]
    assert np.all(np.isfinite(local_scores))
    assert meta_local["mode"] == "independent_tile"

    # Must produce different anomaly maps because tile sample means and covariances differ
    diff = np.max(np.abs(global_scores - local_scores))
    assert diff > 1.0, f"Independent tile mode unexpectedly identical to global: diff={diff}"


def test_tiled_coordinate_preservation():
    """Ensure spatial peak coordinates are preserved exactly across tiles."""
    h, w, b = 35, 45, 10
    cube = np.random.uniform(50.0, 100.0, size=(h, w, b)).astype(np.float32)

    # Plant a stark anomaly at row 22, col 33 (inside a boundary tile when tile_size is (15, 15))
    target_pos = (22, 33)
    cube[target_pos[0], target_pos[1], :] += 300.0

    det = RXDetector()
    tiled_scores, _ = process_tiled(
        cube, detector=det, tile_size=(15, 15), mode="global_statistics"
    )

    peak_pos = np.unravel_index(np.argmax(tiled_scores), tiled_scores.shape)
    assert peak_pos == target_pos, f"Expected peak at {target_pos}, got {peak_pos}"


def test_tiled_detector_adapter(synthetic_cube):
    rx = RXDetector()
    adapter = TiledDetectorAdapter(rx, tile_size=(20, 20), mode="global_statistics")

    scores = adapter.fit_predict(synthetic_cube)
    assert scores.shape == synthetic_cube.shape[:2]
    assert np.all(np.isfinite(scores))

    # Compatibility with evaluate_anomaly_map
    gt = np.zeros(synthetic_cube.shape[:2], dtype=np.int32)
    gt[5:10, 5:10] = 1
    metrics = evaluate_anomaly_map(gt, scores)
    assert "auc_roc" in metrics
    assert 0.0 <= metrics["auc_roc"] <= 1.0


def test_compare_tile_sizes(synthetic_cube):
    rx = RXDetector()
    comp = compare_tile_sizes(
        synthetic_cube, detector=rx, tile_sizes=((20, 25), (40, 50))
    )
    assert "comparisons" in comp
    assert len(comp["comparisons"]) == 2
    for entry in comp["comparisons"]:
        assert entry["reproduces_full_scene"] is True
        assert entry["max_abs_difference"] < 1e-8


def test_tiled_input_validation():
    det = RXDetector()
    # 2D input
    with pytest.raises(ValueError, match="3D numpy array"):
        process_tiled(np.random.rand(20, 20), det)

    # Non-finite values
    nan_cube = np.ones((20, 20, 5), dtype=np.float32)
    nan_cube[0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="non-finite values"):
        process_tiled(nan_cube, det)

    # Invalid mode
    with pytest.raises(ValueError, match="mode must be"):
        process_tiled(np.ones((20, 20, 5)), det, mode="invalid_mode")


# ============================================================================
# 3. Feature 11: Runtime & Memory Profiling Tests
# ============================================================================


def test_profile_detector_run(synthetic_cube):
    rx = RXDetector()
    report = profile_detector_run(
        rx, synthetic_cube, warmup_runs=1, repeat_runs=2, profile_memory=True
    )

    assert report["status"] == "success"
    assert report["cube_shape"] == synthetic_cube.shape
    assert report["total_pixels"] == synthetic_cube.shape[0] * synthetic_cube.shape[1]

    # Timing metrics
    timing = report["timing"]
    assert timing["fit_time_mean_sec"] > 0
    assert timing["predict_time_mean_sec"] > 0
    assert timing["total_time_mean_sec"] > 0
    assert timing["throughput_pixels_per_sec"] > 0
    assert len(timing["individual_predict_times_sec"]) == 2

    # Memory metrics
    memory = report["memory"]
    assert memory["tracemalloc_available"] is True
    assert memory["peak_traced_mb"] is not None
    assert memory["peak_traced_mb"] >= 0.0


def test_profile_detector_run_failure():
    failing = FailingDetector()
    cube = np.ones((10, 10, 5), dtype=np.float32)

    report = profile_detector_run(failing, cube, warmup_runs=1, repeat_runs=2)
    # Warm-up fails and is explicitly logged without crashing
    assert report["status"] == "warmup_failed"
    assert "Simulated detector fitting failure" in report["error"]


def test_benchmark_multiple_detectors(synthetic_cube):
    det1 = RXDetector(method="cholesky", name="RX-Cholesky")
    det2 = RXDetector(method="pinv", name="RX-Pinv")

    bench = benchmark_multiple_detectors(
        [det1, det2], synthetic_cube, warmup_runs=1, repeat_runs=2
    )

    assert bench["total_detectors_benchmarked"] == 2
    assert bench["successful_runs"] == 2
    assert bench["fastest_detector"] in ("RX-Cholesky", "RX-Pinv")
    assert len(bench["benchmarks"]) == 2


def test_profiling_parameter_validation(synthetic_cube):
    rx = RXDetector()
    with pytest.raises(ValueError, match="repeat_runs must be at least 1"):
        profile_detector_run(rx, synthetic_cube, repeat_runs=0)

    with pytest.raises(ValueError, match="warmup_runs must be non-negative"):
        profile_detector_run(rx, synthetic_cube, warmup_runs=-1)

    with pytest.raises(ValueError, match="3D numpy array"):
        profile_detector_run(rx, np.random.rand(10, 10))
