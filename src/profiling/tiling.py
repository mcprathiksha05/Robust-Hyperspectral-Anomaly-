"""Feature 12: Spatial tiled processing for hyperspectral anomaly detection.

Implements chunked inference for large hyperspectral cubes (H, W, B), preserving
the exact spatial output coordinate system (H, W).

Algorithm Modes:
    1. 'global_statistics' (Tiled Inference):
       The detector is fitted once using the full scene to estimate global background
       statistics (e.g., mean vector mu and covariance matrix Sigma). Then, spatial
       tiles are passed through detector.predict() using the fixed global parameters.
       For point-wise Mahalanobis distance detectors (such as global RX), this mode
       strictly reproduces full-scene predictions down to floating-point precision.

    2. 'independent_tile' (Tile-wise Detection):
       A fresh detector is fitted and evaluated independently on each spatial tile,
       estimating purely local background statistics. Output scores will naturally
       differ from global RX because each tile has its own unique sample mean and covariance.

Memory Clarification & Constraints:
    When passing an in-memory numpy array to process_tiled(), the input cube and
    assembled 2D output map remain allocated in memory. Tiling bounds the PEAK
    EPHEMERAL WORKING ALLOCATIONS during matrix operations (centering, triangular
    solvers, and intermediate scratch arrays) to O(T_H * T_W * B) instead of O(H * W * B).
"""

import time
from typing import Any, Callable, Dict, List, Literal, Optional, Sequence, Tuple, Type, Union
import numpy as np

from src.models.base import BaseDetector
from src.models.rx import RXDetector
from src.profiling.runtime import profile_detector_run


def get_tile_slices(
    h: int, w: int, tile_h: int, tile_w: int
) -> List[Tuple[int, int, int, int]]:
    """Compute spatial bounding boxes for tiling an (H, W) grid.

    Handles:
        - Non-square scenes.
        - Partial boundary tiles along the right and bottom edges.
        - Scenes smaller than tile dimensions (returns a single full-size tile).

    Args:
        h: Spatial height.
        w: Spatial width.
        tile_h: Desired tile height.
        tile_w: Desired tile width.

    Returns:
        List of (r_start, r_end, c_start, c_end) integer tuples.
    """
    if tile_h <= 0 or tile_w <= 0:
        raise ValueError(f"Tile dimensions must be positive, got ({tile_h}, {tile_w})")

    row_bounds = [(r, min(r + tile_h, h)) for r in range(0, h, tile_h)]
    col_bounds = [(c, min(c + tile_w, w)) for c in range(0, w, tile_w)]

    tiles = []
    for r0, r1 in row_bounds:
        for c0, c1 in col_bounds:
            tiles.append((r0, r1, c0, c1))
    return tiles


def process_tiled(
    cube: np.ndarray,
    detector: BaseDetector,
    tile_size: Tuple[int, int] = (50, 50),
    mode: Literal["global_statistics", "independent_tile"] = "global_statistics",
    detector_factory: Optional[Callable[[], BaseDetector]] = None,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Process a hyperspectral cube in spatial tiles and reassemble the full score map.

    Args:
        cube: Hyperspectral cube of shape (H, W, B).
        detector: BaseDetector instance.
        tile_size: Tuple (tile_height, tile_width).
        mode:
            - 'global_statistics': Detector is fitted once globally (if not already fitted);
              tiles are evaluated using the fixed global statistics.
            - 'independent_tile': Each tile is fitted and evaluated independently.
        detector_factory: Callable returning a fresh BaseDetector instance.
                          Used in 'independent_tile' mode to instantiate fresh detectors per tile.

    Returns:
        Tuple[np.ndarray, Dict[str, Any]]:
            - score_map: Assembled 2D float64 anomaly score map of shape (H, W).
            - metadata: Dictionary detailing tile count, mode, timing, and dimensions.
    """
    if not isinstance(cube, np.ndarray) or cube.ndim != 3:
        raise ValueError(f"Cube must be a 3D numpy array, got {type(cube)}")

    h, w, b = cube.shape
    if h <= 0 or w <= 0 or b <= 0:
        raise ValueError(f"Cube dimensions must be positive, got ({h}, {w}, {b})")

    if not np.all(np.isfinite(cube)):
        raise ValueError("Cube contains non-finite values (NaN or Inf)")

    tile_h, tile_w = tile_size
    if tile_h <= 0 or tile_w <= 0:
        raise ValueError(f"Tile dimensions must be positive, got {tile_size}")

    if mode not in ("global_statistics", "independent_tile"):
        raise ValueError(
            f"mode must be 'global_statistics' or 'independent_tile', got '{mode}'"
        )

    t0 = time.perf_counter()
    tile_boxes = get_tile_slices(h, w, tile_h, tile_w)
    score_map = np.empty((h, w), dtype=np.float64)

    # 1. Mode A: Global Statistics
    if mode == "global_statistics":
        # Fit globally if detector not already fitted
        if not getattr(detector, "is_fitted", False):
            detector.fit(cube)

        tile_times = []
        for r0, r1, c0, c1 in tile_boxes:
            t_tile_start = time.perf_counter()
            tile_cube = cube[r0:r1, c0:c1, :]
            # Predict using globally fitted statistics
            tile_scores = detector.predict(tile_cube)
            score_map[r0:r1, c0:c1] = tile_scores
            tile_times.append(time.perf_counter() - t_tile_start)

    # 2. Mode B: Independent Tile-wise Detection
    else:
        factory = detector_factory or (lambda: detector.__class__())
        tile_times = []
        for r0, r1, c0, c1 in tile_boxes:
            t_tile_start = time.perf_counter()
            tile_cube = cube[r0:r1, c0:c1, :]
            local_det = factory()
            tile_scores = local_det.fit_predict(tile_cube)
            score_map[r0:r1, c0:c1] = tile_scores
            tile_times.append(time.perf_counter() - t_tile_start)

    total_elapsed_ms = (time.perf_counter() - t0) * 1000.0

    metadata: Dict[str, Any] = {
        "mode": mode,
        "input_shape": (h, w, b),
        "output_shape": (h, w),
        "tile_size": (tile_h, tile_w),
        "total_tiles": len(tile_boxes),
        "tile_boxes": tile_boxes,
        "total_elapsed_ms": float(total_elapsed_ms),
        "mean_tile_ms": float(np.mean(tile_times) * 1000.0) if tile_times else 0.0,
        "memory_note": (
            "Peak ephemeral working memory during predict is bounded to "
            f"~{tile_h}x{tile_w}x{b} elements per tile."
        ),
    }

    return score_map, metadata


class TiledDetectorAdapter(BaseDetector):
    """Adapter wrapping any BaseDetector to perform spatial tiled inference automatically.

    Adheres strictly to the BaseDetector contract (fit, predict, fit_predict).
    """

    def __init__(
        self,
        base_detector: BaseDetector,
        tile_size: Tuple[int, int] = (50, 50),
        mode: Literal["global_statistics", "independent_tile"] = "global_statistics",
        name: Optional[str] = None,
    ):
        super().__init__(name=name or f"Tiled({base_detector.name}, tile={tile_size})")
        self.base_detector = base_detector
        self.tile_size = tile_size
        self.mode = mode

    def fit(self, cube: np.ndarray) -> "TiledDetectorAdapter":
        """Fit the underlying detector."""
        cube_val = self.validate_cube(cube)
        if self.mode == "global_statistics":
            self.base_detector.fit(cube_val)
        self.is_fitted = True
        return self

    def predict(self, cube: np.ndarray) -> np.ndarray:
        """Compute tiled score map and validate output shape."""
        if not self.is_fitted:
            raise RuntimeError("Adapter must be fitted before predict.")
        cube_val = self.validate_cube(cube)
        h, w, _ = cube_val.shape
        scores, _ = process_tiled(
            cube_val,
            detector=self.base_detector,
            tile_size=self.tile_size,
            mode=self.mode,
        )
        return self.validate_score_map(scores, expected_shape=(h, w))


def compare_tile_sizes(
    cube: np.ndarray,
    detector: BaseDetector,
    tile_sizes: Sequence[Tuple[int, int]] = ((25, 25), (50, 50), (100, 100)),
    mode: Literal["global_statistics", "independent_tile"] = "global_statistics",
    warmup_runs: int = 1,
    repeat_runs: int = 3,
) -> Dict[str, Any]:
    """Benchmark full-scene prediction versus tiled inference across various tile sizes.

    Timing Protocol:
        - Both full-scene and tiled inference execute identical warm-up runs
          (warmup_runs) and measured repetitions (repeat_runs).
        - Separately isolates fit latency and predict latency.
        - Tiled prediction latency is compared directly against full-scene prediction latency.

    Memory Clarification:
        - Reports theoretical ephemeral working buffer sizes O(T_H * T_W * B).
        - Clarifies that input and output storage remain in memory for numpy arrays.

    Args:
        cube: Hyperspectral cube of shape (H, W, B).
        detector: BaseDetector instance.
        tile_sizes: Sequence of (tile_h, tile_w) dimensions to compare.
        mode: 'global_statistics' or 'independent_tile'.
        warmup_runs: Number of unmeasured warm-up iterations (default: 1).
        repeat_runs: Number of measured repetitions (default: 3).

    Returns:
        dict: Detailed comparison report containing full-scene baseline and per-tile-size metrics.
    """
    h, w, b = cube.shape

    # 1. Warm-up fit & predict on full scene
    for _ in range(max(1, warmup_runs)):
        detector.fit(cube)
        _ = detector.predict(cube)

    # 2. Measure full-scene fit latency
    fit_times = []
    for _ in range(max(1, repeat_runs)):
        t_fit_start = time.perf_counter()
        detector.fit(cube)
        fit_times.append(time.perf_counter() - t_fit_start)
    full_fit_ms = float(np.mean(fit_times) * 1000.0)

    # 3. Measure full-scene predict latency
    pred_times = []
    full_scores = None
    for _ in range(max(1, repeat_runs)):
        t_pred_start = time.perf_counter()
        full_scores = detector.predict(cube)
        pred_times.append(time.perf_counter() - t_pred_start)
    full_predict_ms = float(np.mean(pred_times) * 1000.0)
    full_total_ms = full_fit_ms + full_predict_ms

    tile_comparisons = []

    for th, tw in tile_sizes:
        # Warm-up tiled inference
        for _ in range(max(1, warmup_runs)):
            _ = process_tiled(cube, detector=detector, tile_size=(th, tw), mode=mode)

        # Measured repetitions
        tiled_times = []
        last_tiled_scores = None
        last_meta = None
        for _ in range(max(1, repeat_runs)):
            t_start = time.perf_counter()
            last_tiled_scores, last_meta = process_tiled(
                cube, detector=detector, tile_size=(th, tw), mode=mode
            )
            tiled_times.append(time.perf_counter() - t_start)

        tiled_predict_ms = float(np.mean(tiled_times) * 1000.0)
        tiled_predict_std_ms = float(np.std(tiled_times) * 1000.0)

        max_diff = float(np.max(np.abs(full_scores - last_tiled_scores)))
        mean_diff = float(np.mean(np.abs(full_scores - last_tiled_scores)))

        # Theoretical ephemeral working buffer size per tile in MB
        # Buffer for centering (T_H * T_W * B * 8 bytes) + intermediate solve (T_H * T_W * B * 8 bytes)
        theoretical_buffer_mb = float(2.0 * th * tw * b * 8.0 / (1024.0 * 1024.0))

        entry = {
            "tile_size": (th, tw),
            "total_tiles": last_meta["total_tiles"],
            "elapsed_ms": tiled_predict_ms,
            "predict_mean_ms": tiled_predict_ms,
            "predict_std_ms": tiled_predict_std_ms,
            "max_abs_difference": max_diff,
            "mean_abs_difference": mean_diff,
            "reproduces_full_scene": (max_diff < 1e-7) if mode == "global_statistics" else False,
            "theoretical_working_buffer_mb": theoretical_buffer_mb,
        }
        tile_comparisons.append(entry)

    theoretical_full_buffer_mb = float(2.0 * h * w * b * 8.0 / (1024.0 * 1024.0))

    return {
        "mode": mode,
        "scene_shape": (h, w, b),
        "full_scene_fit_ms": full_fit_ms,
        "full_scene_predict_ms": full_predict_ms,
        "full_scene_elapsed_ms": full_total_ms,
        "theoretical_full_working_buffer_mb": theoretical_full_buffer_mb,
        "warmup_runs": warmup_runs,
        "repeat_runs": repeat_runs,
        "timing_boundary_note": (
            "full_scene_predict_ms isolates prediction latency under identical warm-up and "
            "repetition counts, enabling direct comparison with tiled elapsed_ms."
        ),
        "comparisons": tile_comparisons,
    }
