"""Feature 11: Runtime and memory profiling utilities for hyperspectral detectors.

Provides high-resolution, monotonic timing and memory profiling:
    - Separate measurements for initialization, fitting, and prediction phases.
    - Repeated-run statistics (mean, std, min, max) with configurable warm-up runs.
    - Pixel throughput (pixels/sec and megabytes/sec).
    - Peak heap memory allocation via Python tracemalloc.
    - Process RSS memory delta via psutil (with explicit unavailability fallbacks).
    - Safe exception handling that records failures without silently discarding them.
"""

import gc
import sys
import time
import tracemalloc
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union
import numpy as np

try:
    import psutil
    _PSUTIL_AVAILABLE = True
except ImportError:
    _PSUTIL_AVAILABLE = False

from src.models.base import BaseDetector
from src.models.rx import RXDetector


class ProfilingError(Exception):
    """Exception raised when profiling fails."""
    pass


def get_current_rss_bytes() -> Optional[int]:
    """Get resident set size (RSS) in bytes for current process, or None if unavailable."""
    if not _PSUTIL_AVAILABLE:
        return None
    try:
        process = psutil.Process()
        return int(process.memory_info().rss)
    except Exception:
        return None


def profile_detector_run(
    detector: BaseDetector,
    cube: np.ndarray,
    warmup_runs: int = 1,
    repeat_runs: int = 3,
    profile_memory: bool = True,
) -> Dict[str, Any]:
    """Profile fitting and prediction latency and memory for a detector on a cube.

    Timing Protocol:
        - Uses time.perf_counter() for high-resolution monotonic wall-clock timing.
        - Executes optional warm-up runs to prime JIT caches and CPU governor states.
        - Repeats fitting and prediction across multiple iterations to report
          mean, standard deviation, min, and max.
        - Does NOT conflate data loading or plotting time with inference latency.

    Memory Protocol:
        - 'peak_traced_mb': Peak memory allocated through Python's allocator
          measured via tracemalloc.
        - 'rss_delta_mb': Change in process Resident Set Size via psutil.
        - Limitations: Operating system memory management and garbage collection
          mean RSS deltas can fluctuate; tracemalloc captures Python-managed allocations.
          Unavailable metrics are explicitly flagged as None with documentation.

    Args:
        detector: Detector instance adhering to BaseDetector.
        cube: Hyperspectral cube of shape (H, W, B).
        warmup_runs: Number of unmeasured warm-up iterations (default: 1).
        repeat_runs: Number of measured repetitions (default: 3, minimum 1).
        profile_memory: Whether to track peak memory with tracemalloc and psutil.

    Returns:
        dict: Detailed profiling report containing timing, throughput, memory,
              input metadata, and run status.
    """
    if not isinstance(cube, np.ndarray) or cube.ndim != 3:
        raise ValueError(f"Cube must be a 3D numpy array, got {type(cube)}")
    if repeat_runs < 1:
        raise ValueError(f"repeat_runs must be at least 1, got {repeat_runs}")
    if warmup_runs < 0:
        raise ValueError(f"warmup_runs must be non-negative, got {warmup_runs}")

    h, w, b = cube.shape
    total_pixels = h * w
    cube_bytes = cube.nbytes
    cube_mb = cube_bytes / (1024.0 * 1024.0)

    report: Dict[str, Any] = {
        "detector_name": getattr(detector, "name", detector.__class__.__name__),
        "cube_shape": (h, w, b),
        "cube_dtype": str(cube.dtype),
        "total_pixels": total_pixels,
        "cube_size_mb": float(cube_mb),
        "warmup_runs": warmup_runs,
        "repeat_runs": repeat_runs,
        "status": "pending",
        "error": None,
    }

    # 1. Warm-up runs (unmeasured)
    try:
        for _ in range(warmup_runs):
            detector.fit(cube)
            _ = detector.predict(cube)
    except Exception as e:
        report["status"] = "warmup_failed"
        report["error"] = f"Warm-up failed: {str(e)}"
        return report

    # 2. Measured timing repetitions
    fit_times: List[float] = []
    predict_times: List[float] = []
    total_times: List[float] = []

    # Memory baseline
    peak_traced_bytes: Optional[int] = None
    rss_start: Optional[int] = None
    rss_end: Optional[int] = None

    if profile_memory:
        gc.collect()
        rss_start = get_current_rss_bytes()
        tracemalloc.start()
        tracemalloc.reset_peak()

    try:
        for _ in range(repeat_runs):
            # Fit measurement
            t_fit_start = time.perf_counter()
            detector.fit(cube)
            t_fit_end = time.perf_counter()
            fit_sec = t_fit_end - t_fit_start

            # Predict measurement
            t_pred_start = time.perf_counter()
            scores = detector.predict(cube)
            t_pred_end = time.perf_counter()
            pred_sec = t_pred_end - t_pred_start

            fit_times.append(fit_sec)
            predict_times.append(pred_sec)
            total_times.append(fit_sec + pred_sec)

        if profile_memory:
            _, peak_traced_bytes = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            rss_end = get_current_rss_bytes()

        report["status"] = "success"

    except Exception as e:
        if profile_memory and tracemalloc.is_tracing():
            tracemalloc.stop()
        report["status"] = "failed"
        report["error"] = str(e)
        return report

    # 3. Aggregate timing metrics
    mean_fit = float(np.mean(fit_times))
    mean_pred = float(np.mean(predict_times))
    mean_total = float(np.mean(total_times))

    throughput_px_sec = float(total_pixels / mean_pred) if mean_pred > 0 else 0.0
    throughput_mb_sec = float(cube_mb / mean_pred) if mean_pred > 0 else 0.0

    report["timing"] = {
        "fit_time_mean_sec": mean_fit,
        "fit_time_std_sec": float(np.std(fit_times)),
        "fit_time_min_sec": float(np.min(fit_times)),
        "fit_time_max_sec": float(np.max(fit_times)),
        "predict_time_mean_sec": mean_pred,
        "predict_time_std_sec": float(np.std(predict_times)),
        "predict_time_min_sec": float(np.min(predict_times)),
        "predict_time_max_sec": float(np.max(predict_times)),
        "total_time_mean_sec": mean_total,
        "total_time_std_sec": float(np.std(total_times)),
        "throughput_pixels_per_sec": throughput_px_sec,
        "throughput_mb_per_sec": throughput_mb_sec,
        "individual_fit_times_sec": fit_times,
        "individual_predict_times_sec": predict_times,
    }

    # 4. Memory measurements
    if profile_memory:
        peak_mb = (
            float(peak_traced_bytes / (1024.0 * 1024.0))
            if peak_traced_bytes is not None
            else None
        )
        rss_delta = (
            float((rss_end - rss_start) / (1024.0 * 1024.0))
            if (rss_start is not None and rss_end is not None)
            else None
        )
        report["memory"] = {
            "peak_traced_mb": peak_mb,
            "rss_delta_mb": rss_delta,
            "tracemalloc_available": True,
            "psutil_available": _PSUTIL_AVAILABLE,
            "notes": (
                "peak_traced_mb measures peak Python heap allocations during execution. "
                "rss_delta_mb measures physical resident memory change via psutil."
            ),
        }
    else:
        report["memory"] = {"status": "unprofiled"}

    return report


def benchmark_multiple_detectors(
    detectors: Sequence[BaseDetector],
    cube: np.ndarray,
    warmup_runs: int = 1,
    repeat_runs: int = 3,
) -> Dict[str, Any]:
    """Benchmark and rank multiple detectors on a single hyperspectral cube.

    Args:
        detectors: Sequence of BaseDetector instances.
        cube: Hyperspectral cube of shape (H, W, B).
        warmup_runs: Warm-up count.
        repeat_runs: Timing repetitions.

    Returns:
        dict: Comparative benchmark table with detector names, latencies, throughputs, and memory.
    """
    results = []
    for det in detectors:
        res = profile_detector_run(
            det, cube, warmup_runs=warmup_runs, repeat_runs=repeat_runs, profile_memory=True
        )
        results.append(res)

    # Sort successful runs by prediction latency
    successful = [r for r in results if r["status"] == "success"]
    successful_sorted = sorted(
        successful, key=lambda x: x["timing"]["predict_time_mean_sec"]
    )

    return {
        "total_detectors_benchmarked": len(detectors),
        "successful_runs": len(successful),
        "failed_runs": len(results) - len(successful),
        "fastest_detector": (
            successful_sorted[0]["detector_name"] if successful_sorted else None
        ),
        "benchmarks": results,
    }
