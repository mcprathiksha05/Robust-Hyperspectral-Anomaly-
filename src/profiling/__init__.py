"""Hyperspectral runtime, memory profiling, and tiled processing framework (Stage 4).

Features:
    - Feature 11: Runtime and memory profiling (runtime.py)
    - Feature 12: Spatial tiled processing & memory-bounded chunking (tiling.py)
"""

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

__all__ = [
    # Feature 11: Profiling
    "profile_detector_run",
    "benchmark_multiple_detectors",
    "get_current_rss_bytes",
    # Feature 12: Tiling
    "get_tile_slices",
    "process_tiled",
    "TiledDetectorAdapter",
    "compare_tile_sizes",
]
