"""Stage 4 Smoke Test: Profiling & Tiled Processing on Real Hyperspectral Data.

Executes real-data runtime and memory profiling comparing full-scene RX with
spatial tiled processing across multiple tile sizes on benchmark datasets.
"""

import sys
import platform
from pathlib import Path
import numpy as np

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loading.load_hsi import load
from src.models.rx import RXDetector
from src.evaluation.metrics import evaluate_anomaly_map
from src.profiling.runtime import profile_detector_run
from src.profiling.tiling import process_tiled, compare_tile_sizes


def run_profiling_smoke_test(dataset_name: str = "abu-airport-1"):
    print("=" * 88)
    print(f" STAGE 4 PROFILING & TILED INFERENCE SMOKE TEST")
    print(f" Dataset: {dataset_name}")
    print("=" * 88)

    # 1. Environment & Hardware Context
    print("\n--- System & Hardware Context ---")
    print(f"  OS Platform          : {platform.system()} {platform.release()} ({platform.architecture()[0]})")
    print(f"  Python Runtime       : Python {platform.python_version()} ({platform.python_implementation()})")
    print(f"  Processor / Machine  : {platform.machine()} - {platform.processor() or 'Standard CPU'}")
    print(f"  Timing Engine        : time.perf_counter() (High-Resolution Monotonic)")

    # 2. Data Loading
    cube, gt, meta = load(dataset_name, as_float=True)
    h, w, b = cube.shape
    total_pixels = h * w
    cube_mb = cube.nbytes / (1024.0 * 1024.0)

    print(f"\n--- Dataset Invariants ---")
    print(f"  Shape (H, W, B)      : ({h}, {w}, {b})")
    print(f"  Dtype                : {cube.dtype}")
    print(f"  Total Pixels         : {total_pixels:,}")
    print(f"  In-Memory Cube Size  : {cube_mb:.2f} MB")
    print(f"  Anomaly Pixels       : {int(np.sum(gt))} ({np.mean(gt)*100:.2f}%)")

    # 3. Full-Scene Baseline Profiling
    print(f"\n--- 1. Full-Scene RX Baseline Profiling ---")
    detector = RXDetector(regularization=1e-4)
    prof_baseline = profile_detector_run(
        detector, cube, warmup_runs=1, repeat_runs=3, profile_memory=True
    )
    t_base = prof_baseline["timing"]
    m_base = prof_baseline["memory"]

    full_scores = detector.predict(cube)
    base_metrics = evaluate_anomaly_map(gt, full_scores, target_tpr=0.95)

    print(f"  Fit Time (Mean)      : {t_base['fit_time_mean_sec']*1000.0:.2f} ms")
    print(f"  Predict Time (Mean)  : {t_base['predict_time_mean_sec']*1000.0:.2f} ms")
    print(f"  Total Latency (Mean) : {t_base['total_time_mean_sec']*1000.0:.2f} ms")
    print(f"  Throughput           : {t_base['throughput_pixels_per_sec']:,.0f} px/sec ({t_base['throughput_mb_per_sec']:.2f} MB/s)")
    print(f"  Peak Traced Memory   : {m_base['peak_traced_mb']:.2f} MB")
    print(f"  Process RSS Delta    : {m_base['rss_delta_mb'] if m_base['rss_delta_mb'] is not None else 'N/A'} MB")
    print(f"  ROC-AUC / F1-max     : {base_metrics['auc_roc']:.4f} / {base_metrics['f1_max']:.4f}")

    # 4. Global Statistics Tiled Inference Across Tile Sizes
    print(f"\n--- 2. Global-Statistics Tiled Inference Comparison ---")
    test_tiles = [(25, 25), (50, 50), (70, 70), (100, 100)]
    tiled_comp = compare_tile_sizes(
        cube, detector, tile_sizes=test_tiles, mode="global_statistics", warmup_runs=1, repeat_runs=3
    )

    print(f"  Timing Protocol       : 1 warm-up run + 3 measured repetitions per configuration")
    print(f"  Full-Scene Fit Time   : {tiled_comp['full_scene_fit_ms']:.2f} ms (Shared across all tiles in global mode)")
    print(f"  Full-Scene Predict    : {tiled_comp['full_scene_predict_ms']:.2f} ms (Direct comparison target for tiled predict)")
    print(f"  Full-Scene Total Time : {tiled_comp['full_scene_elapsed_ms']:.2f} ms\n")

    print(f"{'Tile Size':<14} {'Tiles':<8} {'Predict (ms)':<14} {'Max Abs Diff':<16} {'Reproduces Full?':<18} {'ROC-AUC':<10}")
    print("-" * 88)

    for entry in tiled_comp["comparisons"]:
        th, tw = entry["tile_size"]
        reprod_str = "YES (< 1e-8)" if entry["reproduces_full_scene"] else "NO"
        print(
            f"{f'({th}, {tw})':<14} {entry['total_tiles']:<8} {entry['predict_mean_ms']:<14.2f} "
            f"{entry['max_abs_difference']:<16.2e} {reprod_str:<18} {base_metrics['auc_roc']:<10.4f}"
        )

    # 5. Independent Tile-wise Detection Mode (Distinct Behavior)
    print(f"\n--- 3. Independent Tile-wise Detection (Mode: 'independent_tile') ---")
    ind_scores, ind_meta = process_tiled(
        cube, detector=detector, tile_size=(50, 50), mode="independent_tile"
    )
    ind_metrics = evaluate_anomaly_map(gt, ind_scores, target_tpr=0.95)
    max_diff_ind = float(np.max(np.abs(full_scores - ind_scores)))
    mean_diff_ind = float(np.mean(np.abs(full_scores - ind_scores)))

    print(f"  Tile Dimensions       : (50, 50) -> 4 independent local models")
    print(f"  Execution Time        : {ind_meta['total_elapsed_ms']:.2f} ms (Fitting + predicting 4 separate local models)")
    print(f"  Max Absolute Diff     : {max_diff_ind:.4f} (Scores naturally differ because tile background stats differ)")
    print(f"  Mean Absolute Diff    : {mean_diff_ind:.4f}")
    print(f"  Independent ROC-AUC   : {ind_metrics['auc_roc']:.4f} (vs. Full-Scene Global: {base_metrics['auc_roc']:.4f})")
    print(f"  Independent F1-max    : {ind_metrics['f1_max']:.4f} (vs. Full-Scene Global: {base_metrics['f1_max']:.4f})")

    # 6. Memory Allocation Clarification & Audit
    print(f"\n--- 4. Memory Allocations Audit (Measured vs. Estimated) ---")
    print(f"  [A] Measured Allocations:")
    print(f"      - Python-Traced Heap Peak (tracemalloc) : {m_base['peak_traced_mb']:.2f} MB")
    print(f"      - Process Physical Resident Set (psutil) : Net RSS Delta = {m_base['rss_delta_mb'] if m_base['rss_delta_mb'] is not None else 'N/A'} MB")
    print(f"  [B] Fixed In-Memory Storage:")
    print(f"      - Input HSI Cube (H x W x B, float32)    : {cube_mb:.2f} MB")
    print(f"      - Output Anomaly Map (H x W, float64)   : {h * w * 8 / (1024 * 1024):.2f} MB")
    print(f"  [C] Theoretical Ephemeral Working Buffers during Predict:")
    print(f"      - Full-Scene Predict Working Buffers     : ~ {tiled_comp['theoretical_full_working_buffer_mb']:.2f} MB (centering + solver arrays)")
    for entry in tiled_comp["comparisons"]:
        th, tw = entry["tile_size"]
        print(f"      - Tile ({th:3d}, {tw:3d}) Working Buffers        : ~ {entry['theoretical_working_buffer_mb']:.2f} MB per tile")
    print(f"  [D] Memory Audit Finding:")
    print(f"      Tiling bounds ephemeral working buffers during matrix solves, but does NOT reduce")
    print(f"      total process memory when the full cube is pre-loaded as an in-memory numpy array.")

    print("\n" + "=" * 88)
    print(" STAGE 4 SMOKE TEST COMPLETE")
    print("=" * 88)


if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else "abu-airport-1"
    run_profiling_smoke_test(target)
