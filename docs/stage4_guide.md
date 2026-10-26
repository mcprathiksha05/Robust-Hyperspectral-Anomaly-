# Stage 4 Guide: Runtime/Memory Profiling & Spatial Tiled Processing

This document details the Stage 4 implementation for the research project:
*"Robust Hyperspectral Anomaly and Target Detection for Defense Applications Under Camouflage, Clutter, and Computational Constraints"*.

---

## 1. Module Overview

Stage 4 provides profiling and memory-bounded spatial chunking under `src/profiling/`:

```
src/profiling/
├── __init__.py      # Unified exports
├── runtime.py       # Feature 11: High-resolution timing, throughput, tracemalloc/psutil memory
└── tiling.py        # Feature 12: Spatial tiled chunking, global-statistics vs. independent modes
```

---

## 2. Feature 11: Runtime and Memory Profiling (`src/profiling/runtime.py`)

### Timing Methodology
- **Clock Engine:** Uses `time.perf_counter()` (high-resolution monotonic clock).
- **Phase Separation:** Separates model initialization, background fitting, and prediction phases:
  - `fit_time_mean_sec`: Covariance estimation and Cholesky decomposition time.
  - `predict_time_mean_sec`: Mahalanobis quadratic form computation time.
- **Protocol:**
  - Configurable unmeasured warm-up iterations (`warmup_runs`, default 1) to stabilize JIT/OS caches.
  - Multiple timing repetitions (`repeat_runs`, default 3) reporting mean, std, min, and max.
  - Pixel throughput: $\text{Throughput} = \frac{H \times W}{\text{mean\_predict\_sec}}$ (pixels/sec and MB/sec).

### Memory Profiling Methodology & Limitations
- **`peak_traced_mb`:** Measured using Python standard library `tracemalloc`. Captures the exact peak RAM allocated through Python's memory allocator during execution.
- **`rss_delta_mb`:** Measured using `psutil.Process().memory_info().rss`. Measures the net change in physical process resident memory.
- **Documented Limitations:**
  - OS-level memory recycling and garbage collection mean RSS changes can fluctuate between runs.
  - If `psutil` or platform memory info is unavailable, it is explicitly flagged as `None` rather than fabricated.

---

## 3. Feature 12: Spatial Tiled Processing (`src/profiling/tiling.py`)

### Slicing & Boundary Handling
The `get_tile_slices(h, w, tile_h, tile_w)` utility computes coordinate bounding boxes $[r_0, r_1, c_0, c_1]$:
- Non-square scenes: Supported seamlessly (e.g. $80 \times 100$ or $325 \times 220$).
- Partial boundary tiles: Tiles along the right and bottom boundaries are automatically truncated to scene limits.
- Small scenes: If $(H, W) \le (T_H, T_W)$, returns exactly 1 full-size tile.

### Critical Algorithm Distinction: Two Operational Modes
1. **Mode A: `global_statistics` (Tiled Inference):**
   - The detector is fitted on the full scene, establishing global background parameters ($\mu$ and regularized covariance factor $L$).
   - Spatial tiles $[r_0:r_1, c_0:c_1]$ are evaluated through `detector.predict(tile)` using the fixed global statistics.
   - **Theoretical & Empirical Invariant:** Because Mahalanobis distances in global RX are point-wise evaluations $r(x_i) = (x_i - \mu)^T \Sigma^{-1} (x_i - \mu)$, tiled inference **identically reproduces full-scene predictions** down to machine precision ($\max |S_{\text{full}} - S_{\text{tiled}}| < 10^{-8}$).
2. **Mode B: `independent_tile` (Tile-wise Detection):**
   - A fresh detector instance is fitted exclusively on each tile, estimating purely local background statistics ($\mu_{\text{tile}}, \Sigma_{\text{tile}}$).
   - Anomaly scores naturally differ from full-scene RX because each tile operates with local background parameters.

### Memory Reduction Clarification
> **Important:** When passing an in-memory numpy cube to `process_tiled()`, the full cube $(H, W, B)$ and the assembled output map $(H, W)$ remain allocated in RAM. What spatial tiling reduces is the **peak ephemeral working buffer allocations** during matrix triangular solves and quadratic products, bounding scratch buffers from $O(H \cdot W \cdot B)$ down to $O(T_H \cdot T_W \cdot B)$.

---

## 4. Empirical Smoke Test Results (`abu-airport-1`)

Executing `python tests/smoke_test_profiling.py abu-airport-1` on an Intel 64-bit Windows 11 platform:

| Mode / Configuration | Tiles | Fit Latency | Predict Latency | Total Latency | Max Abs Diff vs. Full | Reproduces Full? | ROC-AUC |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Full-Scene Baseline** | 1 | 41.44 ms | 51.17 ms | 92.61 ms | 0.00 | *baseline* | 0.8830 |
| **Tiled (100, 100)** | 1 | *(shared)* | 48.66 ms | 48.66 ms | 0.00e+00 | **YES (< 1e-8)** | 0.8830 |
| **Tiled (70, 70)** | 4 | *(shared)* | 58.40 ms | 58.40 ms | 0.00e+00 | **YES (< 1e-8)** | 0.8830 |
| **Tiled (50, 50)** | 4 | *(shared)* | 54.16 ms | 54.16 ms | 0.00e+00 | **YES (< 1e-8)** | 0.8830 |
| **Tiled (25, 25)** | 16 | *(shared)* | 64.39 ms | 64.39 ms | 0.00e+00 | **YES (< 1e-8)** | 0.8830 |
| **Independent (50, 50)** | 4 | *(4 local)* | *(4 local)* | 98.15 ms | 595.17 | **NO (Local Models)** | 0.7292 |

> **Timing Boundary Note:** In Mode A (`global_statistics`), the global background covariance is fitted once during the fit phase (41.44 ms), so tiled inference measures pure chunked prediction latency (`tiled_predict_ms`), which is directly comparable to `full_scene_predict_ms` (51.17 ms). In Mode B (`independent_tile`), 4 distinct local models are fitted and predicted per tile, making its total latency (98.15 ms) comparable to `full_scene_total_ms` (92.61 ms).

### Memory Audit Breakdown (`abu-airport-1`)
1. **Resident Storage (In-Memory Arrays):**
   - Input HSI cube ($100 \times 100 \times 205$ float32): **7.82 MB**
   - Output score map ($100 \times 100$ float64): **0.08 MB**
2. **Measured Process Allocations:**
   - Python-Traced Heap Peak (`tracemalloc`): **63.36 MB**
   - Physical Resident Set Size Net Delta (`psutil` RSS): **~1.06 MB**
3. **Theoretical Ephemeral Working Buffers during Predict:**
   - Full-Scene Predict: **~31.28 MB** ($2 \times 10000 \times 205 \times 8$ bytes)
   - Tiled $(25, 25)$ per tile: **~1.96 MB** ($2 \times 625 \times 205 \times 8$ bytes)
   - Tiled $(50, 50)$ per tile: **~7.82 MB** ($2 \times 2500 \times 205 \times 8$ bytes)
   - *Key Distinction:* Tiling bounds temporary scratch allocations during matrix operations; it does not reduce process RSS when the entire cube is preloaded as an in-memory numpy array.

### ROC-AUC Discrepancy Reconciliation (`abu-airport-1`)
- **Stage 1/2 Smoke Tests (`0.8236`):** Explicitly specified `regularization=1e-6` in `RXDetector` to reproduce the unregularized reference benchmark cited in `data/README.md` (`0.822`).
- **Stage 3/4 Baseline (`0.8830`):** Used the class default `regularization=1e-4` in `RXDetector`.
- **Root Cause:** Due to high spectral collinearity across the 205 bands of `abu-airport-1`, minimal regularization ($\lambda=10^{-6}$) inverts noisy near-zero eigenvalues. Moderate regularization ($\lambda=10^{-4}$) dampens ill-conditioned modes, improving background suppression and lifting ROC-AUC from 0.8236 to 0.8830. Both values are mathematically verified on the exact same dataset without any data loading or metric discrepancy.

---

## 5. Verification Commands

Run Stage 4 unit tests:
```bash
python -m pytest tests/test_profiling.py -v
```

Run complete test suite (Stages 1–4):
```bash
python -m pytest tests/ -v
```

Run real-data profiling smoke test:
```bash
python tests/smoke_test_profiling.py abu-airport-1
```

