# Robust Hyperspectral Anomaly and Target Detection

**Research Project:** Robust Hyperspectral Anomaly and Target Detection for Defense Applications Under Camouflage, Clutter, and Computational Constraints.

---

## Quick Start

### 1. Launch the Interactive Research Dashboard
```bash
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

The dashboard discovers scene identifiers from the local `data/processed/**/*.npz` files. Use the shared top-bar selector or the selector in Dataset Explorer to switch scenes; the choice is synchronized across workspaces, and results from a different scene are cleared. The explorer reports the loaded cube's source dtype and dimensions, reference-label availability, compressed archive size, and estimated cube memory. Invalid archive structures are reported instead of being replaced with another scene. The navy/cyan theme is configured in `.streamlit/config.toml`.

The processed `.npz` archives are the datasets currently included in the project. Dataset discovery refreshes periodically while the app is running. Loading preserves source measurements and optional labels; evaluation controls that require labels are unavailable when a dataset has no valid ground-truth mask.

### 2. Run the Complete Automated Test Suite
```bash
python -m pytest tests/ -v
```

### 3. Run the Stage 5 Application Smoke Test
```bash
python tests/smoke_test_app.py
```

---

## Research Capabilities & Stage Guides

- **Stage 1 (Baseline Protocol & Evaluation):** [`docs/stage1_guide.md`](docs/stage1_guide.md)
  - `BaseDetector` interface, Global Reed-Xiaoli (`RXDetector`) with Cholesky factor, shared evaluation metrics (`roc_auc`, `average_precision`, `max_f1_score`, `fpr_at_tpr`, `false_alarms_per_megapixel`).
- **Stage 2 (Robustness Perturbation Engine):** [`docs/stage2_guide.md`](docs/stage2_guide.md)
  - Noise: Gaussian (SNR dB), Poisson, stripe drift, dead-pixel, saturated-pixel, salt-and-pepper.
  - Clutter: Linear sub-pixel spectral mixing, localized camouflage netting, shadow attenuation, tree canopy occlusion.
  - Bands: Random channel dropout, explicit masking, localized band noise, atmospheric absorption masking.
- **Stage 3 (Advanced Testing & Analysis):** [`docs/stage3_guide.md`](docs/stage3_guide.md)
  - Ablation studies, cross-scene background transfer, sensitivity sweeps, multi-seed stability audits, Monte Carlo uncertainty maps.
- **Stage 4 (Profiling & Spatial Tiled Processing):** [`docs/stage4_guide.md`](docs/stage4_guide.md)
  - Monotonic clock timing (`time.perf_counter`), `tracemalloc` peak heap allocation, `psutil` RSS physical memory delta.
  - Spatial chunking (`process_tiled`, `compare_tile_sizes`) bounding scratch memory to $O(T_H T_W B)$ while reproducing full-scene global RX scores ($< 10^{-8}$).
- **Stage 5 (Integrated Application & Analytics):** [`docs/stage5_guide.md`](docs/stage5_guide.md)
  - Interactive anomaly heatmaps with pixel coordinates and ground-truth overlays (Feature 9).
  - Genuine model leaderboard with multi-metric sorting and definitions (Feature 10).
  - Persistent SQLite experiment tracking and run comparisons (Feature 13).
  - Multi-format result export: CSV, JSON, NPZ, PNG, Excel XLSX, and Markdown (Feature 14).
  - Integrated 9-section Streamlit research platform (Feature 15).
- **Additional model and inspection workspaces:**
  - Unsupervised PCA reconstruction-residual detection alongside the existing Global RX baseline.
  - Side-by-side model maps, measured runtimes, label-based metrics, and local experiment-history records.
  - Shared pixel and spectral-band selection, exploded sampled-band cube view, and scaled anomaly-score surface.
  - Cached data-quality diagnostics, bounded-sample PCA component maps/loadings, and a local evidence-grounded research assistant.

PCA fitting is unsupervised; ground-truth labels are used only for post-hoc evaluation. The dashboard has no trained CNN/autoencoder weights or remote language-model provider. See the stage guide for assumptions and limitations.
