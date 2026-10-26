# Stage 5 Guide: Integrated Hyperspectral Defense Analytics Platform

This guide documents the Stage 5 final integrated research application for the project:
*"Robust Hyperspectral Anomaly and Target Detection for Defense Applications Under Camouflage, Clutter, and Computational Constraints"*.

---

## 1. Architecture Overview

Stage 5 unifies the experimental capabilities of Stages 1–4 into an interactive, modular Streamlit research dashboard with persistent SQLite experiment tracking and multi-format result exporting.

```
Robust-Hyperspectral-Anomaly/
├── app.py                      # Streamlit application entry point
├── src/
│   ├── dashboard/              # Modular UI & Visualization
│   │   ├── __init__.py         # Package exports
│   │   ├── state.py            # Session state, caching, & stale-result invalidation
│   │   ├── components.py       # Feature 9: Interactive Plotly heatmaps, RGB synthesis, curves
│   │   └── pages.py            # Feature 15: 9 modular dashboard page renderers
│   ├── experiments/            # Persistent Tracking & Leaderboard
│   │   ├── __init__.py         # Package exports
│   │   ├── history.py          # Feature 13: SQLite experiment history & audit trail
│   │   └── leaderboard.py      # Feature 10: Multi-metric ranking, filtering, definitions
│   ├── exports/                # Result Export Engine
│   │   ├── __init__.py         # Package exports
│   │   └── results.py          # Feature 14: CSV, JSON, NPZ, PNG, Excel XLSX, Markdown
│   ├── models/                 # BaseDetector, RXDetector & PCAResidualDetector
│   ├── evaluation/             # Shared evaluation metrics
│   ├── robustness/             # Features 1-3: Perturbation engine
│   ├── analysis/               # Features 4-8: Advanced testing & analysis
│   └── profiling/              # Features 11-12: Timing, memory, & spatial tiling
├── data/
│   ├── processed/              # 16 processed benchmark .npz datasets
│   └── history/                # SQLite persistent experiment database (experiments.db)
├── exports/                    # Default directory for disk-saved experiment artifacts
└── tests/
    ├── test_stage1.py          # Stage 1 unit tests
    ├── test_robustness.py      # Stage 2 unit tests
    ├── test_analysis.py        # Stage 3 unit tests
    ├── test_profiling.py       # Stage 4 unit tests
    ├── test_stage5.py          # Stage 5 unit tests
    └── smoke_test_app.py       # Stage 5 end-to-end integration smoke test
```

The model package also includes `PCAResidualDetector`, an unsupervised PCA subspace
model that reports squared spectral reconstruction residuals. Its fitting sample is
bounded and seeded; it is not a supervised classifier and does not load pretrained
weights. The Dataset Explorer includes non-destructive input diagnostics and optional
PCA component images/loadings fitted on a deterministic sample.

---

## 2. Launching the Application

### Prerequisites
Verify that the existing environment dependencies are present:
```bash
python -c "import streamlit, plotly, pandas, numpy, scipy, sklearn, openpyxl, matplotlib; print('All dependencies verified!')"
```

### Starting the Dashboard
From the root workspace directory, run:
```bash
streamlit run app.py
```
By default, Streamlit opens locally at `http://localhost:8501`.

---

## 3. Dashboard Sections & Feature Reference

The sidebar radio navigates across 9 research modules:

### 1. Overview
- Comprehensive project abstract and system architecture summary.
- Scientific and defense application disclaimers.
- Quick start step-by-step workflow guide.

### 2. Dataset Explorer
- Dynamic dataset selector populated from all 16 local `.npz` benchmark cubes via `load_hsi`.
- Displays spatial height, width, spectral band count, data type, ground truth target count, and cube RAM size.
- **RGB Composite Viewer:** Synthesizes high-contrast pseudo-RGB from selectable spectral channels or native metadata RGB (e.g. `muufl-gulfport-campus1`) with ground truth target overlay.
- **Single Band Slicer:** Inspects any individual wavelength channel with continuous colormaps.
- **Spectral Signature Inspector:** Interactive pixel coordinate input $(r, c)$ plotting continuous spectral response curves with wavelength axis annotations.

### 3. Anomaly Detection (Feature 9: Interactive Anomaly Map)
- Real-time detector execution on any benchmark scene using the agreed `BaseDetector` interface.
- Configurable covariance regularization ($\lambda = 10^{-4}$ default, $\lambda = 10^{-6}$ minimal baseline, $\lambda = 10^{-2}$, or custom float) and solver (`cholesky`, `solve`, `pinv`).
- **Interactive Plotly Heatmap:**
  - Real-time zooming, panning, and coordinate hover displaying `(row, col, anomaly_score)`.
  - Toggleable ground-truth target overlay (red markers).
  - Clear non-probability disclaimer in subtitle.
- **Quantitative Metrics Cards:** ROC-AUC, PR-AUC, Max F1, FPR @ 95% TPR, and False Alarms / Megapixel.
- **Operational Curves:** Side-by-side ROC and Precision-Recall curves.
- **Persistent Tracking:** One-click "Save to Experiment History" button storing genuine run metrics to SQLite.
- **Stale State Invalidation:** Automatically flushes stale maps and metrics when datasets or parameters are changed.

### 4. Robustness Experiments (Features 1, 2, 3)
- **Feature 1 (Noise Robustness):** Additive Gaussian noise (SNR dB), horizontal stripe drift, and dead pixels.
- **Feature 2 (Clutter Robustness):** Linear sub-pixel clutter mixing and shadow attenuation.
- **Feature 3 (Band Corruption):** Random spectral channel dropout, masking, and localized band noise.
- Side-by-side metric degradation benchmarks comparing clean vs. perturbed scenes.

### 5. Research Analysis (Features 4, 5, 6, 7, 8)
- **Feature 4 (Ablation Studies):** Controlled comparisons of input normalizations, retained band fractions, and solvers.
- **Feature 5 (Cross-Scene Generalization):** Multi-target transfer evaluations with band-compatibility checks.
- **Feature 6 (Sensitivity Sweeps):** Parametric sweeps across noise SNR (dB) with degradation plots.
- **Feature 8 (Uncertainty Maps):** Monte Carlo perturbation variability maps (mean score, standard deviation sensitivity map).

### 6. Performance & Scalability (Features 11, 12)
- **Feature 11 (Runtime & Memory):** Phase-separated monotonic timing (`fit_ms`, `predict_ms`), throughput (`px/s`, `MB/s`), `tracemalloc` peak heap MB, and `psutil` physical RSS delta.
- **Feature 12 (Spatial Tiling):** Chunked inference benchmarking ($(25, 25), (50, 50), (100, 100)$), verifying that global-statistics mode reproduces full-scene scores ($< 10^{-8}$) while bounding temporary scratch buffers.

### 7. Model Leaderboard (Feature 10)
- Multi-metric comparative ranking constructed exclusively from genuine records in `ExperimentHistoryDB`.
- Interactive sorting by ROC-AUC, PR-AUC, Max F1, Total Time (ms), Peak Traced RAM, or Timestamp.
- Filtering by dataset, detector, and execution status.
- Verification baseline loader to run and record the verified `abu-airport-1` baselines ($\lambda=10^{-6} \to 0.8236$ and $\lambda=10^{-4} \to 0.8830$).
- Expandable metric definitions table detailing mathematical formulas and ground-truth dependencies.

### 8. Experiment History (Feature 13)
- Complete audit trail of past runs stored in `data/history/experiments.db`.
- Detailed JSON inspection of full hyperparameters, software versions, and runtime metrics.
- Side-by-side run comparison calculating exact metric deltas between configurations.
- Run deletion and database management.

### 9. Export Results (Feature 14)
- **Active Run Exports:** Download active anomaly score maps as compressed `.npz` archives, high-resolution `.png` heatmaps, tabular `.csv` metrics, provenance `.json`, and publication-ready `.md` reports.
- **Database History Exports:** Export the complete experimental repository to `.csv`, `.json`, or Microsoft Excel `.xlsx`.
- **Local Disk Saving:** Direct export to `exports/` with automatic directory creation and overwrite collision protection.

### Additional model and inspection workspaces

- **Model laboratory:** Select Global RX or PCA residual for the active run, configure
  model-specific parameters, compare both output maps, and record comparison runs in
  SQLite. Score maps retain their own scales; RX Mahalanobis scores and PCA squared
  reconstruction errors are not directly comparable as probabilities.
- **Shared pixel/band inspection:** The Overview, Dataset Explorer, and detection
  workspace share selected coordinates and band index. Coordinate controls update
  spectral plots and selection markers; plot clicks are not wired to coordinate state.
- **3D visualization:** The cube renders sampled measured spectral planes; optional
  exploded offsets are illustrative and do not change wavelength spacing. Large scenes
  are spatially downsampled for display only. The anomaly surface uses detector scores
  for height, not physical elevation, and exposes a display-only height scale.
- **Dataset diagnostics:** Reports invalid values, constant bands, per-band statistics,
  label values, memory estimates, and sampled inter-band correlation. Low variance is
  not treated as proof of a noisy band. PCA component maps preserve spatial dimensions
  but can distort distances and do not establish material identity.
- **Local assistant:** Answers supported questions from the current dataset, active
  score map, and local experiment history. It uses deterministic rules and sends no
  arrays to an external service; unsupported questions are explicitly declined.

The application still has no asynchronous experiment queue/cancellation, trained deep
learning models, remote LLM integration, automatic natural-language experiment runner,
or end-to-end guided presentation. Model runs in the current Streamlit interface execute
synchronously after explicit button activation. Metrics that optimize a threshold
against reference labels are post-hoc summaries, not independent deployment thresholds.

---

## 4. How Experiment History is Stored

Experiment history is managed by `src.experiments.history.ExperimentHistoryDB`:
- **Engine:** SQLite 3 (`data/history/experiments.db`).
- **Schema:**
  - `experiment_id`: Unique timestamp-prefixed UUID string.
  - `timestamp`: UTC ISO-8601 string.
  - `dataset_name`, `detector_name`, `status`: Indexed query text fields.
  - `parameters`, `preprocessing`, `corruption`, `metrics`, `runtime`, `memory`, `software_info`: Deserialized JSON strings with automatic handling of NumPy scalars.
  - `error_message`: Full stack trace / exception details for failed runs.
- **Integrity:** Failed runs are explicitly recorded with `status='failed'` rather than falsely marked successful or discarded.
- **Efficiency:** Bulky 3D cubes are never stored directly in the database; lightweight summaries and filesystem references are maintained.

---

## 5. Metric Interpretation & Scientific Guidance

1. **ROC-AUC (Receiver Operating Characteristic Area Under Curve):**
   - Measures global ranking discrimination across all possible decision thresholds.
   - Range: $[0.0, 1.0]$. A score of $0.5$ represents random chance; $1.0$ is perfect separation.
   - *Requirement:* Supervised metric requiring binary target ground truth.
2. **PR-AUC (Precision-Recall Area Under Curve / Average Precision):**
   - Emphasizes performance on rare positive targets against overwhelming background pixels.
   - Particularly sensitive in defense surveillance where targets occupy $< 1\%$ of pixels.
3. **Max F1 Score ($F_{1,\max}$):**
   - The optimal harmonic mean of Precision and Recall discovered post-hoc across operational thresholds.
   - *Operational Warning:* Cannot be assumed in real unsupervised deployments without an empirical threshold selection technique.
4. **FPR @ 95% TPR:**
   - The false alarm rate incurred when operating at an operational probability of detection $P_D \ge 0.95$. Lower is better.
5. **False Alarms per Megapixel:**
   - The raw number of false alarm pixels per $1,000,000$ spatial pixels at the 95% detection threshold.
6. **Raw Anomaly Scores:**
   - Mahalanobis distances $(x - \mu)^T \Sigma_{\text{reg}}^{-1} (x - \mu)$ quantify statistical deviation from the background model. They are **not** probabilities and do not guarantee the presence of a real tactical target.

---

## 6. Verification Commands

Run complete automated test suite (Stages 1–5, 66 tests):
```bash
python -m pytest tests/ -v
```

Run application integration smoke test:
```bash
python tests/smoke_test_app.py
```

Launch the interactive application:
```bash
streamlit run app.py
```
