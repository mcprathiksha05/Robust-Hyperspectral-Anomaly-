# Stage 3 Guide: Advanced Testing & Analysis Protocols

This document details the Stage 3 implementation for the project:
*"Robust Hyperspectral Anomaly and Target Detection for Defense Applications Under Camouflage, Clutter, and Computational Constraints"*.

---

## 1. Module Overview

Stage 3 implements diagnostic and scientific evaluation protocols across 5 core modules in `src/analysis/`:

```
src/analysis/
├── __init__.py           # Unified exports
├── ablation.py           # Feature 4: Controlled configuration comparisons
├── cross_scene.py        # Feature 5: Cross-scene generalization & transfer protocols
├── sensitivity.py        # Feature 6: Parametric sensitivity & operational threshold sweeps
├── stability.py          # Feature 7: Multi-seed repeatability & failure auditing
└── uncertainty.py        # Feature 8: Pixel-level Monte Carlo perturbation uncertainty maps
```

---

## 2. Feature 4: Ablation Studies (`src/analysis/ablation.py`)

### Purpose & Scope
Provides controlled, reproducible evaluations of processing configurations without fabricating unavailable components.

### Evaluated Configurations
1. **Input Normalization:**
   - `'raw'`: Unscaled spectral radiance/reflectance.
   - `'minmax'`: Per-band $[0.0, 1.0]$ normalization.
   - `'zscore'`: Per-band zero-mean, unit-variance standardization.
   - `'l2'`: Spatial pixel vector unit L2-norm scaling.
2. **Spectral Band Subsampling:** Retaining $100\%$, $75\%$, or $50\%$ of channels via uniform spectral decimation.
3. **Regularization $\lambda$:** Covariance diagonal shrinkage from $10^{-6}$ to $10^{-1}$.
4. **Solver:** Cholesky factorization vs. Moore-Penrose pseudo-inverse (`pinv`).

### Key Function
```python
results = run_ablation_study(cube, gt, configurations, detector_cls=RXDetector)
```

---

## 3. Feature 5: Cross-Scene Testing (`src/analysis/cross_scene.py`)

### Protocol Distinction: Within-Scene vs. Cross-Scene Transfer
- **Within-Scene Benchmark:** Model fits background on Scene $S$ and evaluates on Scene $S$.
- **Cross-Scene Transfer:** Model fits background parameters exclusively on Source Scene $S$, and predicts on Target Scene $T$ ($S \ne T$) without re-fitting.

### Spectral Band Compatibility Rules
- Incompatible band counts ($B_{\text{source}} \ne B_{\text{target}}$) are **strictly flagged and skipped** by default under `band_mismatch_strategy="skip"`.
- Alternatively, `band_mismatch_strategy="truncate_min"` evaluates over the common first $\min(B_{\text{src}}, B_{\text{tgt}})$ channels with documented proxy disclaimers.

### Key Functions
- `evaluate_within_scene_benchmark(dataset_names, detector_cls)`: Evaluates within-scene performance across multiple scenes.
- `evaluate_cross_scene_transfer(source_scene, target_scenes)`: Evaluates transfer from one source to multiple targets.
- `evaluate_cross_scene_matrix(scene_names)`: Computes pairwise transfer matrix $M[i, j]$ across band-compatible scenes.

---

## 4. Feature 6: Sensitivity Analysis (`src/analysis/sensitivity.py`)

### Categorization of Swept Parameters
1. **Corruption Parameters:** External environmental/sensor stresses (`snr_db`, `dropped_band_fraction`, `clutter_abundance`).
2. **Detector Parameters:** Internal algorithmic hyperparameters (`regularization`, solver choice).
3. **Operational Threshold Trade-offs:** Score thresholds and their impact on TPR, FPR, Precision, Recall, and False Alarms per Megapixel.

### Protocol Distinction for Decision Thresholds
- $F_{1,\max}$ requires ground truth labels to discover post-hoc.
- `sweep_decision_thresholds(scores, gt)` maps the complete trade-off curve across threshold percentiles without falsely claiming post-hoc optimal thresholds as unsupervised deployment results.

---

## 5. Feature 7: Stability Testing (`src/analysis/stability.py`)

### Sources of Variability
1. **Stochastic Corruption:** Fixed deterministic detector evaluated across $K$ random seeds of input noise or clutter.
2. **Stochastic Detector:** Stochastic model (e.g. random subspace or deep autoencoders) evaluated across $K$ initialization seeds on fixed input.

### Failure Auditing
Failed or non-finite trials are **never silently discarded**; they are recorded with status and error messages and accounted for in `failed_trials` and `failure_rate`.

---

## 6. Feature 8: Uncertainty Estimation (`src/analysis/uncertainty.py`)

### Formulation
Monte Carlo perturbation score variability across $T$ iterations ($s_t = \text{base\_seed} + t$):
- **Mean Score Map:** $\bar{S}(i, j) = \frac{1}{T} \sum_{t=1}^T S_t(i, j)$
- **Variance Map:** $\sigma^2(i, j) = \frac{1}{T-1} \sum_{t=1}^T (S_t(i, j) - \bar{S}(i, j))^2$
- **Standard Deviation Map:** $\sigma(i, j) = \sqrt{\sigma^2(i, j)}$
- **Relative Uncertainty Map (CV):** $\frac{\sigma(i, j)}{\bar{S}(i, j) + \epsilon}$
- **90% Inter-Percentile Spread:** $Q_{95}(i, j) - Q_{05}(i, j)$

### Scientific Disclaimer
> **Important:** This uncertainty map quantifies empirical score sensitivity and fragility under input perturbation. It does NOT represent Bayesian posterior probabilities or calibrated detection confidence.

---

## 7. Verification Commands

Run unit tests for Stage 3:
```bash
python -m pytest tests/test_analysis.py -v
```

Run complete repository test suite:
```bash
python -m pytest tests/ -v
```

Run real-data validation experiments:
```bash
python tests/smoke_test_analysis.py
```
