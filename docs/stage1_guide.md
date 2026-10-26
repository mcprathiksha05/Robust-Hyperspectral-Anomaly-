# Stage 1 Guide: Base Architecture & Baseline Evaluation

This document outlines the Stage 1 implementation for the research project:
*"Robust Hyperspectral Anomaly and Target Detection for Defense Applications Under Camouflage, Clutter, and Computational Constraints"*.

---

## 1. Architecture Overview

### Base Detector Interface (`src/models/base.py`)
All anomaly and target detectors must inherit from `BaseDetector` and implement:
- `fit(self, cube: np.ndarray) -> BaseDetector`: Learns background representations or estimates background statistics.
- `predict(self, cube: np.ndarray) -> np.ndarray`: Produces a 2D float anomaly score map of shape `(H, W)`.
- `fit_predict(self, cube: np.ndarray) -> np.ndarray`: Convenience method combining `fit` and `predict`.

**Contract Invariants:**
- **Input:** 3D numpy array of shape `(H, W, B)` representing spatial height $H$, width $W$, and $B$ spectral bands. Integer cubes are automatically sanitized to float32.
- **Output:** 2D numpy array of shape `(H, W)` where higher values strictly indicate stronger anomaly evidence. All values must be finite.

### Baseline RX Detector (`src/models/rx.py`)
Provides a reference implementation of the Reed-Xiaoli (RX) algorithm:
$$r(x) = (x - \mu)^T \Sigma_{\text{reg}}^{-1} (x - \mu)$$
- **Numerical Stability:** Uses adaptive scale-invariant diagonal regularization $\Sigma_{\text{reg}} = \Sigma + (\lambda \cdot \frac{\text{tr}(\Sigma)}{B}) I$ with Cholesky factorization and automatic fallbacks to pseudoinverse for ill-conditioned covariance matrices.

### Standardized Evaluation Suite (`src/evaluation/metrics.py`)
- `roc_auc`: Standard ROC-AUC (TPR vs. FPR).
- `average_precision`: Step-wise Average Precision / PR-AUC:
  $$\text{AP} = \sum_{n} (R_n - R_{n-1}) P_n$$
- `max_f1_score`: Peak F1-score across all potential decision thresholds.
- `fpr_at_tpr`: False positive rate at target true positive rate (default 95% detection).
- `false_alarms_per_megapixel`: False alarm density per 1,000,000 pixels at the operational threshold.
- `evaluate_anomaly_map`: Convenience wrapper returning all above metrics in a standardized dictionary.

---

## 2. Running the Automated Tests

From the repository root (`Robust-Hyperspectral-Anomaly--main/Robust-Hyperspectral-Anomaly--main`):

```bash
python -m pytest tests/test_stage1.py -v
```

This validates:
1. `BaseDetector` dimension, type, and finite-value checks.
2. `RXDetector` output shape, non-negativity, and anomaly ranking.
3. Collinear / singular covariance handling (e.g., duplicated bands).
4. Analytical metric correctness against known ground truth cases.
5. Invariance and edge case safety (constant scores, single-class labels).

---

## 3. Running the Smoke Test

To verify end-to-end integration with the real `.npz` benchmark datasets:

```bash
# Run on default dataset (abu-airport-1)
python tests/smoke_test_rx.py

# Run on specific benchmark scenes
python tests/smoke_test_rx.py san-diego
python tests/smoke_test_rx.py hydice-urban
```

### Verified Smoke Test Results:
- `abu-airport-1`: ROC-AUC = `0.8236` (Baseline documented in `data/README.md`: `0.822`)
- `san-diego`: ROC-AUC = `0.9411` (Baseline documented in `data/README.md`: `0.940`)
- `hydice-urban`: ROC-AUC = `0.9857` (Baseline documented in `data/README.md`: `0.986`)
