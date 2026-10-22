# Backend guide (Member 1)

This covers tasks 1–8, 17–22 and 29: data loading, preprocessing, detection, evaluation, explanation and error handling. Everything is plain NumPy/SciPy/scikit-learn. The autoencoder is written in NumPy, so **PyTorch is not needed** for this part.

Run every command from the repo root.

```
pip install -r requirements.txt
python -m pytest -q                      # 10 tests, about 3 seconds
python experiments/run_benchmark.py      # full comparison on all 16 datasets, about 5 minutes
```

## Where each task lives

| # | Task | File | Main function(s) |
|---|---|---|---|
| 1 | Data loader | `src/data_loading/load_hsi.py` | `load(name)`, `load_file(path_or_upload)`, `describe(cube, gt)` |
| 2 | Preprocessing | `src/preprocessing/preprocess.py` | `preprocess(cube, normalize="robust")` |
| 3 | PCA | `src/preprocessing/pca.py` | `PCA(n_components).fit_transform(cube)` |
| 4 | RX detector | `src/models/rx.py` | `RXDetector().fit_score(cube)`, `local_rx(cube)` |
| 5 | Spatial residual | `src/models/spatial.py` | `spatial_residual(cube, inner, outer)` |
| 6 | Score fusion | `src/models/fusion.py` | `fuse_scores({...}, method="mean")` |
| 7 | Autoencoder | `src/models/autoencoder.py` | `SpectralAutoencoder().fit_score(cube)` |
| 8 | Thresholding | `src/models/threshold.py` | `apply_threshold(scores, method)`, `calibrate_threshold(...)` |
| 17 | Ground-truth comparison | `src/evaluation/metrics.py` | `compare_to_gt(mask, gt)` |
| 18 | Evaluation metrics | `src/evaluation/metrics.py` | `mask_metrics`, `score_metrics` |
| 19 | False-alarm analysis | `src/evaluation/metrics.py` | `false_alarm_analysis(mask, gt, scores)` |
| 20 | Model comparison | `src/evaluation/compare.py`, `experiments/run_benchmark.py` | `compare_methods`, `benchmark` |
| 21 | Spectral signature inspector | `src/explain/spectral.py` | `compare_spectra`, `plot_spectra` |
| 22 | Pixel explanation | `src/explain/pixel.py` | `explain_pixel(result, row, col)` |
| 29 | Validation and recovery | `src/utils/validation.py` | `HSIValidationError`, `safe_call`, `validate_*` |
| — | Whole pipeline | `src/pipeline.py` | `run_pipeline(cube, gt, PipelineConfig(...))` |

## For the frontend: one call does everything

```python
import streamlit as st
from src.data_loading.load_hsi import load_file, describe
from src.pipeline import PipelineConfig, run_pipeline
from src.explain import explain_pixel, compare_spectra, plot_spectra
from src.utils.validation import HSIValidationError

up = st.file_uploader("Hyperspectral file", type=["npz", "npy", "mat"])
if up:
    try:
        cube, gt, meta = load_file(up)                 # works directly on the upload
    except HSIValidationError as e:
        st.error(str(e)); st.stop()                    # messages are written for users
    st.json(describe(cube, gt))                        # dataset information panel

    cfg = PipelineConfig(
        pca_components=st.slider("PCA variance kept", 0.90, 0.9999, 0.999),
        detectors=tuple(st.multiselect("Detectors", ["rx", "spatial", "autoencoder", "local_rx"],
                                       ["rx", "spatial", "autoencoder"])),
        threshold_percent=st.slider("Flag top % of pixels", 0.1, 10.0, 1.0),
    )
    bar = st.progress(0.0)
    result = run_pipeline(cube, gt, cfg, progress=lambda step, f: bar.progress(f, text=step))

    st.image(result["fused"])        # anomaly heatmap (0-1)
    st.image(result["mask"] * 255)   # binary detection map
    for w in result["warnings"]: st.warning(w)
    for k, v in result["errors"].items(): st.error(f"{k}: {v}")
```

### What `run_pipeline` returns

| Key | Contents |
|---|---|
| `scores` | `{"rx": map, "spatial": map, "autoencoder": map, ...}`, one (H, W) map per detector |
| `fused` | fused anomaly map scaled to 0–1 (the heatmap) |
| `mask` | binary detection map (uint8) |
| `confidence` | share of detectors that place each pixel in their top results; a relative indicator, **not** a probability |
| `valid_mask` | False where the input had NaN/inf values (these pixels are never flagged) |
| `stats` | detected pixels, detected %, threshold, pixels per second |
| `metrics` | only when ground truth is given: `metrics["fused"]` has roc_auc, ap, precision, recall, f1, fpr, false alarms per megapixel; every detector also gets its own entry |
| `comparison` | `category_map` (0 TN, 1 TP, 2 false alarm, 3 missed) for a colour-coded overlay, plus object counts |
| `false_alarms` | false-alarm counts, region sizes, per-material breakdown (MUUFL), and a top-p% trade-off table |
| `timings`, `peak_memory_mb` | seconds per step and peak memory, for the efficiency section |
| `pca`, `preprocess_info`, `fusion_info` | what each step did (components kept, bands dropped, weights) |
| `warnings`, `errors` | anything that went wrong; failed optional steps are skipped, not fatal |

### Pixel explanation and spectral inspector

```python
exp = explain_pixel(result, row, col, wavelengths=meta.get("wavelengths"))
st.write(exp["summary"])                 # plain-English explanation
st.bar_chart({d["wavelength"] or d["band"]: d["value"] for d in exp["rx_bands"]})

comp = compare_spectra(cube, [(row, col)], wavelengths=meta.get("wavelengths"), exclude_mask=result["mask"])
st.pyplot(plot_spectra(comp))
```

## Notes on methods and honest evaluation

- **Robust RX** fits the background twice: the second fit leaves out the top 1% most anomalous pixels, so targets don't distort the background model.
- **Local RX and the spatial residual** compare each pixel only with a ring of neighbours, with a guard window that keeps the target out of its own background. They use integral-image sums, so there's no Python loop over pixels.
- **Autoencoder training** drops the worst-reconstructed 2% of each batch after a warm-up, so it doesn't learn to reproduce the anomalies.
- **Thresholds.** `percentile`, `sigma` and `otsu` need no labels. `calibrate_threshold` picks a threshold for a target false-positive rate, but **only use it on a different image from the one you report results for**. Calibrating and testing on the same image gives over-optimistic numbers.
- **Defaults** (`PipelineConfig`) were set before running the benchmark and were not tuned per dataset.
