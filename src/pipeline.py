"""End-to-end detection pipeline - the single function the Streamlit frontend calls.

    from src.pipeline import PipelineConfig, run_pipeline
    result = run_pipeline(cube, gt, PipelineConfig(threshold_percent=1.0))
    result["fused"], result["mask"], result["metrics"]["fused"], result["warnings"]

Steps: validate -> preprocess -> PCA -> detectors (RX, spatial residual,
autoencoder, optional local RX) -> fusion -> threshold -> evaluation.

Every step runs through ``safe_call``: if an optional step fails (e.g. PCA,
the autoencoder, or a window that is too large), the error is recorded in
``result['errors']`` and the pipeline continues with what still works.
It only stops if the input itself is unusable or every detector fails.
"""
from __future__ import annotations

import time
import tracemalloc
from dataclasses import asdict, dataclass, field
from typing import Callable

import numpy as np

from src.evaluation.metrics import compare_to_gt, false_alarm_analysis, mask_metrics, score_metrics
from src.models.autoencoder import SpectralAutoencoder
from src.models.fusion import FUSIONS, NORMALIZATIONS, fuse_scores, normalize_scores
from src.models.rx import RXDetector, local_rx
from src.models.spatial import METRICS as SPATIAL_METRICS
from src.models.spatial import spatial_residual
from src.models.threshold import METHODS as THRESHOLD_METHODS
from src.models.threshold import apply_threshold
from src.preprocessing.pca import PCA
from src.preprocessing.preprocess import NORMALIZERS, preprocess
from src.utils.validation import (HSIValidationError, check_choice, check_float, check_int, safe_call,
                                  validate_cube, validate_gt)

DETECTORS = ("rx", "spatial", "autoencoder", "local_rx")


@dataclass
class PipelineConfig:
    # preprocessing
    normalize: str = "robust"
    clip_percentiles: tuple | None = (0.1, 99.9)
    drop_constant_bands: bool = True
    # PCA: int = number of components, float = fraction of variance, None = off
    pca_components: int | float | None = 0.999
    # detectors to run
    detectors: tuple = ("rx", "spatial", "autoencoder")
    rx_robust: bool = True
    rx_reg: float = 1e-6
    spatial_inner: int = 3
    spatial_outer: int = 9
    spatial_metric: str = "euclidean"
    local_inner: int = 5
    local_outer: int = 15
    ae_hidden: int = 64
    ae_latent: int = 8
    ae_epochs: int = 40
    # fusion
    fusion_method: str = "mean"
    fusion_normalize: str = "robust"
    fusion_weights: dict | None = None
    # thresholding
    threshold_method: str = "percentile"
    threshold_percent: float = 1.0
    threshold_k: float = 3.0
    threshold_value: float | None = None
    min_region_size: int = 1
    random_state: int = 0
    profile_memory: bool = True
    extra: dict = field(default_factory=dict)

    def validate(self) -> "PipelineConfig":
        check_choice(self.normalize, "normalize", NORMALIZERS)
        if self.pca_components is not None:
            if isinstance(self.pca_components, float) and not self.pca_components.is_integer():
                check_float(self.pca_components, "pca_components", 0, 1, lo_inclusive=False)
            else:
                check_int(self.pca_components, "pca_components", lo=1)
        self.detectors = tuple(self.detectors)
        if not self.detectors:
            raise HSIValidationError("Select at least one detector.")
        for d in self.detectors:
            check_choice(d, "detector", set(DETECTORS))
        check_choice(self.spatial_metric, "spatial_metric", SPATIAL_METRICS)
        check_choice(self.fusion_method, "fusion_method", FUSIONS)
        check_choice(self.fusion_normalize, "fusion_normalize", NORMALIZATIONS)
        check_choice(self.threshold_method, "threshold_method", THRESHOLD_METHODS)
        check_float(self.threshold_percent, "threshold_percent", 0, 100, lo_inclusive=False)
        check_int(self.ae_epochs, "ae_epochs", 1, 1000)
        check_int(self.min_region_size, "min_region_size", 1)
        if self.threshold_method == "fixed" and self.threshold_value is None:
            raise HSIValidationError("threshold_method='fixed' needs threshold_value.")
        return self


def run_pipeline(cube, gt=None, config: PipelineConfig | None = None, scene_labels=None,
                 label_names=None, progress: Callable[[str, float], None] | None = None) -> dict:
    """Run the full detection pipeline. See module docstring for the returned keys."""
    cfg = (config or PipelineConfig()).validate()
    report = progress or (lambda step, frac: None)
    cube = validate_cube(cube)
    if gt is not None:
        gt = validate_gt(gt, cube.shape)

    res: dict = {"config": asdict(cfg), "scores": {}, "timings": {}, "errors": {}, "warnings": [],
                 "models": {}, "metrics": {}}
    own_trace = cfg.profile_memory and not tracemalloc.is_tracing()
    if own_trace:
        tracemalloc.start()
    t_start = time.perf_counter()

    def step(name, fn, *a, fallback=None, **kw):
        r = safe_call(name, fn, *a, fallback=fallback, **kw)
        res["timings"][name] = round(r.seconds, 4)
        if r.error and r.ok:          # succeeded via a fallback
            res["warnings"].append(f"{name}: {r.error}")
        elif r.error:                 # failed; pipeline decides whether to continue
            res["errors"][name] = r.error
        return r

    try:
        # 1. preprocessing ------------------------------------------------------
        report("Preprocessing", 0.05)
        r = step("preprocess", preprocess, cube, normalize=cfg.normalize,
                 drop_constant_bands=cfg.drop_constant_bands, clip_percentiles=cfg.clip_percentiles)
        if not r.ok:
            raise HSIValidationError(f"Preprocessing failed: {r.error}")
        clean, pinfo = r.value
        valid = pinfo["valid_mask"]
        res["warnings"].extend(pinfo["warnings"])
        res["valid_mask"], res["preprocess_info"] = valid, pinfo

        # 2. PCA ----------------------------------------------------------------
        reduced = clean
        if cfg.pca_components is not None:
            report("PCA", 0.15)
            pca = PCA(cfg.pca_components, random_state=cfg.random_state)
            r = step("pca", pca.fit_transform, clean, mask=valid)
            if r.ok:
                reduced = r.value
                res["models"]["pca"] = pca
                res["pca"] = pca.summary()
            else:
                res["warnings"].append("PCA failed; detectors use all bands instead.")
        res["data"] = {"clean": clean, "reduced": reduced}

        # 3. detectors ------------------------------------------------------------
        frac = {"rx": 0.25, "spatial": 0.35, "local_rx": 0.45, "autoencoder": 0.55}
        if "rx" in cfg.detectors:
            report("RX detector", frac["rx"])
            rx = RXDetector(reg=cfg.rx_reg, robust=cfg.rx_robust, random_state=cfg.random_state)
            r = step("rx", rx.fit_score, reduced, mask=valid)
            if r.ok:
                res["scores"]["rx"], res["models"]["rx"] = r.value, rx
        if "spatial" in cfg.detectors:
            report("Spatial residual", frac["spatial"])
            r = step("spatial", spatial_residual, reduced, inner=cfg.spatial_inner,
                     outer=cfg.spatial_outer, metric=cfg.spatial_metric)
            if r.ok:
                res["scores"]["spatial"], res["data"]["spatial_residual"] = r.value
        if "local_rx" in cfg.detectors:
            report("Local RX", frac["local_rx"])
            r = step("local_rx", local_rx, reduced, inner=cfg.local_inner, outer=cfg.local_outer)
            if r.ok:
                res["scores"]["local_rx"] = r.value
        if "autoencoder" in cfg.detectors:
            report("Autoencoder (training)", frac["autoencoder"])
            ae = SpectralAutoencoder(hidden_dim=cfg.ae_hidden, latent_dim=cfg.ae_latent,
                                     epochs=cfg.ae_epochs, random_state=cfg.random_state)
            r = step("autoencoder", ae.fit_score, clean, mask=valid)
            if r.ok:
                res["scores"]["autoencoder"], res["models"]["autoencoder"] = r.value, ae
        if not res["scores"]:
            raise HSIValidationError("Every detector failed: " +
                                     "; ".join(f"{k}: {v}" for k, v in res["errors"].items()))
        for k in res["scores"]:  # invalid pixels must never be flagged
            res["scores"][k] = np.where(valid, res["scores"][k], np.nanmin(res["scores"][k]))

        # 4. fusion -----------------------------------------------------------------
        report("Score fusion", 0.8)
        r = step("fusion", fuse_scores, res["scores"], method=cfg.fusion_method,
                 weights=cfg.fusion_weights, normalize=cfg.fusion_normalize, valid_mask=valid,
                 agreement_top_percent=max(cfg.threshold_percent, 0.1))
        if r.ok:
            res["fused"], finfo = r.value
            res["confidence"] = finfo.pop("agreement")
            res["fusion_info"] = finfo
        else:  # fall back to the first detector that worked
            first = next(iter(res["scores"]))
            res["fused"] = normalize_scores(res["scores"][first], "minmax", valid)
            res["confidence"] = None
            res["warnings"].append(f"Fusion failed; using '{first}' scores only.")

        # 5. threshold -----------------------------------------------------------
        report("Thresholding", 0.85)
        r = step("threshold", apply_threshold, res["fused"], method=cfg.threshold_method,
                 percent=cfg.threshold_percent, k=cfg.threshold_k, value=cfg.threshold_value,
                 valid_mask=valid, min_region_size=cfg.min_region_size)
        if not r.ok:
            r = step("threshold_fallback", apply_threshold, res["fused"], method="percentile",
                     percent=1.0, valid_mask=valid)
            res["warnings"].append("Threshold settings failed; used top 1% instead.")
        res["mask"], res["threshold_info"] = r.value
        if "warning" in res["threshold_info"]:
            res["warnings"].append(res["threshold_info"]["warning"])
        res["stats"] = {"detected_pixels": int(res["mask"].sum()),
                        "detected_percent": round(100 * float(res["mask"].mean()), 4),
                        "threshold": res["threshold_info"]["threshold"]}

        # 6. evaluation (only with labels) --------------------------------------
        if gt is not None and 0 < gt.sum() < gt.size:
            report("Evaluation", 0.92)
            for name, s in {**res["scores"], "fused": res["fused"]}.items():
                res["metrics"][name] = score_metrics(s, gt)
            res["metrics"]["fused"].update(mask_metrics(res["mask"], gt))
            res["comparison"] = compare_to_gt(res["mask"], gt, valid)
            res["false_alarms"] = false_alarm_analysis(res["mask"], gt, scores=res["fused"],
                                                       scene_labels=scene_labels, label_names=label_names)
        elif gt is not None:
            res["warnings"].append("Ground truth has no targets (or no background); metrics skipped.")
    finally:
        res["timings"]["total"] = round(time.perf_counter() - t_start, 4)
        if own_trace:
            res["peak_memory_mb"] = round(tracemalloc.get_traced_memory()[1] / 1e6, 2)
            tracemalloc.stop()
    h, w = cube.shape[:2]
    res["stats"]["pixels_per_second"] = round(h * w / max(res["timings"]["total"], 1e-9))
    report("Done", 1.0)
    return res
