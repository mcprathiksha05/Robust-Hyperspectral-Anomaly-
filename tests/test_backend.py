"""Tests for Member 1's backend. Run from the repo root:  python -m pytest -q

They use a small synthetic scene (smooth background + implanted targets) so
they run in seconds and do not need the datasets.
"""
import io

import numpy as np
import pytest

from src.data_loading.load_hsi import describe, load_file
from src.evaluation.compare import compare_methods
from src.evaluation.metrics import compare_to_gt, false_alarm_analysis, mask_metrics, score_metrics
from src.explain import compare_spectra, explain_pixel
from src.models import (RXDetector, SpectralAutoencoder, apply_threshold, calibrate_threshold,
                        fuse_scores, local_rx, normalize_scores, spatial_residual)
from src.pipeline import PipelineConfig, run_pipeline
from src.preprocessing import PCA, preprocess
from src.utils.validation import HSIValidationError, safe_call, validate_cube


def make_scene(h=40, w=40, b=30, seed=0):
    rng = np.random.default_rng(seed)
    base = np.linspace(0.2, 0.8, b)
    cube = base + 0.02 * rng.standard_normal((h, w, b))
    cube += np.linspace(0, 0.1, w)[None, :, None]           # brightness gradient
    gt = np.zeros((h, w), np.uint8)
    target = base[::-1] * 0.9                                 # different spectral shape
    for r, c in [(10, 10), (25, 30), (32, 8)]:
        cube[r:r + 2, c:c + 2] = 0.5 * cube[r:r + 2, c:c + 2] + 0.5 * target
        gt[r:r + 2, c:c + 2] = 1
    return cube.astype(np.float32), gt


FAST = PipelineConfig(ae_epochs=3, ae_hidden=16, ae_latent=4, local_inner=3, local_outer=9)


# ------------------------------------------------------------ 29 validation
def test_validation_rejects_bad_inputs():
    with pytest.raises(HSIValidationError, match="3 dimensions"):
        validate_cube(np.zeros((10, 10)))
    with pytest.raises(HSIValidationError, match="numeric"):
        validate_cube(np.array([["a"] * 10] * 10)[..., None])
    with pytest.raises(HSIValidationError, match="no finite"):
        validate_cube(np.full((10, 10, 5), np.nan))
    with pytest.raises(HSIValidationError, match="odd"):
        spatial_residual(make_scene()[0], inner=2, outer=9)
    with pytest.raises(HSIValidationError, match="threshold_method"):
        PipelineConfig(threshold_method="magic").validate()


def test_safe_call_recovers():
    def boom(x):
        raise RuntimeError("kaput")
    r = safe_call("step", boom, 1, fallback=lambda x: x * 2)
    assert r.ok and r.value == 2 and "kaput" in r.error
    r = safe_call("step", boom, 1)
    assert not r.ok and "kaput" in r.error


# --------------------------------------------------------------- 1 loader
def test_loader_roundtrip_npz_and_mat(tmp_path):
    cube, gt = make_scene()
    buf = io.BytesIO()
    np.savez(buf, data=cube, map=gt)
    c, g, meta = load_file(buf.getvalue(), filename="scene.npz")
    assert c.shape == cube.shape and (g == gt).all()
    import scipy.io as sio
    sio.savemat(tmp_path / "scene.mat", {"hsi": cube, "groundtruth": gt})
    c, g, meta = load_file(tmp_path / "scene.mat")
    assert np.allclose(c, cube) and (g == gt).all() and meta["cube_key"] == "hsi"
    np.save(tmp_path / "bands_first.npy", cube.transpose(2, 0, 1))
    c, g, _ = load_file(tmp_path / "bands_first.npy", bands_first=True)
    assert c.shape == cube.shape and g is None
    with pytest.raises(HSIValidationError, match="Unsupported"):
        load_file(b"abc", filename="x.txt")
    assert describe(cube, gt)["target_pixels"] == 12


# ---------------------------------------------------- 2-3 preprocessing, PCA
def test_preprocess_repairs_invalid_values():
    cube, _ = make_scene()
    cube[0, 0, 3] = np.nan
    cube[1, 1] = np.inf
    cube[..., 5] = 7.0                                      # constant band
    x, info = preprocess(cube)
    assert np.isfinite(x).all()
    assert info["invalid_pixels"] == 2 and not info["valid_mask"][0, 0]
    assert 5 in info["constant_bands_dropped"] and x.shape[2] == cube.shape[2] - 1


def test_pca_variance_and_inverse():
    cube, _ = make_scene()
    p = PCA(0.99)
    z = p.fit_transform(cube)
    assert z.shape[2] == p.n_components_ < cube.shape[2]
    assert p.cumulative_variance_ >= 0.99
    full = PCA(cube.shape[2]).fit(cube)
    assert np.allclose(full.inverse_transform(full.transform(cube)), cube, atol=1e-4)


# --------------------------------------------------------- 4-8 detectors
def test_detectors_find_implanted_targets():
    cube, gt = make_scene()
    y = gt.astype(bool)
    rx = RXDetector(robust=True).fit(cube)
    s = rx.score(cube)
    assert s[y].mean() > 5 * np.median(s)
    pix = cube[10, 10]
    assert np.isclose(rx.contributions(pix).sum(), rx.score(pix[None, None])[0, 0], rtol=1e-6)
    assert score_metrics(local_rx(PCA(5).fit_transform(cube), 3, 9), gt)["roc_auc"] > 0.9
    sp, resid = spatial_residual(cube, 3, 9)
    assert resid.shape == cube.shape and score_metrics(sp, gt)["roc_auc"] > 0.9
    ae = SpectralAutoencoder(hidden_dim=16, latent_dim=3, epochs=15).fit(cube)
    assert score_metrics(ae.score(cube), gt)["roc_auc"] > 0.9


def test_fusion_and_thresholds():
    cube, gt = make_scene()
    a = RXDetector().fit_score(cube)
    b, _ = spatial_residual(cube, 3, 9)
    for method in ("mean", "max", "geometric", "rank_mean"):
        f, info = fuse_scores({"rx": a, "spatial": b}, method=method)
        assert f.min() >= 0 and f.max() <= 1 and info["agreement"].shape == a.shape
    with pytest.raises(HSIValidationError):
        fuse_scores({"rx": a, "bad": b[:10]})
    assert normalize_scores(a, "rank").max() == 1
    m, info = apply_threshold(a, "percentile", percent=1)
    assert abs(m.mean() - 0.01) < 0.005
    for method in ("sigma", "otsu"):
        apply_threshold(a, method)
    cal = calibrate_threshold(a, gt, target="fpr", value=0.01)
    assert cal["fpr"] <= 0.01
    m, _ = apply_threshold(a, "fixed", value=cal["threshold"])
    assert mask_metrics(m, gt)["fpr"] <= 0.01 + 1e-9


# ---------------------------------------------------- 17-19 evaluation
def test_metrics_hand_checked():
    gt = np.zeros((10, 10), np.uint8); gt[0, :4] = 1
    mask = np.zeros((10, 10), np.uint8); mask[0, :2] = 1; mask[9, 9] = 1
    m = mask_metrics(mask, gt)
    assert (m["tp"], m["fp"], m["fn"]) == (2, 1, 2)
    assert np.isclose(m["precision"], 2 / 3) and np.isclose(m["recall"], 0.5)
    c = compare_to_gt(mask, gt)
    assert c["gt_objects"] == 1 and c["objects_detected"] == 1 and c["false_alarm_regions"] == 1
    fa = false_alarm_analysis(mask, gt, scores=mask.astype(float))
    assert fa["false_alarms"] == 1 and fa["single_pixel_false_alarms"] == 1
    perfect = gt.astype(float)
    assert score_metrics(perfect, gt)["roc_auc"] == 1.0


# ------------------------------------------- pipeline, 20 comparison, 21-22 explain
def test_pipeline_end_to_end_and_recovery():
    cube, gt = make_scene()
    cube[0, 0] = np.nan
    cfg = PipelineConfig(**{**FAST.__dict__, "detectors": ("rx", "spatial", "autoencoder", "local_rx"),
                            "local_outer": 99})              # too big -> local_rx fails, rest continue
    r = run_pipeline(cube, gt, cfg)
    assert "local_rx" in r["errors"] and set(r["scores"]) == {"rx", "spatial", "autoencoder"}
    assert r["mask"][0, 0] == 0                              # invalid pixel never flagged
    assert r["metrics"]["fused"]["roc_auc"] > 0.9 and r["timings"]["total"] > 0
    e = explain_pixel(r, 10, 10)
    assert e["flagged"] and "summary" in e and len(e["rx_bands"]) == 5
    comp = compare_spectra(cube, [(10, 10), (5, 5)], exclude_mask=gt)
    assert comp["spectra"][0]["sam_to_background_deg"] > comp["spectra"][1]["sam_to_background_deg"]


def test_compare_methods_table():
    cube, gt = make_scene()
    rows = compare_methods(cube, gt, variants={"RX": dict(detectors=("rx",)),
                                               "Fusion": dict(detectors=("rx", "spatial"))}, base=FAST)
    assert [r["method"] for r in rows] == ["RX", "Fusion"] and all(r["roc_auc"] > 0.9 for r in rows)
