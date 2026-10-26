import numpy as np
import pytest

from src.analysis.diagnostics import compute_pca_component_views, inspect_cube_quality
from src.dashboard.components import (
    create_3d_anomaly_topography_figure,
    create_3d_hyperspectral_cube_figure,
)
from src.models.pca_residual import PCAResidualDetector


def test_pca_residual_detector_finds_spectral_reconstruction_outlier():
    rng = np.random.default_rng(10)
    latent = rng.normal(size=(24, 20, 1))
    background = np.ones((8,), dtype=np.float32)
    cube = (
        latent * background
        + rng.normal(scale=0.02, size=(24, 20, 8))
    ).astype(np.float32)
    cube[12, 10, :] += np.array([2, -2, 2, -2, 2, -2, 2, -2], dtype=np.float32)

    detector = PCAResidualDetector(n_components=1, random_state=4)
    scores = detector.fit_predict(cube)

    assert scores.shape == cube.shape[:2]
    assert np.all(np.isfinite(scores))
    assert np.unravel_index(np.argmax(scores), scores.shape) == (12, 10)


def test_pca_residual_detector_validates_fit_state_and_band_count():
    detector = PCAResidualDetector(n_components=2)
    with pytest.raises(RuntimeError, match="must be fitted"):
        detector.predict(np.ones((5, 5, 4)))

    detector.fit(np.random.default_rng(1).normal(size=(5, 5, 4)).astype(np.float32))
    with pytest.raises(ValueError, match="band count mismatch"):
        detector.predict(np.ones((5, 5, 5), dtype=np.float32))


def test_pca_residual_detector_rejects_unachievable_component_count():
    with pytest.raises(ValueError, match="n_components"):
        PCAResidualDetector(n_components=5).fit(np.ones((3, 3, 4), dtype=np.float32))


def test_dataset_diagnostics_report_invalid_values_and_labels():
    cube = np.ones((4, 5, 3), dtype=np.float32)
    cube[:, :, 0] = np.arange(20, dtype=np.float32).reshape(4, 5)
    cube[:, :, 1] = np.arange(20, dtype=np.float32).reshape(4, 5) * 2
    cube[0, 0, 0] = np.nan
    cube[1, 0, 1] = np.inf
    gt = np.zeros((4, 5), dtype=np.uint8)
    gt[2, 3] = 1

    report = inspect_cube_quality(cube, gt)

    assert report["shape"] == (4, 5, 3)
    assert report["nan_values"] == 1
    assert report["infinite_values"] == 1
    assert report["constant_band_indices"] == [2]
    assert report["positive_label_count"] == 1
    assert report["binary_labels"] is True


def test_dataset_diagnostics_validate_ground_truth_shape():
    with pytest.raises(ValueError, match="does not match cube spatial shape"):
        inspect_cube_quality(np.ones((4, 5, 3)), np.zeros((5, 4)))


def test_pca_component_views_keep_spatial_shape_and_are_reproducible():
    rng = np.random.default_rng(8)
    cube = rng.normal(size=(12, 10, 6)).astype(np.float32)
    first = compute_pca_component_views(cube, n_components=3, random_state=7)
    second = compute_pca_component_views(cube, n_components=3, random_state=7)

    assert first["component_maps"].shape == (3, 12, 10)
    assert first["components"].shape == (3, 6)
    assert first["fit_sample_count"] == 120
    np.testing.assert_allclose(first["component_maps"], second["component_maps"])


def test_pca_component_views_reject_nonfinite_cube():
    cube = np.ones((4, 5, 3), dtype=np.float32)
    cube[0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="without NaN or infinite"):
        compute_pca_component_views(cube)


def test_cube_figure_explodes_sampled_bands_and_marks_selected_pixel():
    cube = np.arange(12 * 10 * 8, dtype=np.float32).reshape(12, 10, 8)
    figure = create_3d_hyperspectral_cube_figure(
        cube,
        current_slice_band=3,
        layer_count=4,
        exploded=True,
        layer_separation=1.5,
        selected_pixel=(5, 6),
    )

    surfaces = [trace for trace in figure.data if trace.type == "surface"]
    assert len(surfaces) == 5
    assert any(trace.name == "Selected pixel" for trace in figure.data)
    assert "exploded spacing is illustrative" in figure.layout.title.text
    assert surfaces[-1].z[0][0] > surfaces[0].z[0][0]


def test_anomaly_topography_scales_scores_without_changing_score_data():
    scores = np.arange(20, dtype=np.float64).reshape(4, 5)
    figure = create_3d_anomaly_topography_figure(
        scores,
        height_scale=2.5,
        selected_pixel=(2, 3),
    )

    np.testing.assert_allclose(figure.data[0].z, scores * 2.5)
    assert any(trace.name == "Selected pixel" for trace in figure.data)
    assert "not physical elevation" in figure.layout.title.text
