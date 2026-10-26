from pathlib import Path

import numpy as np
import pytest
from streamlit.testing.v1 import AppTest

from src.data_loading import load_hsi
from src.data_loading.load_hsi import DatasetLoadError
from src.dashboard import state


def _write_scene(path: Path, shape, dtype, include_labels=True):
    path.parent.mkdir(parents=True, exist_ok=True)
    cube = np.arange(np.prod(shape), dtype=dtype).reshape(shape)
    if include_labels:
        labels = np.zeros(shape[:2], dtype=np.uint8)
        labels[0, 0] = 1
        np.savez_compressed(path, data=cube, gt=labels, wavelengths=np.arange(shape[2]))
    else:
        np.savez_compressed(path, data=cube)


def test_discovery_finds_multiple_local_dataset_files(monkeypatch, tmp_path):
    _write_scene(tmp_path / "anomaly" / "synthetic-first.npz", (4, 5, 3), np.int16)
    _write_scene(tmp_path / "target" / "synthetic-second.npz", (6, 3, 2), np.float64)
    monkeypatch.setattr(load_hsi, "ROOT", tmp_path)

    assert load_hsi.list_datasets() == ["synthetic-first", "synthetic-second"]


def test_dataset_loader_preserves_variable_shapes_and_source_dtype(monkeypatch, tmp_path):
    _write_scene(tmp_path / "anomaly" / "synthetic-first.npz", (4, 5, 3), np.int16)
    _write_scene(tmp_path / "target" / "synthetic-second.npz", (6, 3, 2), np.float64)
    monkeypatch.setattr(load_hsi, "ROOT", tmp_path)

    first, first_labels, _ = load_hsi.load("synthetic-first", as_float=False)
    second, second_labels, _ = load_hsi.load("synthetic-second", as_float=False)

    assert first.shape == (4, 5, 3)
    assert first.dtype == np.int16
    assert first_labels.shape == (4, 5)
    assert second.shape == (6, 3, 2)
    assert second.dtype == np.float64
    assert second_labels.shape == (6, 3)


def test_missing_reference_labels_are_returned_as_unavailable(monkeypatch, tmp_path):
    _write_scene(
        tmp_path / "anomaly" / "synthetic-unlabeled.npz",
        (3, 2, 4),
        np.float32,
        include_labels=False,
    )
    monkeypatch.setattr(load_hsi, "ROOT", tmp_path)

    cube, labels, _ = load_hsi.load("synthetic-unlabeled", as_float=False)

    assert cube.shape == (3, 2, 4)
    assert labels is None


def test_dataset_cache_invalidates_when_file_changes(monkeypatch, tmp_path):
    dataset_path = tmp_path / "anomaly" / "synthetic-versioned.npz"
    _write_scene(dataset_path, (3, 2, 4), np.float32)
    monkeypatch.setattr(load_hsi, "ROOT", tmp_path)
    state._load_dataset_cached.clear()

    original, _, _ = state.load_dataset_cached("synthetic-versioned")
    _write_scene(dataset_path, (5, 4, 2), np.float64)
    updated, _, _ = state.load_dataset_cached("synthetic-versioned")

    assert original.shape == (3, 2, 4)
    assert updated.shape == (5, 4, 2)
    assert updated.dtype == np.float64


def test_invalid_cube_and_mismatched_labels_raise_explanatory_errors(monkeypatch, tmp_path):
    folder = tmp_path / "anomaly"
    folder.mkdir()
    np.savez_compressed(folder / "invalid-cube.npz", data=np.ones((3, 4)))
    np.savez_compressed(
        folder / "invalid-labels.npz",
        data=np.ones((3, 4, 2)),
        gt=np.zeros((4, 3), dtype=np.uint8),
    )
    monkeypatch.setattr(load_hsi, "ROOT", tmp_path)

    with pytest.raises(DatasetLoadError, match="non-empty numeric"):
        load_hsi.load("invalid-cube")
    with pytest.raises(DatasetLoadError, match="does not match"):
        load_hsi.load("invalid-labels")


def test_corrupted_archive_raises_an_explanatory_error(monkeypatch, tmp_path):
    folder = tmp_path / "anomaly"
    folder.mkdir()
    (folder / "corrupted.npz").write_bytes(b"not a numpy archive")
    monkeypatch.setattr(load_hsi, "ROOT", tmp_path)

    with pytest.raises(DatasetLoadError, match="Could not read corrupted.npz"):
        load_hsi.load("corrupted")


def test_all_discovered_project_scenes_load_with_matching_dimensions():
    datasets = load_hsi.list_datasets()

    assert len(datasets) > 1
    assert "abu-airport-1" in datasets
    for dataset_name in datasets:
        cube, labels, _ = load_hsi.load(dataset_name, as_float=False)
        assert cube.ndim == 3
        assert all(size > 0 for size in cube.shape)
        if labels is not None:
            assert labels.shape == cube.shape[:2]


def test_dataset_explorer_switch_updates_scene_metadata_and_band_range():
    app_path = Path(__file__).resolve().parents[1] / "app.py"
    app = AppTest.from_file(str(app_path), default_timeout=180).run()
    assert not app.exception
    expected_datasets = load_hsi.list_datasets()
    assert next(widget for widget in app.selectbox if widget.label == "Active dataset").options == expected_datasets

    explorer = next(option for option in app.radio[0].options if "Dataset Explorer" in option)
    app.radio[0].set_value(explorer).run()
    band_slider = next(widget for widget in app.slider if widget.label == "Spectral band depth")
    band_slider.set_value(102).run()
    app.session_state["scores"] = np.ones((100, 100))
    app.session_state["scores_dataset"] = "abu-airport-1"
    app.session_state["metrics"] = {"auc_roc": 0.5}

    dataset_selector = next(widget for widget in app.selectbox if widget.label == "Dataset")
    assert dataset_selector.options == expected_datasets
    dataset_selector.set_value("muufl-gulfport-campus1").run()

    assert app.session_state["selected_dataset"] == "muufl-gulfport-campus1"
    assert app.session_state["shell_dataset"] == "muufl-gulfport-campus1"
    assert app.session_state["explorer_dataset"] == "muufl-gulfport-campus1"
    assert app.session_state["scores"] is None
    assert app.session_state["metrics"] is None
    band_slider = next(widget for widget in app.slider if widget.label == "Spectral band depth")
    assert band_slider.max == 63
    assert 0 <= band_slider.value <= band_slider.max
    assert not app.exception
    assert not app.error

    detection = next(option for option in app.radio[0].options if "Anomaly Detection" in option)
    app.radio[0].set_value(detection).run()
    run_detector = next(button for button in app.button if "Run Detector" in button.label)
    run_detector.click().run()
    assert app.session_state["scores"].shape == (325, 220)
    assert app.session_state["scores_dataset"] == "muufl-gulfport-campus1"
    assert not app.exception

    header_selector = next(widget for widget in app.selectbox if widget.label == "Active dataset")
    header_selector.set_value("hydice-urban").run()
    assert app.session_state["selected_dataset"] == "hydice-urban"
    assert app.session_state["scores"] is None
    assert app.session_state["metrics"] is None
    assert not app.exception
    assert not app.error
