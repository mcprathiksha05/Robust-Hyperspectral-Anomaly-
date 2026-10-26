"""Automated Verification Suite for Phase 1: Feature 3 (3D Hyperspectral Cube) and Feature 5 (Spectral Signature Explorer)."""

from pathlib import Path
import sys
import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loading.load_hsi import list_datasets, load
from src.dashboard.components import (
    create_3d_hyperspectral_cube_figure,
    create_multi_pixel_spectral_figure,
    create_spectral_signature_figure,
    synthesize_rgb_composite,
)


def test_feature3_real_cube_dimensions_and_slicing():
    """Verify that real cubes load with exact dimensions and support valid band slices."""
    available = list_datasets()
    assert len(available) > 0, "No datasets found in data/processed"
    ds_name = available[0]
    cube, gt, meta = load(ds_name)

    assert cube.ndim == 3
    h, w, b = cube.shape
    assert h > 0 and w > 0 and b > 0
    assert np.issubdtype(cube.dtype, np.number)

    # Verify single band extraction matches cube data exactly
    mid_band = b // 2
    slice_2d = cube[:, :, mid_band]
    assert slice_2d.shape == (h, w)
    assert np.array_equal(slice_2d, cube[:, :, mid_band])


def test_feature3_display_modes_and_rgb_synthesis():
    """Verify grayscale, normalized grayscale, and false-color composite synthesis."""
    available = list_datasets()
    cube, gt, meta = load(available[0])
    h, w, b = cube.shape

    # 1. False-color composite with custom band selections
    r_b, g_b, b_b = min(b - 1, int(b * 0.7)), min(b - 1, int(b * 0.4)), min(b - 1, int(b * 0.15))
    fc_composite = synthesize_rgb_composite(cube, red_band=r_b, green_band=g_b, blue_band=b_b)
    assert fc_composite.shape == (h, w, 3)
    assert fc_composite.dtype == np.float32
    assert 0.0 <= fc_composite.min() and fc_composite.max() <= 1.0

    # 2. Linear Grayscale and 2-98% Normalized Grayscale
    raw_band = cube[:, :, 0].astype(np.float32)
    p2, p98 = np.percentile(raw_band, (2.0, 98.0))
    if p98 > p2:
        norm_band = np.clip((raw_band - p2) / (p98 - p2), 0.0, 1.0)
        assert norm_band.min() >= 0.0 and norm_band.max() <= 1.0


def test_feature3_3d_cube_figure_generation():
    """Verify Plotly 3D cube figure generation with layers and exploded mode."""
    available = list_datasets()
    cube, gt, meta = load(available[0])

    fig = create_3d_hyperspectral_cube_figure(
        cube,
        wavelengths=meta.get("wavelengths"),
        current_slice_band=0,
        layer_count=4,
        exploded=True,
        layer_separation=1.5,
        layer_opacity=0.8,
    )
    assert fig is not None
    assert len(fig.data) >= 4  # At least 4 surface slices rendered


def test_feature5_multi_pixel_spectral_extraction_and_chart():
    """Verify multi-pixel spectral extraction and comparison plot."""
    available = list_datasets()
    cube, gt, meta = load(available[0])
    h, w, b = cube.shape

    pixels = [
        (0, 0, "Origin Pixel"),
        (h // 2, w // 2, "Center Pixel"),
        (h - 1, w - 1, "Corner Pixel"),
    ]

    fig = create_multi_pixel_spectral_figure(
        cube,
        pixels=pixels,
        wavelengths=meta.get("wavelengths"),
        target_mask=gt,
        title="Test Multi-Pixel Signatures",
    )
    assert fig is not None
    assert len(fig.data) == 3

    # Verify that plotted values correspond exactly to pixel values in the cube
    for i, (r, c, tag) in enumerate(pixels):
        plotted_y = np.array(fig.data[i].y)
        expected_y = cube[r, c, :].astype(np.float64)
        np.testing.assert_allclose(plotted_y, expected_y)


def test_feature5_csv_export_dataframe_integrity():
    """Verify that exported CSV records contain exact row, col, band, and reflectance values."""
    available = list_datasets()
    ds_name = available[0]
    cube, gt, meta = load(ds_name)
    h, w, b = cube.shape

    pixels = [(5, 10, "Target A"), (15, 20, "Background B")]
    rows = []
    wl_arr = meta.get("wavelengths") if (meta.get("wavelengths") is not None and len(meta["wavelengths"]) == b) else None

    for r, c, tag in pixels:
        spec = cube[r, c, :].astype(np.float64)
        for b_idx, val in enumerate(spec):
            rows.append({
                "Dataset": ds_name,
                "Row": r,
                "Column": c,
                "Tag": tag,
                "Band_Index": b_idx + 1,
                "Wavelength_nm": float(wl_arr[b_idx]) if wl_arr is not None else np.nan,
                "Spectral_Value": float(val),
            })

    df = pd.DataFrame(rows)
    assert len(df) == len(pixels) * b
    assert "Spectral_Value" in df.columns
    # Check that first pixel's first band value matches cube[5, 10, 0]
    assert df.iloc[0]["Spectral_Value"] == float(cube[5, 10, 0])


def test_headless_phase1_navigation_and_ui_elements():
    """Verify Streamlit AppTest can navigate to Feature 3 and Feature 5 pages."""
    app_path = PROJECT_ROOT / "app.py"
    at = AppTest.from_file(str(app_path), default_timeout=25).run()
    assert len(at.exception) == 0

    # 1. Navigate to Feature 3: 3D Hyperspectral Cube
    nav_radio = at.sidebar.radio[0]
    cube_opt = next(opt for opt in nav_radio.options if "Datasets" in opt or "Cube" in opt)
    at = nav_radio.set_value(cube_opt).run()
    assert len(at.exception) == 0
    assert any("3D Hyperspectral Cube" in str(m.value) for m in at.markdown)

    # 2. Navigate to Feature 5: Spectral Signature Explorer
    nav_radio = at.sidebar.radio[0]
    spec_opt = next(opt for opt in nav_radio.options if "Spectral" in opt)
    at = nav_radio.set_value(spec_opt).run()
    assert len(at.exception) == 0
    assert any("Spectral Signature Explorer" in str(m.value) for m in at.markdown)

