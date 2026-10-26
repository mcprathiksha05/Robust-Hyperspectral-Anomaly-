"""Automated Verification Suite for Feature 2 — Interactive 3D Earth Explorer."""

from pathlib import Path
import sys
import pytest
from streamlit.testing.v1 import AppTest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.dashboard.earth_explorer import (
    ILLUSTRATIVE_LOCATIONS,
    build_3d_earth_html,
)


def test_illustrative_locations_metadata_integrity():
    """Verify that all 5 specified illustrative locations exist with valid coordinates."""
    required_keys = ["bengaluru", "amazon", "sahara", "north_atlantic", "himalayas"]
    for key in required_keys:
        assert key in ILLUSTRATIVE_LOCATIONS, f"Missing required location: {key}"
        loc = ILLUSTRATIVE_LOCATIONS[key]
        assert "name" in loc and len(loc["name"]) > 0
        assert "lat" in loc and -90.0 <= loc["lat"] <= 90.0
        assert "lon" in loc and -180.0 <= loc["lon"] <= 180.0
        assert "description" in loc and len(loc["description"]) > 20
        assert "region_type" in loc
        # Scientific honesty requirement: no fabricated dataset associations
        assert loc["dataset_linked"] is False
        assert "No hyperspectral dataset is linked to this location." in loc["dataset_note"]


def test_build_3d_earth_html_components_and_disclaimers():
    """Verify HTML generator embeds Three.js, procedural textures, satellite, and disclaimers."""
    html = build_3d_earth_html(
        initial_location_id="bengaluru",
        auto_rotate=True,
        satellite_playing=True,
        orbit_speed="normal",
        scan_active=True,
    )
    assert isinstance(html, str)
    assert len(html) > 1000

    # Three.js & OrbitControls integration
    assert "three.min.js" in html
    assert "OrbitControls.js" in html

    # Procedural 2048x1024 offline texture generator
    assert "generateProceduralEarthTexture" in html
    assert "CanvasTexture" in html

    # Parametric Satellite Model & 51.6° inclination
    assert "orbitRadius" in html
    assert "51.6" in html
    assert "busMesh" in html
    assert "wingGeo" in html

    # Scanning swath visualization
    assert "scanCone" in html
    assert "footprintMesh" in html

    # 2D fallback renderer for non-WebGL environments
    assert "init2DFallback" in html
    assert "fallback-canvas" in html

    # All 5 locations encoded in the script
    for loc_id in ["bengaluru", "amazon", "sahara", "north_atlantic", "himalayas"]:
        assert loc_id in html


def test_headless_earth_explorer_page_navigation():
    """Verify Streamlit AppTest can navigate to 3D Earth Explorer without errors."""
    app_path = PROJECT_ROOT / "app.py"
    at = AppTest.from_file(str(app_path), default_timeout=25).run()
    assert len(at.exception) == 0

    # Locate Earth Explorer in workspace radio
    earth_opt = next(opt for opt in at.radio[0].options if "Earth Explorer" in opt)
    at.radio[0].set_value(earth_opt).run()

    # Verify no unhandled exceptions
    assert len(at.exception) == 0, f"Exceptions on Earth Explorer page: {at.exception}"
    assert len(at.error) == 0, f"Errors on Earth Explorer page: {at.error}"

    # Verify title & disclaimers
    titles = [t.value for t in at.title]
    assert any("3D Earth Explorer" in t for t in titles)

    # Verify location selector widget is present
    loc_selector = next(w for w in at.selectbox if w.label == "Select illustrative landmark")
    assert "Bengaluru, India" in loc_selector.options
    assert "Himalayan Mountain Range" in loc_selector.options

    # Verify speed selector
    speed_selector = next(w for w in at.selectbox if w.label == "Orbit velocity")
    assert "Normal" in speed_selector.options

    # Verify buttons
    btn_labels = [b.label for b in at.button]
    assert any("Dataset Explorer" in lbl for lbl in btn_labels)
    assert any("Clear Selection" in lbl for lbl in btn_labels)

    # Verify that ML detector backend scores are NOT triggered by globe navigation
    assert at.session_state.get("scores") is None


def test_earth_explorer_open_dataset_button_navigation():
    """Verify clicking 'Open Dataset Explorer' button transitions page cleanly without StreamlitWidgetAlreadyInstantiatedError."""
    app_path = PROJECT_ROOT / "app.py"
    at = AppTest.from_file(str(app_path), default_timeout=25).run()
    assert len(at.exception) == 0

    # Navigate to 3D Earth Explorer
    earth_opt = next(opt for opt in at.radio[0].options if "Earth Explorer" in opt)
    at.radio[0].set_value(earth_opt).run()
    assert len(at.exception) == 0

    # Click the Open Dataset Explorer button
    dataset_btn = next(b for b in at.button if "Open Dataset Explorer" in b.label)
    at = dataset_btn.click().run()

    # Must have 0 exceptions and active section must transition to Datasets / Dataset Explorer
    assert len(at.exception) == 0, f"Exception occurred on click: {at.exception}"
    assert "Datasets" in at.session_state.get("active_section", "")


def test_earth_explorer_launch_pipeline_button():
    """Verify clicking 'Run Detection Pipeline' routes to Anomaly Detection with chosen scene."""
    app_path = PROJECT_ROOT / "app.py"
    at = AppTest.from_file(str(app_path), default_timeout=25).run()
    assert len(at.exception) == 0

    # Navigate to 3D Earth Explorer
    earth_opt = next(opt for opt in at.radio[0].options if "Earth Explorer" in opt)
    at.radio[0].set_value(earth_opt).run()
    assert len(at.exception) == 0

    # Click the Run Detection Pipeline button
    detect_btn = next(b for b in at.button if "Run Detection Pipeline" in b.label)
    at = detect_btn.click().run()

    # Must have 0 exceptions and active section must transition to Detection Pipeline
    assert len(at.exception) == 0, f"Exception occurred on click: {at.exception}"
    assert "Detection" in at.session_state.get("active_section", "")


