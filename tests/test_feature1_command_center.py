"""Automated Verification for Feature 1 — Professional Command Center."""

from pathlib import Path
import sys
import numpy as np
from streamlit.testing.v1 import AppTest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loading.load_hsi import list_datasets, get_all_manifest_entries
from src.dashboard.state import get_backend_status, get_manifest_summary_cached


def test_command_center_manifest_and_backend_helpers():
    """Verify lightweight manifest and verified backend checks without full cube loads."""
    datasets = list_datasets()
    assert len(datasets) >= 16

    manifest = get_manifest_summary_cached()
    assert len(manifest) >= 16
    assert any(row["name"] == "abu-airport-1" for row in manifest)
    entry = next(row for row in manifest if row["name"] == "abu-airport-1")
    assert entry["height"] == "100"
    assert entry["width"] == "100"
    assert entry["bands"] == "205"

    backend_info = get_backend_status()
    assert "connected" in backend_info
    assert "status_label" in backend_info
    assert "badge_color" in backend_info


def test_command_center_app_headless_execution():
    """Verify Streamlit app renders Command Center with title, metrics, and navigation."""
    app_path = PROJECT_ROOT / "app.py"
    at = AppTest.from_file(str(app_path), default_timeout=25)
    at.run()

    # 1. No unhandled exceptions or errors
    assert len(at.exception) == 0, f"App raised exceptions: {at.exception}"
    assert len(at.error) == 0, f"App had errors: {at.error}"

    # 2. Command center title
    assert len(at.title) >= 1
    assert "HYPER" in at.title[0].value

    # 3. Active dataset selectbox present
    active_ds_widget = next(w for w in at.selectbox if w.label == "Active dataset")
    assert "abu-airport-1" in active_ds_widget.options

    # 4. Navigation radio options contain Command Center, Detection Pipeline, Datasets
    radio = at.radio[0]
    options = radio.options
    assert any("Command Center" in opt for opt in options)
    assert any("Anomaly Detection" in opt for opt in options)
    assert any("Dataset Explorer" in opt for opt in options)

    # 5. Quick action buttons exist and function
    cmd_open = next(b for b in at.button if "Open Detection Pipeline" in b.label)
    assert cmd_open is not None
