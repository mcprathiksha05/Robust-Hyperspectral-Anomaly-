"""Session State and Caching Manager for Hyperspectral Dashboard.

Ensures:
    - Safe lazy-loading of benchmark datasets via load_hsi.
    - Systematic cache invalidation to prevent stale results when datasets or parameters change.
    - Persistent connection to ExperimentHistoryDB.
    - Clean separation of UI state from core detector and evaluation modules.
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import streamlit as st
from streamlit.errors import StreamlitWidgetAlreadyInstantiatedError

from src.analysis.diagnostics import inspect_cube_quality
from src.data_loading.load_hsi import (
    get_all_manifest_entries,
    get_dataset_path,
    list_datasets,
    load,
)
from src.experiments.history import ExperimentHistoryDB

# Cache dataset loading across Streamlit reruns
@st.cache_data(show_spinner=False, ttl=30, max_entries=1)
def get_available_datasets() -> List[str]:
    """Return sorted list of available processed .npz dataset names."""
    return list_datasets()


@st.cache_data(show_spinner=False, max_entries=1)
def get_manifest_summary_cached() -> List[Dict[str, str]]:
    """Cached accessor for all manifest records."""
    return get_all_manifest_entries()


def get_backend_status() -> Dict[str, Any]:
    """Check connectivity to the external backend runner without executing it."""
    from src.dashboard.backend_bridge import DEFAULT_BACKEND_ROOT

    backend_root = Path(
        os.environ.get("HSI_BACKEND_ROOT", str(DEFAULT_BACKEND_ROOT))
    ).resolve()
    py_exe = backend_root / ".venv" / "Scripts" / "python.exe"
    runner = backend_root / "scripts" / "dashboard_runner.py"
    connected = py_exe.is_file() and runner.is_file()
    return {
        "connected": connected,
        "backend_root": str(backend_root),
        "status_label": "Online · Dual-Engine" if connected else "Local Engine Only",
        "badge_color": "#10B981" if connected else "#38BDF8",
    }


def get_dataset_version(dataset_name: str) -> str:
    """Return a cache/result version derived from the local archive file."""
    dataset_path = get_dataset_path(dataset_name).resolve()
    file_stat = dataset_path.stat()
    return f"{dataset_path}:{file_stat.st_size}:{file_stat.st_mtime_ns}"


def reconcile_dataset_selection() -> Optional[str]:
    """Normalize the canonical dataset selection without firing widget policy errors.

    The canonical state is ``selected_dataset``. Mirrored widget-backed values may
    lag behind during a rerun, so they should only fill in the canonical state when
    the canonical key is absent or invalid. Once the canonical value is known, all
    mirrored aliases should be brought back in sync so header/explorer selectors do
    not drift away from the active scene.
    """
    datasets = get_available_datasets()
    if not datasets:
        return None

    active = st.session_state.get("selected_dataset")
    if active not in datasets:
        active = None

    if active is None:
        for key in ("header_active_dataset", "explorer_dataset", "shell_dataset"):
            candidate = st.session_state.get(key)
            if candidate in datasets:
                active = candidate
                break

    if active is None:
        active = datasets[0]

    st.session_state["selected_dataset"] = active
    st.session_state["selection_dataset"] = active
    for key in ("header_active_dataset", "explorer_dataset", "shell_dataset"):
        try:
            if st.session_state.get(key) != active:
                st.session_state[key] = active
        except StreamlitWidgetAlreadyInstantiatedError:
            pass
    return active


def set_active_dataset(target_dataset: str, *, preserve_widget_key: Optional[str] = None) -> None:
    """Set the canonical dataset and mirror it without mutating widget-backed state.

    ``preserve_widget_key`` identifies a selectbox key whose value is already managed
    by Streamlit in the current rerun. We intentionally leave that key alone and only
    update the canonical selection plus the non-widget mirrors. If a widget-backed key
    has already been instantiated, we skip it instead of raising a policy error.
    """
    datasets = get_available_datasets()
    if target_dataset not in datasets:
        raise ValueError(f"Unknown dataset selected: {target_dataset!r}")

    previous_dataset = st.session_state.get("selected_dataset")
    st.session_state["selected_dataset"] = target_dataset
    st.session_state["selection_dataset"] = target_dataset
    for key in ("header_active_dataset", "shell_dataset", "explorer_dataset"):
        if key == preserve_widget_key:
            continue
        try:
            if st.session_state.get(key) != target_dataset:
                st.session_state[key] = target_dataset
        except StreamlitWidgetAlreadyInstantiatedError:
            pass

    if previous_dataset != target_dataset:
        st.session_state["scores"] = None
        st.session_state["metrics"] = None
        st.session_state["runtime_info"] = None
        st.session_state["memory_info"] = None
        st.session_state["scores_dataset"] = None
        st.session_state["last_run_config_hash"] = None
        st.session_state["active_detector"] = None
        st.session_state["active_detector_name"] = "Global-RX"
        st.session_state["active_detector_config"] = {}
        st.session_state["model_comparison"] = None
        st.session_state["pca_projection"] = None
        st.session_state["research_assistant_messages"] = []

    # Keep closely related aliases aligned even when the widget-backed key is the
    # active source of truth for this rerun. This avoids header/explorer drift after
    # a route or detection-page dataset change.
    for key in ("header_active_dataset", "shell_dataset", "explorer_dataset"):
        if key == preserve_widget_key:
            continue
        try:
            st.session_state[key] = target_dataset
        except StreamlitWidgetAlreadyInstantiatedError:
            pass


def sync_selected_dataset(widget_key: str) -> None:
    """Update the canonical scene selection and invalidate stale results.

    We intentionally avoid overwriting the source widget key from within its
    callback because Streamlit disallows writing widget-backed state values after
    the widget has been instantiated in the same rerun.
    """
    datasets = get_available_datasets()
    selected_dataset = st.session_state.get(widget_key)
    if selected_dataset is None:
        reconcile_dataset_selection()
        return
    if selected_dataset not in datasets:
        selected_dataset = reconcile_dataset_selection()
        if selected_dataset is None:
            raise ValueError(f"Unknown dataset selected: {selected_dataset!r}")

    set_active_dataset(selected_dataset, preserve_widget_key=widget_key)


@st.cache_data(show_spinner=False, max_entries=16)
def _load_dataset_cached(
    dataset_name: str,
    file_path: str,
    file_size: int,
    modified_ns: int,
) -> Tuple[np.ndarray, Optional[np.ndarray], Dict[str, Any]]:
    """Load using a cache key that changes when the dataset file changes."""
    del file_path, file_size, modified_ns
    return load(dataset_name, as_float=False)


def load_dataset_cached(
    dataset_name: str,
) -> Tuple[np.ndarray, Optional[np.ndarray], Dict[str, Any]]:
    """Cached loader for hyperspectral cube, ground truth, and metadata.

    Returns:
        cube: source dtype, shape (H, W, B)
        gt: optional (H, W) reference-label mask
        meta: dict of metadata arrays
    """
    dataset_path = get_dataset_path(dataset_name)
    file_stat = dataset_path.stat()
    return _load_dataset_cached(
        dataset_name,
        str(dataset_path.resolve()),
        file_stat.st_size,
        file_stat.st_mtime_ns,
    )


@st.cache_data(show_spinner=False, max_entries=16)
def _get_dataset_diagnostics_cached(
    dataset_name: str,
    file_path: str,
    file_size: int,
    modified_ns: int,
) -> Dict[str, Any]:
    """Cache non-destructive diagnostics for one local dataset."""
    del file_path, file_size, modified_ns
    cube, gt, _ = load_dataset_cached(dataset_name)
    return inspect_cube_quality(cube, gt)


def get_dataset_diagnostics_cached(dataset_name: str) -> Dict[str, Any]:
    dataset_path = get_dataset_path(dataset_name)
    file_stat = dataset_path.stat()
    return _get_dataset_diagnostics_cached(
        dataset_name,
        str(dataset_path.resolve()),
        file_stat.st_size,
        file_stat.st_mtime_ns,
    )


def get_history_db() -> ExperimentHistoryDB:
    """Retrieve or initialize the persistent SQLite ExperimentHistoryDB."""
    if "history_db" not in st.session_state:
        st.session_state["history_db"] = ExperimentHistoryDB()
    return st.session_state["history_db"]


def init_session_state() -> None:
    """Initialize all expected session state keys with default values."""
    datasets = get_available_datasets()
    default_dataset = datasets[0] if datasets else None
    defaults = {
        "selected_dataset": default_dataset,
        "header_active_dataset": default_dataset,
        "shell_dataset": default_dataset,
        "explorer_dataset": default_dataset,
        "loaded_dataset_name": None,
        "cube": None,
        "gt": None,
        "meta": None,
        "detector_regularization": 1e-4,
        "detector_solver": "cholesky",
        "scores": None,
        "scores_dataset": None,
        "metrics": None,
        "runtime_info": None,
        "memory_info": None,
        "last_run_config_hash": None,
        "active_detector_name": "Global-RX",
        "active_detector_config": {},
        "active_detector": None,
        "selection_dataset": None,
        "active_section": "Overview",
        "show_cinematic_intro": True,
        "intro_has_played": False,
    }

    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val
    if st.session_state.shell_dataset not in datasets:
        st.session_state.shell_dataset = st.session_state.selected_dataset
    if st.session_state.explorer_dataset not in datasets:
        st.session_state.explorer_dataset = st.session_state.selected_dataset


def invalidate_results_if_stale(
    current_dataset: str,
    current_reg: float,
    current_solver: str,
    detector_name: str = "Global-RX",
    detector_config: Optional[Dict[str, Any]] = None,
) -> bool:
    """Check if parameters or dataset changed, and invalidate computed results if stale.

    Returns:
        bool: True if results were invalidated, False otherwise.
    """
    current_hash = result_config_hash(
        current_dataset, current_reg, current_solver, detector_name, detector_config
    )
    last_hash = st.session_state.get("last_run_config_hash")

    if last_hash is not None and last_hash != current_hash:
        st.session_state["scores"] = None
        st.session_state["scores_dataset"] = None
        st.session_state["metrics"] = None
        st.session_state["runtime_info"] = None
        st.session_state["memory_info"] = None
        st.session_state["last_run_config_hash"] = None
        st.session_state["active_detector"] = None
        st.session_state["model_comparison"] = None
        st.session_state["pca_projection"] = None
        return True
    return False


def set_active_results(
    dataset_name: str,
    reg: float,
    solver: str,
    scores: np.ndarray,
    metrics: Optional[Dict[str, Any]] = None,
    runtime_info: Optional[Dict[str, Any]] = None,
    memory_info: Optional[Dict[str, Any]] = None,
    detector_name: str = "Global-RX",
    detector_config: Optional[Dict[str, Any]] = None,
    detector: Optional[Any] = None,
) -> None:
    """Store fresh inference results and update configuration hash."""
    st.session_state["scores"] = scores
    st.session_state["scores_dataset"] = dataset_name
    st.session_state["metrics"] = metrics
    st.session_state["runtime_info"] = runtime_info
    st.session_state["memory_info"] = memory_info
    st.session_state["active_detector_name"] = detector_name
    st.session_state["active_detector_config"] = detector_config or {}
    st.session_state["active_detector"] = detector
    st.session_state["last_run_config_hash"] = result_config_hash(
        dataset_name, reg, solver, detector_name, detector_config
    )


def result_config_hash(
    dataset_name: str,
    reg: float,
    solver: str,
    detector_name: str,
    detector_config: Optional[Dict[str, Any]],
) -> str:
    dataset_version = get_dataset_version(dataset_name)
    base_hash = f"{dataset_name}_{dataset_version}_{reg}_{solver}"
    if detector_name == "Global-RX" and not detector_config:
        return base_hash
    config_hash = json.dumps(detector_config or {}, sort_keys=True, separators=(",", ":"))
    return f"{base_hash}_{detector_name}_{config_hash}"


def prepare_shared_inspection(dataset_name: str, height: int, width: int, bands: int) -> None:
    """Keep the selected pixel and band in range and synchronized across sections."""
    dataset_version = get_dataset_version(dataset_name)
    if (
        st.session_state.get("selection_dataset") != dataset_name
        or st.session_state.get("selection_dataset_version") != dataset_version
    ):
        st.session_state["selection_dataset"] = dataset_name
        st.session_state["selection_dataset_version"] = dataset_version
        st.session_state["shared_pixel_row"] = height // 2
        st.session_state["shared_pixel_col"] = width // 2
        st.session_state["shared_spectral_band"] = bands // 2
        st.session_state["pca_projection"] = None
    else:
        st.session_state["shared_pixel_row"] = min(
            max(int(st.session_state.get("shared_pixel_row", height // 2)), 0),
            height - 1,
        )
        st.session_state["shared_pixel_col"] = min(
            max(int(st.session_state.get("shared_pixel_col", width // 2)), 0),
            width - 1,
        )
        st.session_state["shared_spectral_band"] = min(
            max(int(st.session_state.get("shared_spectral_band", bands // 2)), 0),
            bands - 1,
        )


def step_shared_band(step: int, bands: int) -> None:
    """Move the active spectral band without leaving the current dataset range."""
    current = int(st.session_state.get("shared_spectral_band", 0))
    st.session_state["shared_spectral_band"] = min(max(current + step, 0), bands - 1)
