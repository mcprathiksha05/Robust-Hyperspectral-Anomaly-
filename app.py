"""HSI Defense Analytics Streamlit application entry point."""

import os
import sys
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.dashboard.cinematic_intro import render_cinematic_intro_overlay
from src.dashboard.pages import (
    render_anomaly_detection_page,
    render_dataset_explorer_page,
    render_exports_page,
    render_history_page,
    render_leaderboard_page,
    render_overview_page,
    render_performance_page,
    render_research_analysis_page,
    render_robustness_page,
    render_upload_page,
)

try:
    from src.dashboard.pages import render_earth_explorer_page
except ImportError:
    from src.dashboard.earth_explorer import (
        render_earth_explorer_view as render_earth_explorer_page,
    )

from src.dashboard.state import (
    get_available_datasets,
    get_backend_status,
    get_history_db,
    init_session_state,
    reconcile_dataset_selection,
    sync_selected_dataset,
)
from src.dashboard.theme import (
    apply_custom_theme,
    render_application_header,
    render_horizontal_sliding_nav,
)


def set_active_section(section: str) -> None:
    """Set the sidebar route from a quick-action button callback or direct string."""
    target = section.strip().lower()

    valid_sections = [
        "⌂  Command Center / Overview",
        "🌐  3D Earth Explorer",
        "◉  Detection Pipeline / Anomaly Detection",
        "▦  Datasets / Dataset Explorer",
        "◈  Robustness Laboratory",
        "∿  Spectral Analysis",
        "◷  Runtime / Profiling",
        "≋  Model Comparison",
        "◴  Experiment History",
        "⇩  Export Results",
        "📤  Upload Data",
    ]

    resolved = None

    # Exact or substring match against full labels.
    for name in valid_sections:
        if target == name.lower() or target in name.lower():
            resolved = name
            break

    # Keyword-based routing fallback for backwards compatibility.
    if not resolved:
        for name in valid_sections:
            name_lower = name.lower()

            if (
                ("overview" in target or "command" in target)
                and ("overview" in name_lower or "command" in name_lower)
            ):
                resolved = name
                break

            if (
                ("earth" in target or "globe" in target)
                and "earth" in name_lower
            ):
                resolved = name
                break

            if (
                ("detect" in target or "pipeline" in target or "anomaly" in target)
                and ("detect" in name_lower or "anomaly" in name_lower)
            ):
                resolved = name
                break

            if (
                ("dataset" in target or "explore" in target)
                and ("dataset" in name_lower or "explore" in name_lower)
            ):
                resolved = name
                break

            if "robust" in target and "robust" in name_lower:
                resolved = name
                break

            if (
                ("spectral" in target or "research" in target)
                and ("spectral" in name_lower or "research" in name_lower)
            ):
                resolved = name
                break

            if (
                ("runtime" in target or "profil" in target or "perf" in target)
                and ("runtime" in name_lower or "profil" in name_lower)
            ):
                resolved = name
                break

            if (
                ("leaderboard" in target or "compar" in target)
                and ("leaderboard" in name_lower or "compar" in name_lower)
            ):
                resolved = name
                break

            if "history" in target and "history" in name_lower:
                resolved = name
                break

            if "export" in target and "export" in name_lower:
                resolved = name
                break

            if "upload" in target and "upload" in name_lower:
                resolved = name
                break

    if not resolved:
        resolved = section

    try:
        st.session_state.active_section = resolved
    except Exception:
        pass

    st.session_state["target_section"] = resolved


def main() -> None:
    st.set_page_config(
        page_title="HYPER — Hyperspectral Intelligence Platform | Command Center",
        page_icon="🛰️",
        layout="wide",
        initial_sidebar_state="collapsed",
    )

    apply_custom_theme()
    init_session_state()

    # Title check for testing and semantic structure.
    st.title("HYPER — Hyperspectral Intelligence Platform", anchor=False)

    # Detect headless automated testing environment.
    is_testing = (
        "pytest" in sys.modules
        or "PYTEST_CURRENT_TEST" in os.environ
        or st.session_state.get("_is_testing", False)
    )

    # Direct query parameter navigation and intro controls.
    if "intro" in st.query_params:
        param = str(st.query_params["intro"]).strip().lower()

        if param in ("1", "true", "yes", "replay"):
            st.session_state["show_cinematic_intro"] = True
            st.session_state["intro_has_played"] = False
        elif param in ("0", "false", "no", "skip", "done"):
            st.session_state["show_cinematic_intro"] = False
            st.session_state["intro_has_played"] = True

    if "nointro" in st.query_params:
        st.session_state["show_cinematic_intro"] = False
        st.session_state["intro_has_played"] = True

    if "nav" in st.query_params and st.query_params["nav"]:
        set_active_section(st.query_params["nav"])

    # Full-screen Cinematic AI Introduction (0–18 seconds).
    if (
        not is_testing
        and st.session_state.get("show_cinematic_intro", True)
        and not st.session_state.get("intro_has_played", False)
    ):
        intro_bar_col1, intro_bar_col2 = st.columns([7, 3])

        with intro_bar_col1:
            st.markdown(
                """
                <div style="display: flex; align-items: center; gap: 10px; padding: 6px 0;">
                    <span style="display: inline-block; width: 8px; height: 8px; border-radius: 50%; background: #00E5FF; box-shadow: 0 0 10px #00E5FF;"></span>
                    <span style="font-family: 'JetBrains Mono', monospace; font-size: 13px; font-weight: 700; color: #00E5FF; letter-spacing: 2px;">HYPER // CINEMATIC INTRODUCTION</span>
                    <span style="font-family: 'JetBrains Mono', monospace; font-size: 11px; color: #94A3B8;">(18s Immersive Scientific Journey)</span>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with intro_bar_col2:
            if st.button(
                "⏭ Skip Intro & Launch Platform",
                key="btn_skip_cinematic_intro",
                type="primary",
                width="stretch",
            ):
                st.session_state["show_cinematic_intro"] = False
                st.session_state["intro_has_played"] = True
                st.rerun()

        render_cinematic_intro_overlay(height=840)
        return

    datasets = get_available_datasets()
    reconcile_dataset_selection()

    if datasets and st.session_state.selected_dataset not in datasets:
        missing_dataset = st.session_state.selected_dataset
        st.session_state["selected_dataset"] = datasets[0]
        st.session_state["selection_dataset"] = datasets[0]

        st.warning(
            f"The previously selected dataset {missing_dataset!r} is no longer available. "
            f"Selected {datasets[0]!r}; choose another dataset from the selector."
        )

    history_count = len(get_history_db().list_runs())
    backend_status = get_backend_status()

    engine_label = (
        "Backend Connected"
        if backend_status["connected"]
        else "Local Engine Ready"
    )
    engine_color = "#10B981" if backend_status["connected"] else "#00CFFF"

    sections = {
        "⌂  Command Center / Overview": render_overview_page,
        "🌐  3D Earth Explorer": render_earth_explorer_page,
        "◉  Detection Pipeline / Anomaly Detection": render_anomaly_detection_page,
        "▦  Datasets / Dataset Explorer": render_dataset_explorer_page,
        "◈  Robustness Laboratory": render_robustness_page,
        "∿  Spectral Analysis": render_research_analysis_page,
        "◷  Runtime / Profiling": render_performance_page,
        "≋  Model Comparison": render_leaderboard_page,
        "◴  Experiment History": render_history_page,
        "⇩  Export Results": render_exports_page,
        "📤  Upload Data": render_upload_page,
    }

    with st.sidebar:
        st.markdown(
            """
            <div class="brand-lockup">
                <svg class="brand-mark" viewBox="0 0 40 46" role="img" aria-label="Satellite Orbit Badge">
                    <circle cx="20" cy="23" r="14" fill="none" stroke="#00CFFF" stroke-width="1.8" stroke-dasharray="3 3"/>
                    <ellipse cx="20" cy="23" rx="18" ry="7" fill="none" stroke="#168BFF" stroke-width="1.6" transform="rotate(-28 20 23)"/>
                    <path d="M20 7v6M20 33v6M4 23h6M30 23h6" stroke="#29E7A4" stroke-width="1.6" stroke-linecap="round"/>
                    <circle cx="20" cy="23" r="3.2" fill="#00F0FF"/>
                </svg>
                <div>
                    <div class="brand-name">HYPER</div>
                    <div class="brand-subtitle">Hyperspectral Intelligence Platform</div>
                    <div class="brand-tagline">“See Beyond the Visible.”</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            f"""
            <div class="sidebar-dataset-status">
                <span>BENCHMARK SCENES</span>
                <strong>{len(datasets)} local datasets</strong>
                <div class="sidebar-status-line"><span></span> {history_count} saved experiment runs</div>
                <div class="sidebar-engine-badge" style="color: {engine_color};">
                    <span style="background: {engine_color}; box-shadow: 0 0 6px {engine_color};"></span> {engine_label}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            '<div class="sidebar-section-label">WORKSPACE</div>',
            unsafe_allow_html=True,
        )

        section_names = list(sections)

        # Apply any pending section navigation request before the radio widget is instantiated.
        if "target_section" in st.session_state and st.session_state["target_section"]:
            pending_target = st.session_state.pop("target_section")

            if pending_target in section_names:
                st.session_state.active_section = pending_target
            else:
                for name in section_names:
                    if str(pending_target).lower() in name.lower():
                        st.session_state.active_section = name
                        break

        if st.session_state.active_section not in section_names:
            mapped = False

            for name in section_names:
                if str(st.session_state.active_section).lower() in name.lower():
                    st.session_state.active_section = name
                    mapped = True
                    break

            if not mapped:
                st.session_state.active_section = section_names[0]

        selected_section = st.radio(
            "Workspace",
            section_names,
            key="active_section",
            label_visibility="collapsed",
        )

        st.markdown(
            '<div class="sidebar-section-label">QUICK ACTIONS</div>',
            unsafe_allow_html=True,
        )

        quick_actions = st.columns(2)

        quick_actions[0].button(
            "Explore data",
            width="stretch",
            key="quick_explore",
            on_click=set_active_section,
            args=("▦  Datasets / Dataset Explorer",),
        )

        quick_actions[1].button(
            "Run detection",
            width="stretch",
            key="quick_detection",
            on_click=set_active_section,
            args=("◉  Detection Pipeline / Anomaly Detection",),
        )

        if st.button(
            "🎬 Replay Cinematic Intro",
            width="stretch",
            key="quick_replay_intro",
        ):
            st.session_state["show_cinematic_intro"] = True
            st.session_state["intro_has_played"] = False
            st.rerun()

        st.markdown(
            """
            <div class="sidebar-terrain" aria-hidden="true">
                <svg viewBox="0 0 260 84" preserveAspectRatio="none">
                    <path d="M0 70 25 55 42 63 68 30 88 48 110 18 131 44 152 26 177 56 197 35 220 62 242 46 260 60"
                          fill="none" stroke="rgba(0,207,255,.7)" stroke-width="1"/>
                    <path d="M0 77 25 64 42 69 68 41 88 57 110 32 131 53 152 39 177 65 197 48 220 70 242 58 260 68"
                          fill="none" stroke="rgba(22,139,255,.45)" stroke-width="1"/>
                    <path d="M0 82 25 71 42 75 68 53 88 65 110 45 131 63 152 50 177 73 197 60 220 77 242 68 260 76"
                          fill="none" stroke="rgba(22,139,255,.24)" stroke-width="1"/>
                </svg>
                <span>SCIENTIFIC WORKSTATION • COMMAND CENTER</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    render_application_header(
        selected_section.split("  ", 1)[1],
        datasets,
        dataset_widget_key="shell_dataset",
        on_dataset_change=sync_selected_dataset,
    )

    # Initialize and render the active-dataset selector using its session-state key.
    if "header_active_dataset" not in st.session_state:
        st.session_state["header_active_dataset"] = st.session_state.get(
            "selected_dataset", datasets[0]
        )

    if st.session_state["header_active_dataset"] not in datasets:
        st.session_state["header_active_dataset"] = st.session_state.get(
            "selected_dataset", datasets[0]
        )

    st.selectbox(
        "Active dataset",
        options=datasets,
        key="header_active_dataset",
        on_change=sync_selected_dataset,
        args=("header_active_dataset",),
        help="Choose the active benchmark scene for the current workspace.",
    )

    render_horizontal_sliding_nav(
        selected_section,
        on_select=set_active_section,
    )

    sections[selected_section]()


if __name__ == "__main__":
    main()