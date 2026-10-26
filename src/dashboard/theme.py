"""Design System & Aesthetic Styling for Hyperspectral Intelligence Platform (HYPER).

Provides:
    - Global dark space/scientific theme CSS injection with deep navy background and starfield.
    - Uniform cinematic brand header & real-time telemetry banner.
    - Premium horizontal sliding navigation bar (carousel) across all 10 modules.
    - High-information-density metric HUD cards and telemetry indicators.
    - Standardized Plotly figure templates for consistent 2D and 3D rendering.
"""

import textwrap
from typing import Any, Callable, Dict, List, Optional, Tuple
import streamlit as st


def apply_custom_theme() -> None:
    """Inject premium scientific visualization stylesheet into Streamlit."""
    css = textwrap.dedent(
        """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600;700&display=swap');

    :root {
        --bg-base: #030914;
        --bg-surface: #061224;
        --bg-sidebar: #07182B;
        --bg-card: #091B30;
        --bg-card-elevated: #0D233B;
        --bg-card-hover: #102C49;
        --accent-cyan: #00E5FF;
        --accent-cyan-sec: #00BFEA;
        --accent-blue: #176BFF;
        --accent-blue-sec: #3478F6;
        --accent-purple: #9B7BFF;
        --accent-green: #35E0A1;
        --accent-amber: #FFBE55;
        --accent-rose: #FF6178;
        --text-primary: #F2F7FF;
        --text-secondary: #9CB0C8;
        --text-muted: #6F849E;
        --border-subtle: #18334F;
        --border-cyan-bright: #00AEEA;
    }

    /* Overall Space Base & Celestial Atmosphere */
    .stApp {
        background-color: #030914 !important;
        background-image: 
            radial-gradient(circle at 50% 0%, rgba(23, 107, 255, 0.16) 0%, transparent 62%),
            radial-gradient(circle at 85% 15%, rgba(0, 229, 255, 0.09) 0%, transparent 40%),
            radial-gradient(circle at 15% 75%, rgba(155, 123, 255, 0.05) 0%, transparent 45%),
            radial-gradient(1.5px 1.5px at 30px 40px, #ffffff, rgba(0,0,0,0)),
            radial-gradient(1.5px 1.5px at 120px 180px, #00E5FF, rgba(0,0,0,0)),
            radial-gradient(2px 2px at 260px 90px, #ffffff, rgba(0,0,0,0)),
            radial-gradient(1.5px 1.5px at 360px 290px, #9CB0C8, rgba(0,0,0,0)),
            radial-gradient(2px 2px at 480px 190px, #00E5FF, rgba(0,0,0,0)),
            radial-gradient(1px 1px at 580px 70px, #ffffff, rgba(0,0,0,0)),
            radial-gradient(2px 2px at 720px 330px, #ffffff, rgba(0,0,0,0)),
            radial-gradient(1.5px 1.5px at 870px 150px, #3478F6, rgba(0,0,0,0)),
            radial-gradient(2px 2px at 990px 270px, #ffffff, rgba(0,0,0,0)) !important;
        background-repeat: repeat !important;
        background-size: auto, auto, auto, 550px 550px, 700px 700px, 780px 780px, 620px 620px, 860px 860px, 480px 480px, 820px 820px, 680px 680px, 920px 920px !important;
        color: var(--text-primary) !important;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
    }

    /* Container Spacing */
    [data-testid="stMainBlockContainer"] {
        padding-top: calc(3.75rem - 1rem) !important;
    }
    [data-testid="stMainBlockContainer"],
    .block-container {
        padding-bottom: 2.2rem !important;
        padding-left: 1.5rem !important;
        padding-right: 1.5rem !important;
        max-width: 100% !important;
    }

    /* Keep semantic h1 for tests & accessibility while letting custom brand header render */
    [data-testid="stMainBlockContainer"] > div:first-child h1,
    .stTitle {
        display: none !important;
    }

    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background-color: var(--bg-sidebar) !important;
        border-right: 1px solid var(--border-subtle) !important;
        box-shadow: 4px 0 24px rgba(0, 0, 0, 0.5) !important;
    }
    [data-testid="stSidebar"] hr {
        border-color: var(--border-subtle) !important;
        margin: 0.8rem 0 !important;
    }

    /* Custom Radio in Sidebar */
    [data-testid="stSidebar"] .stRadio div[role="radiogroup"] > label {
        min-height: 2.15rem;
        padding: 0.42rem 0.7rem !important;
        margin: 0 !important;
        border-radius: 6px !important;
        background: transparent !important;
        border: 1px solid transparent !important;
        color: var(--text-secondary) !important;
        font-size: 0.81rem !important;
        letter-spacing: 0 !important;
        text-transform: none !important;
        font-family: 'Inter', sans-serif !important;
        transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1) !important;
    }
    [data-testid="stSidebar"] .stRadio div[role="radiogroup"] > label:hover {
        background: rgba(0, 229, 255, 0.08) !important;
        border-color: rgba(0, 229, 255, 0.25) !important;
        color: #FFFFFF !important;
    }
    [data-testid="stSidebar"] .stRadio div[role="radiogroup"] > label[data-checked="true"] {
        color: #FFFFFF !important;
        background: linear-gradient(90deg, rgba(0, 229, 255, 0.18), rgba(23, 107, 255, 0.12)) !important;
        border: 1px solid rgba(0, 229, 255, 0.4) !important;
        border-left: 3px solid var(--accent-cyan) !important;
        box-shadow: inset 0 0 18px rgba(0, 229, 255, 0.08) !important;
        font-weight: 600 !important;
    }

    /* Global Typography */
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Inter', sans-serif !important;
        color: var(--text-primary) !important;
        font-weight: 700 !important;
        letter-spacing: -0.02em !important;
    }

    /* Global Brand Header */
    .hyper-global-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 0.45rem 0.2rem 0.75rem 0.2rem;
        margin-bottom: 0.45rem;
        border-bottom: 1px solid var(--border-subtle);
        flex-wrap: wrap;
        gap: 0.85rem;
    }
    .hyper-brand-group {
        display: flex;
        align-items: center;
        gap: 0.75rem;
    }
    .hyper-logo-svg {
        width: 38px;
        height: 38px;
        flex: 0 0 auto;
        filter: drop-shadow(0 0 8px rgba(0, 229, 255, 0.5));
        transition: transform 0.3s ease, filter 0.3s ease;
    }
    .hyper-logo-svg:hover {
        transform: rotate(15deg) scale(1.05);
        filter: drop-shadow(0 0 14px rgba(0, 229, 255, 0.85));
    }
    .hyper-brand-titles {
        display: flex;
        flex-direction: column;
    }
    .hyper-brand-main {
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    .hyper-brand-name {
        font-size: 1.35rem;
        font-weight: 800;
        letter-spacing: 0.06em;
        background: linear-gradient(135deg, #FFFFFF 0%, #E2E8F0 50%, var(--accent-cyan) 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        line-height: 1.1;
    }
    .hyper-brand-sub {
        font-size: 0.76rem;
        color: var(--text-secondary);
        font-weight: 500;
        letter-spacing: 0.02em;
    }
    .hyper-brand-tagline {
        font-size: 0.72rem;
        color: var(--accent-cyan);
        font-style: italic;
        letter-spacing: 0.02em;
    }
    .hyper-header-telemetry {
        display: flex;
        align-items: center;
        gap: 0.65rem;
        flex-wrap: wrap;
    }
    .header-status-pill {
        display: inline-flex;
        align-items: center;
        gap: 0.4rem;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.68rem;
        font-weight: 600;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        padding: 0.28rem 0.65rem;
        border-radius: 6px;
        background: rgba(9, 27, 48, 0.7);
        border: 1px solid var(--border-subtle);
    }
    .header-status-pill.online {
        color: var(--accent-green);
        border-color: rgba(53, 224, 161, 0.35);
        background: rgba(53, 224, 161, 0.08);
    }
    .header-status-pill.online .pulse-dot {
        width: 6px;
        height: 6px;
        border-radius: 50%;
        background: var(--accent-green);
        box-shadow: 0 0 8px var(--accent-green);
        display: inline-block;
    }
    .header-status-pill.cyan {
        color: var(--accent-cyan);
        border-color: rgba(0, 229, 255, 0.35);
        background: rgba(0, 229, 255, 0.08);
    }
    .header-status-pill.cyan .pulse-dot {
        width: 6px;
        height: 6px;
        border-radius: 50%;
        background: var(--accent-cyan);
        box-shadow: 0 0 8px var(--accent-cyan);
        display: inline-block;
    }

    /* =========================================================================
       HORIZONTAL SLIDING COMMAND NAVIGATION BAR (CAROUSEL)
       ========================================================================= */
    .hyper-nav-wrapper {
        background: rgba(7, 24, 43, 0.85);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border: 1px solid var(--border-subtle);
        box-shadow: 0 4px 24px rgba(0, 0, 0, 0.5), inset 0 0 16px rgba(0, 229, 255, 0.04);
        border-radius: 12px;
        padding: 0.35rem 0.45rem;
        margin: 0.2rem 0 1.15rem 0;
        position: relative;
        z-index: 50;
    }
    .hyper-nav-arrow-btn {
        background: rgba(9, 27, 48, 0.9) !important;
        border: 1px solid var(--border-subtle) !important;
        color: var(--accent-cyan) !important;
        border-radius: 8px !important;
        width: 2.15rem !important;
        height: 2.25rem !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        cursor: pointer !important;
        font-size: 1.2rem !important;
        font-weight: 700 !important;
        transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1) !important;
        padding: 0 !important;
        margin: 0 auto !important;
    }
    .hyper-nav-arrow-btn:hover {
        background: var(--bg-card-hover) !important;
        border-color: var(--accent-cyan) !important;
        box-shadow: 0 0 12px rgba(0, 229, 255, 0.4) !important;
        color: #FFFFFF !important;
        transform: scale(1.05) !important;
    }
    .hyper-nav-arrow-btn:active {
        transform: scale(0.95) !important;
    }

    /* Inner Container for Navigation Track */
    .st-key-hyper_nav_slider {
        width: 100% !important;
    }
    .st-key-hyper_nav_slider > div {
        display: flex !important;
        flex-direction: row !important;
        flex-wrap: nowrap !important;
        overflow-x: auto !important;
        scroll-behavior: smooth !important;
        scrollbar-width: none !important;
        -ms-overflow-style: none !important;
        gap: 0.42rem !important;
        padding: 0.15rem 0.1rem !important;
        align-items: center !important;
    }
    .st-key-hyper_nav_slider > div::-webkit-scrollbar {
        display: none !important;
    }
    .st-key-hyper_nav_slider button {
        min-width: max-content !important;
        padding: 0.42rem 0.85rem !important;
        border-radius: 8px !important;
        font-family: 'Inter', sans-serif !important;
        font-size: 0.80rem !important;
        font-weight: 600 !important;
        letter-spacing: 0.01em !important;
        white-space: nowrap !important;
        transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1) !important;
        border: 1px solid var(--border-subtle) !important;
    }
    /* Inactive module buttons */
    .st-key-hyper_nav_slider button[kind="secondary"] {
        background: rgba(9, 27, 48, 0.65) !important;
        border: 1px solid var(--border-subtle) !important;
        color: var(--text-secondary) !important;
    }
    .st-key-hyper_nav_slider button[kind="secondary"]:hover {
        background: var(--bg-card-hover) !important;
        border-color: var(--accent-cyan-sec) !important;
        color: #FFFFFF !important;
        box-shadow: 0 0 14px rgba(0, 191, 234, 0.25) !important;
        transform: translateY(-1px) !important;
    }
    /* Active module button: Glowing Cyan Pill */
    .st-key-hyper_nav_slider button[kind="primary"] {
        background: linear-gradient(135deg, rgba(0, 229, 255, 0.24) 0%, rgba(23, 107, 255, 0.35) 100%) !important;
        border: 1px solid var(--accent-cyan) !important;
        border-bottom: 2px solid var(--accent-cyan) !important;
        color: #FFFFFF !important;
        font-weight: 700 !important;
        box-shadow: 0 0 16px rgba(0, 229, 255, 0.35), inset 0 0 12px rgba(0, 229, 255, 0.15) !important;
        transform: translateY(-1px) !important;
    }

    /* Overview Introduction Hero */
    .overview-intro-box {
        background: linear-gradient(135deg, rgba(9, 27, 48, 0.85) 0%, rgba(6, 18, 36, 0.95) 100%);
        border: 1px solid var(--border-subtle);
        border-radius: 12px;
        padding: 1.25rem 1.45rem;
        margin-bottom: 1.15rem;
        position: relative;
        overflow: hidden;
        box-shadow: 0 8px 30px rgba(0, 0, 0, 0.35);
    }
    .overview-intro-box::after {
        content: '';
        position: absolute;
        top: 0;
        right: 0;
        width: 320px;
        height: 100%;
        background: radial-gradient(circle at 100% 50%, rgba(0, 229, 255, 0.08) 0%, transparent 70%);
        pointer-events: none;
    }
    .overview-eyebrow {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.68rem;
        font-weight: 700;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        color: var(--accent-cyan);
        display: inline-block;
        margin-bottom: 0.35rem;
    }
    .overview-heading {
        font-size: 1.65rem;
        font-weight: 800;
        color: #FFFFFF;
        letter-spacing: -0.02em;
        line-height: 1.25;
        margin: 0.15rem 0 0.45rem 0;
        background: linear-gradient(135deg, #FFFFFF 0%, #EAF6FF 60%, var(--accent-cyan) 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .overview-description {
        font-size: 0.86rem;
        color: var(--text-secondary);
        line-height: 1.5;
        max-width: 820px;
        margin-bottom: 0.75rem;
    }

    /* Futuristic KPI Cards */
    .overview-kpi-card {
        background: var(--bg-card);
        border: 1px solid var(--border-subtle);
        border-radius: 10px;
        padding: 0.85rem 1rem 0.75rem 1rem;
        position: relative;
        overflow: hidden;
        transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1);
        box-shadow: 0 4px 18px rgba(0, 0, 0, 0.3);
    }
    .overview-kpi-card:hover {
        transform: translateY(-2px);
        border-color: var(--accent-cyan);
        box-shadow: 0 8px 25px rgba(0, 229, 255, 0.15);
        background: var(--bg-card-elevated);
    }
    .overview-kpi-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 2px;
        background: linear-gradient(90deg, var(--accent-cyan) 0%, transparent 80%);
    }
    .kpi-top-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 0.45rem;
    }
    .kpi-icon-pill {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 30px;
        height: 30px;
        border-radius: 6px;
        background: rgba(0, 229, 255, 0.1);
        border: 1px solid rgba(0, 229, 255, 0.25);
        color: var(--accent-cyan);
    }
    .kpi-badge {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.65rem;
        font-weight: 600;
        color: var(--text-secondary);
        background: rgba(14, 34, 60, 0.7);
        padding: 0.15rem 0.45rem;
        border-radius: 4px;
        border: 1px solid rgba(24, 51, 79, 0.6);
    }
    .kpi-value-text {
        font-family: 'JetBrains Mono', monospace;
        font-size: 1.55rem;
        font-weight: 800;
        color: #FFFFFF;
        line-height: 1.1;
        letter-spacing: -0.02em;
    }
    .kpi-label-text {
        font-family: 'Inter', sans-serif;
        font-size: 0.74rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: var(--accent-cyan);
        margin: 0.2rem 0 0.15rem 0;
    }
    .kpi-caption-text {
        font-size: 0.72rem;
        color: var(--text-muted);
        line-height: 1.35;
    }

    /* Panels & Containers */
    .overview-panel {
        background: var(--bg-card);
        border: 1px solid var(--border-subtle);
        border-radius: 12px;
        padding: 1.05rem 1.25rem;
        margin-bottom: 1.1rem;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.35);
        transition: border-color 0.2s ease;
    }
    .overview-panel:hover {
        border-color: rgba(0, 229, 255, 0.35);
    }
    .overview-panel-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 0.75rem;
        border-bottom: 1px solid rgba(24, 51, 79, 0.6);
        padding-bottom: 0.5rem;
    }
    .overview-panel-title {
        font-size: 0.96rem;
        font-weight: 700;
        color: #FFFFFF;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    .overview-panel-subtitle {
        font-size: 0.74rem;
        color: var(--text-secondary);
    }

    /* Activity Timeline */
    .activity-timeline {
        display: flex;
        flex-direction: column;
        gap: 0.55rem;
    }
    .activity-item {
        background: rgba(6, 18, 36, 0.7);
        border: 1px solid var(--border-subtle);
        border-radius: 8px;
        padding: 0.55rem 0.75rem;
        display: flex;
        align-items: center;
        gap: 0.65rem;
        transition: all 0.2s ease;
    }
    .activity-item:hover {
        background: var(--bg-card-hover);
        border-color: rgba(0, 229, 255, 0.3);
        transform: translateX(2px);
    }
    .activity-dot {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background: var(--accent-green);
        box-shadow: 0 0 8px var(--accent-green);
        flex: 0 0 auto;
    }
    .activity-content {
        flex: 1 1 auto;
        min-width: 0;
    }
    .activity-title {
        font-size: 0.78rem;
        font-weight: 600;
        color: #FFFFFF;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }
    .activity-meta {
        font-size: 0.68rem;
        color: var(--text-muted);
        font-family: 'JetBrains Mono', monospace;
    }
    .activity-badge {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.64rem;
        font-weight: 600;
        padding: 0.15rem 0.45rem;
        border-radius: 4px;
        background: rgba(53, 224, 161, 0.12);
        color: var(--accent-green);
        border: 1px solid rgba(53, 224, 161, 0.3);
        flex: 0 0 auto;
    }

    /* System Telemetry Meters */
    .telemetry-row {
        margin-bottom: 0.65rem;
    }
    .telemetry-label-row {
        display: flex;
        justify-content: space-between;
        font-size: 0.72rem;
        font-family: 'JetBrains Mono', monospace;
        margin-bottom: 0.22rem;
        color: var(--text-secondary);
    }
    .telemetry-label-row strong {
        color: #FFFFFF;
    }
    .telemetry-meter-bar {
        height: 6px;
        border-radius: 3px;
        background: rgba(24, 51, 79, 0.6);
        overflow: hidden;
    }
    .telemetry-meter-fill {
        height: 100%;
        border-radius: 3px;
        background: linear-gradient(90deg, var(--accent-blue) 0%, var(--accent-cyan) 100%);
        box-shadow: 0 0 8px rgba(0, 229, 255, 0.4);
    }

    /* Buttons Styling */
    .stButton > button {
        background: linear-gradient(135deg, #0A4384 0%, #176BFF 100%) !important;
        color: #FFFFFF !important;
        font-family: 'Inter', sans-serif !important;
        font-weight: 600 !important;
        border: 1px solid rgba(0, 229, 255, 0.35) !important;
        border-radius: 8px !important;
        padding: 0.45rem 1rem !important;
        box-shadow: 0 4px 14px rgba(23, 107, 255, 0.25) !important;
        transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1) !important;
    }
    .stButton > button:hover {
        transform: translateY(-1px) !important;
        box-shadow: 0 6px 20px rgba(0, 229, 255, 0.35) !important;
        border-color: var(--accent-cyan) !important;
        background: linear-gradient(135deg, #0D54A3 0%, #2077FF 100%) !important;
    }
    .stButton > button:active {
        transform: scale(0.98) !important;
    }

    /* Native Streamlit metric adjustments */
    [data-testid="stMetric"] {
        padding: 0.65rem 0.85rem !important;
        border-radius: 8px !important;
        background: var(--bg-card) !important;
        border: 1px solid var(--border-subtle) !important;
        box-shadow: none !important;
    }
    [data-testid="stMetricLabel"] {
        color: var(--text-secondary) !important;
        font-size: 0.70rem !important;
        font-family: 'JetBrains Mono', monospace !important;
        text-transform: uppercase !important;
    }
    [data-testid="stMetricValue"] {
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 1.35rem !important;
        color: #FFFFFF !important;
    }

    /* Plotly chart borders */
    [data-testid="stPlotlyChart"] {
        border: 1px solid var(--border-subtle);
        border-radius: 8px;
        overflow: hidden;
        background: var(--bg-card);
    }

    /* Selectbox styling */
    .stSelectbox div[data-baseweb="select"] > div {
        background-color: var(--bg-surface) !important;
        border-color: var(--border-subtle) !important;
        color: #FFFFFF !important;
        border-radius: 8px !important;
    }

    /* Compact System Footer */
    .hyper-system-footer {
        border-top: 1px solid var(--border-subtle);
        padding: 0.85rem 0.2rem 0.2rem 0.2rem;
        margin-top: 1rem;
        display: flex;
        justify-content: space-between;
        align-items: center;
        flex-wrap: wrap;
        gap: 0.5rem;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.68rem;
        color: var(--text-muted);
    }
    .footer-left {
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    .footer-dot {
        width: 6px;
        height: 6px;
        border-radius: 50%;
        background: var(--accent-cyan);
        box-shadow: 0 0 6px var(--accent-cyan);
    }

    /* Responsive Breakpoints */
    @media (max-width: 900px) {
        [data-testid="stMainBlockContainer"],
        .block-container { padding-left: 0.8rem !important; padding-right: 0.8rem !important; }
        .hyper-brand-tagline { display: none; }
        .overview-heading { font-size: 1.35rem; }
    }
    @media (prefers-reduced-motion: reduce) {
        *, *::before, *::after {
            animation-duration: 0.01ms !important;
            animation-iteration-count: 1 !important;
            scroll-behavior: auto !important;
            transition-duration: 0.01ms !important;
        }
    }
    </style>
    """
    )
    st.markdown(css, unsafe_allow_html=True)


def get_scientific_plotly_layout(**kwargs) -> Dict[str, Any]:
    """Return unified dark scientific Plotly layout specifications."""
    layout: Dict[str, Any] = dict(
        template="plotly_dark",
        paper_bgcolor="#030914",
        plot_bgcolor="#091B30",
        font=dict(family="Inter, sans-serif", color="#9CB0C8", size=11),
        margin=dict(l=45, r=35, t=45, b=40),
        xaxis=dict(
            gridcolor="#18334F",
            zerolinecolor="#18334F",
            tickfont=dict(family="JetBrains Mono, monospace", size=10, color="#6F849E"),
        ),
        yaxis=dict(
            gridcolor="#18334F",
            zerolinecolor="#18334F",
            tickfont=dict(family="JetBrains Mono, monospace", size=10, color="#6F849E"),
        ),
        hoverlabel=dict(
            bgcolor="#0D233B",
            bordercolor="#00E5FF",
            font=dict(family="JetBrains Mono, monospace", size=11, color="#FFFFFF"),
        ),
    )
    layout.update(kwargs)
    return layout


def render_application_header(
    section: str,
    datasets: List[str],
    dataset_widget_key: str,
    on_dataset_change: Callable[..., None],
) -> None:
    """Render the cinematic global brand header with real connection & dataset telemetry."""
    from src.dashboard.state import get_backend_status

    backend_info = get_backend_status()
    is_connected = backend_info.get("connected", False)

    st.markdown(
        f"""
        <div class="hyper-global-header">
            <div class="hyper-brand-group">
                <svg class="hyper-logo-svg" viewBox="0 0 40 40" fill="none" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="HYPER Orbital Brandmark">
                    <circle cx="20" cy="20" r="17" stroke="#18334F" stroke-width="1.2" stroke-dasharray="3 3"/>
                    <ellipse cx="20" cy="20" rx="18" ry="7.5" stroke="#176BFF" stroke-width="1.8" transform="rotate(-30 20 20)"/>
                    <ellipse cx="20" cy="20" rx="14" ry="5.5" stroke="#00E5FF" stroke-width="1.2" transform="rotate(25 20 20)" opacity="0.8"/>
                    <circle cx="20" cy="20" r="5" fill="#030914" stroke="#00E5FF" stroke-width="2"/>
                    <circle cx="20" cy="20" r="2.2" fill="#00E5FF"/>
                    <circle cx="34" cy="13" r="1.8" fill="#35E0A1"/>
                </svg>
                <div class="hyper-brand-titles">
                    <div class="hyper-brand-main">
                        <span class="hyper-brand-name">HYPER</span>
                        <span class="hyper-brand-sub">Hyperspectral Intelligence Platform</span>
                    </div>
                    <span class="hyper-brand-tagline">“See Beyond the Visible.”</span>
                </div>
            </div>
            <div class="hyper-header-telemetry">
                <div class="header-status-pill {'online' if is_connected else 'cyan'}">
                    <span class="pulse-dot"></span>
                    <span>{'BACKEND CONNECTED' if is_connected else 'LOCAL ENGINE READY'}</span>
                </div>
                <div class="header-status-pill cyan">
                    <span>{len(datasets)} BENCHMARK SCENES</span>
                </div>
                <a href="?intro=1" target="_self" class="header-status-pill cyan" style="text-decoration: none; cursor: pointer; transition: all 0.2s ease;">
                    <span>🎬 REPLAY INTRO</span>
                </a>
                <div class="header-status-pill" style="color: #F2F7FF; background: rgba(9,27,48,0.85);">
                    <span style="color: #6F849E;">WORKSPACE /</span> <strong>{section}</strong>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_horizontal_sliding_nav(active_section: str, on_select: Callable[[str], None]) -> None:
    """Render the premium horizontal sliding navigation carousel containing all 10 modules."""
    NAV_MODULES = [
        ("Overview", "⌂  Command Center / Overview", ":material/dashboard:"),
        ("3D Earth Explorer", "🌐  3D Earth Explorer", ":material/public:"),
        ("Anomaly Detection", "◉  Detection Pipeline / Anomaly Detection", ":material/radar:"),
        ("Dataset Explorer", "▦  Datasets / Dataset Explorer", ":material/database:"),
        ("Robustness Laboratory", "◈  Robustness Laboratory", ":material/shield:"),
        ("Spectral Analysis", "∿  Spectral Analysis", ":material/graphic_eq:"),
        ("Runtime / Profiling", "◷  Runtime / Profiling", ":material/speed:"),
        ("Model Comparison", "≋  Model Comparison", ":material/compare_arrows:"),
        ("Experiment History", "◴  Experiment History", ":material/history:"),
        ("Export Results", "⇩  Export Results", ":material/download:"),
    ]

    st.markdown('<div class="hyper-nav-wrapper">', unsafe_allow_html=True)
    nav_cols = st.columns([0.45, 11.1, 0.45], vertical_alignment="center")

    with nav_cols[0]:
        st.markdown(
            """
            <button class="hyper-nav-arrow-btn" onclick="
                const el = document.querySelector('.st-key-hyper_nav_slider > div');
                if (el) el.scrollBy({left: -280, behavior: 'smooth'});
            " title="Slide Left" aria-label="Slide Left">‹</button>
            """,
            unsafe_allow_html=True,
        )

    with nav_cols[1]:
        with st.container(key="hyper_nav_slider", horizontal=True):
            for label, section_key, icon in NAV_MODULES:
                # Active if matches exact key or friendly keyword
                is_active = (section_key == active_section) or (label.lower() in active_section.lower())
                st.button(
                    label,
                    key=f"hnav_{label.lower().replace(' ', '_').replace('/', '_')}",
                    icon=icon,
                    type="primary" if is_active else "secondary",
                    on_click=on_select,
                    args=(section_key,),
                )

    with nav_cols[2]:
        st.markdown(
            """
            <button class="hyper-nav-arrow-btn" onclick="
                const el = document.querySelector('.st-key-hyper_nav_slider > div');
                if (el) el.scrollBy({left: 280, behavior: 'smooth'});
            " title="Slide Right" aria-label="Slide Right">›</button>
            """,
            unsafe_allow_html=True,
        )

    st.markdown('</div>', unsafe_allow_html=True)


def render_page_header(
    badge: str,
    title: str,
    subtitle: str,
    telemetry: Optional[List[Tuple[str, str]]] = None,
) -> None:
    """Render a unified, high-tech section banner across modules."""
    tel_html = ""
    if telemetry:
        tel_items = "".join(
            f'<div class="telemetry-item"><strong>{label}:</strong> {val}</div>'
            for label, val in telemetry
        )
        tel_html = f'<div class="telemetry-bar">{tel_items}</div>'

    html = textwrap.dedent(
        f"""<div class="section-header-banner">
<div class="badge-pill"><span class="status-dot"></span> {badge}</div>
<div style="font-size: 1.5rem; font-weight: 700; color: #FFFFFF; letter-spacing: -0.01em;">{title}</div>
<div style="font-size: 0.86rem; color: #9CB0C8; max-width: 900px; line-height: 1.45;">{subtitle}</div>
{tel_html}
</div>"""
    )
    st.markdown(html, unsafe_allow_html=True)


def render_hero_banner() -> None:
    """Render the high-impact Mission Control hero banner on the Overview page."""
    html = textwrap.dedent(
        """<div class="overview-intro-box">
<div class="overview-eyebrow">SCIENTIFIC INTELLIGENCE COMMAND CENTER</div>
<div class="overview-heading">Explore Earth. Discover Patterns. Understand Our Planet.</div>
<div class="overview-description">A unified, production-grade hyperspectral intelligence platform integrating automated unsupervised Mahalanobis detection, physical perturbation stress-testing, ablation sweeps, high-resolution memory/latency profiling, and persistent SQLite audit tracking across verified benchmark scenes.</div>
</div>"""
    )
    st.markdown(html, unsafe_allow_html=True)


def render_hud_metric_cards(metrics_data: List[Dict[str, Any]]) -> None:
    """Render custom-styled HUD metric cards using native responsive Streamlit columns."""
    n = len(metrics_data)
    if n == 0:
        return
    cols = st.columns(n)
    for col, m in zip(cols, metrics_data):
        with col:
            col.metric(
                label=m.get("label", ""),
                value=m.get("value", "N/A"),
                delta=m.get("delta"),
                help=m.get("help", m.get("caption")),
            )
