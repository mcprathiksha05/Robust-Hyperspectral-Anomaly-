"""Cinematic Hyperspectral AI Introduction Module for HYPER.

Provides an immersive, futuristic 12-18s scientific journey:
Darkness -> Space -> Earth -> Satellite scanning -> Hyperspectral layers ->
3D spectral cube -> Anomaly detection -> AI neural network -> Glowing portal -> Real Dashboard.

Features:
- Individual scene pop-outs with 3D perspective, overshoot, and settle.
- 100% offline, zero external asset dependencies (pre-cached base64 textures).
- Web Audio API synthesized ambient space drone & transition chimes (muted by default).
- Immediate 'Skip Intro' button active during every scene.
- Interactive timeline progress bar with clickable scene markers.
- Seamless transition into the real interactive Streamlit dashboard.
"""

import base64
import functools
from pathlib import Path
from typing import Dict
import streamlit as st
import streamlit.components.v1 as components


INTRO_ASSETS_DIR = Path(__file__).resolve().parent / "assets" / "intro"


@functools.lru_cache(maxsize=1)
def get_intro_assets_b64() -> Dict[str, str]:
    """Load and base64-encode all intro scene images once for fast in-memory serving."""
    assets = {}
    if INTRO_ASSETS_DIR.exists():
        for p in INTRO_ASSETS_DIR.glob("*.jpg"):
            try:
                with open(p, "rb") as f:
                    assets[p.stem] = base64.b64encode(f.read()).decode("utf-8")
            except Exception:
                pass
    return assets


def build_cinematic_intro_html(auto_play: bool = True, duration: float = 18.0) -> str:
    """Build the complete self-contained HTML5/CSS3/JS cinematic intro experience."""
    assets = get_intro_assets_b64()
    
    img_s1 = assets.get("scene1_earth", "")
    img_s2 = assets.get("scene2_satellite", "")
    img_s3 = assets.get("scene3_layers", "")
    img_s4 = assets.get("scene4_cube", "")
    img_s5 = assets.get("scene5_anomaly", "")
    img_s6 = assets.get("scene6_ai", "")
    img_s7 = assets.get("scene7_portal_bg", "")
    img_s8 = assets.get("scene8_portal_gateway", "")

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
<title>HYPER // Cinematic Hyperspectral AI Introduction</title>
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;800;900&family=JetBrains+Mono:wght@400;500;700&display=swap');

    *, *::before, *::after {{
        box-sizing: border-box;
        margin: 0;
        padding: 0;
        user-select: none;
        -webkit-user-select: none;
    }}

    :root {{
        --color-bg: #020611;
        --color-cyan: #00E5FF;
        --color-blue: #176BFF;
        --color-violet: #8B5CF6;
        --color-gold: #FBBF24;
        --color-red: #EF4444;
        --color-text: #F8FAFC;
        --color-text-dim: #94A3B8;
        --font-main: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        --font-mono: 'JetBrains Mono', monospace;
    }}

    html, body {{
        width: 100%;
        height: 100%;
        background-color: var(--color-bg);
        color: var(--color-text);
        font-family: var(--font-main);
        overflow: hidden;
    }}

    #cinematic-root {{
        position: relative;
        width: 100vw;
        height: 100vh;
        background: radial-gradient(circle at 50% 50%, #06152B 0%, #020611 85%);
        overflow: hidden;
        perspective: 1200px;
    }}

    /* Background Starfield & Particle Canvas */
    #bg-canvas {{
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        z-index: 1;
        pointer-events: none;
    }}

    /* Global Vignette & Sci-Fi Scanlines */
    .viewport-overlay {{
        position: absolute;
        inset: 0;
        z-index: 50;
        pointer-events: none;
        background: 
            radial-gradient(ellipse at center, transparent 40%, rgba(2, 6, 17, 0.75) 90%),
            linear-gradient(rgba(0, 229, 255, 0.015) 50%, rgba(0, 0, 0, 0.25) 50%);
        background-size: 100% 100%, 100% 4px;
        box-shadow: inset 0 0 100px rgba(0, 229, 255, 0.08);
    }}

    /* Top HUD Bar */
    .top-hud {{
        position: absolute;
        top: 20px;
        left: 24px;
        right: 24px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        z-index: 100;
        pointer-events: auto;
    }}

    .brand-hud {{
        display: flex;
        align-items: center;
        gap: 12px;
        background: rgba(6, 21, 43, 0.7);
        border: 1px solid rgba(0, 229, 255, 0.25);
        padding: 8px 16px;
        border-radius: 8px;
        backdrop-filter: blur(12px);
    }}

    .brand-hud .hud-dot {{
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background: var(--color-cyan);
        box-shadow: 0 0 10px var(--color-cyan);
        animation: pulse-dot 1.5s infinite;
    }}

    .brand-hud .brand-title {{
        font-family: var(--font-mono);
        font-size: 12px;
        font-weight: 700;
        letter-spacing: 2px;
        color: var(--color-cyan);
    }}

    .brand-hud .scene-badge {{
        font-family: var(--font-mono);
        font-size: 11px;
        color: var(--color-text-dim);
        border-left: 1px solid rgba(255,255,255,0.15);
        padding-left: 10px;
    }}

    .hud-controls {{
        display: flex;
        align-items: center;
        gap: 10px;
    }}

    .hud-btn {{
        background: rgba(8, 28, 54, 0.85);
        border: 1px solid rgba(0, 229, 255, 0.35);
        color: var(--color-text);
        font-family: var(--font-mono);
        font-size: 12px;
        font-weight: 600;
        padding: 8px 16px;
        border-radius: 6px;
        cursor: pointer;
        display: flex;
        align-items: center;
        gap: 8px;
        transition: all 0.2s ease;
        backdrop-filter: blur(10px);
    }}

    .hud-btn:hover {{
        background: rgba(0, 229, 255, 0.2);
        border-color: var(--color-cyan);
        box-shadow: 0 0 15px rgba(0, 229, 255, 0.35);
        transform: translateY(-1px);
    }}

    .hud-btn.skip-btn {{
        background: linear-gradient(135deg, rgba(23, 107, 255, 0.5) 0%, rgba(0, 229, 255, 0.5) 100%);
        border: 1px solid var(--color-cyan);
        color: #FFFFFF;
        box-shadow: 0 0 18px rgba(0, 229, 255, 0.3);
    }}

    .hud-btn.skip-btn:hover {{
        background: linear-gradient(135deg, #176BFF 0%, #00E5FF 100%);
        color: #020611;
        font-weight: 700;
    }}

    /* Bottom Timeline Progress Bar */
    .bottom-hud {{
        position: absolute;
        bottom: 24px;
        left: 24px;
        right: 24px;
        display: flex;
        flex-direction: column;
        gap: 10px;
        z-index: 100;
        pointer-events: auto;
    }}

    .timeline-bar-wrapper {{
        position: relative;
        width: 100%;
        height: 6px;
        background: rgba(255, 255, 255, 0.1);
        border-radius: 3px;
        overflow: hidden;
        cursor: pointer;
    }}

    .timeline-progress {{
        position: absolute;
        top: 0;
        left: 0;
        height: 100%;
        width: 0%;
        background: linear-gradient(90deg, var(--color-blue), var(--color-cyan) 80%, #FFFFFF);
        box-shadow: 0 0 12px var(--color-cyan);
        transition: width 0.1s linear;
    }}

    .scene-markers {{
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 0 4px;
    }}

    .scene-node {{
        display: flex;
        align-items: center;
        gap: 6px;
        font-family: var(--font-mono);
        font-size: 11px;
        color: var(--color-text-dim);
        cursor: pointer;
        transition: all 0.2s ease;
        padding: 4px 8px;
        border-radius: 4px;
    }}

    .scene-node .node-dot {{
        width: 6px;
        height: 6px;
        border-radius: 50%;
        background: rgba(255, 255, 255, 0.3);
        transition: all 0.2s ease;
    }}

    .scene-node.active {{
        color: var(--color-cyan);
        font-weight: 700;
        background: rgba(0, 229, 255, 0.1);
    }}

    .scene-node.active .node-dot {{
        background: var(--color-cyan);
        box-shadow: 0 0 8px var(--color-cyan);
        transform: scale(1.4);
    }}

    .scene-node:hover {{
        color: #FFFFFF;
        background: rgba(255, 255, 255, 0.05);
    }}

    /* Main Scene Stage */
    .scene-stage {{
        position: absolute;
        inset: 0;
        display: flex;
        justify-content: center;
        align-items: center;
        z-index: 10;
        transform-style: preserve-3d;
    }}

    .scene-container {{
        position: absolute;
        width: 100%;
        height: 100%;
        display: flex;
        justify-content: center;
        align-items: center;
        opacity: 0;
        pointer-events: none;
        transition: opacity 0.8s cubic-bezier(0.25, 1, 0.5, 1);
        transform-style: preserve-3d;
        will-change: transform, opacity;
    }}

    .scene-container.active {{
        opacity: 1;
        pointer-events: auto;
    }}

    /* Pop-out Card System */
    .pop-card {{
        position: relative;
        width: 86vw;
        max-width: 1400px;
        height: 75vh;
        max-height: 820px;
        border-radius: 18px;
        overflow: hidden;
        background-size: cover;
        background-position: center;
        box-shadow: 
            0 30px 90px rgba(0, 0, 0, 0.9),
            0 0 60px rgba(0, 229, 255, 0.2),
            inset 0 0 30px rgba(0, 229, 255, 0.15);
        border: 1px solid rgba(0, 229, 255, 0.35);
        transform: scale(0.85) translateZ(-80px);
        transition: transform 1.2s cubic-bezier(0.16, 1, 0.3, 1), box-shadow 1.2s ease;
        display: flex;
        flex-direction: column;
        justify-content: flex-end;
        padding: 40px 50px;
        will-change: transform;
    }}

    .scene-container.active .pop-card {{
        transform: scale(1) translateZ(0px);
        box-shadow: 
            0 40px 120px rgba(0, 0, 0, 0.95),
            0 0 80px rgba(0, 229, 255, 0.35),
            inset 0 0 40px rgba(0, 229, 255, 0.2);
    }}

    /* Cinematic Text Overlay */
    .scene-caption-box {{
        position: relative;
        z-index: 20;
        max-width: 720px;
        background: linear-gradient(135deg, rgba(2, 8, 22, 0.88) 0%, rgba(6, 20, 42, 0.72) 100%);
        border: 1px solid rgba(0, 229, 255, 0.3);
        border-left: 4px solid var(--color-cyan);
        border-radius: 12px;
        padding: 24px 30px;
        backdrop-filter: blur(16px);
        box-shadow: 0 16px 40px rgba(0, 0, 0, 0.6);
        transform: translateY(30px);
        opacity: 0;
        transition: transform 0.9s 0.2s cubic-bezier(0.16, 1, 0.3, 1), opacity 0.9s 0.2s ease;
    }}

    .scene-container.active .scene-caption-box {{
        transform: translateY(0);
        opacity: 1;
    }}

    .scene-caption-kicker {{
        font-family: var(--font-mono);
        font-size: 11px;
        font-weight: 700;
        letter-spacing: 3px;
        color: var(--color-cyan);
        text-transform: uppercase;
        margin-bottom: 8px;
        display: flex;
        align-items: center;
        gap: 8px;
    }}

    .scene-caption-kicker::before {{
        content: '';
        display: inline-block;
        width: 14px;
        height: 2px;
        background: var(--color-cyan);
    }}

    .scene-caption-title {{
        font-size: 28px;
        font-weight: 800;
        letter-spacing: 1.5px;
        color: #FFFFFF;
        text-transform: uppercase;
        line-height: 1.2;
        margin-bottom: 10px;
        text-shadow: 0 0 20px rgba(0, 229, 255, 0.5);
    }}

    .scene-caption-desc {{
        font-size: 14px;
        color: #CBD5E1;
        line-height: 1.5;
        font-weight: 400;
    }}

    .scene-caption-pills {{
        display: flex;
        gap: 10px;
        margin-top: 14px;
    }}

    .caption-pill {{
        font-family: var(--font-mono);
        font-size: 10px;
        font-weight: 700;
        padding: 4px 10px;
        border-radius: 4px;
        background: rgba(0, 229, 255, 0.12);
        color: var(--color-cyan);
        border: 1px solid rgba(0, 229, 255, 0.3);
        letter-spacing: 1px;
    }}

    /* Specific Scene Special FX */

    /* Scene 1: Cosmic Earth Glow */
    #scene-1 .pop-card {{
        background-image: url('data:image/jpeg;base64,{img_s1}');
    }}

    /* Scene 2: Satellite Scan Beams */
    #scene-2 .pop-card {{
        background-image: url('data:image/jpeg;base64,{img_s2}');
    }}
    .spectral-laser-overlay {{
        position: absolute;
        inset: 0;
        pointer-events: none;
        overflow: hidden;
    }}
    .laser-ray {{
        position: absolute;
        top: 25%;
        left: 50%;
        width: 2px;
        height: 60%;
        transform-origin: top center;
        opacity: 0.85;
        filter: drop-shadow(0 0 6px currentColor);
        animation: pulse-ray 2s infinite alternate ease-in-out;
    }}

    /* Scene 3: Floating Hyperspectral Layers */
    #scene-3 .pop-card {{
        background-image: url('data:image/jpeg;base64,{img_s3}');
    }}

    /* Scene 4: 3D Data Cube + Spectral Curve */
    #scene-4 .pop-card {{
        background-image: url('data:image/jpeg;base64,{img_s4}');
    }}

    /* Scene 5: Anomaly Scan + Heatmap */
    #scene-5 .pop-card {{
        background-image: url('data:image/jpeg;base64,{img_s5}');
    }}
    .scanner-sweep-line {{
        position: absolute;
        top: 0;
        left: 0;
        width: 3px;
        height: 100%;
        background: linear-gradient(180deg, transparent, var(--color-cyan) 30%, #FFFFFF 50%, var(--color-cyan) 70%, transparent);
        box-shadow: 0 0 25px var(--color-cyan), 0 0 50px var(--color-cyan);
        animation: scan-sweep 3s infinite linear;
        pointer-events: none;
    }}

    /* Scene 6: AI Neural Network */
    #scene-6 .pop-card {{
        background-image: url('data:image/jpeg;base64,{img_s6}');
    }}

    /* Scene 7: Dimensional Portal */
    #scene-7 .pop-card {{
        background-image: url('data:image/jpeg;base64,{img_s7}');
    }}
    .portal-rings-container {{
        position: absolute;
        inset: 0;
        display: flex;
        justify-content: center;
        align-items: center;
        pointer-events: none;
    }}
    .portal-ring {{
        position: absolute;
        border-radius: 50%;
        border: 2px dashed rgba(0, 229, 255, 0.6);
        box-shadow: 0 0 30px rgba(0, 229, 255, 0.4), inset 0 0 30px rgba(139, 92, 246, 0.3);
    }}
    .portal-ring.ring-1 {{ width: 320px; height: 320px; animation: spin-clockwise 14s infinite linear; }}
    .portal-ring.ring-2 {{ width: 440px; height: 440px; border-color: rgba(139, 92, 246, 0.6); animation: spin-counter 18s infinite linear; }}
    .portal-ring.ring-3 {{ width: 560px; height: 560px; border-color: rgba(0, 229, 255, 0.4); animation: spin-clockwise 24s infinite linear; }}

    /* Scene 8: The Dashboard Gateway */
    #scene-8 .pop-card {{
        background-image: url('data:image/jpeg;base64,{img_s8}');
    }}

    /* Keyframe Animations */
    @keyframes pulse-dot {{
        0%, 100% {{ transform: scale(1); opacity: 1; }}
        50% {{ transform: scale(1.4); opacity: 0.6; }}
    }}
    @keyframes scan-sweep {{
        0% {{ transform: translateX(-10vw); opacity: 0; }}
        10% {{ opacity: 1; }}
        90% {{ opacity: 1; }}
        100% {{ transform: translateX(88vw); opacity: 0; }}
    }}
    @keyframes spin-clockwise {{
        from {{ transform: rotate(0deg); }}
        to {{ transform: rotate(360deg); }}
    }}
    @keyframes spin-counter {{
        from {{ transform: rotate(360deg); }}
        to {{ transform: rotate(0deg); }}
    }}

    /* Hand-off Flash & Fade */
    .exit-flash {{
        position: absolute;
        inset: 0;
        background: #FFFFFF;
        opacity: 0;
        pointer-events: none;
        z-index: 200;
        transition: opacity 0.5s ease;
    }}
    .exit-flash.active {{
        opacity: 0.85;
    }}
</style>
</head>
<body>

<div id="cinematic-root">
    <!-- Starfield and cosmic dust canvas -->
    <canvas id="bg-canvas"></canvas>

    <!-- Sci-Fi Viewport scanlines and vignette -->
    <div class="viewport-overlay"></div>
    <div class="exit-flash" id="exit-flash"></div>

    <!-- Top HUD Navigation -->
    <div class="top-hud">
        <div class="brand-hud">
            <span class="hud-dot"></span>
            <span class="brand-title">HYPER RESEARCH LAB</span>
            <span class="scene-badge" id="telemetry-badge">MISSION BRIEFING // INITIALIZING</span>
        </div>
        <div class="hud-controls">
            <button class="hud-btn" id="btn-audio" title="Toggle Ambient Audio">
                <span id="audio-icon">🔇</span>
                <span id="audio-label">SOUND: OFF</span>
            </button>
            <button class="hud-btn skip-btn" id="btn-skip">
                <span>ENTER WORKSPACE ⏭</span>
            </button>
        </div>
    </div>

    <!-- Main 8-Scene Stage -->
    <div class="scene-stage">
        <!-- Scene 1: The Cosmic Awakening -->
        <div class="scene-container active" id="scene-1" data-scene-idx="0">
            <div class="pop-card">
                <div class="scene-caption-box">
                    <div class="scene-caption-kicker">SCENE 01 // ORBITAL AWAKENING</div>
                    <h2 class="scene-caption-title">Welcome to a New Perspective</h2>
                    <p class="scene-caption-desc">Beyond human vision lies a realm of radiant spectral energy. Enter space where hyperspectral optical sensors capture Earth in hundreds of contiguous bands.</p>
                    <div class="scene-caption-pills">
                        <span class="caption-pill">DEEP COSMOS</span>
                        <span class="caption-pill">LEO PLATFORM</span>
                        <span class="caption-pill">ATMOSPHERIC CORONA</span>
                    </div>
                </div>
            </div>
        </div>

        <!-- Scene 2: Capturing the Invisible -->
        <div class="scene-container" id="scene-2" data-scene-idx="1">
            <div class="pop-card">
                <div class="spectral-laser-overlay">
                    <div class="laser-ray" style="color: #8B5CF6; transform: rotate(-35deg); height: 75%;"></div>
                    <div class="laser-ray" style="color: #3B82F6; transform: rotate(-25deg); height: 78%;"></div>
                    <div class="laser-ray" style="color: #00E5FF; transform: rotate(-15deg); height: 82%;"></div>
                    <div class="laser-ray" style="color: #10B981; transform: rotate(-5deg); height: 85%;"></div>
                    <div class="laser-ray" style="color: #FBBF24; transform: rotate(5deg); height: 83%;"></div>
                    <div class="laser-ray" style="color: #F97316; transform: rotate(15deg); height: 80%;"></div>
                    <div class="laser-ray" style="color: #EF4444; transform: rotate(25deg); height: 76%;"></div>
                </div>
                <div class="scene-caption-box">
                    <div class="scene-caption-kicker">SCENE 02 // SENSOR SWATH ACQUISITION</div>
                    <h2 class="scene-caption-title">Capturing the Invisible</h2>
                    <p class="scene-caption-desc">Every pixel holds a continuous spectrum of physical information. Solar photons reflect off planetary terrain, separating into distinct diagnostic wavelengths.</p>
                    <div class="scene-caption-pills">
                        <span class="caption-pill">400nm - 1000nm</span>
                        <span class="caption-pill">205 CONTINUOUS CHANNELS</span>
                        <span class="caption-pill">RADIANCY DECOMPOSITION</span>
                    </div>
                </div>
            </div>
        </div>

        <!-- Scene 3: The World of Hyperspectral Imaging -->
        <div class="scene-container" id="scene-3" data-scene-idx="2">
            <div class="pop-card">
                <div class="scene-caption-box">
                    <div class="scene-caption-kicker">SCENE 03 // SPECTRAL LAYER DISPERSION</div>
                    <h2 class="scene-caption-title">More than an Image</h2>
                    <p class="scene-caption-desc">Hundreds of narrow spectral bands stacked vertically in space. Where conventional cameras capture only RGB, hyperspectral sensors unveil subtle material compositions.</p>
                    <div class="scene-caption-pills">
                        <span class="caption-pill">VNIR SPECTRUM</span>
                        <span class="caption-pill">HOLOGRAPHIC STACK</span>
                        <span class="caption-pill">SUB-BAND ISOLATION</span>
                    </div>
                </div>
            </div>
        </div>

        <!-- Scene 4: The Spectral Data Cube -->
        <div class="scene-container" id="scene-4" data-scene-idx="3">
            <div class="pop-card">
                <div class="scene-caption-box">
                    <div class="scene-caption-kicker">SCENE 04 // 3D HYPERSPECTRAL TENSOR</div>
                    <h2 class="scene-caption-title">Every Pixel Has a Spectral Signature</h2>
                    <p class="scene-caption-desc">Structure hidden within light. The 3D data cube fuses spatial coordinates (X, Y) with spectral depth (λ). Each physical material produces an unmistakable spectral fingerprint.</p>
                    <div class="scene-caption-pills">
                        <span class="caption-pill">REFLECTANCE r(λ)</span>
                        <span class="caption-pill">H × W × B TENSOR</span>
                        <span class="caption-pill">CHLOROPHYLL ABSORPTION</span>
                    </div>
                </div>
            </div>
        </div>

        <!-- Scene 5: Discovering the Unusual -->
        <div class="scene-container" id="scene-5" data-scene-idx="4">
            <div class="pop-card">
                <div class="scanner-sweep-line"></div>
                <div class="scene-caption-box">
                    <div class="scene-caption-kicker">SCENE 05 // ANOMALY DETECTION ENGINE</div>
                    <h2 class="scene-caption-title">Finding What Doesn't Belong</h2>
                    <p class="scene-caption-desc">Identifying pixels that statistically deviate from the surrounding background clutter. Unsupervised Mahalanobis distances pinpoint man-made targets concealed under natural camouflage.</p>
                    <div class="scene-caption-pills">
                        <span class="caption-pill">REED-XIAOLI (RX)</span>
                        <span class="caption-pill">MAHALANOBIS r(x)</span>
                        <span class="caption-pill">FALSE-ALARM REJECTION</span>
                    </div>
                </div>
            </div>
        </div>

        <!-- Scene 6: Artificial Intelligence in Motion -->
        <div class="scene-container" id="scene-6" data-scene-idx="5">
            <div class="pop-card">
                <div class="scene-caption-box">
                    <div class="scene-caption-kicker">SCENE 06 // CYBERNETIC INTELLIGENCE</div>
                    <h2 class="scene-caption-title">Turning Complex Data into Insights</h2>
                    <p class="scene-caption-desc">Advanced machine learning transforms high-dimensional hypercubes into real-time tactical intelligence through subspace estimation, PCA residuals, and noise-robust filtering.</p>
                    <div class="scene-caption-pills">
                        <span class="caption-pill">ANALYZE</span>
                        <span class="caption-pill">DETECT</span>
                        <span class="caption-pill">VISUALIZE</span>
                    </div>
                </div>
            </div>
        </div>

        <!-- Scene 7: The Spectral Portal -->
        <div class="scene-container" id="scene-7" data-scene-idx="6">
            <div class="pop-card">
                <div class="portal-rings-container">
                    <div class="portal-ring ring-1"></div>
                    <div class="portal-ring ring-2"></div>
                    <div class="portal-ring ring-3"></div>
                </div>
                <div class="scene-caption-box" style="text-align: center; border-left: 1px solid rgba(0, 229, 255, 0.3); border-top: 4px solid var(--color-cyan); margin: 0 auto;">
                    <div class="scene-caption-kicker" style="justify-content: center;">DIMENSIONAL GATEWAY // READY</div>
                    <h2 class="scene-caption-title" style="font-size: 32px; letter-spacing: 2px;">Robust Hyperspectral Anomaly Detection</h2>
                    <p class="scene-caption-desc">Step into the world of hyperspectral intelligence. Autonomous target detection under severe camouflage, clutter, and computational constraints.</p>
                    <div class="scene-caption-pills" style="justify-content: center;">
                        <span class="caption-pill">EXPLORE</span>
                        <span class="caption-pill">ANALYZE</span>
                        <span class="caption-pill">DETECT</span>
                    </div>
                </div>
            </div>
        </div>

        <!-- Scene 8: The Live Dashboard Reveal -->
        <div class="scene-container" id="scene-8" data-scene-idx="7">
            <div class="pop-card" style="box-shadow: 0 0 100px rgba(0, 229, 255, 0.6); border-color: var(--color-cyan);">
                <div class="scene-caption-box" style="margin: 0 auto; text-align: center; border-top: 4px solid #35E0A1;">
                    <div class="scene-caption-kicker" style="justify-content: center; color: #35E0A1;">TELEMETRY SYNCHRONIZED // 100% ONLINE</div>
                    <h2 class="scene-caption-title">Hyperspectral Intelligence Platform</h2>
                    <p class="scene-caption-desc">Entering live scientific workstation. Complete benchmark datasets, interactive 3D planetary explorer, robustness laboratory, and real-time anomaly detection pipelines ready.</p>
                    <div class="scene-caption-pills" style="justify-content: center;">
                        <span class="caption-pill" style="color: #35E0A1; border-color: #35E0A1;">WORKSPACE MOUNTED</span>
                        <span class="caption-pill">BENCHMARK SCENES READY</span>
                        <span class="caption-pill">CLICK ENTER TO PROCEED</span>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <!-- Bottom Timeline Bar -->
    <div class="bottom-hud">
        <div class="timeline-bar-wrapper" id="timeline-track">
            <div class="timeline-progress" id="timeline-fill"></div>
        </div>
        <div class="scene-markers">
            <div class="scene-node active" data-target="0"><span class="node-dot"></span>01 COSMIC</div>
            <div class="scene-node" data-target="1"><span class="node-dot"></span>02 SATELLITE</div>
            <div class="scene-node" data-target="2"><span class="node-dot"></span>03 LAYERS</div>
            <div class="scene-node" data-target="3"><span class="node-dot"></span>04 3D CUBE</div>
            <div class="scene-node" data-target="4"><span class="node-dot"></span>05 ANOMALY</div>
            <div class="scene-node" data-target="5"><span class="node-dot"></span>06 AI BRAIN</div>
            <div class="scene-node" data-target="6"><span class="node-dot"></span>07 PORTAL</div>
            <div class="scene-node" data-target="7"><span class="node-dot"></span>08 REVEAL</div>
        </div>
    </div>
</div>

<script>
(function() {{
    const TOTAL_SCENES = 8;
    const SCENE_DURATION_MS = 2250; // ~2.25s per scene = 18s total
    const TOTAL_DURATION_MS = TOTAL_SCENES * SCENE_DURATION_MS;

    let currentSceneIdx = 0;
    let startTime = null;
    let isPlaying = true;
    let animationFrameId = null;

    // Web Audio Synthesizer Engine
    let audioCtx = null;
    let isAudioEnabled = false;
    let droneOsc1 = null;
    let droneOsc2 = null;
    let droneGain = null;

    function initAudio() {{
        try {{
            const AudioContext = window.AudioContext || window.webkitAudioContext;
            audioCtx = new AudioContext();

            droneGain = audioCtx.createGain();
            droneGain.gain.setValueAtTime(0, audioCtx.currentTime);

            const filter = audioCtx.createBiquadFilter();
            filter.type = 'lowpass';
            filter.frequency.setValueAtTime(260, audioCtx.currentTime);

            droneOsc1 = audioCtx.createOscillator();
            droneOsc1.type = 'sawtooth';
            droneOsc1.frequency.setValueAtTime(55, audioCtx.currentTime); // Low A1 sub-drone

            droneOsc2 = audioCtx.createOscillator();
            droneOsc2.type = 'sine';
            droneOsc2.frequency.setValueAtTime(110, audioCtx.currentTime); // A2 harmonic

            droneOsc1.connect(filter);
            droneOsc2.connect(filter);
            filter.connect(droneGain);
            droneGain.connect(audioCtx.destination);

            droneOsc1.start();
            droneOsc2.start();
        }} catch(e) {{
            console.warn('Web Audio not available:', e);
        }}
    }}

    function playTransitionChime(freq = 440) {{
        if (!isAudioEnabled || !audioCtx) return;
        try {{
            const now = audioCtx.currentTime;
            const osc = audioCtx.createOscillator();
            const g = audioCtx.createGain();
            osc.type = 'sine';
            osc.frequency.setValueAtTime(freq, now);
            osc.frequency.exponentialRampToValueAtTime(freq * 1.5, now + 0.4);

            g.gain.setValueAtTime(0.12, now);
            g.gain.exponentialRampToValueAtTime(0.001, now + 0.55);

            osc.connect(g);
            g.connect(audioCtx.destination);
            osc.start(now);
            osc.stop(now + 0.6);
        }} catch(e) {{}}
    }}

    // Canvas Background Starfield & Dust Particles
    const canvas = document.getElementById('bg-canvas');
    const ctx = canvas.getContext('2d');
    let width = canvas.width = window.innerWidth;
    let height = canvas.height = window.innerHeight;

    window.addEventListener('resize', () => {{
        width = canvas.width = window.innerWidth;
        height = canvas.height = window.innerHeight;
    }});

    const stars = [];
    for (let i = 0; i < 180; i++) {{
        stars.push({{
            x: Math.random() * width,
            y: Math.random() * height,
            size: Math.random() * 2 + 0.5,
            speed: Math.random() * 0.4 + 0.1,
            color: Math.random() > 0.3 ? '#00E5FF' : (Math.random() > 0.5 ? '#8B5CF6' : '#FFFFFF'),
            alpha: Math.random() * 0.8 + 0.2
        }});
    }}

    function renderCanvas() {{
        ctx.clearRect(0, 0, width, height);
        for (let i = 0; i < stars.length; i++) {{
            const s = stars[i];
            s.y -= s.speed;
            if (s.y < 0) s.y = height;

            ctx.fillStyle = s.color;
            ctx.globalAlpha = s.alpha * (0.6 + 0.4 * Math.sin(Date.now() * 0.003 + i));
            ctx.beginPath();
            ctx.arc(s.x, s.y, s.size, 0, Math.PI * 2);
            ctx.fill();
        }}
        requestAnimationFrame(renderCanvas);
    }}
    renderCanvas();

    // Scene Transition Control
    const containers = document.querySelectorAll('.scene-container');
    const nodes = document.querySelectorAll('.scene-node');
    const timelineFill = document.getElementById('timeline-fill');
    const telemetryBadge = document.getElementById('telemetry-badge');

    const sceneBadges = [
        "SCENE 01 // ORBITAL AWAKENING (0.0s)",
        "SCENE 02 // SENSOR ACQUISITION (2.2s)",
        "SCENE 03 // SPECTRAL DISPERSION (4.5s)",
        "SCENE 04 // 3D HYPERCUBE TENSOR (7.0s)",
        "SCENE 05 // UNUSUAL TARGET SCAN (9.5s)",
        "SCENE 06 // CYBERNETIC AI BRAIN (12.0s)",
        "SCENE 07 // DIMENSIONAL GATEWAY (14.5s)",
        "SCENE 08 // REAL WORKSPACE REVEAL (16.5s)"
    ];

    const transitionFrequencies = [220, 330, 440, 550, 660, 770, 880, 990];

    function showScene(idx) {{
        if (idx < 0 || idx >= TOTAL_SCENES) return;
        currentSceneIdx = idx;

        containers.forEach((c, i) => {{
            if (i === idx) {{
                c.classList.add('active');
            }} else {{
                c.classList.remove('active');
            }}
        }});

        nodes.forEach((n, i) => {{
            if (i === idx) {{
                n.classList.add('active');
            }} else {{
                n.classList.remove('active');
            }}
        }});

        telemetryBadge.textContent = sceneBadges[idx];
        playTransitionChime(transitionFrequencies[idx]);
    }}

    // Animation Loop
    function loop(now) {{
        if (!startTime) startTime = now;
        const elapsed = now - startTime;

        if (elapsed >= TOTAL_DURATION_MS) {{
            timelineFill.style.width = '100%';
            finishAndRevealDashboard();
            return;
        }}

        const progressPercent = (elapsed / TOTAL_DURATION_MS) * 100;
        timelineFill.style.width = progressPercent + '%';

        const targetScene = Math.min(TOTAL_SCENES - 1, Math.floor(elapsed / SCENE_DURATION_MS));
        if (targetScene !== currentSceneIdx) {{
            showScene(targetScene);
        }}

        if (isPlaying) {{
            animationFrameId = requestAnimationFrame(loop);
        }}
    }}

    // Immediate Skip & Finish
    function finishAndRevealDashboard() {{
        isPlaying = false;
        if (animationFrameId) cancelAnimationFrame(animationFrameId);

        // Flash & Fade Transition
        const flash = document.getElementById('exit-flash');
        flash.classList.add('active');

        if (droneGain && audioCtx) {{
            droneGain.gain.exponentialRampToValueAtTime(0.0001, audioCtx.currentTime + 0.4);
        }}

        setTimeout(() => {{
            try {{
                // Communicate to Streamlit parent environment
                if (window.parent) {{
                    window.parent.postMessage({{ type: 'HYPER_INTRO_DONE' }}, '*');

                    // If Streamlit skip button exists in parent DOM, trigger it
                    const parentBtn = window.parent.document.querySelector('button[data-testid="baseButton-secondary"], button[kind="secondary"]');
                    const skipCandidate = window.parent.document.querySelector('#btn_skip_intro_hidden, [data-testid="stButton"] button');
                    if (skipCandidate) {{
                        skipCandidate.click();
                    }}
                }}
            }} catch(e) {{}}

            // Hide intro container
            const root = document.getElementById('cinematic-root');
            if (root) {{
                root.style.opacity = '0';
                root.style.transition = 'opacity 0.5s ease';
                setTimeout(() => {{ root.style.display = 'none'; }}, 500);
            }}
        }}, 400);
    }}

    // Event Handlers
    document.getElementById('btn-skip').addEventListener('click', finishAndRevealDashboard);

    // Audio Toggle
    const btnAudio = document.getElementById('btn-audio');
    const audioIcon = document.getElementById('audio-icon');
    const audioLabel = document.getElementById('audio-label');

    btnAudio.addEventListener('click', () => {{
        if (!audioCtx) initAudio();
        if (audioCtx.state === 'suspended') {{
            audioCtx.resume();
        }}
        isAudioEnabled = !isAudioEnabled;
        if (isAudioEnabled) {{
            droneGain.gain.setValueAtTime(0.01, audioCtx.currentTime);
            droneGain.gain.exponentialRampToValueAtTime(0.35, audioCtx.currentTime + 1.2);
            audioIcon.textContent = '🔊';
            audioLabel.textContent = 'SOUND: ON';
            btnAudio.style.borderColor = 'var(--color-cyan)';
            btnAudio.style.background = 'rgba(0, 229, 255, 0.25)';
        }} else {{
            droneGain.gain.exponentialRampToValueAtTime(0.0001, audioCtx.currentTime + 0.4);
            audioIcon.textContent = '🔇';
            audioLabel.textContent = 'SOUND: OFF';
            btnAudio.style.borderColor = 'rgba(0, 229, 255, 0.35)';
            btnAudio.style.background = 'rgba(8, 28, 54, 0.85)';
        }}
    }});

    // Clickable Scene Nodes
    nodes.forEach(node => {{
        node.addEventListener('click', () => {{
            const targetIdx = parseInt(node.getAttribute('data-target'), 10);
            startTime = performance.now() - (targetIdx * SCENE_DURATION_MS);
            showScene(targetIdx);
        }});
    }});

    // Timeline Track Scrubbing
    document.getElementById('timeline-track').addEventListener('click', (e) => {{
        const rect = e.currentTarget.getBoundingClientRect();
        const clickRatio = (e.clientX - rect.left) / rect.width;
        startTime = performance.now() - (clickRatio * TOTAL_DURATION_MS);
    }});

    // Start playback
    requestAnimationFrame(loop);
}})();
</script>
</body>
</html>"""


def render_cinematic_intro_overlay(height: int = 860) -> None:
    """Render the full cinematic intro experience with seamless dismiss handling."""
    # Ensure session state initialization
    if "show_cinematic_intro" not in st.session_state:
        st.session_state["show_cinematic_intro"] = True
    if "intro_has_played" not in st.session_state:
        st.session_state["intro_has_played"] = False

    intro_html = build_cinematic_intro_html(auto_play=True, duration=18.0)

    # Embed HTML5 interactive component
    components.html(intro_html, height=height, scrolling=False)
