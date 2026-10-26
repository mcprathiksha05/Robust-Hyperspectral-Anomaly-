"""Feature 2 — Interactive 3D Earth Explorer for HYPER Platform.

Provides:
    - Rotatable, zoomable genuine 3D Earth visualization using WebGL/Three.js.
    - Parametric simulated orbiting satellite with solar panels and visible orbital trajectory.
    - Controllable animations (Play/Pause, Slow/Normal/Fast orbit speed).
    - Controllable Earth rotation and scanning beam animation.
    - 5 Selectable illustrative geographic locations with 3D markers and raycasting.
    - Selected-location information panel with coordinates, regional description, and dataset linkage status.
    - Scientific simulation status panel and explicit disclaimers (zero fabricated telemetry).
    - Seamless, decoupled integration with the HYPER detection pipeline.
"""

import json
from typing import Any, Dict, List, Optional
import streamlit as st
import streamlit.components.v1 as components

from src.dashboard.state import get_available_datasets, set_active_dataset
from src.data_loading.load_hsi import get_dataset_manifest_entry

# -----------------------------------------------------------------------------
# 1. Illustrative Geographic Locations Data
# -----------------------------------------------------------------------------
ILLUSTRATIVE_LOCATIONS: Dict[str, Dict[str, Any]] = {
    "bengaluru": {
        "id": "bengaluru",
        "name": "Bengaluru, India",
        "lat": 12.9716,
        "lon": 77.5946,
        "region_type": "Deccan Plateau / Urban Transition",
        "description": (
            "Southern Indian plateau region characterized by high-density urban infrastructure, "
            "interspersed industrial corridors, and seasonal dry-deciduous agricultural fringes."
        ),
        "elevation_m": "920 m",
        "climate": "Tropical savanna (Aw)",
        "dataset_linked": False,
        "dataset_note": "No hyperspectral dataset is linked to this location.",
    },
    "amazon": {
        "id": "amazon",
        "name": "Amazon Basin, South America",
        "lat": -3.4653,
        "lon": -62.2159,
        "region_type": "Tropical Rainforest Canopy",
        "description": (
            "Extensive equatorial rainforest basin exhibiting dense multi-tier canopy architecture, "
            "heavy atmospheric water vapor absorption, persistent cloud cover, and high organic matter reflectance."
        ),
        "elevation_m": "65 m",
        "climate": "Tropical rainforest (Af)",
        "dataset_linked": False,
        "dataset_note": "No hyperspectral dataset is linked to this location.",
    },
    "sahara": {
        "id": "sahara",
        "name": "Sahara Region, North Africa",
        "lat": 23.4162,
        "lon": 25.6628,
        "region_type": "Hyper-Arid Mineral Sand & Bedrock",
        "description": (
            "Vast subtropical desert plateau with quartz sand ergs, exposed rocky reg, and minimal "
            "vegetation clutter. Ideal for absolute radiometric calibration and mineralogical spectral baselines."
        ),
        "elevation_m": "450 m",
        "climate": "Hyper-arid desert (BWh)",
        "dataset_linked": False,
        "dataset_note": "No hyperspectral dataset is linked to this location.",
    },
    "north_atlantic": {
        "id": "north_atlantic",
        "name": "North Atlantic Pelagic Zone",
        "lat": 38.0000,
        "lon": -40.0000,
        "region_type": "Deep Marine / Open Ocean",
        "description": (
            "Open oceanic mid-latitude pelagic zone characterized by high specular sea-surface reflectance, "
            "wind-induced wave scattering, marine aerosol attenuation, and low near-infrared terrestrial return."
        ),
        "elevation_m": "0 m (Sea level)",
        "climate": "Marine temperate",
        "dataset_linked": False,
        "dataset_note": "No hyperspectral dataset is linked to this location.",
    },
    "himalayas": {
        "id": "himalayas",
        "name": "Himalayan Mountain Range",
        "lat": 27.9881,
        "lon": 86.9250,
        "region_type": "High-Altitude Glacial & Alpine Ridge",
        "description": (
            "Extreme relief alpine topography featuring permanent glacial ice, compacted firn snow packs, "
            "steep-slope topographic shadow occlusions, and severe bidirectional reflectance distribution (BRDF) effects."
        ),
        "elevation_m": "8,848 m (Peak elevation)",
        "climate": "Alpine tundra (ET)",
        "dataset_linked": False,
        "dataset_note": "No hyperspectral dataset is linked to this location.",
    },
}


# -----------------------------------------------------------------------------
# 2. 3D WebGL Earth Explorer HTML/JavaScript Component Generator
# -----------------------------------------------------------------------------
def build_3d_earth_html(
    initial_location_id: Optional[str] = "bengaluru",
    auto_rotate: bool = True,
    satellite_playing: bool = True,
    orbit_speed: str = "normal",  # slow, normal, fast
    scan_active: bool = True,
) -> str:
    """Generate self-contained Three.js 3D Earth HTML/WebGL component.

    Includes:
        - Procedural high-resolution vector continent texture fallback (100% offline).
        - Parametric 3D satellite with solar arrays, nadir payload horn, and cyan indicator.
        - Realistic orbit path line with Keplerian inclination (51.6°).
        - Translucent scanning cone and ground track swath with toggle.
        - 5 Geographic location pins with animated beacon rings and labels.
        - Raycaster click interaction and in-canvas HUD controls.
        - Graceful Canvas 2D fallback if WebGL is unavailable.
    """
    locations_json = json.dumps(ILLUSTRATIVE_LOCATIONS)
    speed_factor = 0.015
    if orbit_speed == "slow":
        speed_factor = 0.006
    elif orbit_speed == "fast":
        speed_factor = 0.038

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>HYPER 3D Earth Explorer</title>
<style>
    * {{ box-sizing: border-box; margin: 0; padding: 0; user-select: none; }}
    body {{
        background: #020914;
        color: #E2E8F0;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Inter", sans-serif;
        overflow: hidden;
        width: 100vw;
        height: 100vh;
    }}
    #webgl-container {{
        width: 100%;
        height: 100%;
        position: absolute;
        top: 0;
        left: 0;
    }}
    /* On-Canvas Telemetry HUD Overlay */
    .hud-overlay {{
        position: absolute;
        pointer-events: none;
        z-index: 10;
    }}
    .hud-top-left {{
        top: 14px;
        left: 14px;
        display: flex;
        flex-direction: column;
        gap: 6px;
    }}
    .hud-badge {{
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: rgba(6, 20, 38, 0.85);
        border: 1px solid rgba(0, 207, 255, 0.35);
        border-radius: 4px;
        padding: 4px 9px;
        font-size: 11px;
        font-family: "JetBrains Mono", monospace;
        letter-spacing: 0.06em;
        color: #00F0FF;
        backdrop-filter: blur(8px);
    }}
    .hud-dot {{
        width: 7px;
        height: 7px;
        border-radius: 50%;
        background: #00F0FF;
        box-shadow: 0 0 8px #00F0FF;
    }}
    .hud-sub-badge {{
        font-size: 10px;
        color: #94A3B8;
        font-family: "JetBrains Mono", monospace;
        background: rgba(2, 9, 20, 0.7);
        padding: 2px 7px;
        border-radius: 3px;
        border-left: 2px solid #10B981;
    }}
    .hud-top-right {{
        top: 14px;
        right: 14px;
        text-align: right;
    }}
    .location-banner {{
        background: rgba(8, 26, 44, 0.88);
        border: 1px solid rgba(0, 207, 255, 0.3);
        border-radius: 6px;
        padding: 6px 12px;
        font-size: 11px;
        color: #FFFFFF;
        font-family: "Inter", sans-serif;
        backdrop-filter: blur(8px);
        max-width: 260px;
    }}
    .location-banner strong {{
        display: block;
        color: #00F0FF;
        font-size: 12px;
        margin-bottom: 2px;
    }}
    /* Bottom Controls Toolbar */
    .hud-bottom-bar {{
        bottom: 12px;
        left: 14px;
        right: 14px;
        display: flex;
        justify-content: space-between;
        align-items: flex-end;
        pointer-events: auto;
    }}
    .hud-btn-group {{
        display: flex;
        gap: 6px;
        background: rgba(6, 20, 38, 0.88);
        border: 1px solid rgba(0, 207, 255, 0.22);
        padding: 5px 8px;
        border-radius: 6px;
        backdrop-filter: blur(8px);
    }}
    .hud-btn {{
        background: rgba(14, 34, 60, 0.8);
        border: 1px solid rgba(0, 207, 255, 0.25);
        color: #E2E8F0;
        padding: 4px 10px;
        border-radius: 4px;
        font-size: 11px;
        font-family: "Inter", sans-serif;
        cursor: pointer;
        transition: all 0.15s ease;
    }}
    .hud-btn:hover {{
        background: rgba(0, 207, 255, 0.18);
        border-color: #00F0FF;
        color: #FFFFFF;
        box-shadow: 0 0 8px rgba(0, 207, 255, 0.3);
    }}
    .hud-btn.active {{
        background: #00CFFF;
        color: #020914;
        font-weight: 700;
        border-color: #00F0FF;
    }}
    /* Fallback Display */
    #fallback-container {{
        display: none;
        width: 100%;
        height: 100%;
        padding: 24px;
        text-align: center;
        background: radial-gradient(circle at center, #0B2545 0%, #020914 80%);
    }}
    #fallback-canvas {{
        margin: 20px auto;
        display: block;
        border: 1px solid rgba(0, 207, 255, 0.3);
        border-radius: 50%;
        box-shadow: 0 0 30px rgba(0, 207, 255, 0.2);
    }}
</style>
<!-- Load Three.js and OrbitControls from standard unpkg CDN with inline fallback -->
<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>
</head>
<body>

<div id="webgl-container"></div>

<!-- Telemetry HUD -->
<div class="hud-overlay hud-top-left">
    <div class="hud-badge">
        <span class="hud-dot"></span>
        <span>HYPER 3D SATELLITE EXPLORER</span>
    </div>
    <div class="hud-sub-badge">SIMULATION MODE · LEO 51.6° INCLINATION</div>
</div>

<div class="hud-overlay hud-top-right">
    <div class="location-banner" id="location-banner">
        <strong>TARGET LOCATION</strong>
        <span id="target-loc-text">Select marker or control</span>
    </div>
</div>

<div class="hud-overlay hud-bottom-bar">
    <div class="hud-btn-group">
        <button class="hud-btn" id="btn-play-pause">⏸ Pause</button>
        <button class="hud-btn" id="btn-speed">Speed: Normal</button>
        <button class="hud-btn" id="btn-rotate">Earth: Rotating</button>
        <button class="hud-btn" id="btn-scan">Scan: ON</button>
    </div>
    <div class="hud-btn-group">
        <button class="hud-btn" id="btn-zoom-in" title="Zoom in">＋</button>
        <button class="hud-btn" id="btn-zoom-out" title="Zoom out">－</button>
        <button class="hud-btn" id="btn-reset" title="Reset Camera">⟲ Reset</button>
    </div>
</div>

<!-- 2D Canvas Fallback if WebGL fails -->
<div id="fallback-container">
    <h3 style="color:#00F0FF; margin-bottom: 8px;">HYPER Earth Explorer (2D Scientific Fallback)</h3>
    <p style="font-size: 12px; color: #94A3B8;">WebGL is unavailable in this view. Rendering 2D orthographic projection.</p>
    <canvas id="fallback-canvas" width="360" height="360"></canvas>
</div>

<script>
(function() {{
    const LOCATIONS = {locations_json};
    let initialLocationKey = "{initial_location_id or 'bengaluru'}";

    // State Variables
    let isSatellitePlaying = {str(satellite_playing).lower()};
    let isEarthRotating = {str(auto_rotate).lower()};
    let isScanActive = {str(scan_active).lower()};
    let orbitSpeedMultiplier = {speed_factor};
    let currentSpeedLabel = "{orbit_speed.capitalize()}";
    let selectedLocation = LOCATIONS[initialLocationKey] || null;

    // Check WebGL availability
    function isWebGLSupported() {{
        try {{
            const canvas = document.createElement('canvas');
            return !!(window.WebGLRenderingContext && (canvas.getContext('webgl') || canvas.getContext('experimental-webgl')));
        }} catch(e) {{
            return false;
        }}
    }}

    if (!window.THREE || !isWebGLSupported()) {{
        document.getElementById('webgl-container').style.display = 'none';
        document.getElementById('fallback-container').style.display = 'block';
        init2DFallback();
        return;
    }}

    // -------------------------------------------------------------------------
    // 1. Procedural 2048x1024 High-Resolution Earth Vector Map (100% Offline)
    // -------------------------------------------------------------------------
    function generateProceduralEarthTexture() {{
        const canvas = document.createElement('canvas');
        canvas.width = 2048;
        canvas.height = 1024;
        const ctx = canvas.getContext('2d');

        // Deep ocean gradient
        const oceanGrad = ctx.createLinearGradient(0, 0, 0, canvas.height);
        oceanGrad.addColorStop(0, '#041226');
        oceanGrad.addColorStop(0.5, '#082142');
        oceanGrad.addColorStop(1, '#041226');
        ctx.fillStyle = oceanGrad;
        ctx.fillRect(0, 0, canvas.width, canvas.height);

        // Faint latitude / longitude graticule grid
        ctx.strokeStyle = 'rgba(0, 207, 255, 0.08)';
        ctx.lineWidth = 1;
        for (let lat = 0; lat <= canvas.height; lat += canvas.height / 12) {{
            ctx.beginPath();
            ctx.moveTo(0, lat);
            ctx.lineTo(canvas.width, lat);
            ctx.stroke();
        }}
        for (let lon = 0; lon <= canvas.width; lon += canvas.width / 24) {{
            ctx.beginPath();
            ctx.moveTo(lon, 0);
            ctx.lineTo(lon, canvas.height);
            ctx.stroke();
        }}

        // Helper: Convert lat (-90..90), lon (-180..180) to canvas (x, y)
        function toXY(lat, lon) {{
            const x = ((lon + 180) / 360) * canvas.width;
            const y = ((90 - lat) / 180) * canvas.height;
            return [x, y];
        }}

        function drawPolygon(coords, fillColor, strokeColor) {{
            ctx.fillStyle = fillColor;
            ctx.strokeStyle = strokeColor || 'rgba(0, 207, 255, 0.25)';
            ctx.lineWidth = 1.2;
            ctx.beginPath();
            for (let i = 0; i < coords.length; i++) {{
                const [cx, cy] = toXY(coords[i][0], coords[i][1]);
                if (i === 0) ctx.moveTo(cx, cy);
                else ctx.lineTo(cx, cy);
            }}
            ctx.closePath();
            ctx.fill();
            ctx.stroke();
        }}

        // Continental shapes (Equirectangular geographic paths)
        // North America
        drawPolygon([
            [70, -165], [72, -130], [68, -85], [58, -60], [48, -53], [42, -70],
            [30, -80], [25, -80], [15, -90], [8, -78], [15, -100], [25, -110],
            [32, -118], [48, -125], [60, -140], [65, -168]
        ], '#274b35');

        // Greenland
        drawPolygon([
            [82, -45], [78, -18], [70, -22], [60, -45], [65, -55], [78, -68]
        ], '#dbeafe');

        // South America
        drawPolygon([
            [12, -75], [8, -55], [-5, -35], [-20, -40], [-35, -55], [-55, -65],
            [-54, -72], [-35, -73], [-15, -75], [0, -80], [10, -75]
        ], '#1e482b');

        // Amazon rainforest highlight
        drawPolygon([
            [3, -60], [2, -50], [-8, -48], [-12, -65], [-5, -70]
        ], '#166534');

        // Eurasia (Europe + Asia)
        drawPolygon([
            [71, 28], [75, 100], [70, 170], [60, 165], [52, 140], [40, 120],
            [22, 118], [10, 105], [15, 95], [22, 88], [8, 77], [22, 70],
            [25, 60], [12, 45], [30, 32], [42, 28], [36, -5], [45, -5],
            [55, 5], [60, 20], [68, 20]
        ], '#2e5a3c');

        // India Subcontinent
        drawPolygon([
            [28, 70], [24, 88], [18, 83], [8, 77], [16, 73], [24, 68]
        ], '#3d6b47');

        // Himalayas mountain ridge
        drawPolygon([
            [32, 75], [30, 85], [27, 90], [26, 85], [29, 76]
        ], '#e2e8f0');

        // Africa
        drawPolygon([
            [35, -5], [37, 10], [32, 32], [12, 45], [10, 50], [0, 42],
            [-25, 32], [-35, 20], [-34, 18], [-15, 12], [4, 8], [5, -2],
            [15, -17], [25, -15], [35, -5]
        ], '#3f5636');

        // Sahara Desert Ochre Zone
        drawPolygon([
            [30, -10], [32, 28], [20, 32], [16, -10]
        ], '#9c7b44');

        // Australia
        drawPolygon([
            [-12, 130], [-12, 142], [-22, 150], [-35, 148], [-38, 142],
            [-32, 115], [-22, 114], [-15, 125]
        ], '#8c6b38');

        // Antarctica
        drawPolygon([
            [-68, -180], [-65, -90], [-68, 0], [-65, 90], [-68, 180],
            [-90, 180], [-90, -180]
        ], '#eff6ff');

        // Atmospheric cloud swirls
        ctx.fillStyle = 'rgba(255, 255, 255, 0.16)';
        for (let i = 0; i < 28; i++) {{
            const cx = (i * 73) % canvas.width;
            const cy = ((i * 37) % (canvas.height - 180)) + 90;
            ctx.beginPath();
            ctx.ellipse(cx, cy, 90 + (i % 5) * 20, 18 + (i % 3) * 6, (i * 0.2), 0, Math.PI * 2);
            ctx.fill();
        }}

        return canvas;
    }}

    // -------------------------------------------------------------------------
    // 2. Three.js Scene, Camera, and Lighting
    // -------------------------------------------------------------------------
    const container = document.getElementById('webgl-container');
    const width = container.clientWidth || window.innerWidth;
    const height = container.clientHeight || window.innerHeight;

    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x020914);

    const camera = new THREE.PerspectiveCamera(45, width / height, 0.1, 1000);
    camera.position.set(0, 4, 18);

    const renderer = new THREE.WebGLRenderer({{ antialias: true, alpha: false }});
    renderer.setSize(width, height);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    container.appendChild(renderer.domElement);

    const controls = new THREE.OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.05;
    controls.minDistance = 7.5;
    controls.maxDistance = 35.0;
    controls.enablePan = false;

    // Ambient & Directional Sun Lighting
    const ambientLight = new THREE.AmbientLight(0x1a2e45, 1.2);
    scene.add(ambientLight);

    const sunLight = new THREE.DirectionalLight(0xEAF6FF, 2.0);
    sunLight.position.set(25, 12, 18);
    scene.add(sunLight);

    // Deep space stars
    const starsGeo = new THREE.BufferGeometry();
    const starCount = 1200;
    const starPos = new Float32Array(starCount * 3);
    for (let i = 0; i < starCount * 3; i += 3) {{
        starPos[i] = (Math.random() - 0.5) * 350;
        starPos[i + 1] = (Math.random() - 0.5) * 350;
        starPos[i + 2] = (Math.random() - 0.5) * 350;
    }}
    starsGeo.setAttribute('position', new THREE.BufferAttribute(starPos, 3));
    const starsMat = new THREE.PointsMaterial({{ color: 0x88AAFF, size: 0.9, transparent: true, opacity: 0.85 }});
    const starField = new THREE.Points(starsGeo, starsMat);
    scene.add(starField);

    // -------------------------------------------------------------------------
    // 3. Earth Globe Mesh & Atmospheric Glow
    // -------------------------------------------------------------------------
    const earthRadius = 5.0;
    const earthGeo = new THREE.SphereGeometry(earthRadius, 64, 64);

    // Primary procedural texture
    const proceduralCanvas = generateProceduralEarthTexture();
    const earthTexture = new THREE.CanvasTexture(proceduralCanvas);
    earthTexture.wrapS = THREE.RepeatWrapping;
    earthTexture.wrapT = THREE.ClampToEdgeWrapping;

    const earthMat = new THREE.MeshStandardMaterial({{
        map: earthTexture,
        roughness: 0.65,
        metalness: 0.15,
    }});
    const earthMesh = new THREE.Mesh(earthGeo, earthMat);
    scene.add(earthMesh);

    // Optional online texture enhancement (Public Domain NASA Visible Earth)
    const textureLoader = new THREE.TextureLoader();
    textureLoader.crossOrigin = "anonymous";
    textureLoader.load(
        "https://unpkg.com/three-globe/example/img/earth-blue-marble.jpg",
        function(onlineTex) {{
            onlineTex.wrapS = THREE.RepeatWrapping;
            onlineTex.wrapT = THREE.ClampToEdgeWrapping;
            earthMat.map = onlineTex;
            earthMat.needsUpdate = true;
        }},
        undefined,
        function() {{
            // On failure or offline mode, smoothly retain the procedural canvas
        }}
    );

    // Atmospheric rim glow outer sphere
    const atmosGeo = new THREE.SphereGeometry(earthRadius * 1.025, 48, 48);
    const atmosMat = new THREE.MeshBasicMaterial({{
        color: 0x00F0FF,
        transparent: true,
        opacity: 0.16,
        side: THREE.BackSide,
        blending: THREE.AdditiveBlending,
    }});
    const atmosMesh = new THREE.Mesh(atmosGeo, atmosMat);
    scene.add(atmosMesh);

    // -------------------------------------------------------------------------
    // 4. Selectable Geographic Markers
    // -------------------------------------------------------------------------
    const markerGroup = new THREE.Group();
    earthMesh.add(markerGroup);

    function latLonToVector3(lat, lon, radius) {{
        const phi = (90 - lat) * (Math.PI / 180);
        const theta = (lon + 180) * (Math.PI / 180);
        const x = -radius * Math.sin(phi) * Math.cos(theta);
        const y = radius * Math.cos(phi);
        const z = radius * Math.sin(phi) * Math.sin(theta);
        return new THREE.Vector3(x, y, z);
    }}

    const markerObjects = [];
    Object.values(LOCATIONS).forEach(loc => {{
        const pos = latLonToVector3(loc.lat, loc.lon, earthRadius * 1.01);

        // Core marker pin
        const pinGeo = new THREE.SphereGeometry(0.12, 16, 16);
        const pinMat = new THREE.MeshBasicMaterial({{
            color: (loc.id === initialLocationKey) ? 0x00F0FF : 0x38BDF8,
        }});
        const pin = new THREE.Mesh(pinGeo, pinMat);
        pin.position.copy(pos);
        pin.userData = {{ location: loc }};

        // Pulsing outer ring
        const ringGeo = new THREE.RingGeometry(0.16, 0.26, 24);
        const ringMat = new THREE.MeshBasicMaterial({{
            color: 0x00F0FF,
            side: THREE.DoubleSide,
            transparent: true,
            opacity: 0.65,
        }});
        const ring = new THREE.Mesh(ringGeo, ringMat);
        ring.position.copy(pos);
        ring.lookAt(new THREE.Vector3(0, 0, 0));
        pin.add(ring);
        pin.userData.ring = ring;

        markerGroup.add(pin);
        markerObjects.push(pin);
    }});

    function updateMarkerHighlight(targetLocId) {{
        markerObjects.forEach(m => {{
            const isMatch = m.userData.location.id === targetLocId;
            m.material.color.setHex(isMatch ? 0x00F0FF : 0x22D3EE);
            if (m.userData.ring) {{
                m.userData.ring.material.opacity = isMatch ? 0.95 : 0.35;
                m.userData.ring.scale.set(isMatch ? 1.4 : 1.0, isMatch ? 1.4 : 1.0, 1.0);
            }}
        }});
        if (LOCATIONS[targetLocId]) {{
            selectedLocation = LOCATIONS[targetLocId];
            document.getElementById('target-loc-text').textContent = `${{selectedLocation.name}} (${{selectedLocation.lat.toFixed(2)}}°, ${{selectedLocation.lon.toFixed(2)}}°)`;
        }}
    }}

    updateMarkerHighlight(initialLocationKey);

    // -------------------------------------------------------------------------
    // 5. Parametric Satellite Model & Keplerian Orbit Path (LEO 51.6°)
    // -------------------------------------------------------------------------
    const orbitRadius = 7.6;
    const orbitInclination = 51.6 * (Math.PI / 180);

    // Visible orbital path line
    const orbitSegments = 128;
    const orbitPoints = [];
    for (let i = 0; i <= orbitSegments; i++) {{
        const theta = (i / orbitSegments) * Math.PI * 2;
        const x = orbitRadius * Math.cos(theta);
        const y = orbitRadius * Math.sin(theta) * Math.sin(orbitInclination);
        const z = orbitRadius * Math.sin(theta) * Math.cos(orbitInclination);
        orbitPoints.push(new THREE.Vector3(x, y, z));
    }}
    const orbitGeo = new THREE.BufferGeometry().setFromPoints(orbitPoints);
    const orbitMat = new THREE.LineBasicMaterial({{
        color: 0x00F0FF,
        transparent: true,
        opacity: 0.38,
        linewidth: 1.5,
    }});
    const orbitLine = new THREE.Line(orbitGeo, orbitMat);
    scene.add(orbitLine);

    // Satellite Group Model
    const satelliteGroup = new THREE.Group();

    // Spacecraft Bus (Main body)
    const busGeo = new THREE.BoxGeometry(0.32, 0.32, 0.52);
    const busMat = new THREE.MeshStandardMaterial({{
        color: 0x1E293B,
        metalness: 0.85,
        roughness: 0.2,
    }});
    const busMesh = new THREE.Mesh(busGeo, busMat);
    satelliteGroup.add(busMesh);

    // Gold thermal foil accent
    const foilGeo = new THREE.BoxGeometry(0.24, 0.24, 0.1);
    const foilMat = new THREE.MeshStandardMaterial({{
        color: 0xD97706,
        metalness: 0.9,
        roughness: 0.25,
    }});
    const foilMesh = new THREE.Mesh(foilGeo, foilMat);
    foilMesh.position.set(0, 0, 0.24);
    satelliteGroup.add(foilMesh);

    // Dual Photovoltaic Solar Arrays
    const wingGeo = new THREE.BoxGeometry(0.85, 0.025, 0.36);
    const wingMat = new THREE.MeshStandardMaterial({{
        color: 0x0284C7,
        metalness: 0.9,
        roughness: 0.15,
    }});

    const leftWing = new THREE.Mesh(wingGeo, wingMat);
    leftWing.position.set(-0.62, 0, 0);
    satelliteGroup.add(leftWing);

    const rightWing = new THREE.Mesh(wingGeo, wingMat);
    rightWing.position.set(0.62, 0, 0);
    satelliteGroup.add(rightWing);

    // Nadir-facing Optical Sensor Horn
    const hornGeo = new THREE.ConeGeometry(0.14, 0.22, 16);
    const hornMat = new THREE.MeshStandardMaterial({{ color: 0x00F0FF, metalness: 0.7, roughness: 0.3 }});
    const hornMesh = new THREE.Mesh(hornGeo, hornMat);
    hornMesh.rotation.x = Math.PI;
    hornMesh.position.set(0, -0.22, 0);
    satelliteGroup.add(hornMesh);

    // Pulsing cyan indicator beacon
    const beaconGeo = new THREE.SphereGeometry(0.065, 8, 8);
    const beaconMat = new THREE.MeshBasicMaterial({{ color: 0x00F0FF }});
    const beaconMesh = new THREE.Mesh(beaconGeo, beaconMat);
    beaconMesh.position.set(0, 0.22, 0);
    satelliteGroup.add(beaconMesh);

    scene.add(satelliteGroup);

    // -------------------------------------------------------------------------
    // 6. Scanning Swath & Footprint Visualization
    // -------------------------------------------------------------------------
    const scanHeight = orbitRadius - earthRadius;
    const scanConeGeo = new THREE.ConeGeometry(1.6, scanHeight, 24, 1, true);
    const scanConeMat = new THREE.MeshBasicMaterial({{
        color: 0x00F0FF,
        transparent: true,
        opacity: isScanActive ? 0.16 : 0.0,
        side: THREE.DoubleSide,
        depthWrite: false,
    }});
    const scanCone = new THREE.Mesh(scanConeGeo, scanConeMat);
    scene.add(scanCone);

    // Ground footprint ring
    const footprintGeo = new THREE.RingGeometry(1.3, 1.55, 32);
    const footprintMat = new THREE.MeshBasicMaterial({{
        color: 0x00F0FF,
        transparent: true,
        opacity: isScanActive ? 0.45 : 0.0,
        side: THREE.DoubleSide,
    }});
    const footprintMesh = new THREE.Mesh(footprintGeo, footprintMat);
    scene.add(footprintMesh);

    // -------------------------------------------------------------------------
    // 7. Raycasting & Marker Interaction
    // -------------------------------------------------------------------------
    const raycaster = new THREE.Raycaster();
    const mouse = new THREE.Vector2();

    function onPointerDown(event) {{
        const rect = renderer.domElement.getBoundingClientRect();
        mouse.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
        mouse.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;

        raycaster.setFromCamera(mouse, camera);
        const intersects = raycaster.intersectObjects(markerObjects, false);
        if (intersects.length > 0) {{
            const hitLoc = intersects[0].object.userData.location;
            if (hitLoc) {{
                orientGlobeToLocation(hitLoc.lat, hitLoc.lon);
                updateMarkerHighlight(hitLoc.id);
            }}
        }}
    }}
    renderer.domElement.addEventListener('pointerdown', onPointerDown);

    // Smoothly rotate globe to face selected location
    let targetQuat = null;
    let isOrienting = false;
    function orientGlobeToLocation(lat, lon) {{
        const targetVec = latLonToVector3(lat, lon, 1.0).normalize();
        const cameraDir = camera.position.clone().normalize();
        // Compute rotation needed to bring targetVec to camera direction
        const q = new THREE.Quaternion().setFromUnitVectors(targetVec, cameraDir);
        targetQuat = q.multiply(earthMesh.quaternion.clone());
        isOrienting = true;
    }}

    // -------------------------------------------------------------------------
    // 8. In-Canvas Toolbar Buttons Event Binding
    // -------------------------------------------------------------------------
    const btnPlayPause = document.getElementById('btn-play-pause');
    btnPlayPause.addEventListener('click', () => {{
        isSatellitePlaying = !isSatellitePlaying;
        btnPlayPause.textContent = isSatellitePlaying ? '⏸ Pause' : '▶ Play';
        btnPlayPause.classList.toggle('active', !isSatellitePlaying);
    }});

    const btnSpeed = document.getElementById('btn-speed');
    btnSpeed.addEventListener('click', () => {{
        if (orbitSpeedMultiplier < 0.01) {{
            orbitSpeedMultiplier = 0.015;
            btnSpeed.textContent = 'Speed: Normal';
        }} else if (orbitSpeedMultiplier < 0.025) {{
            orbitSpeedMultiplier = 0.038;
            btnSpeed.textContent = 'Speed: Fast';
        }} else {{
            orbitSpeedMultiplier = 0.006;
            btnSpeed.textContent = 'Speed: Slow';
        }}
    }});

    const btnRotate = document.getElementById('btn-rotate');
    btnRotate.addEventListener('click', () => {{
        isEarthRotating = !isEarthRotating;
        btnRotate.textContent = isEarthRotating ? 'Earth: Rotating' : 'Earth: Static';
        btnRotate.classList.toggle('active', !isEarthRotating);
    }});

    const btnScan = document.getElementById('btn-scan');
    btnScan.addEventListener('click', () => {{
        isScanActive = !isScanActive;
        btnScan.textContent = isScanActive ? 'Scan: ON' : 'Scan: OFF';
        btnScan.classList.toggle('active', isScanActive);
        scanConeMat.opacity = isScanActive ? 0.16 : 0.0;
        footprintMat.opacity = isScanActive ? 0.45 : 0.0;
    }});

    document.getElementById('btn-zoom-in').addEventListener('click', () => {{
        camera.position.multiplyScalar(0.85);
        controls.update();
    }});

    document.getElementById('btn-zoom-out').addEventListener('click', () => {{
        camera.position.multiplyScalar(1.18);
        controls.update();
    }});

    document.getElementById('btn-reset').addEventListener('click', () => {{
        camera.position.set(0, 4, 18);
        controls.target.set(0, 0, 0);
        controls.update();
    }});

    // -------------------------------------------------------------------------
    // 9. Main Animation Loop (60 FPS)
    // -------------------------------------------------------------------------
    let orbitTheta = 0.0;
    let clock = new THREE.Clock();

    function animate() {{
        requestAnimationFrame(animate);

        const delta = clock.getDelta();
        const elapsedTime = clock.getElapsedTime();

        // Earth rotation
        if (isEarthRotating && !isOrienting) {{
            earthMesh.rotation.y += 0.0012;
        }}

        // Slerp globe orientation if smoothly transitioning
        if (isOrienting && targetQuat) {{
            earthMesh.quaternion.slerp(targetQuat, 0.06);
            if (earthMesh.quaternion.angleTo(targetQuat) < 0.01) {{
                isOrienting = false;
            }}
        }}

        // Satellite orbital progression
        if (isSatellitePlaying) {{
            orbitTheta += orbitSpeedMultiplier;
            if (orbitTheta > Math.PI * 2) orbitTheta -= Math.PI * 2;
        }}

        const satX = orbitRadius * Math.cos(orbitTheta);
        const satY = orbitRadius * Math.sin(orbitTheta) * Math.sin(orbitInclination);
        const satZ = orbitRadius * Math.sin(orbitTheta) * Math.cos(orbitInclination);
        satelliteGroup.position.set(satX, satY, satZ);

        // Nadir orientation (Point sensor horn to Earth center)
        satelliteGroup.lookAt(0, 0, 0);

        // Position scanning cone between satellite and surface
        if (isScanActive) {{
            const satPos = satelliteGroup.position;
            const nadirDir = satPos.clone().normalize();
            const surfacePos = nadirDir.clone().multiplyScalar(earthRadius);
            const midPos = satPos.clone().add(surfacePos).multiplyScalar(0.5);

            scanCone.position.copy(midPos);
            scanCone.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), nadirDir);

            footprintMesh.position.copy(surfacePos.clone().multiplyScalar(1.002));
            footprintMesh.lookAt(surfacePos.clone().add(nadirDir));

            // Dynamic scan pulse
            scanConeMat.opacity = 0.12 + Math.sin(elapsedTime * 4.0) * 0.05;
        }}

        // Pulsing marker rings
        markerObjects.forEach((m, idx) => {{
            if (m.userData.ring) {{
                const pulse = 1.0 + Math.sin(elapsedTime * 3.0 + idx) * 0.18;
                m.userData.ring.scale.set(pulse, pulse, 1.0);
            }}
        }});

        controls.update();
        renderer.render(scene, camera);
    }}

    animate();

    // Responsive resize handler
    window.addEventListener('resize', () => {{
        const newW = container.clientWidth || window.innerWidth;
        const newH = container.clientHeight || window.innerHeight;
        camera.aspect = newW / newH;
        camera.updateProjectionMatrix();
        renderer.setSize(newW, newH);
    }});

    // -------------------------------------------------------------------------
    // 10. 2D Orthographic Fallback Renderer (For non-WebGL clients)
    // -------------------------------------------------------------------------
    function init2DFallback() {{
        const fCanvas = document.getElementById('fallback-canvas');
        const fCtx = fCanvas.getContext('2d');
        let fAngle = 0;

        function draw2D() {{
            fCtx.clearRect(0, 0, fCanvas.width, fCanvas.height);
            const cx = fCanvas.width / 2;
            const cy = fCanvas.height / 2;
            const r = fCanvas.width * 0.44;

            // Globe sphere
            const gGrad = fCtx.createRadialGradient(cx - r*0.3, cy - r*0.3, r*0.1, cx, cy, r);
            gGrad.addColorStop(0, '#0D3B66');
            gGrad.addColorStop(0.7, '#082142');
            gGrad.addColorStop(1, '#041226');
            fCtx.fillStyle = gGrad;
            fCtx.beginPath();
            fCtx.arc(cx, cy, r, 0, Math.PI * 2);
            fCtx.fill();

            // Atmosphere rim
            fCtx.strokeStyle = 'rgba(0, 207, 255, 0.4)';
            fCtx.lineWidth = 3;
            fCtx.stroke();

            // Orbit path ellipse
            fCtx.strokeStyle = 'rgba(0, 207, 255, 0.35)';
            fCtx.lineWidth = 1.5;
            fCtx.beginPath();
            fCtx.ellipse(cx, cy, r * 1.35, r * 0.55, -0.4, 0, Math.PI * 2);
            fCtx.stroke();

            // Satellite dot
            fAngle += 0.02;
            const satX = cx + Math.cos(fAngle) * (r * 1.35) * Math.cos(-0.4) - Math.sin(fAngle) * (r * 0.55) * Math.sin(-0.4);
            const satY = cy + Math.cos(fAngle) * (r * 1.35) * Math.sin(-0.4) + Math.sin(fAngle) * (r * 0.55) * Math.cos(-0.4);

            fCtx.fillStyle = '#00F0FF';
            fCtx.beginPath();
            fCtx.arc(satX, satY, 5, 0, Math.PI * 2);
            fCtx.fill();

            requestAnimationFrame(draw2D);
        }}
        draw2D();
    }}
}})();
</script>
</body>
</html>
"""


# -----------------------------------------------------------------------------
# 3. Main Streamlit Page View Renderer
# -----------------------------------------------------------------------------
def render_earth_explorer_view() -> None:
    """Render Feature 2 Interactive 3D Earth Explorer."""
    # Top Section Banner
    st.title("HYPER — 3D Earth Explorer", anchor=False)
    st.markdown(
        """
        <div style="display: flex; align-items: center; gap: 0.75rem; margin: -0.2rem 0 0.85rem 0; flex-wrap: wrap;">
            <span style="font-family: 'JetBrains Mono', monospace; font-size: 0.72rem; font-weight: 700; color: #00F0FF; background: rgba(0, 240, 255, 0.1); border: 1px solid rgba(0, 240, 255, 0.28); padding: 0.18rem 0.55rem; border-radius: 4px; letter-spacing: 0.08em;">ORBITAL VISUALIZATION</span>
            <span style="color: #00CFFF; font-style: italic; font-weight: 500; font-size: 0.88rem;">“See Beyond the Visible.”</span>
            <span style="color: #64748B;">·</span>
            <span style="color: #94A3B8; font-size: 0.84rem;">Interactive planetary observation station, simulated LEO orbital track, and regional terrestrial reconnaissance.</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Scientific Simulation Disclaimers Banner (Prominently Rendered)
    st.markdown(
        """
        <div style="background: rgba(8, 26, 44, 0.78); border: 1px solid rgba(0, 207, 255, 0.22); border-left: 3px solid #00F0FF; border-radius: 6px; padding: 0.65rem 0.95rem; margin-bottom: 1.1rem; font-size: 0.81rem; color: #CBD5E1; line-height: 1.45;">
            <strong style="color: #00F0FF; font-family: 'JetBrains Mono', monospace; letter-spacing: 0.05em;">SCIENTIFIC SIMULATION ADVISORY:</strong>
            The 3D globe, satellite spacecraft model, orbital path, and sweeping sensor swath are illustrative visualizations designed for educational orientation and spatial context.
            <em>“Simulated satellite and illustrative orbit — not real spacecraft or actual orbital trajectory.”</em> &nbsp;·&nbsp;
            <em>“Illustrative scan animation — not live satellite acquisition.”</em>
            No telemetry or spectral detection results are fabricated.
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Manage local session state for control synchronization
    if "earth_selected_loc" not in st.session_state:
        st.session_state["earth_selected_loc"] = "bengaluru"
    if "earth_satellite_play" not in st.session_state:
        st.session_state["earth_satellite_play"] = True
    if "earth_speed" not in st.session_state:
        st.session_state["earth_speed"] = "normal"
    if "earth_rotate" not in st.session_state:
        st.session_state["earth_rotate"] = True
    if "earth_scan" not in st.session_state:
        st.session_state["earth_scan"] = True

    # 2-Column Responsive Layout: Globe (Center/Left) & Controls/Info (Right)
    globe_col, info_col = st.columns([1.85, 1.15], gap="large")

    with globe_col:
        st.markdown(
            """
            <div class="command-panel-header" style="margin-bottom: 0.4rem;">
                <div class="command-panel-title">
                    <span>🌍</span> Planetary Earth Observation Workstation
                </div>
                <div style="font-size: 0.75rem; color: #94A3B8; font-family: 'JetBrains Mono', monospace;">
                    WebGL 60 FPS · 3D OrbitControls
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Generate and embed the 3D WebGL HTML Canvas
        earth_html = build_3d_earth_html(
            initial_location_id=st.session_state["earth_selected_loc"],
            auto_rotate=st.session_state["earth_rotate"],
            satellite_playing=st.session_state["earth_satellite_play"],
            orbit_speed=st.session_state["earth_speed"],
            scan_active=st.session_state["earth_scan"],
        )

        components.html(earth_html, height=590, scrolling=False)

        st.caption(
            "💡 **3D Interaction Controls:** Click and drag with mouse or touch to rotate the globe · "
            "Scroll or pinch to zoom in/out · Click directly on any glowing marker to focus on that region · "
            "Use on-canvas HUD toolbar buttons for instant zero-latency toggles."
        )

    with info_col:
        # Card 1: Visualization & Satellite Control Panel
        st.markdown(
            """
            <div class="command-panel-header">
                <div class="command-panel-title">
                    <span>🎛️</span> Satellite & Explorer Controls
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        ctrl_col1, ctrl_col2 = st.columns(2)
        with ctrl_col1:
            sat_btn_label = "⏸ Pause Orbit" if st.session_state["earth_satellite_play"] else "▶ Resume Orbit"
            if st.button(sat_btn_label, key="ctrl_toggle_sat", width="stretch"):
                st.session_state["earth_satellite_play"] = not st.session_state["earth_satellite_play"]
                st.rerun()

            rot_btn_label = "Earth Spin: ON" if st.session_state["earth_rotate"] else "Earth Spin: OFF"
            if st.button(rot_btn_label, key="ctrl_toggle_rot", width="stretch"):
                st.session_state["earth_rotate"] = not st.session_state["earth_rotate"]
                st.rerun()

        with ctrl_col2:
            speed_options = ["Slow", "Normal", "Fast"]
            curr_speed_name = st.session_state["earth_speed"].capitalize()
            curr_speed_idx = speed_options.index(curr_speed_name) if curr_speed_name in speed_options else 1
            selected_speed = st.selectbox(
                "Orbit velocity",
                speed_options,
                index=curr_speed_idx,
                key="ctrl_speed_select",
                help="Adjust the simulated satellite's angular orbital velocity around the Earth.",
            )
            if selected_speed.lower() != st.session_state["earth_speed"]:
                st.session_state["earth_speed"] = selected_speed.lower()
                st.rerun()

            scan_btn_label = "Sensor Swath: ON" if st.session_state["earth_scan"] else "Sensor Swath: OFF"
            if st.button(scan_btn_label, key="ctrl_toggle_scan", width="stretch"):
                st.session_state["earth_scan"] = not st.session_state["earth_scan"]
                st.rerun()

        st.markdown("<div style='height: 0.85rem;'></div>", unsafe_allow_html=True)

        # Card 2: Selected-Location Information Panel
        st.markdown(
            """
            <div class="command-panel-header">
                <div class="command-panel-title">
                    <span>📍</span> Selected Geographic Region
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        loc_keys = list(ILLUSTRATIVE_LOCATIONS.keys())
        loc_names = [ILLUSTRATIVE_LOCATIONS[k]["name"] for k in loc_keys]
        curr_loc_key = st.session_state.get("earth_selected_loc", "bengaluru")
        if curr_loc_key not in loc_keys:
            curr_loc_key = "bengaluru"
        curr_idx = loc_keys.index(curr_loc_key)

        chosen_loc_name = st.selectbox(
            "Select illustrative landmark",
            loc_names,
            index=curr_idx,
            key="earth_loc_selector",
            help="Choose an illustrative geographic region to inspect its terrain properties.",
        )
        # Update session state if dropdown changed
        new_loc_key = loc_keys[loc_names.index(chosen_loc_name)]
        if new_loc_key != st.session_state["earth_selected_loc"]:
            st.session_state["earth_selected_loc"] = new_loc_key
            st.rerun()

        loc_data = ILLUSTRATIVE_LOCATIONS[st.session_state["earth_selected_loc"]]

        # Display location telemetry card
        st.markdown(
            f"""
            <div style="background: rgba(14, 22, 38, 0.7); border: 1px solid rgba(0, 207, 255, 0.18); border-radius: 8px; padding: 0.85rem 1rem; margin-bottom: 0.75rem;">
                <div style="font-size: 1.05rem; font-weight: 700; color: #FFFFFF; margin-bottom: 0.25rem;">
                    {loc_data['name']}
                </div>
                <div style="font-family: 'JetBrains Mono', monospace; font-size: 0.78rem; color: #00F0FF; margin-bottom: 0.5rem;">
                    Lat: {loc_data['lat']:.4f}° &nbsp;·&nbsp; Lon: {loc_data['lon']:.4f}° &nbsp;·&nbsp; Elev: {loc_data['elevation_m']}
                </div>
                <div style="font-size: 0.82rem; color: #CBD5E1; line-height: 1.45; margin-bottom: 0.65rem;">
                    {loc_data['description']}
                </div>
                <div style="font-size: 0.76rem; color: #94A3B8; font-family: 'JetBrains Mono', monospace; padding-top: 0.45rem; border-top: 1px solid rgba(0, 207, 255, 0.1);">
                    <strong>CLIMATE / TERRAIN:</strong> {loc_data['climate']} · {loc_data['region_type']}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Dataset linkage notification (Honest reporting as required by Section 8 & 13)
        if loc_data.get("dataset_linked"):
            st.markdown(
                f"""
                <div style="background: rgba(16, 185, 129, 0.12); border: 1px solid rgba(16, 185, 129, 0.4); border-radius: 6px; padding: 0.65rem 0.85rem; margin-bottom: 0.85rem;">
                    <div style="font-size: 0.76rem; font-family: 'JetBrains Mono', monospace; color: #10B981; font-weight: 700; margin-bottom: 0.25rem;">
                        DATASET ASSOCIATION: VERIFIED
                    </div>
                    <div style="font-size: 0.80rem; color: #E2E8F0;">
                        Linked scene: <strong>{loc_data.get('dataset_name', 'Verified Scene')}</strong>
                    </div>
                    <div style="font-size: 0.74rem; color: #94A3B8; margin-top: 0.25rem;">
                        Workflow: Open Dataset → Run Detection → View Results
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                """
                <div style="background: rgba(2, 9, 20, 0.65); border: 1px solid rgba(56, 189, 248, 0.15); border-radius: 6px; padding: 0.65rem 0.85rem; margin-bottom: 0.85rem;">
                    <div style="font-size: 0.76rem; font-family: 'JetBrains Mono', monospace; color: #94A3B8; margin-bottom: 0.35rem;">
                        DATASET ASSOCIATION STATUS
                    </div>
                    <div style="font-size: 0.80rem; color: #E2E8F0; margin-bottom: 0.25rem;">
                        <em>“No hyperspectral dataset is linked to this location.”</em>
                    </div>
                    <div style="font-size: 0.72rem; color: #64748B; font-family: 'JetBrains Mono', monospace;">
                        STATUS: Visual Scan Only · Live Satellite Data: Not Connected
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        def _route_to_datasets() -> None:
            target = "▦  Datasets / Dataset Explorer"
            try:
                st.session_state.active_section = target
            except Exception:
                pass
            st.session_state["target_section"] = target

        def _reset_earth_selection() -> None:
            st.session_state["earth_selected_loc"] = "bengaluru"

        nav_b1, nav_b2 = st.columns([1.4, 1.0])
        with nav_b1:
            st.button(
                "▦ Open Dataset Explorer",
                key="earth_open_datasets",
                width="stretch",
                help="Navigate to verified local benchmark scenes in Dataset Explorer",
                on_click=_route_to_datasets,
            )
        with nav_b2:
            st.button(
                "Clear Selection",
                key="earth_clear_selection",
                width="stretch",
                on_click=_reset_earth_selection,
            )

        st.markdown("<div style='height: 0.65rem;'></div>", unsafe_allow_html=True)

        # Card 2B: Local Benchmark Scenes Workflow (Section 13, 14, 15)
        available_ds = get_available_datasets()
        if available_ds:
            st.markdown(
                """
                <div class="command-panel-header">
                    <div class="command-panel-title">
                        <span>🔬</span> Benchmark Scene Pipeline Link
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            default_ds_idx = 0
            curr_selected = st.session_state.get("selected_dataset", available_ds[0])
            if curr_selected in available_ds:
                default_ds_idx = available_ds.index(curr_selected)

            chosen_bench_ds = st.selectbox(
                "Choose local benchmark scene",
                available_ds,
                index=default_ds_idx,
                key="earth_bench_ds_selector",
                help="Select a verified local hyperspectral cube to inspect or run through the detection pipeline.",
            )

            manifest_entry = get_dataset_manifest_entry(chosen_bench_ds)
            if manifest_entry:
                sensor_name = manifest_entry.get("sensor", "Unknown")
                shape_str = f"{manifest_entry.get('height', '?')}x{manifest_entry.get('width', '?')}x{manifest_entry.get('bands', '?')}"
                targets_str = f"{manifest_entry.get('target_pixels', '?')} px ({manifest_entry.get('target_pct', '?')}%)"
                st.markdown(
                    f"""
                    <div style="background: rgba(14, 22, 38, 0.55); border: 1px solid rgba(0, 207, 255, 0.12); border-radius: 6px; padding: 0.5rem 0.75rem; font-family: 'JetBrains Mono', monospace; font-size: 0.72rem; margin-bottom: 0.6rem;">
                        <span style="color: #94A3B8;">SENSOR:</span> <strong style="color: #00F0FF;">{sensor_name}</strong> &nbsp;·&nbsp;
                        <span style="color: #94A3B8;">DIM:</span> <strong style="color: #FFFFFF;">{shape_str}</strong> &nbsp;·&nbsp;
                        <span style="color: #94A3B8;">TARGETS:</span> <strong style="color: #38BDF8;">{targets_str}</strong>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            def _route_bench_to_explorer(ds_name: str) -> None:
                set_active_dataset(ds_name)
                target = "▦  Datasets / Dataset Explorer"
                try:
                    st.session_state.active_section = target
                except Exception:
                    pass
                st.session_state["target_section"] = target

            def _route_bench_to_pipeline(ds_name: str) -> None:
                set_active_dataset(ds_name)
                target = "◉  Detection Pipeline / Anomaly Detection"
                try:
                    st.session_state.active_section = target
                except Exception:
                    pass
                st.session_state["target_section"] = target

            b_col1, b_col2 = st.columns(2)
            with b_col1:
                st.button(
                    "▦ Inspect in Explorer",
                    key="earth_bench_inspect",
                    width="stretch",
                    on_click=_route_bench_to_explorer,
                    args=(chosen_bench_ds,),
                    help="Open this benchmark cube in the Dataset Explorer for spectral band browsing",
                )
            with b_col2:
                st.button(
                    "◉ Run Detection Pipeline",
                    key="earth_bench_detect",
                    width="stretch",
                    on_click=_route_bench_to_pipeline,
                    args=(chosen_bench_ds,),
                    help="Open the Detection Pipeline with this scene to run RX, PCA, or Backend Pipeline",
                )

            st.markdown("<div style='height: 0.85rem;'></div>", unsafe_allow_html=True)

        # Card 3: Scientific Honesty & Telemetry Status (Section 13)
        st.markdown(
            """
            <div class="command-panel-header">
                <div class="command-panel-title">
                    <span>🛰️</span> Simulation System Specifications
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            f"""
            <div style="background: rgba(8, 26, 44, 0.55); border: 1px solid rgba(0, 207, 255, 0.12); border-radius: 8px; padding: 0.75rem 0.95rem; font-family: 'JetBrains Mono', monospace; font-size: 0.74rem;">
                <div style="display: flex; justify-content: space-between; margin-bottom: 0.35rem;">
                    <span style="color: #94A3B8;">VISUALIZATION MODE:</span>
                    <strong style="color: #38BDF8;">Simulated Planetary Explorer</strong>
                </div>
                <div style="display: flex; justify-content: space-between; margin-bottom: 0.35rem;">
                    <span style="color: #94A3B8;">SPACECRAFT MODEL:</span>
                    <strong style="color: #FFFFFF;">Parametric LEO Satellite</strong>
                </div>
                <div style="display: flex; justify-content: space-between; margin-bottom: 0.35rem;">
                    <span style="color: #94A3B8;">ORBITAL TRAJECTORY:</span>
                    <strong style="color: #FFFFFF;">51.6° Inclination (~600 km LEO)</strong>
                </div>
                <div style="display: flex; justify-content: space-between; margin-bottom: 0.35rem;">
                    <span style="color: #94A3B8;">SWATH SENSOR CONE:</span>
                    <strong style="color: {'#10B981' if st.session_state['earth_scan'] else '#64748B'};">{'Active Beam' if st.session_state['earth_scan'] else 'Standby'}</strong>
                </div>
                <div style="display: flex; justify-content: space-between; margin-bottom: 0.35rem;">
                    <span style="color: #94A3B8;">LIVE SATELLITE DATA:</span>
                    <strong style="color: #F59E0B;">Not Connected (Simulation)</strong>
                </div>
                <div style="display: flex; justify-content: space-between; border-top: 1px solid rgba(0,207,255,0.1); padding-top: 0.35rem;">
                    <span style="color: #94A3B8;">ML DETECTOR STATUS:</span>
                    <strong style="color: #10B981;">Idle / Isolated from Globe</strong>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_overview_earth_preview(height: int = 430) -> None:
    """Render the interactive 3D Earth Preview for the Overview command center."""
    earth_html = build_3d_earth_html(
        initial_location_id="bengaluru",
        auto_rotate=True,
        satellite_playing=True,
        orbit_speed="normal",
        scan_active=True,
    )
    components.html(earth_html, height=height, scrolling=False)
