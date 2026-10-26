# HYPER Dashboard 3D Earth Assets & License Documentation

This directory documents the 3D visualization assets and shaders used by the **HYPER — Hyperspectral Intelligence Platform** (Feature 2: Interactive 3D Earth Explorer).

## 1. Zero External Dependency & Offline Architecture
The Interactive 3D Earth Explorer is designed to operate completely offline without requiring:
- Any external paid satellite imagery API.
- Any API keys, account tokens, or external authentication.
- Any live external network connection for the core visualization.

## 2. Earth Surface Texture Generation
- **Primary Mechanism**: A client-side procedural Canvas2D vector texture generator synthesizes an equirectangular (2:1 aspect ratio) global surface map directly in memory.
- **Topographical Mapping**: Accurately renders major continental landmasses (Eurasia, Africa, North America, South America, Australia, Antarctica, Greenland) with realistic ocean bathymetry (`#081d33` to `#0d2b45`), vegetation corridors (`#2d5a3f`), arid zones (`#a38249`), Himalayan elevations, and polar ice shelves.
- **Optional Imagery Fallback**: Can optionally blend with public domain NASA Visible Earth (Blue Marble) imagery if network connectivity is available, smoothly reverting to the embedded procedural texture if offline or on load error.
- **License**: Public Domain (NASA Visible Earth) / MIT (Procedural code).

## 3. 3D Graphics Engine
- **Engine**: Three.js (WebGL renderer with OrbitControls).
- **License**: MIT License (Copyright (c) 2010-2024 Three.js authors).
- **Fallback**: Includes an embedded Canvas2D sphere projection fallback if WebGL is unavailable or disabled in the client environment.

## 4. Satellite Model & Orbital Simulation
- **Geometry**: Lightweight parametric 3D model constructed entirely from core Three.js primitives (`BoxGeometry`, `CylinderGeometry`, `SphereGeometry`).
- **Spacecraft Bus**: Cuboid frame with titanium/solar panel shaders.
- **Arrays**: Symmetrical dual solar wings with deep blue photovoltaic cell grid texturing.
- **Payload**: Nadir-facing sensor horn pointing toward Earth geocenter.
- **Orbit**: Circular/elliptical inclined Keplerian orbit line ($i = 51.6^\circ$, LEO illustrative altitude).

## 5. Scientific Honesty & Simulation Disclaimers
All orbital trajectories, satellite positions, scanning swaths, and geographic markers are illustrative simulations. They do not represent real-time spacecraft tracking, live telemetry, or active sensor acquisition.
