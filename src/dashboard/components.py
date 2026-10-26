"""Feature 9 & Reusable Visualization Components for Hyperspectral Dashboard.

Provides interactive Plotly visualizations and UI components:
    - Interactive 2D anomaly heatmaps with coordinates, score inspection, and ground-truth overlays.
    - Genuine interactive 3D Hyperspectral Data Cube volumetric slicer.
    - Genuine interactive 3D Anomaly Topographic Mesh surface.
    - Spectral channel slices and synthesized pseudo-RGB / true-RGB composites.
    - Pixel-level spectral signature curves with wavelength annotations.
    - Side-by-side ROC and Precision-Recall operating curves with AUC fills.
    - High-information-density metric HUD cards with tooltips and scientific disclaimers.
"""

import textwrap
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import precision_recall_curve, roc_curve
import streamlit as st

from src.dashboard.theme import get_scientific_plotly_layout, render_hud_metric_cards
from src.experiments.leaderboard import get_metric_definitions


def synthesize_rgb_composite(
    cube: np.ndarray,
    meta: Optional[Dict[str, Any]] = None,
    red_band: Optional[int] = None,
    green_band: Optional[int] = None,
    blue_band: Optional[int] = None,
) -> np.ndarray:
    """Generate an (H, W, 3) float32 RGB composite from a hyperspectral cube.

    Uses pre-rendered RGB from metadata if available; otherwise selects
    representative red, green, and blue spectral channels with percentile contrast stretching.
    """
    h, w, b = cube.shape

    # Check for precomputed RGB in dataset metadata (e.g. MUUFL)
    if meta and "rgb" in meta:
        rgb_meta = meta["rgb"]
        if isinstance(rgb_meta, np.ndarray) and rgb_meta.ndim == 3 and rgb_meta.shape[2] == 3:
            rgb_f = rgb_meta.astype(np.float32)
            if rgb_f.max() > 1.0:
                rgb_f = rgb_f / 255.0
            return np.clip(rgb_f, 0.0, 1.0)

    # Determine default band indices across the spectrum
    r_idx = red_band if red_band is not None else int(round(b * 0.70))
    g_idx = green_band if green_band is not None else int(round(b * 0.40))
    b_idx = blue_band if blue_band is not None else int(round(b * 0.15))

    r_idx = max(0, min(b - 1, r_idx))
    g_idx = max(0, min(b - 1, g_idx))
    b_idx = max(0, min(b - 1, b_idx))

    r_band = cube[:, :, r_idx].astype(np.float32)
    g_band = cube[:, :, g_idx].astype(np.float32)
    b_band = cube[:, :, b_idx].astype(np.float32)

    def _stretch(channel: np.ndarray) -> np.ndarray:
        p2, p98 = np.percentile(channel, (2.0, 98.0))
        if p98 > p2:
            stretched = (channel - p2) / (p98 - p2)
        else:
            stretched = np.zeros_like(channel)
        return np.clip(stretched, 0.0, 1.0)

    rgb = np.stack([_stretch(r_band), _stretch(g_band), _stretch(b_band)], axis=-1)
    return rgb.astype(np.float32)


def create_rgb_composite_figure(
    cube: np.ndarray,
    meta: Optional[Dict[str, Any]] = None,
    red_band: Optional[int] = None,
    green_band: Optional[int] = None,
    blue_band: Optional[int] = None,
    gt: Optional[np.ndarray] = None,
    show_ground_truth: bool = False,
    title: Optional[str] = "RGB Composite",
) -> go.Figure:
    """Create an interactive Plotly RGB figure with optional ground truth overlay."""
    rgb = synthesize_rgb_composite(
        cube, meta=meta, red_band=red_band, green_band=green_band, blue_band=blue_band
    )
    rgb_uint8 = (rgb * 255.0).astype(np.uint8)

    fig = go.Figure()
    fig.add_trace(go.Image(z=rgb_uint8))

    if show_ground_truth and gt is not None:
        tr, tc = np.where(gt > 0)
        if len(tr) > 0:
            fig.add_trace(
                go.Scatter(
                    x=tc,
                    y=tr,
                    mode="markers",
                    marker=dict(symbol="circle-open", size=8, color="#F43F5E", line=dict(width=2.5)),
                    name=f"Targets ({len(tr)} px)",
                    hovertemplate="<b>Tactical Target</b><br>Row: %{y}<br>Col: %{x}<extra></extra>",
                )
            )

    fig.update_layout(
        **get_scientific_plotly_layout(
            title=dict(text=f"<b>{title}</b>", x=0.02, y=0.96, font=dict(color="#FFFFFF", size=14)),
            xaxis=dict(title="Spatial Column (X)", showgrid=True, gridcolor="rgba(255,255,255,0.05)"),
            yaxis=dict(title="Spatial Row (Y)", autorange="reversed", showgrid=True, gridcolor="rgba(255,255,255,0.05)"),
            height=490,
            margin=dict(l=45, r=35, t=55, b=45),
        )
    )
    return fig


def create_single_band_figure(
    band_data: np.ndarray,
    title: str = "Original Band Image",
    colormap: str = "Viridis",
    selected_pixel: Optional[Tuple[int, int]] = None,
) -> go.Figure:
    """Render a single spatial band image in interactive Plotly."""
    fig = go.Figure()
    fig.add_trace(
        go.Heatmap(
            z=band_data,
            colorscale=colormap,
            colorbar=dict(
                title=dict(text="Reflectance / Digital Count", side="right"),
                thickness=14,
                len=0.9,
                tickfont=dict(size=10, family="JetBrains Mono"),
            ),
            hovertemplate=(
                "<b>Row (Y):</b> %{y}<br>"
                "<b>Col (X):</b> %{x}<br>"
                "<b>Spectral Value:</b> %{z:.4f}<extra></extra>"
            ),
            name="Band Image",
        )
    )
    if selected_pixel is not None:
        row, col = selected_pixel
        if 0 <= row < band_data.shape[0] and 0 <= col < band_data.shape[1]:
            fig.add_trace(
                go.Scatter(
                    x=[col],
                    y=[row],
                    mode="markers",
                    marker=dict(size=11, color="#00CFFF", symbol="x", line=dict(width=2)),
                    name="Selected pixel",
                    hovertemplate="Selected pixel<br>Row: %{y}<br>Column: %{x}<extra></extra>",
                )
            )
    fig.update_layout(
        **get_scientific_plotly_layout(
            title=dict(text=f"<b>{title}</b>", x=0.02, y=0.96, font=dict(color="#FFFFFF", size=14)),
            xaxis=dict(title="Spatial Column (X)", constrain="domain", showgrid=True, gridcolor="rgba(255,255,255,0.05)"),
            yaxis=dict(title="Spatial Row (Y)", autorange="reversed", scaleanchor="x", showgrid=True, gridcolor="rgba(255,255,255,0.05)"),
            margin=dict(l=45, r=45, t=55, b=45),
            height=490,
        )
    )
    return fig


def create_ground_truth_mask_figure(
    gt: np.ndarray,
    title: str = "Ground-Truth Reference Mask",
    selected_pixel: Optional[Tuple[int, int]] = None,
) -> go.Figure:
    """Render a binary ground truth mask with high contrast styling and target statistics."""
    fig = go.Figure()
    gt_binary = (gt > 0).astype(np.int32)
    custom_colorscale = [[0.0, "#0B132B"], [1.0, "#00F0FF"]]

    fig.add_trace(
        go.Heatmap(
            z=gt_binary,
            colorscale=custom_colorscale,
            showscale=False,
            hovertemplate=(
                "<b>Row (Y):</b> %{y}<br>"
                "<b>Col (X):</b> %{x}<br>"
                "<b>Class:</b> %{customdata}<extra></extra>"
            ),
            customdata=np.where(gt_binary > 0, "Tactical Target / Anomaly", "Confirmed Background"),
            name="Ground Truth Mask",
        )
    )

    tr, tc = np.where(gt_binary > 0)
    target_count = len(tr)
    total_pixels = gt.size
    prevalence = (target_count / max(total_pixels, 1)) * 100.0

    if target_count > 0:
        fig.add_trace(
            go.Scatter(
                x=tc,
                y=tr,
                mode="markers",
                marker=dict(symbol="circle-open", size=7, color="#F43F5E", line=dict(width=1.8)),
                name=f"Targets ({target_count:,} px / {prevalence:.2f}%)",
                hovertemplate="<b>Target Pixel</b><br>Row: %{y}<br>Col: %{x}<extra></extra>",
            )
        )

    if selected_pixel is not None:
        row, col = selected_pixel
        if 0 <= row < gt.shape[0] and 0 <= col < gt.shape[1]:
            fig.add_trace(
                go.Scatter(
                    x=[col],
                    y=[row],
                    mode="markers",
                    marker=dict(size=11, color="#F59E0B", symbol="x", line=dict(width=2)),
                    name="Selected pixel",
                    hovertemplate="Selected pixel<br>Row: %{y}<br>Column: %{x}<extra></extra>",
                )
            )

    fig.update_layout(
        **get_scientific_plotly_layout(
            title=dict(
                text=f"<b>{title}</b><br><span style='font-size:11px;color:#94A3B8;'>Targets: {target_count:,} px ({prevalence:.2f}% prevalence) · Background: {total_pixels - target_count:,} px</span>",
                x=0.02,
                y=0.96,
                font=dict(color="#FFFFFF", size=14),
            ),
            xaxis=dict(title="Spatial Column (X)", constrain="domain", showgrid=True, gridcolor="rgba(255,255,255,0.05)"),
            yaxis=dict(title="Spatial Row (Y)", autorange="reversed", scaleanchor="x", showgrid=True, gridcolor="rgba(255,255,255,0.05)"),
            margin=dict(l=45, r=45, t=65, b=45),
            height=490,
        )
    )
    return fig


def create_anomaly_heatmap_figure(
    score_map: np.ndarray,
    gt: Optional[np.ndarray] = None,
    title: str = "Anomaly Score Map",
    colormap: str = "Viridis",
    show_ground_truth: bool = False,
    selected_pixel: Optional[Tuple[int, int]] = None,
    zmin: Optional[float] = None,
    zmax: Optional[float] = None,
) -> go.Figure:
    """Create an interactive Plotly heatmap for pixel-level anomaly score inspection.

    Features:
        - Interactive zoom, pan, hover displaying (row, col, score).
        - Optional percentile-based display limits (zmin, zmax) preserving raw score data.
        - Optional ground truth target pixel overlay (red scatter markers).
        - Prominent non-probability disclaimer in subtitle.
    """
    fig = go.Figure()

    # Heatmap trace (Trace 0)
    fig.add_trace(
        go.Heatmap(
            z=score_map,
            colorscale=colormap,
            zmin=zmin,
            zmax=zmax,
            colorbar=dict(
                title=dict(text="Anomaly Score<br><span style='font-size:10px;'>Mahalanobis Dist</span>", side="right"),
                thickness=14,
                len=0.9,
                tickfont=dict(size=10, family="JetBrains Mono"),
            ),
            hovertemplate=(
                "<b>Row (Y):</b> %{y}<br>"
                "<b>Col (X):</b> %{x}<br>"
                "<b>Anomaly Score:</b> %{z:.4f}<extra></extra>"
            ),
            name="Anomaly Score Map",
        )
    )

    # Optional ground truth overlay (Trace 1)
    if show_ground_truth and gt is not None:
        target_rows, target_cols = np.where(gt > 0)
        if len(target_rows) > 0:
            fig.add_trace(
                go.Scatter(
                    x=target_cols,
                    y=target_rows,
                    mode="markers",
                    marker=dict(
                        symbol="circle-open",
                        size=8,
                        color="#F43F5E",
                        line=dict(width=2.2, color="#F43F5E"),
                    ),
                    name=f"Ground Truth Targets ({len(target_rows)} px)",
                    hovertemplate=(
                        "<b>Ground Truth Target</b><br>"
                        "Row: %{y}<br>Col: %{x}<extra></extra>"
                    ),
                )
            )

    if selected_pixel is not None:
        row, col = selected_pixel
        if 0 <= row < score_map.shape[0] and 0 <= col < score_map.shape[1]:
            fig.add_trace(
                go.Scatter(
                    x=[col],
                    y=[row],
                    mode="markers",
                    marker=dict(size=11, color="#00CFFF", symbol="x", line=dict(width=2)),
                    name="Selected pixel",
                    hovertemplate="Selected pixel<br>Row: %{y}<br>Column: %{x}<extra></extra>",
                )
            )

    fig.update_layout(
        **get_scientific_plotly_layout(
            title=dict(
                text=f"<b>{title}</b><br><span style='font-size:11px;color:#94A3B8;'>Scores represent Mahalanobis statistical distance; NOT calibrated target probability.</span>",
                x=0.02,
                y=0.96,
                font=dict(color="#FFFFFF", size=14),
            ),
            xaxis=dict(title="Spatial Column (X)", constrain="domain", showgrid=True, gridcolor="rgba(255,255,255,0.05)"),
            yaxis=dict(title="Spatial Row (Y)", autorange="reversed", scaleanchor="x", showgrid=True, gridcolor="rgba(255,255,255,0.05)"),
            margin=dict(l=45, r=45, t=70, b=45),
            height=520,
        )
    )
    return fig


def create_3d_hyperspectral_cube_figure(
    cube: np.ndarray,
    wavelengths: Optional[np.ndarray] = None,
    current_slice_band: Optional[int] = None,
    colormap: str = "Viridis",
    title: str = "3D Hyperspectral Data Cube Explorer",
    gt: Optional[np.ndarray] = None,
    show_grid: bool = True,
    layer_count: Optional[int] = None,
    exploded: bool = False,
    layer_separation: float = 0.0,
    layer_opacity: Optional[float] = None,
    selected_pixel: Optional[Tuple[int, int]] = None,
) -> go.Figure:
    """Render sampled measured spectral-band planes in interactive Plotly WebGL.

    Exploded mode offsets planes for inspection and is a visualization transform,
    not a change to the measured spectral spacing.
    """
    h, w, b = cube.shape

    if layer_count is not None:
        if not isinstance(layer_count, int) or not 1 <= layer_count <= min(b, 16):
            raise ValueError(f"layer_count must be between 1 and min(number of bands, 16), got {layer_count}")
        base_bands = np.linspace(0, b - 1, layer_count, dtype=int).tolist()
    else:
        base_bands = [
            int(round(b * 0.05)),
            int(round(b * 0.35)),
            int(round(b * 0.65)),
            int(round(b * 0.92)),
        ]
    if current_slice_band is not None and 0 <= current_slice_band < b:
        if current_slice_band not in base_bands:
            base_bands.append(current_slice_band)
    base_bands = sorted(list(set(base_bands)))
    if not np.isfinite(layer_separation) or layer_separation < 0:
        raise ValueError("layer_separation must be a finite, non-negative number")
    if layer_opacity is not None and (
        not np.isfinite(layer_opacity) or not 0.0 < layer_opacity <= 1.0
    ):
        raise ValueError("layer_opacity must be in the interval (0, 1]")

    fig = go.Figure()

    # Downsample spatially for ultra-smooth 60fps WebGL rendering if cube is large
    step = 1 if max(h, w) <= 120 else 2
    y_coords = np.arange(0, h, step)
    x_coords = np.arange(0, w, step)
    xx, yy = np.meshgrid(x_coords, y_coords)

    max_z = float(b - 1)
    for idx, band_idx in enumerate(base_bands):
        band_data = cube[::step, ::step, band_idx].astype(np.float32)
        # Contrast stretch for visualization
        p2, p98 = np.percentile(band_data, (2.0, 98.0))
        if p98 > p2:
            band_norm = (band_data - p2) / (p98 - p2)
        else:
            band_norm = np.zeros_like(band_data)

        # Depth coordinate on Z-axis
        z_level = float(band_idx)
        if exploded:
            z_level += idx * layer_separation
        max_z = max(max_z, z_level)
        zz = np.full_like(xx, fill_value=z_level, dtype=np.float32)

        wl_label = f"{wavelengths[band_idx]:.1f} nm" if wavelengths is not None and len(wavelengths) > band_idx else f"Band {band_idx + 1}"
        is_active = (current_slice_band is not None and band_idx == current_slice_band)

        fig.add_trace(
            go.Surface(
                x=xx,
                y=yy,
                z=zz,
                surfacecolor=band_norm,
                colorscale="Plasma" if is_active else colormap,
                cmin=0.0,
                cmax=1.0,
                opacity=(
                    layer_opacity
                    if layer_opacity is not None
                    else (0.92 if is_active else 0.82)
                ),
                showscale=(idx == 0),
                colorbar=dict(
                    title=dict(text="Normalized<br>Reflectance", side="right"),
                    thickness=12,
                    len=0.75,
                ) if idx == 0 else None,
                name=f"{wl_label} (Z={band_idx})",
                hovertemplate=(
                    f"<b>{wl_label}</b><br>"
                    "Col (X): %{x}<br>"
                    "Row (Y): %{y}<br>"
                    "Band Index (Z): " + str(band_idx) + "<br>"
                    "Radiance: %{surfacecolor:.3f}<extra></extra>"
                ),
            )
        )

    if selected_pixel is not None:
        selected_row, selected_col = selected_pixel
        if 0 <= selected_row < h and 0 <= selected_col < w:
            selected_band = current_slice_band if current_slice_band is not None else base_bands[0]
            selected_z = float(selected_band)
            if exploded and selected_band in base_bands:
                selected_z += base_bands.index(selected_band) * layer_separation
            fig.add_trace(
                go.Scatter3d(
                    x=[selected_col],
                    y=[selected_row],
                    z=[selected_z],
                    mode="markers",
                    marker=dict(size=7, color="#00CFFF", symbol="circle", line=dict(color="#FFFFFF", width=1.5)),
                    name="Selected pixel",
                    hovertemplate="Selected pixel<br>Column: %{x}<br>Row: %{y}<extra></extra>",
                )
            )

    # Add 3D wireframe bounding box edges
    box_x = [0, w - 1, w - 1, 0, 0, 0, w - 1, w - 1, 0, 0, 0, 0, w - 1, w - 1, w - 1, w - 1]
    box_y = [0, 0, h - 1, h - 1, 0, 0, 0, h - 1, h - 1, 0, h - 1, h - 1, h - 1, h - 1, 0, 0]
    box_z = [0, 0, 0, 0, 0, max_z, max_z, max_z, max_z, max_z, 0, max_z, max_z, 0, 0, max_z]

    fig.add_trace(
        go.Scatter3d(
            x=box_x,
            y=box_y,
            z=box_z,
            mode="lines",
            line=dict(color="rgba(56, 189, 248, 0.4)", width=2),
            hoverinfo="none",
            showlegend=False,
            name="Cube Bounds",
        )
    )

    if gt is not None:
        target_rows, target_cols = np.where(gt > 0)
        if target_rows.size > 1500:
            selected = np.linspace(0, target_rows.size - 1, 1500, dtype=int)
            target_rows = target_rows[selected]
            target_cols = target_cols[selected]
        if target_rows.size:
            fig.add_trace(
                go.Scatter3d(
                    x=target_cols,
                    y=target_rows,
                    z=np.full(target_rows.shape, max_z if exploded else b - 1),
                    mode="markers",
                    marker=dict(
                        symbol="diamond",
                        size=4,
                        color="#FF5364",
                        line=dict(color="#EAF6FF", width=1),
                    ),
                    name="Ground-truth reference",
                    hovertemplate=(
                        "<b>Dataset reference label</b><br>"
                        "Column: %{x}<br>Row: %{y}<extra></extra>"
                    ),
                )
            )

    cube_description = "Sampled measured spectral-band planes · drag to orbit · scroll to zoom"
    if exploded:
        cube_description += " · exploded spacing is illustrative"
    if step > 1:
        cube_description += f" · spatial sampling step {step}"

    fig.update_layout(
        **get_scientific_plotly_layout(
            title=dict(
                text=f"<b>{title}</b><br><span style='font-size:11px;color:#94A3B8;'>{cube_description}</span>",
                x=0.02,
                y=0.96,
                font=dict(color="#FFFFFF", size=14),
            ),
            scene=dict(
                xaxis=dict(title="Spatial X (Cols)", backgroundcolor="#020914", gridcolor="rgba(0,207,255,0.12)", showbackground=True, showgrid=show_grid),
                yaxis=dict(title="Spatial Y (Rows)", backgroundcolor="#020914", gridcolor="rgba(0,207,255,0.12)", showbackground=True, showgrid=show_grid),
                zaxis=dict(title="Spectral Channel (Z)", backgroundcolor="#020914", gridcolor="rgba(0,207,255,0.12)", showbackground=True, showgrid=show_grid),
                camera=dict(
                    eye=dict(x=-1.5, y=-1.6, z=1.3),
                    up=dict(x=0, y=0, z=1),
                ),
                aspectratio=dict(x=1.0, y=1.0, z=0.8),
            ),
            height=460,
            margin=dict(l=20, r=20, t=60, b=20),
        )
    )
    return fig


def create_3d_anomaly_topography_figure(
    score_map: np.ndarray,
    gt: Optional[np.ndarray] = None,
    threshold: Optional[float] = None,
    colormap: str = "Viridis",
    title: str = "3D Anomaly Score Topography",
    height_scale: float = 1.0,
    selected_pixel: Optional[Tuple[int, int]] = None,
) -> go.Figure:
    """Render anomaly scores as an interactive, downsampled 3D visualization.

    Height is the anomaly score multiplied by ``height_scale``; it is not physical elevation.
    """
    if not isinstance(score_map, np.ndarray) or score_map.ndim != 2:
        raise ValueError("score_map must be a two-dimensional numpy array")
    if not np.isfinite(height_scale) or height_scale <= 0:
        raise ValueError("height_scale must be a finite, positive number")
    h, w = score_map.shape
    step = 1 if max(h, w) <= 120 else 2

    x = np.arange(0, w, step)
    y = np.arange(0, h, step)
    xx, yy = np.meshgrid(x, y)
    zz = score_map[::step, ::step].astype(np.float64)
    scaled_z = zz * height_scale

    fig = go.Figure()

    # 3D Topographic Surface Mesh
    fig.add_trace(
        go.Surface(
            x=xx,
            y=yy,
            z=scaled_z,
            surfacecolor=zz,
            colorscale=colormap,
            contours=dict(
                z=dict(show=True, usecolormap=True, highlightcolor="#00F0FF", project_z=True)
            ),
            lighting=dict(ambient=0.55, diffuse=0.85, roughness=0.45, specular=0.6, fresnel=0.2),
            colorbar=dict(
                title=dict(text="Anomaly<br>Score", side="right"),
                thickness=14,
                len=0.8,
                tickfont=dict(size=10, family="JetBrains Mono"),
            ),
            name="Anomaly Surface",
            customdata=zz,
            hovertemplate=(
                "<b>Anomaly Score Surface</b><br>"
                "Col (X): %{x}<br>"
                "Row (Y): %{y}<br>"
                "Anomaly score: %{customdata:.4g}<br>"
                "Displayed height: %{z:.4g}<extra></extra>"
            ),
        )
    )

    # Optional 3D Decision Threshold Cut Plane
    if threshold is not None and np.isfinite(threshold):
        z_plane = np.full_like(zz, fill_value=threshold * height_scale)
        fig.add_trace(
            go.Surface(
                x=xx,
                y=yy,
                z=z_plane,
                colorscale=[[0, "rgba(244, 63, 94, 0.35)"], [1, "rgba(244, 63, 94, 0.35)"]],
                showscale=False,
                name=f"Threshold Plane ({threshold:.2f})",
                hoverinfo="none",
                opacity=0.35,
            )
        )

    # Optional Ground Truth Target Markers floating on top of the 3D peaks
    if gt is not None:
        tr, tc = np.where(gt > 0)
        if tr.size > 1500:
            selected = np.linspace(0, tr.size - 1, 1500, dtype=int)
            tr = tr[selected]
            tc = tc[selected]
        if len(tr) > 0:
            gt_scores = score_map[tr, tc]
            tz = (gt_scores + (score_map.max() - score_map.min()) * 0.03) * height_scale
            fig.add_trace(
                go.Scatter3d(
                    x=tc,
                    y=tr,
                    z=tz,
                    customdata=gt_scores,
                    mode="markers",
                    marker=dict(
                        symbol="diamond",
                        size=5,
                        color="#F43F5E",
                        line=dict(color="#FFFFFF", width=1.5),
                    ),
                    name=f"Target Coordinates ({len(tr)} px)",
                    hovertemplate="<b>Ground-truth reference</b><br>Col: %{x}<br>Row: %{y}<br>Score: %{customdata:.4g}<extra></extra>",
                )
            )

    if selected_pixel is not None:
        selected_row, selected_col = selected_pixel
        if 0 <= selected_row < h and 0 <= selected_col < w:
            selected_score = float(score_map[selected_row, selected_col])
            fig.add_trace(
                go.Scatter3d(
                    x=[selected_col],
                    y=[selected_row],
                    z=[selected_score * height_scale],
                    mode="markers",
                    marker=dict(size=8, color="#00CFFF", symbol="circle", line=dict(color="#FFFFFF", width=2)),
                    name="Selected pixel",
                    customdata=[selected_score],
                    hovertemplate="Selected pixel (%{y}, %{x})<br>Score: %{customdata:.4g}<extra></extra>",
                )
            )

    topography_description = (
        f"Displayed height = anomaly score × {height_scale:g}; not physical elevation."
    )
    if step > 1:
        topography_description += f" Spatial sampling step {step}."

    fig.update_layout(
        **get_scientific_plotly_layout(
            title=dict(
                text=f"<b>{title}</b><br><span style='font-size:11px;color:#94A3B8;'>{topography_description}</span>",
                x=0.02,
                y=0.96,
                font=dict(color="#FFFFFF", size=14),
            ),
            scene=dict(
                xaxis=dict(title="Spatial Column (X)", backgroundcolor="#080D18", gridcolor="rgba(255,255,255,0.06)", showbackground=True),
                yaxis=dict(title="Spatial Row (Y)", backgroundcolor="#080D18", gridcolor="rgba(255,255,255,0.06)", showbackground=True),
                zaxis=dict(title=f"Scaled anomaly score (×{height_scale:g})", backgroundcolor="#080D18", gridcolor="rgba(255,255,255,0.06)", showbackground=True),
                camera=dict(
                    eye=dict(x=-1.6, y=-1.6, z=1.2),
                ),
            ),
            height=580,
            margin=dict(l=20, r=20, t=60, b=20),
        )
    )
    return fig


def create_spectral_slice_figure(
    cube: np.ndarray,
    band_index: int,
    wavelengths: Optional[np.ndarray] = None,
    title: Optional[str] = None,
    selected_pixel: Optional[Tuple[int, int]] = None,
) -> go.Figure:
    """Create an interactive Plotly heatmap of a single spectral channel."""
    h, w, b = cube.shape
    band_idx = max(0, min(b - 1, int(band_index)))
    band_data = cube[:, :, band_idx]

    wl_str = f" ({wavelengths[band_idx]:.1f} nm)" if wavelengths is not None and len(wavelengths) > band_idx else ""
    fig_title = title or f"Spectral Band {band_idx + 1} / {b}{wl_str}"

    fig = go.Figure(
        data=go.Heatmap(
            z=band_data,
            colorscale="Inferno",
            colorbar=dict(
                title=dict(text="Radiance /<br>Reflectance", side="right"),
                thickness=14,
                tickfont=dict(size=10, family="JetBrains Mono"),
            ),
            hovertemplate="<b>Row (Y):</b> %{y}<br><b>Col (X):</b> %{x}<br><b>Value:</b> %{z:.4f}<extra></extra>",
        )
    )
    if selected_pixel is not None:
        row, col = selected_pixel
        if 0 <= row < h and 0 <= col < w:
            fig.add_trace(
                go.Scatter(
                    x=[col],
                    y=[row],
                    mode="markers",
                    marker=dict(size=10, color="#00CFFF", symbol="x", line=dict(width=2)),
                    name="Selected pixel",
                    hovertemplate="Selected pixel<br>Column: %{x}<br>Row: %{y}<extra></extra>",
                )
            )
    fig.update_layout(
        **get_scientific_plotly_layout(
            title=dict(text=f"<b>{fig_title}</b>", x=0.02, y=0.96, font=dict(color="#FFFFFF", size=14)),
            xaxis=dict(title="Spatial Column (X)", showgrid=True, gridcolor="rgba(255,255,255,0.05)"),
            yaxis=dict(title="Spatial Row (Y)", autorange="reversed", scaleanchor="x", showgrid=True, gridcolor="rgba(255,255,255,0.05)"),
            margin=dict(l=45, r=45, t=60, b=45),
            height=490,
        )
    )
    return fig


def create_spectral_signature_figure(
    cube: np.ndarray,
    row: int,
    col: int,
    wavelengths: Optional[np.ndarray] = None,
    target_mask: Optional[np.ndarray] = None,
) -> go.Figure:
    """Plot the continuous spectral signature curve at a specific pixel."""
    h, w, b = cube.shape
    r = max(0, min(h - 1, int(row)))
    c = max(0, min(w - 1, int(col)))
    spectrum = cube[r, c, :].astype(np.float64)

    is_target = target_mask[r, c] > 0 if target_mask is not None else False
    target_tag = " [TARGET ANOMALY PIXEL]" if is_target else " [BACKGROUND PIXEL]"

    if wavelengths is not None and len(wavelengths) == b:
        x_vals = wavelengths
        x_label = "Wavelength (nm)"
    else:
        x_vals = np.arange(1, b + 1)
        x_label = "Spectral Band Index"

    fig = go.Figure()

    trace_color = "#F43F5E" if is_target else "#38BDF8"
    fig.add_trace(
        go.Scatter(
            x=x_vals,
            y=spectrum,
            mode="lines+markers",
            marker=dict(size=4, color=trace_color),
            line=dict(color=trace_color, width=2.2),
            name=f"Pixel ({r}, {c})",
            hovertemplate="<b>%{x}</b>: %{y:.4f}<extra></extra>",
        )
    )

    fig.update_layout(
        **get_scientific_plotly_layout(
            title=dict(
                text=f"<b>Spectral Signature at Pixel ({r}, {c}){target_tag}</b>",
                x=0.02,
                font=dict(color="#FFFFFF", size=13),
            ),
            xaxis=dict(title=x_label, showgrid=True, gridcolor="rgba(255,255,255,0.06)"),
            yaxis=dict(title="Reflectance / Radiance", showgrid=True, gridcolor="rgba(255,255,255,0.06)"),
            margin=dict(l=45, r=35, t=55, b=45),
            height=360,
        )
    )
    return fig


def create_multi_pixel_spectral_figure(
    cube: np.ndarray,
    pixels: List[Tuple[int, int, str]],
    wavelengths: Optional[np.ndarray] = None,
    target_mask: Optional[np.ndarray] = None,
    title: str = "Multi-Pixel Spectral Signatures Comparison",
) -> go.Figure:
    """Plot multiple continuous spectral signatures on a shared interactive chart."""
    h, w, b = cube.shape
    if wavelengths is not None and len(wavelengths) == b:
        x_vals = wavelengths
        x_label = "Wavelength (nm)"
    else:
        x_vals = np.arange(1, b + 1)
        x_label = "Spectral Band Index"

    palette = [
        "#00F0FF", "#F43F5E", "#10B981", "#F59E0B",
        "#A855F7", "#EC4899", "#38BDF8", "#F97316",
        "#6366F1", "#14B8A6",
    ]

    fig = go.Figure()

    for i, (r, c, tag) in enumerate(pixels):
        r_clamped = max(0, min(h - 1, int(r)))
        c_clamped = max(0, min(w - 1, int(c)))
        spectrum = cube[r_clamped, c_clamped, :].astype(np.float64)

        color = palette[i % len(palette)]
        is_target = target_mask[r_clamped, c_clamped] > 0 if target_mask is not None else None
        gt_suffix = ""
        if is_target is True:
            gt_suffix = " [Target]"
        elif is_target is False:
            gt_suffix = " [Background]"

        display_name = f"{tag}: ({r_clamped}, {c_clamped}){gt_suffix}" if tag else f"Pixel ({r_clamped}, {c_clamped}){gt_suffix}"

        fig.add_trace(
            go.Scatter(
                x=x_vals,
                y=spectrum,
                mode="lines+markers",
                marker=dict(size=4, color=color),
                line=dict(color=color, width=2.0),
                name=display_name,
                hovertemplate=f"<b>{display_name}</b><br>{x_label}: %{{x}}<br>Value: %{{y:.4f}}<extra></extra>",
            )
        )

    fig.update_layout(
        **get_scientific_plotly_layout(
            title=dict(
                text=f"<b>{title}</b>",
                x=0.02,
                font=dict(color="#FFFFFF", size=13),
            ),
            xaxis=dict(title=x_label, showgrid=True, gridcolor="rgba(255,255,255,0.06)"),
            yaxis=dict(title="Reflectance / Radiance", showgrid=True, gridcolor="rgba(255,255,255,0.06)"),
            margin=dict(l=45, r=35, t=55, b=45),
            height=380,
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1,
                font=dict(size=10, color="#CBD5E1"),
            ),
        )
    )
    return fig


def create_roc_pr_curves_figure(gt: np.ndarray, scores: np.ndarray) -> go.Figure:
    """Generate side-by-side ROC and Precision-Recall curves using Plotly with AUC shading."""
    y_true = (gt.ravel() > 0).astype(int)
    y_scores = scores.ravel()

    fpr, tpr, _ = roc_curve(y_true, y_scores)
    precision, recall, _ = precision_recall_curve(y_true, y_scores)

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("<b>Receiver Operating Characteristic (ROC)</b>", "<b>Precision-Recall (PR) Curve</b>"),
    )

    # ROC Curve with emerald shading
    fig.add_trace(
        go.Scatter(
            x=fpr,
            y=tpr,
            mode="lines",
            name="ROC Curve",
            line=dict(color="#10B981", width=2.4),
            fill="tozeroy",
            fillcolor="rgba(16, 185, 129, 0.12)",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            name="Random Guess",
            line=dict(dash="dash", color="rgba(255,255,255,0.3)"),
        ),
        row=1,
        col=1,
    )

    # PR Curve with amber shading
    fig.add_trace(
        go.Scatter(
            x=recall,
            y=precision,
            mode="lines",
            name="PR Curve",
            line=dict(color="#F59E0B", width=2.4),
            fill="tozeroy",
            fillcolor="rgba(245, 158, 11, 0.12)",
        ),
        row=1,
        col=2,
    )

    fig.update_xaxes(title_text="False Positive Rate (FPR)", row=1, col=1, gridcolor="rgba(255,255,255,0.06)")
    fig.update_yaxes(title_text="True Positive Rate (TPR)", row=1, col=1, gridcolor="rgba(255,255,255,0.06)")
    fig.update_xaxes(title_text="Recall (TPR)", row=1, col=2, gridcolor="rgba(255,255,255,0.06)")
    fig.update_yaxes(title_text="Precision (PPV)", row=1, col=2, gridcolor="rgba(255,255,255,0.06)")

    fig.update_layout(
        **get_scientific_plotly_layout(
            height=390,
            margin=dict(l=45, r=35, t=55, b=45),
            showlegend=False,
        )
    )
    return fig


def create_detection_overlay_figure(
    cube: np.ndarray,
    scores: np.ndarray,
    threshold: float,
    gt: Optional[np.ndarray] = None,
    meta: Optional[Dict[str, Any]] = None,
    red_band: Optional[int] = None,
    green_band: Optional[int] = None,
    blue_band: Optional[int] = None,
    alpha: float = 0.55,
    title: str = "Detection Overlay & Ground-Truth Alignment",
    selected_pixel: Optional[Tuple[int, int]] = None,
) -> go.Figure:
    """Render detection overlay on top of RGB composite with ground-truth confusion coloring."""
    rgb = synthesize_rgb_composite(cube, meta=meta, red_band=red_band, green_band=green_band, blue_band=blue_band)
    h, w = scores.shape
    base_rgb = (rgb * 255.0).astype(np.uint8)

    y_pred = (scores >= threshold).astype(np.int32)
    fig = go.Figure()

    if gt is not None and gt.shape == (h, w):
        gt_bin = (gt > 0).astype(np.int32)
        tp_mask = (y_pred == 1) & (gt_bin == 1)
        fp_mask = (y_pred == 1) & (gt_bin == 0)
        fn_mask = (y_pred == 0) & (gt_bin == 1)

        blended = base_rgb.copy().astype(np.float32)
        mask_any = tp_mask | fp_mask | fn_mask
        color_map = np.zeros((h, w, 3), dtype=np.float32)
        color_map[tp_mask] = [53, 224, 161]   # True Positives: Green
        color_map[fp_mask] = [255, 97, 120]   # False Alarms: Red
        color_map[fn_mask] = [255, 190, 85]   # Misses: Amber/Yellow

        blended[mask_any] = (1.0 - alpha) * base_rgb[mask_any] + alpha * color_map[mask_any]
        blended = np.clip(blended, 0, 255).astype(np.uint8)

        fig.add_trace(go.Image(z=blended))

        tp_count = int(np.sum(tp_mask))
        fp_count = int(np.sum(fp_mask))
        fn_count = int(np.sum(fn_mask))

        fig.add_trace(go.Scatter(
            x=[None], y=[None], mode="markers",
            marker=dict(size=10, color="#35E0A1", symbol="square"),
            name=f"True Positives ({tp_count:,} px)",
            showlegend=True,
        ))
        fig.add_trace(go.Scatter(
            x=[None], y=[None], mode="markers",
            marker=dict(size=10, color="#FF6178", symbol="square"),
            name=f"False Alarms ({fp_count:,} px)",
            showlegend=True,
        ))
        fig.add_trace(go.Scatter(
            x=[None], y=[None], mode="markers",
            marker=dict(size=10, color="#FFBE55", symbol="square"),
            name=f"Misses ({fn_count:,} px)",
            showlegend=True,
        ))
    else:
        det_mask = (y_pred == 1)
        blended = base_rgb.copy().astype(np.float32)
        blended[det_mask] = (1.0 - alpha) * base_rgb[det_mask] + alpha * np.array([0, 229, 255], dtype=np.float32)
        blended = np.clip(blended, 0, 255).astype(np.uint8)
        fig.add_trace(go.Image(z=blended))

        det_count = int(np.sum(det_mask))
        fig.add_trace(go.Scatter(
            x=[None], y=[None], mode="markers",
            marker=dict(size=10, color="#00E5FF", symbol="square"),
            name=f"Detected Anomalies ({det_count:,} px)",
            showlegend=True,
        ))

    if selected_pixel is not None:
        row, col = selected_pixel
        if 0 <= row < h and 0 <= col < w:
            fig.add_trace(
                go.Scatter(
                    x=[col],
                    y=[row],
                    mode="markers",
                    marker=dict(size=11, color="#00E5FF", symbol="x", line=dict(width=2)),
                    name="Selected pixel",
                    hovertemplate="Selected pixel<br>Row: %{y}<br>Column: %{x}<extra></extra>",
                )
            )

    fig.update_layout(
        **get_scientific_plotly_layout(
            title=dict(
                text=f"<b>{title}</b>",
                x=0.02,
                y=0.96,
                font=dict(color="#FFFFFF", size=14),
            ),
            xaxis=dict(title="Spatial Column (X)", showgrid=True, gridcolor="rgba(255,255,255,0.05)"),
            yaxis=dict(title="Spatial Row (Y)", autorange="reversed", showgrid=True, gridcolor="rgba(255,255,255,0.05)"),
            height=490,
            margin=dict(l=45, r=35, t=55, b=45),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1,
                font=dict(color="#F2F7FF", size=10, family="JetBrains Mono"),
            ),
        )
    )
    return fig


def create_binary_mask_figure(
    binary_mask: np.ndarray,
    title: str = "Binary Detection Mask",
    threshold: float = 0.0,
    selected_pixel: Optional[Tuple[int, int]] = None,
) -> go.Figure:
    """Render binary detection mask at operational threshold."""
    fig = go.Figure()
    custom_colorscale = [[0.0, "#030914"], [1.0, "#00E5FF"]]
    fig.add_trace(
        go.Heatmap(
            z=binary_mask,
            colorscale=custom_colorscale,
            showscale=False,
            hovertemplate=(
                "<b>Row:</b> %{y}<br>"
                "<b>Col:</b> %{x}<br>"
                "<b>Prediction:</b> %{customdata}<extra></extra>"
            ),
            customdata=np.where(binary_mask > 0, "Anomaly / Target", "Background"),
            name="Binary Mask",
        )
    )
    if selected_pixel is not None:
        row, col = selected_pixel
        if 0 <= row < binary_mask.shape[0] and 0 <= col < binary_mask.shape[1]:
            fig.add_trace(
                go.Scatter(
                    x=[col],
                    y=[row],
                    mode="markers",
                    marker=dict(size=11, color="#FF6178", symbol="x", line=dict(width=2)),
                    name="Selected pixel",
                    hovertemplate="Selected pixel<br>Row: %{y}<br>Column: %{x}<extra></extra>",
                )
            )
    det_px = int(np.sum(binary_mask > 0))
    tot_px = binary_mask.size
    det_pct = (det_px / max(tot_px, 1)) * 100.0
    fig.update_layout(
        **get_scientific_plotly_layout(
            title=dict(
                text=f"<b>{title}</b><br><span style='font-size:11px;color:#94A3B8;'>Threshold: {threshold:.4f} · Detected: {det_px:,} px ({det_pct:.2f}% of scene)</span>",
                x=0.02,
                y=0.96,
                font=dict(color="#FFFFFF", size=14),
            ),
            xaxis=dict(title="Spatial Column (X)", constrain="domain", showgrid=True, gridcolor="rgba(255,255,255,0.05)"),
            yaxis=dict(title="Spatial Row (Y)", autorange="reversed", scaleanchor="x", showgrid=True, gridcolor="rgba(255,255,255,0.05)"),
            margin=dict(l=45, r=45, t=65, b=45),
            height=490,
        )
    )
    return fig


def create_before_after_comparison_images(
    cube: np.ndarray,
    scores: np.ndarray,
    threshold: float,
    meta: Optional[Dict[str, Any]] = None,
    gt: Optional[np.ndarray] = None,
    alpha: float = 0.55,
) -> Tuple[Any, Any]:
    """Generate PIL Images for before-and-after slider comparison."""
    from PIL import Image
    rgb = synthesize_rgb_composite(cube, meta=meta)
    h, w = scores.shape
    base_rgb = (rgb * 255.0).astype(np.uint8)

    y_pred = (scores >= threshold).astype(np.int32)
    blended = base_rgb.copy().astype(np.float32)

    if gt is not None and gt.shape == (h, w):
        gt_bin = (gt > 0).astype(np.int32)
        tp_mask = (y_pred == 1) & (gt_bin == 1)
        fp_mask = (y_pred == 1) & (gt_bin == 0)
        fn_mask = (y_pred == 0) & (gt_bin == 1)

        mask_any = tp_mask | fp_mask | fn_mask
        color_map = np.zeros((h, w, 3), dtype=np.float32)
        color_map[tp_mask] = [53, 224, 161]
        color_map[fp_mask] = [255, 97, 120]
        color_map[fn_mask] = [255, 190, 85]
        blended[mask_any] = (1.0 - alpha) * base_rgb[mask_any] + alpha * color_map[mask_any]
    else:
        det_mask = (y_pred == 1)
        blended[det_mask] = (1.0 - alpha) * base_rgb[det_mask] + alpha * np.array([0, 229, 255], dtype=np.float32)

    blended = np.clip(blended, 0, 255).astype(np.uint8)
    return Image.fromarray(base_rgb), Image.fromarray(blended)


def render_metrics_cards(metrics: Dict[str, Any]) -> None:
    """Render structured Streamlit metric cards with tooltips and HUD aesthetics."""
    auc_val = metrics.get("auc_roc")
    pr_val = metrics.get("auc_pr")
    f1_val = metrics.get("f1_max")
    fpr_val = metrics.get("fpr_at_95_tpr")
    fa_val = metrics.get("false_alarms_per_mp")
    if fa_val is None:
        fa_val = metrics.get("false_alarms_per_megapixel")

    cards = [
        {
            "label": "ROC-AUC",
            "value": f"{auc_val:.4f}" if auc_val is not None else "N/A",
            "caption": "Area Under ROC Curve [0, 1]",
            "help": "Area Under Receiver Operating Characteristic Curve. Higher is better.",
        },
        {
            "label": "PR-AUC",
            "value": f"{pr_val:.4f}" if pr_val is not None else "N/A",
            "caption": "Precision-Recall Integral",
            "help": "Area Under Precision-Recall Curve. Accounts for extreme target sparsity.",
        },
        {
            "label": "Max F1",
            "value": f"{f1_val:.4f}" if f1_val is not None else "N/A",
            "caption": "Optimal Harmonic Mean",
            "help": "Post-hoc optimal harmonic mean of Precision and Recall across all thresholds.",
        },
        {
            "label": "FPR @ 95% TPR",
            "value": f"{fpr_val:.4f}" if fpr_val is not None else "N/A",
            "caption": "False Alarm Probability",
            "help": "False positive rate required to achieve 95% target recall. Lower is better.",
        },
        {
            "label": "FA / Megapixel",
            "value": f"{fa_val:,.1f}" if fa_val is not None else "N/A",
            "caption": "Density per 10⁶ Pixels",
            "help": "False alarm pixel count per 1,000,000 spatial pixels at 95% TPR. Lower is better.",
        },
    ]
    render_hud_metric_cards(cards)


def render_scientific_disclaimer() -> None:
    """Render a persistent scientific and defense application disclaimer banner."""
    html = textwrap.dedent(
        """<div class="disclaimer-box">
<strong>⚠️ Tactical & Research Disclaimer:</strong>
Hyperspectral anomaly detection scores represent unsupervised Mahalanobis statistical distance from
the estimated background covariance ($r(x) = (x-\\mu)^T \\Sigma^{-1} (x-\\mu)$); they do <strong>not</strong>
indicate verified tactical targets or calibrated probability of detection. Decision thresholds ($F_{1,\\max}$)
require empirical sensor calibration for operational deployment without ground truth.
</div>"""
    )
    st.markdown(html, unsafe_allow_html=True)
