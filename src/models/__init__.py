"""Hyperspectral anomaly detection models and utilities."""

from src.models.base import BaseDetector
from src.models.pca_residual import PCAResidualDetector
from src.models.rx import RXDetector
from src.models.rx_compat import local_rx, rx_scores
from src.models.autoencoder import SpectralAutoencoder
from src.models.fusion import fuse_scores, normalize_scores
from src.models.spatial import spatial_residual
from src.models.threshold import apply_threshold, calibrate_threshold, otsu_threshold

__all__ = [
    "BaseDetector",
    "PCAResidualDetector",
    "RXDetector",
    "SpectralAutoencoder",
    "local_rx",
    "rx_scores",
    "fuse_scores",
    "normalize_scores",
    "spatial_residual",
    "apply_threshold",
    "calibrate_threshold",
    "otsu_threshold",
]