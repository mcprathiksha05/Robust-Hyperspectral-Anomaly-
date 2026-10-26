"""Hyperspectral diagnostic, ablation, and testing framework (Stage 3).

Features:
    - Feature 4: Ablation studies (ablation.py)
    - Feature 5: Cross-scene testing & generalization (cross_scene.py)
    - Feature 6: Sensitivity analysis & parameter sweeps (sensitivity.py)
    - Feature 7: Stability testing & multi-seed audits (stability.py)
    - Feature 8: Uncertainty estimation & sensitivity maps (uncertainty.py)
"""

from src.analysis.ablation import (
    apply_normalization,
    apply_band_subsampling,
    run_ablation_study,
    build_default_ablation_suite,
)
from src.analysis.cross_scene import (
    evaluate_within_scene_benchmark,
    evaluate_cross_scene_transfer,
    evaluate_cross_scene_matrix,
)
from src.analysis.sensitivity import (
    run_parameter_sweep,
    sweep_noise_snr,
    sweep_band_dropout,
    sweep_clutter_abundance,
    sweep_detector_regularization,
    sweep_decision_thresholds,
)
from src.analysis.stability import (
    compute_stability_summary,
    evaluate_corruption_stability,
    evaluate_detector_stability,
)
from src.analysis.uncertainty import (
    estimate_perturbation_uncertainty,
)

__all__ = [
    # Feature 4: Ablation
    "apply_normalization",
    "apply_band_subsampling",
    "run_ablation_study",
    "build_default_ablation_suite",
    # Feature 5: Cross-Scene
    "evaluate_within_scene_benchmark",
    "evaluate_cross_scene_transfer",
    "evaluate_cross_scene_matrix",
    # Feature 6: Sensitivity
    "run_parameter_sweep",
    "sweep_noise_snr",
    "sweep_band_dropout",
    "sweep_clutter_abundance",
    "sweep_detector_regularization",
    "sweep_decision_thresholds",
    # Feature 7: Stability
    "compute_stability_summary",
    "evaluate_corruption_stability",
    "evaluate_detector_stability",
    # Feature 8: Uncertainty
    "estimate_perturbation_uncertainty",
]
