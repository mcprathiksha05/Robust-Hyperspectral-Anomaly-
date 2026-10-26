"""Hyperspectral robustness perturbation engine.

Features:
    - Feature 1: Noise robustness testing (Gaussian, Poisson, stripe, dead, saturated, salt-and-pepper)
    - Feature 2: Clutter robustness testing (Spectral mixing, localized patches, shadow, occlusion)
    - Feature 3: Spectral band corruption (Random band dropout, selected masking, localized noise, absorption)
"""

from src.robustness.noise import (
    add_gaussian_noise,
    add_poisson_noise,
    add_stripe_noise,
    add_dead_pixels,
    add_saturated_pixels,
    add_salt_and_pepper_noise,
)
from src.robustness.clutter import (
    inject_spectral_clutter,
    inject_localized_clutter_patch,
    apply_shadow_attenuation,
    apply_occlusion_simulation,
    get_muufl_occlusion_subsets,
)
from src.robustness.bands import (
    drop_random_bands,
    mask_selected_bands,
    add_localized_band_noise,
    mask_wavelength_ranges,
)

__all__ = [
    # Feature 1: Noise
    "add_gaussian_noise",
    "add_poisson_noise",
    "add_stripe_noise",
    "add_dead_pixels",
    "add_saturated_pixels",
    "add_salt_and_pepper_noise",
    # Feature 2: Clutter
    "inject_spectral_clutter",
    "inject_localized_clutter_patch",
    "apply_shadow_attenuation",
    "apply_occlusion_simulation",
    "get_muufl_occlusion_subsets",
    # Feature 3: Bands
    "drop_random_bands",
    "mask_selected_bands",
    "add_localized_band_noise",
    "mask_wavelength_ranges",
]
