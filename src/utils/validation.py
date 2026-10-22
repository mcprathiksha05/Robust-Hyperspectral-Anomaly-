"""Task 29 - Validation and recovery.

Every public function in the backend validates its inputs with the helpers
here and raises ``HSIValidationError`` with a message a user can act on.
``safe_call`` lets the pipeline try a step, record the failure, and fall back
instead of crashing the whole run.
"""
from __future__ import annotations

import time
import traceback
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

# Limits that keep the dashboard responsive; adjust if your machine allows.
MAX_PIXELS = 4_000_000
MAX_BANDS = 1_000
MIN_PIXELS = 25
MIN_BANDS = 2


class HSIValidationError(ValueError):
    """Raised when data or parameters are unusable. The message is user-facing."""


# --------------------------------------------------------------------------- data
def validate_cube(cube: Any, name: str = "cube", min_pixels: int = MIN_PIXELS) -> np.ndarray:
    """Check that ``cube`` is a usable H x W x B hyperspectral array.

    Returns the cube as a NumPy array (no copy if already one). Raises
    ``HSIValidationError`` describing the first problem found. NaN/inf values
    are *allowed* here (preprocessing repairs them) unless every value is bad.
    """
    if cube is None:
        raise HSIValidationError(f"{name} is empty (None).")
    try:
        cube = np.asarray(cube)
    except Exception as exc:  # pragma: no cover - exotic inputs
        raise HSIValidationError(f"{name} could not be converted to an array: {exc}") from exc
    if cube.dtype == object or not np.issubdtype(cube.dtype, np.number):
        raise HSIValidationError(f"{name} must be numeric, got dtype {cube.dtype}.")
    if np.issubdtype(cube.dtype, np.complexfloating):
        raise HSIValidationError(f"{name} contains complex numbers; expected real reflectance/radiance.")
    if cube.ndim != 3:
        hint = " Did you load a ground-truth mask instead of the image?" if cube.ndim == 2 else ""
        raise HSIValidationError(
            f"{name} must have 3 dimensions (height x width x bands), got shape {cube.shape}.{hint}")
    h, w, b = cube.shape
    if h * w < min_pixels:
        raise HSIValidationError(f"{name} has only {h * w} pixels; at least {min_pixels} are needed.")
    if h * w > MAX_PIXELS:
        raise HSIValidationError(
            f"{name} has {h * w:,} pixels, above the limit of {MAX_PIXELS:,}. Crop or downsample it first.")
    if b < MIN_BANDS:
        raise HSIValidationError(f"{name} has {b} band(s); at least {MIN_BANDS} are needed.")
    if b > MAX_BANDS:
        raise HSIValidationError(
            f"{name} has {b} bands, which looks wrong. If the array is bands-first (B x H x W), "
            "transpose it with cube.transpose(1, 2, 0).")
    if np.issubdtype(cube.dtype, np.floating) and not np.isfinite(cube).any():
        raise HSIValidationError(f"{name} contains no finite values (all NaN or inf).")
    return cube


def validate_gt(gt: Any, cube_shape: tuple, name: str = "ground truth") -> np.ndarray:
    """Check a ground-truth mask matches the cube and is binary. Returns uint8 {0,1}."""
    if gt is None:
        raise HSIValidationError(f"{name} is missing.")
    gt = np.asarray(gt)
    if gt.ndim == 3 and gt.shape[2] == 1:
        gt = gt[..., 0]
    if gt.ndim != 2:
        raise HSIValidationError(f"{name} must be 2-D (height x width), got shape {gt.shape}.")
    if gt.shape != tuple(cube_shape[:2]):
        raise HSIValidationError(
            f"{name} shape {gt.shape} does not match the image size {tuple(cube_shape[:2])}.")
    if np.issubdtype(gt.dtype, np.floating) and not np.isfinite(gt).all():
        raise HSIValidationError(f"{name} contains NaN or inf values.")
    values = np.unique(gt)
    if values.size > 2:
        raise HSIValidationError(
            f"{name} must be a binary mask (0 = background, 1 = target), but has {values.size} "
            "distinct values. Convert class labels to a mask first, e.g. gt = (labels == k).")
    return (gt > 0).astype(np.uint8)


def validate_score_map(scores: Any, name: str = "score map") -> np.ndarray:
    scores = np.asarray(scores, dtype=np.float64)
    if scores.ndim != 2:
        raise HSIValidationError(f"{name} must be 2-D (height x width), got shape {scores.shape}.")
    if not np.isfinite(scores).any():
        raise HSIValidationError(f"{name} has no finite values.")
    return scores


# --------------------------------------------------------------------- parameters
def check_int(value: Any, name: str, lo: int | None = None, hi: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        if isinstance(value, float) and value.is_integer():
            value = int(value)
        else:
            raise HSIValidationError(f"{name} must be a whole number, got {value!r}.")
    value = int(value)
    if lo is not None and value < lo:
        raise HSIValidationError(f"{name} must be at least {lo}, got {value}.")
    if hi is not None and value > hi:
        raise HSIValidationError(f"{name} must be at most {hi}, got {value}.")
    return value


def check_float(value: Any, name: str, lo: float | None = None, hi: float | None = None,
                lo_inclusive: bool = True, hi_inclusive: bool = True) -> float:
    try:
        value = float(value)
    except (TypeError, ValueError):
        raise HSIValidationError(f"{name} must be a number, got {value!r}.") from None
    if not np.isfinite(value):
        raise HSIValidationError(f"{name} must be finite, got {value}.")
    if lo is not None and (value < lo or (not lo_inclusive and value == lo)):
        raise HSIValidationError(f"{name} must be {'>=' if lo_inclusive else '>'} {lo}, got {value}.")
    if hi is not None and (value > hi or (not hi_inclusive and value == hi)):
        raise HSIValidationError(f"{name} must be {'<=' if hi_inclusive else '<'} {hi}, got {value}.")
    return value


def check_choice(value: Any, name: str, choices) -> str:
    if value not in choices:
        raise HSIValidationError(f"{name} must be one of {sorted(choices)}, got {value!r}.")
    return value


def check_windows(inner: int, outer: int, shape: tuple) -> tuple[int, int]:
    """Validate a (guard, outer) dual-window pair for local detectors."""
    inner = check_int(inner, "inner (guard) window", lo=1)
    outer = check_int(outer, "outer window", lo=3)
    if inner % 2 == 0 or outer % 2 == 0:
        raise HSIValidationError(f"Window sizes must be odd, got inner={inner}, outer={outer}.")
    if outer <= inner:
        raise HSIValidationError(f"Outer window ({outer}) must be larger than the inner window ({inner}).")
    if outer > min(shape[:2]):
        raise HSIValidationError(
            f"Outer window ({outer}) is larger than the image ({shape[0]} x {shape[1]}).")
    return inner, outer


# ------------------------------------------------------------------------ recovery
@dataclass
class StepResult:
    """Outcome of one pipeline step run through ``safe_call``."""
    name: str
    ok: bool
    value: Any = None
    error: str | None = None
    seconds: float = 0.0
    detail: str | None = field(default=None, repr=False)


def safe_call(name: str, fn: Callable, *args, fallback: Callable | None = None, **kwargs) -> StepResult:
    """Run ``fn``; on failure try ``fallback`` (same arguments) and report what happened.

    Validation errors keep their friendly message; unexpected errors are
    summarised in one line, with the full traceback kept in ``detail`` for logs.
    """
    t0 = time.perf_counter()
    try:
        return StepResult(name, True, fn(*args, **kwargs), seconds=time.perf_counter() - t0)
    except HSIValidationError as exc:
        err, detail = str(exc), None
    except MemoryError:
        err, detail = (f"{name} ran out of memory. Reduce PCA components, crop the image, "
                       "or use smaller windows."), traceback.format_exc()
    except np.linalg.LinAlgError as exc:
        err, detail = f"{name} failed on a numerical (matrix) step: {exc}", traceback.format_exc()
    except Exception as exc:  # noqa: BLE001 - we want to recover from anything
        err, detail = f"{name} failed: {type(exc).__name__}: {exc}", traceback.format_exc()
    if fallback is not None:
        try:
            value = fallback(*args, **kwargs)
            return StepResult(name, True, value, error=f"{err} (used fallback)",
                              seconds=time.perf_counter() - t0, detail=detail)
        except Exception as exc2:  # noqa: BLE001
            err = f"{err}; fallback also failed: {exc2}"
    return StepResult(name, False, None, error=err, seconds=time.perf_counter() - t0, detail=detail)
