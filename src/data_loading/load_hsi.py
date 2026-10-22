"""Task 1 - Data loader.

Two entry points:

    from src.data_loading.load_hsi import load, load_file, list_datasets, describe

    cube, gt, meta = load("abu-airport-1")          # bundled benchmark datasets
    cube, gt, meta = load_file("my_scene.mat")      # any .npz / .npy / .mat file
    cube, gt, meta = load_file(uploaded_file)       # Streamlit UploadedFile works too

``cube`` is always (height, width, bands) float32, ``gt`` is (height, width)
uint8 {0,1} or ``None`` when the file has no labels, and ``meta`` is a dict
(name, source, wavelengths if present, plus any extra arrays in the file).
"""
from __future__ import annotations

import io
import os
from pathlib import Path
from typing import Any

import numpy as np

from src.utils.validation import HSIValidationError, validate_cube, validate_gt

ROOT = Path(__file__).resolve().parents[2] / "data" / "processed"
SUPPORTED = (".npz", ".npy", ".mat")
MAX_FILE_MB = 500

# Names commonly used for the image and the label mask in public .mat files.
CUBE_KEYS = ("data", "cube", "hsi", "image", "img", "X", "x", "Data", "DATA")
GT_KEYS = ("gt", "map", "mask", "groundtruth", "ground_truth", "GT", "Map", "label", "labels", "y")
WAVELENGTH_KEYS = ("wavelengths", "wavelength", "wl", "bands_nm")


# ---------------------------------------------------------------- bundled data
def list_datasets() -> list[str]:
    """Names of the datasets in data/processed (e.g. 'abu-airport-1', 'san-diego')."""
    return sorted(p.stem for p in ROOT.glob("*/*.npz"))


def load(name: str, as_float: bool = True):
    """Load a bundled dataset by name. Returns (cube, gt, meta)."""
    path = next(ROOT.glob(f"*/{name}.npz"), None)
    if path is None:
        raise HSIValidationError(f"Dataset {name!r} not found. Available: {', '.join(list_datasets())}")
    cube, gt, meta = load_file(path, as_float=as_float)
    meta["name"] = name
    meta["task"] = path.parent.name  # 'anomaly' or 'target'
    return cube, gt, meta


# ------------------------------------------------------------------ any file
def load_file(source: Any, filename: str | None = None, *, cube_key: str | None = None,
              gt_key: str | None = None, bands_first: bool = False, as_float: bool = True):
    """Load a hyperspectral cube (and ground truth if present) from a file.

    Parameters
    ----------
    source : path, bytes, or a file-like object with ``.read()`` (e.g. Streamlit upload).
    filename : needed to detect the format when ``source`` is bytes/file-like without ``.name``.
    cube_key, gt_key : variable names inside .npz/.mat files. Detected automatically if omitted.
    bands_first : set True if the array is stored as (bands, height, width).
    """
    raw, fname = _read_bytes(source, filename)
    ext = Path(fname).suffix.lower()
    if ext not in SUPPORTED:
        raise HSIValidationError(
            f"Unsupported file type {ext or '(none)'!r}. Supported: {', '.join(SUPPORTED)}.")

    if ext == ".npy":
        arrays = {"data": _np_load(raw, fname)}
    elif ext == ".npz":
        arrays = _np_load(raw, fname)
    else:
        arrays = _load_mat(raw, fname)

    ck = cube_key or _find_cube_key(arrays)
    if ck not in arrays:
        raise HSIValidationError(f"Variable {ck!r} not found in {fname}. Found: {', '.join(arrays)}.")
    cube = np.asarray(arrays[ck])
    if bands_first:
        if cube.ndim != 3:
            raise HSIValidationError(f"bands_first=True needs a 3-D array, got shape {cube.shape}.")
        cube = np.moveaxis(cube, 0, -1)
    cube = validate_cube(cube, name=f"'{ck}' in {fname}")

    gk = gt_key or _find_gt_key(arrays, cube.shape, exclude=ck)
    gt = None
    if gk is not None:
        if gk not in arrays:
            raise HSIValidationError(f"Ground-truth variable {gk!r} not found in {fname}.")
        gt = validate_gt(arrays[gk], cube.shape, name=f"'{gk}' in {fname}")

    meta: dict[str, Any] = {"name": Path(fname).stem, "source": fname, "cube_key": ck, "gt_key": gk,
                            "original_dtype": str(cube.dtype)}
    for k in WAVELENGTH_KEYS:
        if k in arrays and np.asarray(arrays[k]).size == cube.shape[2]:
            meta["wavelengths"] = np.asarray(arrays[k], dtype=np.float32).ravel()
            break
    for k, v in arrays.items():
        if k not in (ck, gk) and k not in meta:
            meta[k] = v
    if as_float:
        cube = cube.astype(np.float32, copy=False)
    return cube, gt, meta


def describe(cube: np.ndarray, gt: np.ndarray | None = None) -> dict:
    """Facts for the dataset information panel (dimensions, dtype, invalid pixels...)."""
    cube = validate_cube(cube)
    h, w, b = cube.shape
    flat = cube.reshape(-1, b)
    finite = np.isfinite(flat) if np.issubdtype(flat.dtype, np.floating) else np.ones(flat.shape, bool)
    bad_pixels = int((~finite.all(1)).sum())
    with np.errstate(invalid="ignore"):
        std = np.nanstd(np.where(finite, flat, np.nan).astype(np.float64), axis=0)
    info = {
        "height": h, "width": w, "bands": b, "pixels": h * w, "dtype": str(cube.dtype),
        "size_mb": round(cube.nbytes / 1e6, 2),
        "nan_values": int(np.isnan(flat).sum()) if np.issubdtype(flat.dtype, np.floating) else 0,
        "inf_values": int(np.isinf(flat).sum()) if np.issubdtype(flat.dtype, np.floating) else 0,
        "invalid_pixels": bad_pixels,
        "invalid_pixel_pct": round(100 * bad_pixels / (h * w), 3),
        "constant_bands": [int(i) for i in np.where(~(std > 0))[0]],
        "min": float(np.nanmin(np.where(finite, flat, np.nan))),
        "max": float(np.nanmax(np.where(finite, flat, np.nan))),
    }
    if gt is not None:
        npos = int(np.asarray(gt).sum())
        info.update(target_pixels=npos, target_pct=round(100 * npos / (h * w), 3))
    return info


# ----------------------------------------------------------------- internals
def _read_bytes(source: Any, filename: str | None) -> tuple[bytes, str]:
    if isinstance(source, (str, os.PathLike)):
        path = Path(source)
        if not path.exists():
            raise HSIValidationError(f"File not found: {path}")
        size_mb = path.stat().st_size / 1e6
        if size_mb > MAX_FILE_MB:
            raise HSIValidationError(f"{path.name} is {size_mb:.0f} MB; the limit is {MAX_FILE_MB} MB.")
        return path.read_bytes(), filename or path.name
    if isinstance(source, (bytes, bytearray)):
        if not filename:
            raise HSIValidationError("Pass filename=... when loading from bytes so the format can be detected.")
        return bytes(source), filename
    if hasattr(source, "read"):
        fname = filename or getattr(source, "name", None)
        if not fname:
            raise HSIValidationError("Pass filename=... so the file format can be detected.")
        data = source.read()
        if len(data) > MAX_FILE_MB * 1e6:
            raise HSIValidationError(f"Upload is larger than {MAX_FILE_MB} MB.")
        return data, fname
    raise HSIValidationError(f"Cannot read data from an object of type {type(source).__name__}.")


def _np_load(raw: bytes, fname: str):
    try:
        obj = np.load(io.BytesIO(raw), allow_pickle=False)
    except ValueError as exc:
        raise HSIValidationError(f"{fname} contains Python objects and cannot be loaded safely: {exc}") from exc
    except Exception as exc:
        raise HSIValidationError(f"{fname} is not a valid NumPy file: {exc}") from exc
    if isinstance(obj, np.lib.npyio.NpzFile):
        with obj:
            return {k: obj[k] for k in obj.files}
    return obj


def _load_mat(raw: bytes, fname: str) -> dict:
    if raw[:4] == b"\x89HDF" or b"MATLAB 7.3" in raw[:128]:
        return _load_mat_v73(raw, fname)
    import scipy.io as sio
    try:
        d = sio.loadmat(io.BytesIO(raw), squeeze_me=False)
    except NotImplementedError:
        return _load_mat_v73(raw, fname)
    except Exception as exc:
        raise HSIValidationError(f"{fname} is not a readable MATLAB file: {exc}") from exc
    return {k: v for k, v in d.items() if not k.startswith("__") and isinstance(v, np.ndarray)
            and v.dtype != object and v.dtype.names is None}


def _load_mat_v73(raw: bytes, fname: str) -> dict:
    try:
        import h5py
    except ImportError as exc:
        raise HSIValidationError(f"{fname} is a MATLAB v7.3 file; install h5py to read it (pip install h5py).") from exc
    out = {}
    try:
        with h5py.File(io.BytesIO(raw), "r") as f:
            for k, v in f.items():
                if isinstance(v, h5py.Dataset) and np.issubdtype(v.dtype, np.number):
                    out[k] = np.asarray(v).T  # MATLAB stores column-major; transpose back
    except Exception as exc:
        raise HSIValidationError(f"{fname} could not be read as MATLAB v7.3/HDF5: {exc}") from exc
    return out


def _find_cube_key(arrays: dict) -> str:
    for k in CUBE_KEYS:
        if k in arrays and np.asarray(arrays[k]).ndim == 3:
            return k
    cands = [(np.asarray(v).size, k) for k, v in arrays.items() if np.asarray(v).ndim == 3]
    if not cands:
        shapes = {k: np.asarray(v).shape for k, v in arrays.items()}
        raise HSIValidationError(f"No 3-D image array found. Variables and shapes: {shapes}")
    return max(cands)[1]


def _find_gt_key(arrays: dict, shape: tuple, exclude: str) -> str | None:
    def ok(v):
        v = np.asarray(v)
        return v.ndim == 2 and v.shape == shape[:2] and np.unique(v).size <= 2
    for k in GT_KEYS:
        if k in arrays and k != exclude and ok(arrays[k]):
            return k
    for k, v in arrays.items():
        if k != exclude and ok(v):
            return k
    return None


if __name__ == "__main__":
    for n in list_datasets():
        c, g, _ = load(n, as_float=False)
        print(f"{n:26s} {str(c.shape):16s} {str(c.dtype):8s} targets={int(g.sum())}")
