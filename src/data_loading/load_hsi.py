"""Load hyperspectral datasets discovered in the local processed-data folders."""

import csv
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from src.utils.validation import HSIValidationError

import numpy as np

ROOT = Path(__file__).resolve().parents[2] / "data" / "processed"


class DatasetLoadError(ValueError):
    """Raised when a discovered dataset is missing or structurally invalid."""


def list_datasets() -> List[str]:
    """Return identifiers for local NPZ datasets, independent of dataset names."""
    if not ROOT.is_dir():
        return []
    return sorted(path.stem for path in ROOT.rglob("*.npz"))


def get_dataset_path(name: str) -> Path:
    """Resolve a dataset identifier to exactly one local NPZ file."""
    matches = [
        path for path in ROOT.rglob("*.npz")
        if path.stem == name
    ] if ROOT.is_dir() else []
    if not matches:
        raise FileNotFoundError(
            f"Dataset {name!r} was not found under {ROOT}. "
            f"Available datasets: {list_datasets()}"
        )
    if len(matches) > 1:
        raise DatasetLoadError(
            f"Dataset identifier {name!r} is ambiguous; multiple NPZ files use that name."
        )
    return matches[0]


def get_dataset_manifest_entry(name: str) -> Optional[Dict[str, str]]:
    """Return verified descriptive fields from the optional processed-data manifest."""
    manifest_path = ROOT / "manifest.csv"
    if not manifest_path.is_file():
        return None
    with manifest_path.open("r", encoding="utf-8", newline="") as manifest_file:
        for row in csv.DictReader(manifest_file):
            if row.get("name") == name:
                return row
    return None


def get_all_manifest_entries() -> List[Dict[str, str]]:
    """Return verified descriptive records for all datasets from the manifest."""
    manifest_path = ROOT / "manifest.csv"
    if not manifest_path.is_file():
        return []
    with manifest_path.open("r", encoding="utf-8", newline="") as manifest_file:
        return list(csv.DictReader(manifest_file))


def load(name: str, as_float: bool = True) -> Tuple[np.ndarray, Optional[np.ndarray], Dict[str, Any]]:
    """Load a (H, W, B) numeric cube, optional spatial labels, and NPZ metadata.

    ``as_float`` remains enabled by default for compatibility with existing callers.
    Dashboard loading disables conversion so inspection reports and preserves the
    source array dtype; detectors perform their own numerically safe conversions.
    """
    path = get_dataset_path(name)
    try:
        with np.load(path, allow_pickle=False) as archive:
            if "data" not in archive.files:
                raise DatasetLoadError(f"{path.name} does not contain a 'data' cube.")
            cube = archive["data"]
            if (
                cube.ndim != 3
                or any(size <= 0 for size in cube.shape)
                or not np.issubdtype(cube.dtype, np.number)
            ):
                raise DatasetLoadError(
                    f"{path.name} must contain a non-empty numeric (height, width, bands) "
                    f"'data' array; received shape {cube.shape} and dtype {cube.dtype}."
                )

            gt = archive["gt"] if "gt" in archive.files else None
            if gt is not None:
                if gt.ndim != 2 or gt.shape != cube.shape[:2]:
                    raise DatasetLoadError(
                        f"{path.name} ground-truth shape {gt.shape} does not match "
                        f"the cube's spatial shape {cube.shape[:2]}."
                    )
                if not (
                    np.issubdtype(gt.dtype, np.number)
                    or np.issubdtype(gt.dtype, np.bool_)
                ):
                    raise DatasetLoadError(
                        f"{path.name} ground-truth labels must be numeric or boolean."
                    )
                if not np.all(np.isfinite(gt)):
                    raise DatasetLoadError(
                        f"{path.name} ground-truth labels contain NaN or infinite values."
                    )

            metadata = {
                key: archive[key]
                for key in archive.files
                if key not in ("data", "gt")
            }
    except DatasetLoadError:
        raise
    except (OSError, EOFError, KeyError, ValueError, zipfile.BadZipFile) as error:
        raise DatasetLoadError(f"Could not read {path.name}: {error}") from error

    if as_float:
        cube = cube.astype(np.float32)
    return cube, gt, metadata


def load_mat(path: str, data_key: str = "data", gt_key: str = "map"):
    """Load a supported original .mat file when SciPy is installed."""
    import scipy.io as sio

    data = sio.loadmat(path)
    if data_key not in data:
        raise KeyError(f"MAT file {path!r} does not contain data key {data_key!r}.")
    return data[data_key], data.get(gt_key)


if __name__ == "__main__":
    for dataset_name in list_datasets():
        cube, ground_truth, _ = load(dataset_name, as_float=False)
        label_count = int(np.count_nonzero(ground_truth)) if ground_truth is not None else 0
        print(
            f"{dataset_name:26s} {str(cube.shape):16s} {str(cube.dtype):8s} "
            f"reference_pixels={label_count}"
        )
def load_file(path, filename=None, bands_first=False):
    """Load an HSI cube, optional ground truth, and metadata from a file."""
    import io
    from pathlib import Path

    is_bytes = isinstance(path, (bytes, bytearray))
    if is_bytes:
        filename = filename or "uploaded.npz"
        suffix = Path(filename).suffix.lower()
        source = io.BytesIO(path)
    else:
        file_path = Path(path)
        filename = filename or file_path.name
        suffix = Path(filename).suffix.lower()
        source = file_path

    if suffix not in (".npz", ".npy", ".mat"):
        raise HSIValidationError(f"Unsupported file type: {suffix or filename}")

    if suffix == ".mat":
        import scipy.io as sio
        data = sio.loadmat(source)
        cube_key = next(
            (k for k in ("data", "hsi", "cube", "X")
             if k in data and isinstance(data[k], np.ndarray)
             and data[k].ndim == 3),
            None,
        )
        if cube_key is None:
            raise HSIValidationError("MAT file contains no 3D hyperspectral cube.")
        gt_key = next(
            (k for k in ("map", "groundtruth", "gt", "labels")
             if k in data and isinstance(data[k], np.ndarray)
             and data[k].ndim >= 2),
            None,
        )
        cube = np.asarray(data[cube_key])
        gt = np.asarray(data[gt_key]).squeeze() if gt_key else None
        if bands_first:
            cube = cube.transpose(1, 2, 0)
        return cube, gt, {"cube_key": cube_key, "gt_key": gt_key}

    loaded = np.load(source, allow_pickle=False)
    if suffix == ".npy":
        cube, gt, meta = np.asarray(loaded), None, {}
    else:
        try:
            with loaded as archive:
                cube_key = next(
                    (k for k in ("data", "hsi", "cube", "X")
                     if k in archive.files and archive[k].ndim == 3),
                    None,
                )
                if cube_key is None:
                    raise HSIValidationError(
                        "NPZ file must contain a 3D cube under data, hsi, cube, or X."
                    )
                cube = np.asarray(archive[cube_key])
                gt_key = next(
                    (k for k in ("map", "gt", "groundtruth", "labels")
                     if k in archive.files),
                    None,
                )
                gt = np.asarray(archive[gt_key]) if gt_key else None
                meta = {
                    k: np.asarray(archive[k])
                    for k in archive.files
                    if k not in (cube_key, gt_key)
                }
                meta["cube_key"] = cube_key
                meta["gt_key"] = gt_key
        except ValueError:
            raise

    if cube.ndim != 3:
        raise HSIValidationError(f"Expected a 3D hyperspectral cube; got shape {cube.shape}.")
    if bands_first:
        cube = cube.transpose(1, 2, 0)
    return cube, gt, meta

def describe(cube, gt=None):
    """Return basic descriptive information about a hyperspectral cube."""
    cube = np.asarray(cube)

    if cube.ndim != 3:
        raise DatasetLoadError(
            f"Expected a 3D hyperspectral cube; got shape {cube.shape}."
        )

    valid_pixels = np.all(np.isfinite(cube), axis=-1)
    result = {
        "height": int(cube.shape[0]),
        "width": int(cube.shape[1]),
        "bands": int(cube.shape[2]),
        "shape": tuple(cube.shape),
        "dtype": str(cube.dtype),
        "invalid_pixels": int(np.count_nonzero(~valid_pixels)),
        "target_pixels": 0,
    }

    if gt is not None:
        gt = np.asarray(gt).squeeze()
        if gt.shape != cube.shape[:2]:
            raise DatasetLoadError(
                f"Ground-truth shape {gt.shape} does not match cube shape "
                f"{cube.shape[:2]}."
            )
        result["target_pixels"] = int(np.count_nonzero(gt))

    return result
