"""Minimal loader for this dataset pack.

    from load_hsi import load, list_datasets
    cube, gt, meta = load("abu-airport-1")   # cube: (H, W, B), gt: (H, W) uint8 {0,1}
"""
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2] / "data" / "processed"

def list_datasets():
    return sorted(p.stem for p in ROOT.glob("*/*.npz"))

def load(name, as_float=True):
    path = next(ROOT.glob(f"*/{name}.npz"), None)
    if path is None:
        raise FileNotFoundError(f"{name!r} not found. Available: {list_datasets()}")
    with np.load(path, allow_pickle=False) as d:
        cube, gt = d["data"], d["gt"]
        meta = {k: d[k] for k in d.files if k not in ("data", "gt")}
    if as_float:
        cube = cube.astype(np.float32)
    return cube, gt, meta

def load_mat(path, data_key="data", gt_key="map"):
    """Load one of the original .mat files (ABU / San Diego / HYDICE use keys 'data' and 'map')."""
    import scipy.io as sio
    d = sio.loadmat(path)
    return d[data_key], d[gt_key]

if __name__ == "__main__":
    for n in list_datasets():
        c, g, _ = load(n, as_float=False)
        print(f"{n:26s} {str(c.shape):16s} {str(c.dtype):8s} targets={int(g.sum())}")
