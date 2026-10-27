"""Build the camouflage datasets in data/processed/camouflage/.

    python experiments/build_camouflage_datasets.py --muufl-dir data/raw/MUUFLGulfportDataCollection

1. Real camouflage (needs the MUUFL .mat files from github.com/GatorSense/MUUFLGulfport):
   muufl-flight{1,3,4}-camo : three flights over the same 60 camouflage cloths.
2. Synthetic camouflage (uses the bundled .npz scenes, always built):
   implant-<scene>-a<abundance> : 20 targets mixed into a cleaned background
   at abundance 5%, 10%, 20% and 40% (lower = better camouflaged).
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from src.data_loading.camouflage import implant_targets, muufl_target_masks  # noqa: E402
from src.data_loading.load_hsi import load  # noqa: E402

OUT = ROOT / "data" / "processed" / "camouflage"
SCENES = ["hydice-urban", "san-diego", "abu-urban-1", "abu-beach-2"]
ABUNDANCES = [0.05, 0.10, 0.20, 0.40]
MUUFL_FLIGHTS = {"muufl-flight1-camo": "muufl_gulfport_campus_w_lidar_1.mat",
                 "muufl-flight3-camo": "muufl_gulfport_campus_3.mat",
                 "muufl-flight4-camo": "muufl_gulfport_campus_4.mat"}
DROP_BANDS = 4  # first/last 4 MUUFL bands are noisy (same as the official scene-label file)


def build_muufl(muufl_dir: Path):
    import scipy.io as sio
    if not (muufl_dir / "tgt_img_spectra.mat").exists():
        print(f"MUUFL files not found in {muufl_dir}; skipping real camouflage flights. "
              "Download them from https://github.com/GatorSense/MUUFLGulfport (MUUFLGulfportDataCollection).")
        return
    ref = sio.loadmat(muufl_dir / "tgt_img_spectra.mat", squeeze_me=True, struct_as_record=False)["tgt_img_spectra"]
    for name, fname in MUUFL_FLIGHTS.items():
        path = muufl_dir / fname
        if not path.exists():
            print(f"skip {name}: {path} not found")
            continue
        h = sio.loadmat(path, squeeze_me=True, struct_as_record=False)["hsi"]
        g = h.groundTruth
        keep = slice(DROP_BANDS, len(h.info.wavelength) - DROP_BANDS)
        data = np.asarray(h.Data, np.float32)[..., keep]
        valid = np.asarray(h.valid_mask, bool)
        data[~valid] = 0.0
        gt, halo = muufl_target_masks(g.Targets_rowIndices, g.Targets_colIndices, g.Targets_Size, data.shape)
        gt_all, _ = muufl_target_masks(g.Targets_rowIndices, g.Targets_colIndices, g.Targets_Size, data.shape,
                                       include_calibration=True)
        gt &= valid
        lidar = {}
        if "Lidar" in h._fieldnames:  # flight 1 only: first/last-return height and intensity, 1 m grid
            first, last = h.Lidar[0].z, h.Lidar[1].z
            lidar = dict(lidar_first_height=first[..., 0].astype(np.float32),
                         lidar_last_height=last[..., 0].astype(np.float32),
                         lidar_first_intensity=first[..., 1].astype(np.float32),
                         lidar_last_intensity=last[..., 1].astype(np.float32),
                         canopy_height=(first[..., 0] - last[..., 0]).astype(np.float32))
        np.savez_compressed(
            OUT / f"{name}.npz", data=data, gt=gt, gt_halo=halo, gt_with_calibration=gt_all,
            valid_mask=valid, wavelengths=np.asarray(h.info.wavelength, np.float32)[keep], rgb=h.RGB,
            target_row=np.asarray(g.Targets_rowIndices, np.int32) - 1,
            target_col=np.asarray(g.Targets_colIndices, np.int32) - 1,
            target_type=np.array(list(g.Targets_Type)), target_size_m=np.asarray(g.Targets_Size, np.float32),
            target_occlusion=np.asarray(g.Targets_HumanCat, np.int8),
            target_visibility=np.asarray(g.Targets_HumanConf, np.int8),
            reference_spectra=np.asarray(ref.spectra, np.float32).T[:, keep],
            reference_spectra_names=np.array(list(ref.names)), **lidar)
        print(f"{name:22s} {data.shape}  camouflage pixels {int(gt.sum())}  (with calibration {int(gt_all.sum())})"
              + ("  + LiDAR" if lidar else ""))


def build_implants():
    for scene in SCENES:
        cube, gt, _ = load(scene)
        for a in ABUNDANCES:
            name = f"implant-{scene}-a{int(round(a * 100)):02d}"
            new, new_gt, info = implant_targets(cube, gt, abundance=a, n_targets=20, random_state=0)
            np.savez_compressed(OUT / f"{name}.npz", data=new, gt=new_gt, abundance=np.float32(a),
                                material=info["material"].astype(np.float32),
                                target_row=np.array([t["row"] for t in info["targets"]], np.int32),
                                target_col=np.array([t["col"] for t in info["targets"]], np.int32),
                                target_size_px=np.array([t["size"] for t in info["targets"]], np.int32),
                                mean_sam_deg=np.float32(info["mean_sam_to_background_deg"]))
            print(f"{name:30s} {new.shape}  target px {info['n_target_pixels']:3d}  "
                  f"SAM to background {info['mean_sam_to_background_deg']:.2f} deg")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--muufl-dir", type=Path, default=ROOT / "data" / "raw" / "MUUFLGulfportDataCollection")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    build_muufl(args.muufl_dir)
    build_implants()
