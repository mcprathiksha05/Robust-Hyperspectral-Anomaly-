# Hyperspectral anomaly and target detection datasets

Benchmark data for the project *Robust Hyperspectral Anomaly and Target Detection Under Camouflage, Clutter, and Computational Constraints*. Every scene has a pixel-level ground-truth mask, so all of the evaluation metrics in the proposal (precision, recall, F1, ROC-AUC, AP, FPR, false alarms per megapixel) can be computed.

## Contents

| Folder | What it holds |
|---|---|
| `data/processed/anomaly/` | 15 anomaly-detection scenes as compressed NumPy files (`data` = H×W×B cube, `gt` = H×W mask, 1 = anomaly) |
| `data/processed/target/` | MUUFL Gulfport campus scene for target detection under camouflage and tree occlusion |
| `data/processed/camouflage/` | 19 camouflage scenes: 3 real MUUFL flights and 16 controlled synthetic implants (see below) |
| `data/raw/` | Put the original `.mat` files here locally (git-ignored); see the source repos below |
| `previews/` | False-colour composite plus ground truth, one PNG per scene |
| `manifest.csv` | Size, band count, dtype, target pixel count/percentage and validity checks per scene |
| `src/data_loading/load_hsi.py` | `load(name) -> (cube, gt, meta)` and `list_datasets()` |

## Camouflage datasets (`data/processed/camouflage/`)

Built by `experiments/build_camouflage_datasets.py` and evaluated by `experiments/run_camouflage_eval.py`. Load them like any other scene, e.g. `load("muufl-flight3-camo")`.

**Real camouflage: `muufl-flight1-camo`, `muufl-flight3-camo`, `muufl-flight4-camo`.** These are three separate flights over the same field: about 325×340 pixels, 64 bands (405–1005 nm), 1 m per pixel. Each holds 60 camouflage cloths in brown and four shades of green, sized 0.5, 1 and 3 m, so many are smaller than a pixel. 15 are in the open, 8 in shadow and 37 under trees.
- `gt` marks the camouflage cloths only. The four large calibration cloths (red, black, blue, green) are excluded because they are trivially easy; they are in `gt_with_calibration`.
- `gt_halo` adds a 1-pixel margin around each target. Leave it out of the background when scoring, because GPS positions are accurate to about 1 m and sub-pixel cloths spill into neighbouring pixels.
- `target_occlusion` (0 open, 1 shadow, 2 under trees), `target_visibility`, `target_type`, `target_size_m`, `target_row`, `target_col`, `reference_spectra` (the four camouflage colours), `wavelengths`, `rgb` and `valid_mask` are also included.
- **Flight 1 only: LiDAR**, on the same 1 m grid as the image. It includes `lidar_first_height` and `lidar_last_height` (elevation in m of the first and last laser return), `lidar_first_intensity` and `lidar_last_intensity`, and `canopy_height` (first minus last). Under trees, the first return hits the canopy (median about 14 m at target spots) while the last return reaches the ground (about 5.4 m, the same as open ground), so the laser does see below the leaves. In a quick check, last-return intensity alone separated the 37 under-tree cloths from nearby under-tree ground with AUC 0.68, where hyperspectral detectors score about 0.50.

**Synthetic camouflage: `implant-<scene>-aXX`.** Backgrounds are HYDICE Urban, San Diego, ABU Urban-1 and ABU Beach-2. Their original anomalies are removed and filled with neighbouring background, keeping realistic sensor noise. Then 20 square targets (1–3 px) are planted, using the spectrum of one real object from the same scene mixed in at abundance XX = 05, 10, 20 or 40 %. Lower abundance means the target is closer to its background, i.e. better camouflaged. The files also store `abundance`, `material`, target positions and sizes, and `mean_sam_deg` (spectral angle between the target and its background: about 1–2° at 5 %, 6–13° at 40 %).

With only 20 targets per image, the AUC of a target-free image varies by about ±0.07 between random placements, so smaller differences on a single implant scene are not meaningful. Use `implant_targets(..., random_state=k)` to generate more placements.

Camouflage results (`experiments/results/camouflage_eval.csv`, mean ROC-AUC):

| Method | Implant 5 % | 10 % | 20 % | 40 % | MUUFL open | shadow | under trees |
|---|---|---|---|---|---|---|---|
| RX baseline | 0.49 | 0.53 | 0.60 | 0.72 | 0.87 | 0.56 | 0.53 |
| Robust RX + PCA | 0.60 | 0.68 | 0.81 | 0.92 | 0.83 | 0.62 | 0.51 |
| Spatial residual | 0.53 | 0.60 | 0.76 | 0.88 | 0.76 | 0.60 | 0.49 |
| Fusion RX+spatial | 0.57 | 0.64 | 0.80 | 0.93 | 0.83 | 0.63 | 0.50 |

**Other public camouflage datasets** (too large or not downloadable automatically; download manually if needed):
- **HyperCOD** (AAAI 2026): 350 ground-level camouflage scenes, 200 bands (400–1000 nm), 1240×1680, pixel masks. Download from Baidu Pan via github.com/Baishuyanyan/HyperCOD.
- **HyCamUX** (Zenodo 16737351): airborne camouflage nets and UXO, 224 bands (935–1720 nm), ENVI format with PNG labels, 108 GB, CC BY-NC-SA 4.0.

## Datasets

**ABU: Airport-Beach-Urban (13 scenes).** Airport 1–4, Beach 1–4 and Urban 1–5. These are 100×100 or 150×150 crops with 102–207 bands. Most come from AVIRIS; Beach-4 is ROSIS. The anomalies are aircraft, vehicles, ships and buildings in varied backgrounds, with 0.08 % to 2.7 % of pixels anomalous. Source: Kang et al., "Hyperspectral Anomaly Detection With Attribute and Edge-Preserving Filters," IEEE TGRS 55(10), 2017.

**San Diego airport (AVIRIS).** 100×100×189 with 3 aircraft as anomalies. It is the most widely used single benchmark.

**HYDICE Urban.** 80×100×175 with 21 anomalous pixels (vehicles and roofs). The anomalies are small and the background is cluttered.

**MUUFL Gulfport campus 1 (CASI-1500, 375–1050 nm).** 325×220×64 at 1 m GSD. It contains emplaced cloth targets in four green and brown camouflage-like colours, sized 0.5 m, 1 m and 3 m, so many are sub-pixel. Some sit in shadow or under trees. `gt` is the `cloth_panels` scene-label mask. The `.npz` also holds:
- `target_row` and `target_col` (0-based) for the 50 targets in this image, plus `target_type`, `target_size_m` and `target_elevated`;
- `target_occlusion`: 0 = unoccluded, 1 = shadow, 2 = under trees;
- `target_visibility`: 1 = visible to 4 = not visible;
- `reference_spectra` (4×64, aligned to the scene bands) for matched-filter and ACE target detection;
- `wavelengths`, `rgb`, and `scene_labels` with `scene_label_names` (11 classes) for clutter analysis.

Use this scene for the camouflage, occlusion and target-detection parts of the project. Use ABU, San Diego and HYDICE for anomaly-detection benchmarking.

## Baseline check

A global RX detector on the raw cubes (no PCA, no preprocessing) gives the following ROC-AUC, which matches published RX numbers:

| Scene | AUC | Scene | AUC | Scene | AUC |
|---|---|---|---|---|---|
| airport-1 | 0.822 | beach-1 | 0.981 | urban-1 | 0.991 |
| airport-2 | 0.840 | beach-2 | 0.911 | urban-2 | 0.995 |
| airport-3 | 0.929 | beach-3 | 1.000 | urban-3 | 0.951 |
| airport-4 | 0.953 | beach-4 | 0.959 | urban-4 | 0.989 |
| san-diego | 0.940 | hydice-urban | 0.986 | urban-5 | 0.969 |
| muufl-campus1 | 0.998 | | | | |

Airport-1/2 and San Diego leave the most room for spectral-spatial fusion to show a gain. The beach and urban scenes are close to saturated for AUC, so compare them on AP, FPR and false alarms per megapixel instead.

## Suggested use

- **Tuning versus testing.** Tune thresholds and parameters on one group (for example, Airport) and report on the others, so the evaluation is not over-optimistic.
- **Robustness experiments.** Add Gaussian or stripe noise, scale the background, or implant targets at a chosen abundance using a background spectrum (camouflage-like similarity). ABU and HYDICE backgrounds work well for this.
- **Explainability.** The scene labels in MUUFL let you break false alarms down by material (trees, buildings, shadow and so on).

## Licences and citation

- MUUFL Gulfport: MIT licence (GatorSense). Cite P. Gader, A. Zare et al., "MUUFL Gulfport Hyperspectral and LiDAR Airborne Data Set," Univ. of Florida, Tech. Rep. REP-2013-570, 2013. If you use the scene labels, also cite X. Du and A. Zare, Tech. Rep. 20170417, 2017.
- ABU, San Diego and HYDICE are research benchmarks redistributed through GitHub. Cite the ABU paper above and the AVIRIS (NASA JPL) and HYDICE sources in your report.

Repositories the data was taken from: github.com/sxt1996/Airport-Beach-Urban-ABU, github.com/mahdiyousofun/AED-algorithm, github.com/GatorSense/MUUFLGulfport.
