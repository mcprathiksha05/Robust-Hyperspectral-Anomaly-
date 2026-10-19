# Hyperspectral anomaly and target detection datasets

Benchmark data for the project *Robust Hyperspectral Anomaly and Target Detection Under Camouflage, Clutter, and Computational Constraints*. Every scene has a pixel-level ground-truth mask, so all of the evaluation metrics in the proposal (precision, recall, F1, ROC-AUC, AP, FPR, false alarms per megapixel) can be computed.

## Contents

| Folder | What it holds |
|---|---|
| `data/processed/anomaly/` | 15 anomaly-detection scenes as compressed NumPy files (`data` = H×W×B cube, `gt` = H×W mask, 1 = anomaly) |
| `data/processed/target/` | MUUFL Gulfport campus scene for target detection under camouflage and tree occlusion |
| `data/raw/` | Put the original `.mat` files here locally (git-ignored); see the source repos below |
| `previews/` | False-colour composite plus ground truth, one PNG per scene |
| `manifest.csv` | Size, band count, dtype, target pixel count/percentage and validity checks per scene |
| `src/data_loading/load_hsi.py` | `load(name) -> (cube, gt, meta)` and `list_datasets()` |

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
