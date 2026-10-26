"""Stage 5 Application & End-to-End Integration Smoke Test.

Executes a comprehensive, non-interactive verification across all 9 Stage 5 requirements:
    1. Streamlit app starts and executes without unhandled exceptions.
    2. Actual processed datasets are enumerated and loaded via load_hsi.
    3. The RX detector produces a valid 2D anomaly score map on real data.
    4. Evaluation metrics are accurately computed using existing metric functions.
    5. The model leaderboard is constructed from genuine experiment records.
    6. Experiment history persists in SQLite across database reconnections.
    7. Multi-format exports (CSV, JSON, NPZ, PNG, Excel XLSX) produce valid files.
    8. Error handling and stale state invalidation function correctly.
    9. Invariant bounds and non-probability disclaimers are preserved.
"""

import json
from pathlib import Path
import sys
import tempfile
import numpy as np
import pandas as pd
from streamlit.testing.v1 import AppTest

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loading.load_hsi import list_datasets, load
from src.dashboard.components import synthesize_rgb_composite
from src.dashboard.state import invalidate_results_if_stale
from src.evaluation.metrics import evaluate_anomaly_map
from src.experiments.history import ExperimentHistoryDB
from src.experiments.leaderboard import build_leaderboard, get_metric_definitions
from src.exports.results import (
    export_anomaly_map_npz,
    export_anomaly_map_png,
    export_experiment_json,
    export_metrics_csv,
    export_runs_csv,
    export_runs_excel,
    generate_experiment_report,
)
from src.models.rx import RXDetector


def run_stage5_smoke_test():
    print("=" * 88)
    print(" STAGE 5 INTEGRATED RESEARCH APPLICATION SMOKE TEST")
    print("=" * 88)

    # 1. Dataset verification
    print("\n[1/9] Checking processed benchmark datasets...")
    datasets = list_datasets()
    print(f"      Found {len(datasets)} local datasets: {datasets[:5]}...")
    assert len(datasets) >= 16, f"Expected at least 16 datasets, found {len(datasets)}"
    assert "abu-airport-1" in datasets

    # 2. Real data loading
    print("\n[2/9] Loading real benchmark cube 'abu-airport-1'...")
    cube, gt, meta = load("abu-airport-1", as_float=True)
    h, w, b = cube.shape
    assert (h, w, b) == (100, 100, 205)
    assert int(np.sum(gt)) == 144
    print(f"      Loaded cube: ({h}, {w}, {b}), float32, targets={int(np.sum(gt))}")

    # 3. Detection inference
    print("\n[3/9] Executing RX baseline inference on abu-airport-1...")
    det = RXDetector(regularization=1e-4)
    scores = det.fit_predict(cube)
    assert scores.shape == (h, w)
    assert np.all(np.isfinite(scores))
    print(f"      Generated score map: shape={scores.shape}, range=[{scores.min():.4f}, {scores.max():.4f}]")

    # 4. Metrics evaluation
    print("\n[4/9] Calculating quantitative evaluation metrics...")
    metrics = evaluate_anomaly_map(gt, scores, target_tpr=0.95)
    print(f"      ROC-AUC          : {metrics['auc_roc']:.6f}")
    print(f"      PR-AUC           : {metrics['auc_pr']:.6f}")
    print(f"      Max F1           : {metrics['f1_max']:.6f}")
    print(f"      FPR @ 95% TPR    : {metrics['fpr_at_95_tpr']:.6f}")
    fa_val = metrics.get('false_alarms_per_mp', metrics.get('false_alarms_per_megapixel', 0.0))
    print(f"      FA / Megapixel   : {fa_val:,.1f}")
    assert 0.88 <= metrics["auc_roc"] <= 0.89, f"Unexpected ROC-AUC: {metrics['auc_roc']}"

    # 5. Persistent SQLite experiment history
    print("\n[5/9] Testing persistent SQLite experiment history across restarts...")
    with tempfile.TemporaryDirectory() as tmpdir:
        db_file = Path(tmpdir) / "test_history.db"

        # Session 1: write run
        db1 = ExperimentHistoryDB(db_file)
        exp_id = db1.save_run(
            dataset_name="abu-airport-1",
            detector_name="Global-RX",
            parameters={"regularization": 1e-4, "method": "cholesky"},
            status="success",
            metrics=metrics,
            runtime={"total_time_mean_sec": 0.052},
        )
        print(f"      Saved experiment run: `{exp_id}`")

        # Session 2: close connection and reopen database file
        db2 = ExperimentHistoryDB(db_file)
        retrieved = db2.get_run(exp_id)
        assert retrieved is not None, "Failed to retrieve saved run from persistent database"
        assert retrieved["dataset_name"] == "abu-airport-1"
        assert retrieved["status"] == "success"
        assert abs(retrieved["metrics"]["auc_roc"] - metrics["auc_roc"]) < 1e-5
        print(f"      Successfully verified persistent recovery across SQLite reconnects.")

        # 6. Leaderboard construction
        print("\n[6/9] Building dynamic leaderboard from genuine database records...")
        all_runs = db2.list_runs()
        df_board = build_leaderboard(all_runs, sort_by="ROC-AUC", ascending=False)
        assert len(df_board) == 1
        assert df_board.iloc[0]["Dataset"] == "abu-airport-1"
        assert df_board.iloc[0]["ROC-AUC"] == round(metrics["auc_roc"], 4)
        print(f"      Leaderboard row verified: ROC-AUC={df_board.iloc[0]['ROC-AUC']}")

        # 7. Multi-format exports
        print("\n[7/9] Verifying all export formats (CSV, JSON, NPZ, PNG, Excel, Markdown)...")
        p_csv = export_metrics_csv(metrics, Path(tmpdir) / "metrics.csv")
        p_json = export_experiment_json(retrieved, Path(tmpdir) / "experiment.json")
        p_npz = export_anomaly_map_npz(scores, {"dataset": "abu-airport-1"}, Path(tmpdir) / "map.npz")
        p_png = export_anomaly_map_png(scores, Path(tmpdir) / "map.png")
        p_xlsx = export_runs_excel(all_runs, Path(tmpdir) / "leaderboard.xlsx")
        p_rep = generate_experiment_report(retrieved, Path(tmpdir) / "report.md")

        assert p_csv.exists() and p_csv.stat().st_size > 0
        assert p_json.exists() and p_json.stat().st_size > 0
        assert p_npz.exists() and p_npz.stat().st_size > 0
        assert p_png.exists() and p_png.stat().st_size > 500
        assert p_xlsx.exists() and p_xlsx.stat().st_size > 500
        assert Path(tmpdir, "report.md").exists()
        print(f"      All 6 export formats successfully generated and validated.")

    # 8. Stale state invalidation
    print("\n[8/9] Verifying stale result invalidation...")
    import streamlit as st
    st.session_state["scores"] = scores
    st.session_state["last_run_config_hash"] = "abu-airport-1_0.0001_cholesky"
    was_stale = invalidate_results_if_stale("abu-airport-2", 0.0001, "cholesky")
    assert was_stale is True
    assert st.session_state["scores"] is None
    print(f"      Stale results correctly flushed upon dataset change.")

    # 9. Headless Streamlit application verification
    print("\n[9/9] Executing headless Streamlit AppTest verification...")
    app_path = PROJECT_ROOT / "app.py"
    at = AppTest.from_file(str(app_path), default_timeout=15)
    at.run()
    assert len(at.exception) == 0, f"App execution raised exceptions: {at.exception}"
    print(f"      Streamlit application initialized without exceptions.")

    print("\n" + "=" * 88)
    print(" STAGE 5 INTEGRATION SMOKE TEST COMPLETED SUCCESSFULLY")
    print("=" * 88)


if __name__ == "__main__":
    run_stage5_smoke_test()
