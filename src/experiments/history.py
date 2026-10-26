"""Feature 13: Persistent Experiment History for Hyperspectral Detection.

Provides structured, local SQLite storage for experiment tracking:
    - Unique experiment ID, timestamp, dataset, detector configuration
    - Preprocessing and robustness corruption settings
    - Evaluation metrics and phase-separated runtimes
    - Traced memory and process RSS measurements
    - Full error tracking for failed or non-finite experiments
    - Querying, filtering, comparison, and deletion capabilities

Does NOT store large 3D cubes directly into SQLite; stores lightweight metrics,
metadata, and optional filesystem artifact references.
"""

from contextlib import contextmanager
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import sqlite3
import sys
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union
import uuid
import numpy as np


def _json_serializable(obj: Any) -> Any:
    """Recursively convert NumPy scalars and arrays to native Python types."""
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    elif isinstance(obj, (np.floating, float)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (bool, np.bool_)):
        return bool(obj)
    elif isinstance(obj, dict):
        return {k: _json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_json_serializable(x) for x in obj]
    elif obj is None:
        return None
    return str(obj)


class ExperimentHistoryDB:
    """SQLite-backed persistent experiment history manager."""

    DEFAULT_DB_DIR = Path(__file__).resolve().parents[2] / "data" / "history"
    DEFAULT_DB_PATH = DEFAULT_DB_DIR / "experiments.db"

    def __init__(self, db_path: Optional[Union[str, Path]] = None):
        """Initialize the database connection and schema.

        Args:
            db_path: Path to SQLite database file. If None, uses default location.
        """
        if db_path is None:
            self.db_path = Path(self.DEFAULT_DB_PATH)
        else:
            self.db_path = Path(db_path)

        # In-memory SQLite databases (e.g. :memory:) should not create directories
        if str(self.db_path) != ":memory:":
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            self._memory_conn = None
        else:
            self._memory_conn = sqlite3.connect(":memory:")
            self._memory_conn.row_factory = sqlite3.Row

        self._init_schema()

    @contextmanager
    def _get_connection(self) -> Iterator[sqlite3.Connection]:
        """Yield an SQLite connection with foreign keys and dict-like rows, closing when done."""
        if self._memory_conn is not None:
            yield self._memory_conn
        else:
            conn = sqlite3.connect(str(self.db_path))
            conn.row_factory = sqlite3.Row
            try:
                yield conn
            finally:
                conn.close()

    def close(self) -> None:
        """Close in-memory connection if active."""
        if self._memory_conn is not None:
            self._memory_conn.close()
            self._memory_conn = None

    def _init_schema(self) -> None:
        """Create experiments table and indices if they do not exist."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS experiments (
                    experiment_id TEXT PRIMARY KEY,
                    timestamp TEXT NOT NULL,
                    dataset_name TEXT NOT NULL,
                    detector_name TEXT NOT NULL,
                    parameters TEXT NOT NULL,
                    preprocessing TEXT,
                    corruption TEXT,
                    seed INTEGER,
                    status TEXT NOT NULL,
                    error_message TEXT,
                    metrics TEXT,
                    runtime TEXT,
                    memory TEXT,
                    artifact_path TEXT,
                    software_info TEXT NOT NULL
                )
                """
            )
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_dataset ON experiments(dataset_name)"
            )
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_detector ON experiments(detector_name)"
            )
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_status ON experiments(status)"
            )
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_timestamp ON experiments(timestamp)"
            )
            conn.commit()

    def save_run(
        self,
        dataset_name: str,
        detector_name: str,
        parameters: Dict[str, Any],
        status: str = "success",
        metrics: Optional[Dict[str, Any]] = None,
        runtime: Optional[Dict[str, Any]] = None,
        memory: Optional[Dict[str, Any]] = None,
        preprocessing: Optional[Dict[str, Any]] = None,
        corruption: Optional[Dict[str, Any]] = None,
        seed: Optional[int] = None,
        error_message: Optional[str] = None,
        artifact_path: Optional[str] = None,
        experiment_id: Optional[str] = None,
        timestamp: Optional[str] = None,
    ) -> str:
        """Save a completed or failed experiment run into the database.

        Args:
            dataset_name: Name of benchmark dataset.
            detector_name: Name of detector.
            parameters: Detector hyperparameters and configuration.
            status: 'success', 'failed', or 'error'.
            metrics: Computed evaluation metrics dict (e.g. auc_roc, auc_pr).
            runtime: Runtime timing dict (fit_ms, predict_ms, etc.).
            memory: Memory measurement dict (peak_traced_mb, rss_delta_mb).
            preprocessing: Preprocessing configuration dict (normalization, etc.).
            corruption: Robustness corruption dict (noise_snr, clutter, etc.).
            seed: Random seed used.
            error_message: Error description if status != 'success'.
            artifact_path: Optional filesystem path to saved anomaly map or arrays.
            experiment_id: Custom ID or None for auto-generated UUID.
            timestamp: ISO timestamp string or None for UTC now.

        Returns:
            str: The experiment_id saved.
        """
        exp_id = experiment_id or f"exp_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
        ts = timestamp or datetime.now(timezone.utc).isoformat()

        software_info = {
            "python_version": platform.python_version(),
            "numpy_version": np.__version__,
            "platform": f"{platform.system()} {platform.release()}",
            "machine": platform.machine(),
        }

        # Validate status
        if status not in ("success", "failed", "error"):
            status = "failed" if error_message else "success"

        params_json = json.dumps(_json_serializable(parameters))
        prep_json = json.dumps(_json_serializable(preprocessing)) if preprocessing else None
        corr_json = json.dumps(_json_serializable(corruption)) if corruption else None
        metrics_json = json.dumps(_json_serializable(metrics)) if metrics else None
        runtime_json = json.dumps(_json_serializable(runtime)) if runtime else None
        memory_json = json.dumps(_json_serializable(memory)) if memory else None
        software_json = json.dumps(_json_serializable(software_info))

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT OR REPLACE INTO experiments (
                    experiment_id, timestamp, dataset_name, detector_name,
                    parameters, preprocessing, corruption, seed, status,
                    error_message, metrics, runtime, memory, artifact_path,
                    software_info
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    exp_id,
                    ts,
                    str(dataset_name),
                    str(detector_name),
                    params_json,
                    prep_json,
                    corr_json,
                    int(seed) if seed is not None else None,
                    str(status),
                    str(error_message) if error_message else None,
                    metrics_json,
                    runtime_json,
                    memory_json,
                    str(artifact_path) if artifact_path else None,
                    software_json,
                ),
            )
            conn.commit()

        return exp_id

    def get_run(self, experiment_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a single experiment record by ID."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT * FROM experiments WHERE experiment_id = ?",
                (experiment_id,),
            )
            row = cursor.fetchone()
            if row is None:
                return None
            return self._row_to_dict(row)

    def list_runs(
        self,
        dataset_name: Optional[str] = None,
        detector_name: Optional[str] = None,
        status: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """List experiment runs matching optional filters, ordered newest first."""
        query = "SELECT * FROM experiments WHERE 1=1"
        params: List[Any] = []

        if dataset_name:
            query += " AND dataset_name = ?"
            params.append(dataset_name)
        if detector_name:
            query += " AND detector_name = ?"
            params.append(detector_name)
        if status:
            query += " AND status = ?"
            params.append(status)

        query += " ORDER BY timestamp DESC"
        if limit is not None and limit > 0:
            query += f" LIMIT {int(limit)}"

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, params)
            rows = cursor.fetchall()
            return [self._row_to_dict(r) for r in rows]

    def delete_run(self, experiment_id: str) -> bool:
        """Delete an experiment by ID. Returns True if a record was removed."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "DELETE FROM experiments WHERE experiment_id = ?",
                (experiment_id,),
            )
            conn.commit()
            return cursor.rowcount > 0

    def clear_all(self) -> int:
        """Clear all experiment records. Returns number of rows deleted."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM experiments")
            conn.commit()
            return cursor.rowcount

    def get_statistics(self) -> Dict[str, Any]:
        """Return summary count statistics across all stored experiments."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM experiments")
            total = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(*) FROM experiments WHERE status = 'success'")
            successful = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(*) FROM experiments WHERE status != 'success'")
            failed = cursor.fetchone()[0]

            cursor.execute("SELECT DISTINCT dataset_name FROM experiments")
            datasets = [r[0] for r in cursor.fetchall()]

            cursor.execute("SELECT DISTINCT detector_name FROM experiments")
            detectors = [r[0] for r in cursor.fetchall()]

            return {
                "total_runs": total,
                "successful_runs": successful,
                "failed_runs": failed,
                "unique_datasets": datasets,
                "unique_detectors": detectors,
            }

    def compare_runs(self, id1: str, id2: str) -> Dict[str, Any]:
        """Generate a side-by-side comparison between two experiment runs."""
        run1 = self.get_run(id1)
        run2 = self.get_run(id2)
        if not run1 or not run2:
            raise ValueError(f"One or both run IDs not found: {id1}, {id2}")

        return {
            "run_1": run1,
            "run_2": run2,
            "same_dataset": run1["dataset_name"] == run2["dataset_name"],
            "same_detector": run1["detector_name"] == run2["detector_name"],
            "metrics_delta": {
                k: (
                    (run1["metrics"].get(k, 0.0) - run2["metrics"].get(k, 0.0))
                    if (run1.get("metrics") and run2.get("metrics") and k in run1["metrics"] and k in run2["metrics"])
                    else None
                )
                for k in ("auc_roc", "auc_pr", "f1_max", "fpr_at_95_tpr")
            },
        }

    def _row_to_dict(self, row: sqlite3.Row) -> Dict[str, Any]:
        """Convert a SQLite Row into a cleanly deserialized Python dictionary."""
        d = dict(row)
        for json_field in (
            "parameters",
            "preprocessing",
            "corruption",
            "metrics",
            "runtime",
            "memory",
            "software_info",
        ):
            if d.get(json_field) is not None:
                try:
                    d[json_field] = json.loads(d[json_field])
                except Exception:
                    d[json_field] = None
        return d
