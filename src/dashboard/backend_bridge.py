
import json
import os
from pathlib import Path
import subprocess
import tempfile

import numpy as np


DEFAULT_BACKEND_ROOT = Path(
    r"C:\Users\Hi\Downloads\Robust-Hyperspectral-Anomaly-"
)


def run_backend_pipeline(cube, gt=None, timeout=900):
    """Run the separate backend environment without importing its src package."""
    backend_root = Path(
        os.environ.get("HSI_BACKEND_ROOT", str(DEFAULT_BACKEND_ROOT))
    ).resolve()

    python_exe = backend_root / ".venv" / "Scripts" / "python.exe"
    runner = backend_root / "scripts" / "dashboard_runner.py"

    if not python_exe.is_file():
        raise FileNotFoundError(
            f"Backend Python environment not found: {python_exe}"
        )
    if not runner.is_file():
        raise FileNotFoundError(f"Backend runner not found: {runner}")

    with tempfile.TemporaryDirectory(prefix="hsi_backend_") as temp_dir:
        temp_path = Path(temp_dir)
        input_path = temp_path / "input.npz"
        output_dir = temp_path / "output"
        output_dir.mkdir()

        arrays = {"cube": np.asarray(cube, dtype=np.float32)}
        if gt is not None:
            arrays["gt"] = np.asarray(gt)

        np.savez_compressed(input_path, **arrays)

        completed = subprocess.run(
            [
                str(python_exe),
                str(runner),
                str(input_path),
                str(output_dir),
            ],
            cwd=str(backend_root),
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )

        if completed.returncode != 0:
            details = completed.stderr or completed.stdout
            raise RuntimeError(
                "Backend pipeline failed:\n" + details[-8000:]
            )

        result_path = output_dir / "results.npz"
        metadata_path = output_dir / "metadata.json"

        if not result_path.is_file() or not metadata_path.is_file():
            raise RuntimeError(
                "Backend finished without producing its expected result files."
            )

        with np.load(result_path, allow_pickle=False) as archive:
            results = {
                key: archive[key].copy()
                for key in archive.files
            }

        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

        fused = results["fused"]
        if fused.shape != np.asarray(cube).shape[:2]:
            raise ValueError(
                f"Backend returned score shape {fused.shape}; "
                f"expected {np.asarray(cube).shape[:2]}."
            )

        return {
            "scores": {
                key.removeprefix("score_"): value
                for key, value in results.items()
                if key.startswith("score_")
            },
            "fused": fused,
            "mask": results["mask"],
            "metadata": metadata,
            "stdout": completed.stdout,
        }
