"""Direct command-line entry point for ESKF odometry.

Run Python from the project root without setting PYTHONPATH or installing the package:

    python run_odometry.py --config configs/benchmark_apriltag_init.yaml --split demo
    python run_odometry.py --config configs/deployment_relative_yaw.yaml --record <ABSOLUTE_HDF5_PATH>
    python run_odometry.py --config configs/benchmark_apriltag_init.yaml --split demo --device cpu

Arguments match rov_eskf.cli: --config / --split / --record / --out / --device.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Add src to sys.path for standalone execution (equivalent to python -m rov_eskf.cli).
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from rov_eskf.cli import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
