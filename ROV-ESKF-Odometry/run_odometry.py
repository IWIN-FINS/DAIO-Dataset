"""ESKF 里程计命令行直启入口。

在工程根目录下直接用 python 运行，无需设置 PYTHONPATH、无需 pip install：

    python run_odometry.py --config configs/benchmark_apriltag_init.yaml --split demo
    python run_odometry.py --config configs/deployment_relative_yaw.yaml --record <HDF5绝对路径>
    python run_odometry.py --config configs/benchmark_apriltag_init.yaml --split demo --device cpu

参数同 rov_eskf.cli：--config / --split / --record / --out / --device。
"""
from __future__ import annotations

import sys
from pathlib import Path

# 把 src 目录加入 sys.path，使本脚本可独立启动（也可用 python -m rov_eskf.cli 等价运行）
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from rov_eskf.cli import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
