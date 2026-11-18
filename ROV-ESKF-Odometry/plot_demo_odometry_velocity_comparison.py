"""Plot DAIO velocity figures over complete demonstration scoring intervals for the paper.

The columns show demonstrations 029 and 030, which have the lowest world-frame XY velocity RMSE
over their complete scoring intervals. Rows show world-frame vx, vy, Euler yaw-rate, and XY trajectories.
Each record starts at 0 s to accumulate network history and initialize the ESKF. Figures retain the
complete scoring interval after 8 s and compare offline AprilTag Ground Truth with DAIO (DAIO-Net + 11-state ESKF).

This script reuses existing ESKF replay results without retraining networks or rerunning the filter:
    python plot_demo_odometry_velocity_comparison.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from rov_eskf.data import load_sequence  # noqa: E402
from rov_eskf.evaluate import offline_reference  # noqa: E402
from rov_eskf.yaw_source import causal_yaw_rate_source  # noqa: E402


RUN_NAME = "20260902-174312_v24_solved_yaw_rad"
DEMOS = ("029_demo__segment_001", "030_demo__segment_001")
COLORS = {"ground_truth": "#222222", "daio": "#0F4D92"}


def _record_curves(record: str, run_dir: Path, config: dict) -> dict:
    """Read a full replay initialized over 8 s and construct Ground Truth and DAIO curves."""
    directory = run_dir / record
    metadata = json.loads((directory / "window_metadata.json").read_text(encoding="utf-8"))
    sequence = load_sequence(metadata["source_hdf5"], require_reference=True)
    payload = np.load(directory / "state.npz")
    ts = np.asarray(payload["ts"], dtype=float)
    state = np.asarray(payload["state"], dtype=float)
    reference_xy, reference_yaw, reference_velocity = offline_reference(
        sequence, ts, int(config["evaluation"]["reference_savgol_window"]),
        int(config["evaluation"]["reference_savgol_polyorder"]),
    )
    # ESKF nominal yaw-rate = strictly causal IMU-solved yaw difference + estimated slow correction.
    source = causal_yaw_rate_source(sequence, config["filter"]["yaw_rate_source"])
    imu_rate = np.asarray([source.rate_at(float(time)) for time in ts])
    daio_yaw_rate = imu_rate + state[:, 14]
    ground_truth_yaw_rate = np.gradient(reference_yaw, ts, edge_order=1)
    return {
        "record": record,
        "ts": ts,
        "ground_truth": np.column_stack((reference_velocity[:, :2], ground_truth_yaw_rate)),
        "daio": np.column_stack((state[:, 3:5], daio_yaw_rate)),
        # Use the scoring start as the local origin for both trajectories to overlay their XY shapes.
        "ground_truth_xy": reference_xy - reference_xy[0],
        "daio_xy": state[:, :2] - state[0, :2],
        "startup_s": float(metadata["effective_start_s"]),
        "duration_s": float(ts[-1] - ts[0]),
    }


def _row_limits(records: list[dict], row: int) -> tuple[float, float]:
    """Use shared vertical limits for each quantity across the four demonstrations."""
    values = np.concatenate([np.r_[entry["ground_truth"][:, row], entry["daio"][:, row]] for entry in records])
    span = max(float(values.max() - values.min()), 0.02)
    margin = 0.10 * span
    return float(values.min() - margin), float(values.max() + margin)


def _plot(records: list[dict], pdf: Path, png: Path) -> None:
    """Export a double-column 4x2 figure using complete scoring intervals without selecting local windows."""
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 7.2, "axes.labelsize": 7.2, "axes.titlesize": 7.8,
        "xtick.labelsize": 6.4, "ytick.labelsize": 6.4, "pdf.fonttype": 42, "ps.fonttype": 42,
        "axes.spines.top": False, "axes.spines.right": False,
    })
    figure, axes = plt.subplots(4, 2, figsize=(7.16, 4.85), sharex=False)
    labels = (r"World $v_x^W$ [m/s]", r"World $v_y^W$ [m/s]", r"Yaw rate [rad/s]")
    limits = [_row_limits(records, row) for row in range(3)]
    for column, entry in enumerate(records):
        time = entry["ts"] - entry["ts"][0]
        duration = float(time[-1])
        x_max = max(5.0, np.ceil(duration / 10.0) * 10.0)
        for row in range(3):
            axis = axes[row, column]
            axis.plot(time, entry["ground_truth"][:, row], color=COLORS["ground_truth"], lw=0.85,
                      label="Ground Truth" if row == 0 and column == 0 else None)
            axis.plot(time, entry["daio"][:, row], color=COLORS["daio"], lw=0.9,
                      label="DAIO" if row == 0 and column == 0 else None)
            axis.set(xlim=(0.0, x_max), ylim=limits[row])
            axis.grid(alpha=0.22, linewidth=0.45)
            if row == 0:
                axis.set_title(f"Demo {entry['record'][:3]}\n({duration:.1f} s)", pad=3)
            if column == 0:
                axis.set_ylabel(labels[row])
            if row == 2:
                axis.set_xlabel("Time after initialization [s]")
        # The bottom row shows the planar trajectory over the same complete scoring interval.
        xy_axis = axes[3, column]
        xy_axis.plot(entry["ground_truth_xy"][:, 0], entry["ground_truth_xy"][:, 1],
                     color=COLORS["ground_truth"], lw=0.9)
        xy_axis.plot(entry["daio_xy"][:, 0], entry["daio_xy"][:, 1], color=COLORS["daio"], lw=1.0)
        xy_axis.set_aspect("equal", adjustable="box")
        xy_axis.grid(alpha=0.22, linewidth=0.45)
        xy_axis.set_xlabel(r"World $x^W$ [m]")
        if column == 0:
            xy_axis.set_ylabel(r"World $y^W$ [m]")
    handles, names = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, names, ncol=2, loc="upper center", bbox_to_anchor=(0.5, 1.01),
                  frameon=False, columnspacing=1.8, handlelength=2.2, fontsize=7.2)
    figure.subplots_adjust(left=0.075, right=0.995, bottom=0.09, top=0.875, hspace=0.42, wspace=0.20)
    pdf.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(pdf, bbox_inches="tight", pad_inches=0.02)
    figure.savefig(png, dpi=600, bbox_inches="tight", pad_inches=0.02)
    plt.close(figure)


def main() -> None:
    run_dir = ROOT / "runs" / RUN_NAME
    config = yaml.safe_load((run_dir / "resolved_config.yaml").read_text(encoding="utf-8"))
    records = [_record_curves(record, run_dir, config) for record in DEMOS]
    figure_dir = ROOT.parent / "RA-L Manuscript" / "ieeeconf" / "figures"
    _plot(records, figure_dir / "odometry_velocity_comparison.pdf", figure_dir / "odometry_velocity_comparison.png")
    # Save full trajectory durations to link paper figures to archived data.
    audit_dir = run_dir / "paper_odometry_velocity_figure"
    audit_dir.mkdir(exist_ok=True)
    audit = [{"record": item["record"], "startup_s": item["startup_s"], "plotted_duration_s": item["duration_s"]}
             for item in records]
    (audit_dir / "full_trajectory_windows.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(f"Saved: {figure_dir / 'odometry_velocity_comparison.pdf'}")
    for item in audit:
        print(f"{item['record']}: 0--{item['startup_s']:.1f} s initialization; "
              f"plotted post-initialization duration={item['plotted_duration_s']:.2f} s")


if __name__ == "__main__":
    main()
