"""Export four paper-ready odometry position figures and a separate legend.

Each Test 1--4 figure shows AprilTag and DAIO world-frame X/Y positions over
the full post-initialization scoring interval. The origin is fixed at the
scoring start, matching the existing ESKF benchmark and its position metrics.
Only displayed lines are decimated; box plots and reported errors use every
200 Hz state sample. Existing replay outputs are read without rerunning ESKF.

Run from any directory:
    python plot_paper_odometry_position_individual.py
"""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "rov_odometry_paper_mpl"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml
from matplotlib.lines import Line2D

from plot_demo_odometry_velocity_comparison import ROOT, RUN_NAME, _record_curves


DEMOS = (
    "025_demo__segment_001",
    "028_demo__segment_001",
    "029_demo__segment_001",
    "030_demo__segment_001",
)
FIGURE_DIR = ROOT.parent / "RA-L Manuscript" / "ieeeconf" / "figures" / "odometry_individual"
SPLIT_DIR = ROOT.parent / "dataset" / "paper_split_v24" / "test"
COLORS = {
    "Ground Truth": "#263746",
    "Odometry": "#009E87",
    "x_error": "#6D89B5",
    "y_error": "#59BFA4",
    "xy_error": "#E3A05B",
}
DPI = 1200
MAX_LINE_POINTS = 2400


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _style() -> None:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 13,
        "axes.labelsize": 14,
        "xtick.labelsize": 11.5,
        "ytick.labelsize": 11.5,
        "axes.linewidth": 1.0,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    })


def _display_indices(count: int) -> np.ndarray:
    stride = max(1, int(np.ceil(count / MAX_LINE_POINTS)))
    indices = np.arange(0, count, stride)
    if indices[-1] != count - 1:
        indices = np.r_[indices, count - 1]
    return indices


def _load(test_number: int, record: str, run_dir: Path, config: dict) -> dict:
    metadata = json.loads((run_dir / record / "window_metadata.json").read_text(encoding="utf-8"))
    test_copy = SPLIT_DIR / f"test{test_number}.csv"
    source = Path(metadata["source_hdf5"])
    expected_hash = metadata["source_hdf5_sha256"]
    if _sha256(source) != expected_hash or _sha256(test_copy) != expected_hash:
        raise ValueError(f"Test {test_number}: split CSV differs from the ESKF replay source")

    entry = _record_curves(record, run_dir, config)
    time = np.asarray(entry["ts"] - entry["ts"][0], dtype=float)
    reference = np.asarray(entry["ground_truth_xy"], dtype=float)
    estimate = np.asarray(entry["daio_xy"], dtype=float)
    if reference.shape != estimate.shape or reference.shape != (len(time), 2):
        raise ValueError(f"Test {test_number}: position and timestamp shapes differ")
    if not np.isfinite(reference).all() or not np.isfinite(estimate).all():
        raise ValueError(f"Test {test_number}: non-finite position values")
    if not np.all(np.diff(time) > 0):
        raise ValueError(f"Test {test_number}: time is not strictly increasing")

    residual = estimate - reference
    xy_norm = np.linalg.norm(residual, axis=1)
    return {
        "test": test_number,
        "record": record,
        "time": time,
        "reference": reference,
        "estimate": estimate,
        "absolute_x_error": np.abs(residual[:, 0]),
        "absolute_y_error": np.abs(residual[:, 1]),
        "xy_error_norm": xy_norm,
        "summary": {
            "test": f"Test {test_number}",
            "source_record": record,
            "paper_split_file": str(test_copy),
            "scoring_start_s": float(metadata["effective_start_s"]),
            "scoring_duration_s": float(time[-1]),
            "sample_count": int(len(time)),
            "world_frame": "local world frame, origin at scoring start",
            "x_rmse_m": float(np.sqrt(np.mean(residual[:, 0] ** 2))),
            "y_rmse_m": float(np.sqrt(np.mean(residual[:, 1] ** 2))),
            "xy_vector_rmse_m": float(np.sqrt(np.mean(xy_norm ** 2))),
        },
    }


def _plot_test(entry: dict, path: Path) -> None:
    figure = plt.figure(figsize=(7.0, 3.6), facecolor="white")
    grid = figure.add_gridspec(
        2, 2, width_ratios=(4.1, 1.35), left=0.112, right=0.985,
        top=0.92, bottom=0.155, hspace=0.24, wspace=0.28,
    )
    axes = [figure.add_subplot(grid[0, 0]), figure.add_subplot(grid[1, 0])]
    box_axis = figure.add_subplot(grid[:, 1])
    time, reference, estimate = entry["time"], entry["reference"], entry["estimate"]
    indices = _display_indices(len(time))

    for coordinate, axis in enumerate(axes):
        axis.plot(time[indices], reference[indices, coordinate],
                  color=COLORS["Ground Truth"], lw=1.65, zorder=2)
        axis.plot(time[indices], estimate[indices, coordinate],
                  color=COLORS["Odometry"], lw=1.9, zorder=3)
        axis.set_xlim(0, time[-1])
        combined = np.r_[reference[:, coordinate], estimate[:, coordinate]]
        spread = max(float(np.ptp(combined)), 0.10)
        axis.set_ylim(float(combined.min() - 0.08 * spread),
                      float(combined.max() + 0.08 * spread))
        axis.set_ylabel(rf"World ${'xy'[coordinate]}^W$ [m]")
        axis.grid(axis="y", color="#DCE3E8", lw=0.65)
        axis.tick_params(direction="out", length=3.5, width=0.8)
    axes[0].tick_params(labelbottom=False)
    axes[1].set_xlabel("Time after initialization [s]")
    axes[0].text(0.98, 0.96, f"Test {entry['test']}", transform=axes[0].transAxes,
                 ha="right", va="top", weight="bold", fontsize=15,
                 bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85, "pad": 1.5})

    boxes = box_axis.boxplot(
        [entry["absolute_x_error"], entry["absolute_y_error"], entry["xy_error_norm"]],
        widths=0.52, patch_artist=True, showfliers=True,
        medianprops={"color": "#263746", "linewidth": 1.55},
        whiskerprops={"color": "#4A5965", "linewidth": 1.0},
        capprops={"color": "#4A5965", "linewidth": 1.0},
        flierprops={"marker": ".", "markersize": 1.5, "markeredgecolor": "#4A5965", "alpha": 0.22},
    )
    for patch, color in zip(boxes["boxes"],
                            (COLORS["x_error"], COLORS["y_error"], COLORS["xy_error"])):
        patch.set_facecolor(color)
        patch.set_alpha(0.72)
        patch.set_edgecolor("#4A5965")
        patch.set_linewidth(0.9)
    box_axis.set_xticks((1, 2, 3),
                        (r"$|e_x|$", r"$|e_y|$", r"$\|e_{xy}\|_2$"))
    box_axis.set_ylim(0, float(max(entry["xy_error_norm"]) * 1.06 + 1e-6))
    box_axis.set_ylabel("Absolute position error [m]")
    box_axis.grid(axis="y", color="#DCE3E8", lw=0.65)
    box_axis.tick_params(direction="out", length=3.5, width=0.8)

    figure.savefig(path, dpi=DPI, facecolor="white")
    figure.savefig(path.with_suffix(".pdf"), facecolor="white")
    plt.close(figure)


def _plot_legend(path: Path) -> None:
    figure = plt.figure(figsize=(7.0, 0.54), facecolor="white")
    handles = [
        Line2D([], [], color=COLORS["Ground Truth"], lw=2.5),
        Line2D([], [], color=COLORS["Odometry"], lw=2.7),
    ]
    figure.legend(handles, ("Ground Truth (AprilTag)", "Odometry (DAIO + ESKF)"),
                  loc="center", bbox_to_anchor=(0.5, 0.50), ncol=2, frameon=False,
                  fontsize=14, handlelength=2.5, columnspacing=2.0)
    figure.savefig(path, dpi=DPI, facecolor="white")
    figure.savefig(path.with_suffix(".pdf"), facecolor="white")
    plt.close(figure)


def main() -> None:
    _style()
    run_dir = ROOT / "runs" / RUN_NAME
    config = yaml.safe_load((run_dir / "resolved_config.yaml").read_text(encoding="utf-8"))
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    summaries = []
    for test_number, record in enumerate(DEMOS, start=1):
        entry = _load(test_number, record, run_dir, config)
        destination = FIGURE_DIR / f"test{test_number}_odometry.png"
        _plot_test(entry, destination)
        summaries.append(entry["summary"])
        print(f"Saved {destination}: {entry['summary']['sample_count']} samples, "
              f"{entry['summary']['scoring_duration_s']:.2f} s")
    _plot_legend(FIGURE_DIR / "odometry_legend.png")
    (FIGURE_DIR / "figure_data_summary.json").write_text(
        json.dumps(summaries, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"Saved legend and provenance summary in {FIGURE_DIR}")


if __name__ == "__main__":
    main()
