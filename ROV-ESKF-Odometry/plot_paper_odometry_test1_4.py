"""Plot paper Test 1--4 ESKF evaluation from the current DAIO-Net replay.

The script reads a completed ESKF run. It never reruns the filter. Velocity and
position metrics use every 200 Hz scored sample; only displayed curves are
decimated. AprilTag is an offline reference. Estimated XY is translated once
at the scoring start to the AprilTag world origin, without trajectory fitting.

Example:
    python plot_paper_odometry_test1_4.py --run runs/<paper-test-run>
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "daio_paper_odometry_mpl"))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy.signal import savgol_filter

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
from rov_eskf.data import load_sequence  # noqa: E402
from rov_eskf.evaluate import offline_reference  # noqa: E402


DATA_ROOT = ROOT.parent / "dataset"
DEFAULT_OUTPUT = ROOT.parent / "RA-L Manuscript" / "ieeeconf" / "figures" / "odometry_test1_4"
PAPER_CHECKPOINT = (ROOT.parent / "ROV-IO-Training" / "runs" / "v24_full_val26_27_seed42"
                    / "20260915-144055_seed42" / "best.pt")
TEST_SOURCES = {
    1: "025_demo__segment_001.csv",
    2: "028_demo__segment_001.csv",
    3: "029_demo__segment_001.csv",
    4: "030_demo__segment_001.csv",
}
COLORS = {
    "truth": "#263746",
    "odometry": "#009E87",
    "x": "#6D89B5",
    "y": "#59BFA4",
    "xy": "#E3A05B",
}
TRUTH_VELOCITY_DISPLAY_SMOOTH_S = 1.5
TRUTH_XY_DISPLAY_SMOOTH_S = 1.0


@dataclass(frozen=True)
class Trial:
    number: int
    time: np.ndarray
    truth_xy: np.ndarray
    odometry_xy: np.ndarray
    truth_velocity: np.ndarray
    odometry_velocity: np.ndarray
    position_error: np.ndarray
    velocity_error: np.ndarray
    metrics: dict[str, object]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _rmse(values: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.asarray(values, dtype=float) ** 2)))


def _vector_rmse(values: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.sum(np.asarray(values, dtype=float) ** 2, axis=1))))


def _check_close(actual: float, expected: float, description: str) -> None:
    if not np.isclose(actual, expected, rtol=1e-7, atol=1e-9):
        raise ValueError(f"{description}: recomputed {actual:.10g}, replay report {expected:.10g}")


def load_trials(run_dir: Path, expected_checkpoint: Path = PAPER_CHECKPOINT) -> tuple[list[Trial], dict[str, object]]:
    config = yaml.safe_load((run_dir / "resolved_config.yaml").read_text(encoding="utf-8"))
    run_metadata = json.loads((run_dir / "run_metadata.json").read_text(encoding="utf-8"))
    run_summary = json.loads((run_dir / "replay_summary.json").read_text(encoding="utf-8"))
    expected_checkpoint_hash = sha256(expected_checkpoint)
    if run_metadata["checkpoint_sha256"] != expected_checkpoint_hash:
        raise ValueError("Replay checkpoint hash differs from the requested paper model")
    if Path(config["model"]["checkpoint"]).resolve() != expected_checkpoint.resolve():
        raise ValueError("Replay config points to a different checkpoint")
    if run_summary.get("mode") != "benchmark_apriltag_init":
        raise ValueError("Expected the existing AprilTag-at-start benchmark protocol")
    if [row["record"] for row in run_summary["records"]] != [f"test{i}" for i in range(1, 5)]:
        raise ValueError("Replay must contain Test 1--4, in order, and no other records")
    configured_paths = [Path(item["path"]).resolve() for item in config["data"]["records"]]
    expected_paths = [(DATA_ROOT / "paper_split_v24" / "test" / f"test{i}.csv").resolve()
                      for i in range(1, 5)]
    if configured_paths != expected_paths:
        raise ValueError("Replay inputs do not match the four paper_split_v24 test copies")
    if any(float(item["start_s"]) != 8.0 or item["end_s"] is not None
           for item in config["data"]["records"]):
        raise ValueError("Expected continuous scoring from 8 s through each record end")

    smoothing = config["evaluation"]
    trials: list[Trial] = []
    source_hashes: dict[str, str] = {}
    for number in range(1, 5):
        test_path = expected_paths[number - 1]
        original_path = DATA_ROOT / "processed" / "v2" / TEST_SOURCES[number]
        test_hash = sha256(test_path)
        if test_hash != sha256(original_path):
            raise ValueError(f"Test {number} CSV is not a byte-identical copy of {original_path.name}")
        source_hashes[f"test{number}"] = test_hash
        directory = run_dir / f"test{number}"
        window_metadata = json.loads((directory / "window_metadata.json").read_text(encoding="utf-8"))
        if Path(window_metadata["source_hdf5"]).resolve() != test_path:
            raise ValueError(f"Test {number} replay source differs from the paper split")
        if window_metadata["source_hdf5_sha256"] != test_hash:
            raise ValueError(f"Test {number} replayed CSV hash differs from the current file")

        sequence = load_sequence(test_path, require_reference=True)
        with np.load(directory / "state.npz") as payload:
            ts = np.asarray(payload["ts"], dtype=float)
            state = np.asarray(payload["state"], dtype=float)
        if len(ts) < 1000 or state.shape[0] != len(ts) or state.shape[1] < 5:
            raise ValueError(f"Test {number} state has an unexpected shape")
        if not np.all(np.diff(ts) > 0) or not np.isfinite(state).all():
            raise ValueError(f"Test {number} has invalid state timestamps or values")
        truth_xy, _, truth_velocity = offline_reference(
            sequence, ts, int(smoothing["reference_savgol_window"]),
            int(smoothing["reference_savgol_polyorder"]),
        )
        # Only one position translation, at the scoring start; world axes stay unchanged.
        position_offset = truth_xy[0] - state[0, :2]
        odometry_xy = state[:, :2] + position_offset
        odometry_velocity = state[:, 3:5]
        position_error = odometry_xy - truth_xy
        velocity_error = odometry_velocity - truth_velocity[:, :2]
        time = ts - ts[0]
        lag = int(round(5.0 / float(np.median(np.diff(time)))))
        if lag <= 0 or not np.allclose(time[lag:] - time[:-lag], 5.0, atol=0.01):
            raise ValueError(f"Test {number}: unable to form reliable 5 s relative-displacement pairs")
        relative_error = position_error[lag:] - position_error[:-lag]
        report = json.loads((directory / "metrics.json").read_text(encoding="utf-8"))
        # The legacy XY metric averages the two squared components. The paper
        # table below reports the RMS of the full 2D error vector instead.
        _check_close(_rmse(position_error), float(report["xy_position_rmse_m"]),
                     f"Test {number} legacy position RMSE")
        _check_close(_rmse(velocity_error), float(report["world_xy_velocity_rmse_mps"]),
                     f"Test {number} legacy velocity RMSE")
        _check_close(float(np.linalg.norm(position_error[-1])),
                     float(report["final_position_error_m"]), f"Test {number} final error")
        metrics: dict[str, object] = {
            "test": f"Test {number}",
            "original_record": original_path.stem,
            "duration_s": float(time[-1]),
            "sample_count": int(len(ts)),
            "velocity_x_rmse_mps": _rmse(velocity_error[:, 0]),
            "velocity_y_rmse_mps": _rmse(velocity_error[:, 1]),
            "velocity_xy_vector_rmse_mps": _vector_rmse(velocity_error),
            "position_x_rmse_m": _rmse(position_error[:, 0]),
            "position_y_rmse_m": _rmse(position_error[:, 1]),
            "position_xy_vector_rmse_m": _vector_rmse(position_error),
            "final_position_error_m": float(np.linalg.norm(position_error[-1])),
            "rpe_5s_xy_vector_rmse_m": _vector_rmse(relative_error),
        }
        trials.append(Trial(number, time, truth_xy, odometry_xy, truth_velocity[:, :2],
                            odometry_velocity, position_error, velocity_error, metrics))
    provenance: dict[str, object] = {
        "run_dir": str(run_dir),
        "checkpoint": str(expected_checkpoint),
        "checkpoint_sha256": expected_checkpoint_hash,
        "run_config_sha256": run_metadata["config_sha256"],
        "source_csv_sha256": source_hashes,
        "scoring": "0--8 s startup, then complete scored segment; AprilTag yaw reanchor at 8 s",
        "position_alignment": "single XY translation at scoring start; AprilTag world origin retained",
        "reference": "AprilTag XY Savitzky-Golay smoothing and derived world velocity from the replay config",
        "ground_truth_display_only_smoothing": {
            "method": "cubic Savitzky-Golay applied to plotted AprilTag curves only",
            "world_velocity_window_s": TRUTH_VELOCITY_DISPLAY_SMOOTH_S,
            "xy_trajectory_window_s": TRUTH_XY_DISPLAY_SMOOTH_S,
            "scoring_and_boxplots": "original offline reference; no extra display smoothing",
        },
        "metrics": {
            "component_rmse": "sqrt(mean(e_axis^2)) over all 200 Hz scored samples",
            "xy_vector_rmse": "sqrt(mean(e_x^2 + e_y^2)) over all 200 Hz scored samples",
            "final_position_error": "Euclidean XY position error at last scored sample",
            "rpe_5s_xy_vector_rmse": "sqrt(mean(||[e_p(t+5s)-e_p(t)]||_2^2))",
        },
    }
    return trials, provenance


def configure_style() -> None:
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
    })


def display_indices(count: int, max_points: int = 2500) -> np.ndarray:
    stride = max(1, int(np.ceil(count / max_points)))
    indices = np.arange(0, count, stride)
    return indices if indices[-1] == count - 1 else np.r_[indices, count - 1]


def smooth_truth_for_display(time: np.ndarray, values: np.ndarray,
                             window_s: float) -> np.ndarray:
    """Smooth only the plotted AprilTag line; retain original scored samples."""
    sample_count = int(round(window_s / float(np.median(np.diff(time)))))
    window = max(5, sample_count | 1)
    window = min(window, len(values) if len(values) % 2 else len(values) - 1)
    if window < 5:
        return values.copy()
    smoothed = savgol_filter(values, window_length=window, polyorder=3,
                             axis=0, mode="interp")
    smoothed[0] = values[0]
    smoothed[-1] = values[-1]
    return smoothed


def plot_velocity(trial: Trial, destination: Path, dpi: int) -> None:
    figure, axes = plt.subplots(2, 1, figsize=(8.0, 4.0), sharex=True, facecolor="white")
    indices = display_indices(len(trial.time))
    displayed_truth = smooth_truth_for_display(
        trial.time, trial.truth_velocity, TRUTH_VELOCITY_DISPLAY_SMOOTH_S)
    for axis_index, axis in enumerate(axes):
        axis.plot(trial.time[indices], displayed_truth[indices, axis_index],
                  color=COLORS["truth"], lw=1.6)
        axis.plot(trial.time[indices], trial.odometry_velocity[indices, axis_index],
                  color=COLORS["odometry"], lw=1.85)
        axis.set_ylabel(rf"${{}}^{{W}}v_{{{'xy'[axis_index]}}}$ [m/s]")
        axis.grid(axis="y", color="#DCE3E8", lw=0.65)
        axis.set_xlim(0, float(trial.time[-1]))
        combined = np.r_[trial.truth_velocity[:, axis_index], trial.odometry_velocity[:, axis_index]]
        spread = max(float(np.ptp(combined)), 0.05)
        axis.set_ylim(float(combined.min() - 0.08 * spread), float(combined.max() + 0.08 * spread))
    axes[-1].set_xlabel("Time(s)")
    figure.text(0.135, 0.985, f"Test {trial.number}", fontsize=15,
                fontweight="bold", va="top")
    figure.subplots_adjust(left=0.135, right=0.985, top=0.90, bottom=0.16, hspace=0.28)
    figure.savefig(destination, dpi=dpi, facecolor="white")
    plt.close(figure)


def plot_xy(trial: Trial, destination: Path, dpi: int) -> None:
    figure, axis = plt.subplots(figsize=(5.5, 5.5), facecolor="white")
    indices = display_indices(len(trial.time))
    displayed_truth = smooth_truth_for_display(
        trial.time, trial.truth_xy, TRUTH_XY_DISPLAY_SMOOTH_S)
    axis.plot(displayed_truth[indices, 0], displayed_truth[indices, 1],
              color=COLORS["truth"], lw=1.65)
    axis.plot(trial.odometry_xy[indices, 0], trial.odometry_xy[indices, 1],
              color=COLORS["odometry"], lw=1.85)
    axis.scatter(trial.truth_xy[0, 0], trial.truth_xy[0, 1], s=80, marker="o",
                 facecolors="white", edgecolors=COLORS["truth"], linewidths=2, zorder=5)
    for points, color in ((trial.truth_xy, COLORS["truth"]),
                          (trial.odometry_xy, COLORS["odometry"])):
        axis.scatter(points[-1, 0], points[-1, 1], s=72, marker="s",
                     facecolors=color, edgecolors="white", linewidths=0.9, zorder=6)
    axis.set_aspect("equal", adjustable="box")
    axis.set_xlabel(r"World $x^W$ [m]")
    axis.set_ylabel(r"World $y^W$ [m]")
    axis.grid(color="#DCE3E8", lw=0.65)
    axis.set_title(f"Test {trial.number}", loc="left", fontsize=15, fontweight="bold", pad=12)
    figure.tight_layout(pad=1.5)
    figure.savefig(destination, dpi=dpi, facecolor="white", bbox_inches="tight", pad_inches=0.08)
    plt.close(figure)


def plot_error_boxes(trials: list[Trial], kind: str, destination: Path, dpi: int) -> None:
    figure_width = 8.0 if kind == "velocity" else 7.0
    figure_height = 4.0 if kind == "velocity" else 3.7
    figure, axis = plt.subplots(figsize=(figure_width, figure_height), facecolor="white")
    positions: list[int] = []
    distributions: list[np.ndarray] = []
    box_colors: list[str] = []
    for test_index, trial in enumerate(trials):
        values = trial.velocity_error if kind == "velocity" else trial.position_error
        components = (np.abs(values[:, 0]), np.abs(values[:, 1]), np.linalg.norm(values, axis=1))
        distributions.extend(components)
        positions.extend((4 * test_index + 1, 4 * test_index + 2, 4 * test_index + 3))
        box_colors.extend((COLORS["x"], COLORS["y"], COLORS["xy"]))
    result = axis.boxplot(
        distributions, positions=positions, widths=0.66, patch_artist=True,
        showfliers=True,
        medianprops={"color": COLORS["truth"], "linewidth": 1.4},
        whiskerprops={"color": "#4A5965", "linewidth": 0.9},
        capprops={"color": "#4A5965", "linewidth": 0.9},
        flierprops={"marker": ".", "markersize": 0.9,
                    "markeredgecolor": "#4A5965", "alpha": 0.16},
    )
    for patch, color in zip(result["boxes"], box_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.8)
        patch.set_edgecolor("#4A5965")
    axis.set_xticks((2, 6, 10, 14), [f"Test {i}" for i in range(1, 5)])
    axis.set_xlim(0.2, 15.8)
    if kind == "velocity":
        axis.set_ylim(0, 0.12)
        axis.set_yticks(np.linspace(0, 0.12, 5))
    else:
        axis.set_ylim(bottom=0)
    axis.set_ylabel("Velocity error [m/s]" if kind == "velocity"
                    else "Absolute position error [m]")
    axis.grid(axis="y", color="#DCE3E8", lw=0.65)
    figure.subplots_adjust(left=0.12 if kind == "velocity" else 0.135,
                           right=0.985, top=0.96, bottom=0.16)
    figure.savefig(destination, dpi=dpi, facecolor="white")
    plt.close(figure)


def plot_legend(destination: Path, dpi: int) -> None:
    figure = plt.figure(figsize=(7.0, 0.92), facecolor="white")
    handles = [
        Line2D([], [], color=COLORS["truth"], lw=2.5),
        Line2D([], [], color=COLORS["odometry"], lw=2.7),
        Line2D([], [], color=COLORS["truth"], marker="o", markersize=8,
               markerfacecolor="white", lw=0),
        Line2D([], [], color=COLORS["truth"], marker="s", markersize=8, lw=0),
        Patch(facecolor=COLORS["x"], edgecolor="#4A5965"),
        Patch(facecolor=COLORS["y"], edgecolor="#4A5965"),
        Patch(facecolor=COLORS["xy"], edgecolor="#4A5965"),
    ]
    labels = ("Ground Truth (AprilTag)", "Odometry (DAIO + ESKF)", "Start", "End",
              r"$|e_x|$", r"$|e_y|$", r"$\|e_{xy}\|_2$")
    figure.legend(handles, labels, ncol=4, loc="center", bbox_to_anchor=(0.5, 0.5),
                  frameon=False, fontsize=11.2, handlelength=2.0, columnspacing=1.4)
    figure.savefig(destination, dpi=dpi, facecolor="white")
    plt.close(figure)


def plot_velocity_eval_legend(destination: Path, dpi: int) -> None:
    """Standalone legend for the four velocity traces and velocity-error boxes."""
    figure = plt.figure(figsize=(8.0, 4.0), facecolor="white")
    line_handles = [
        Line2D([], [], color=COLORS["truth"], lw=3.0),
        Line2D([], [], color=COLORS["odometry"], lw=3.0),
    ]
    box_handles = [
        Patch(facecolor=COLORS["x"], edgecolor="#4A5965"),
        Patch(facecolor=COLORS["y"], edgecolor="#4A5965"),
        Patch(facecolor=COLORS["xy"], edgecolor="#4A5965"),
    ]
    lines_axis = figure.add_axes((0.04, 0.15, 0.59, 0.70))
    boxes_axis = figure.add_axes((0.65, 0.15, 0.31, 0.70))
    for axis in (lines_axis, boxes_axis):
        axis.axis("off")
    lines_axis.legend(line_handles,
                      ("Ground Truth (AprilTag)", "Odometry (DAIO + ESKF)"),
                      loc="center left", frameon=False, fontsize=18,
                      handlelength=2.0, labelspacing=1.2)
    boxes_axis.legend(box_handles,
                      (r"$|e_x|$", r"$|e_y|$", r"$\|e_{xy}\|_2$"),
                      loc="center left", frameon=False, fontsize=18,
                      handlelength=2.0, labelspacing=0.8)
    figure.savefig(destination, dpi=dpi, facecolor="white")
    plt.close(figure)


def write_metrics(trials: list[Trial], provenance: dict[str, object], directory: Path) -> None:
    rows = [trial.metrics for trial in trials]
    with (directory / "metrics.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (directory / "provenance.json").write_text(json.dumps(provenance, indent=2, ensure_ascii=False),
                                                encoding="utf-8")
    lines = ["# Test 1--4 ESKF odometry metrics", "",
             "All RMSE values use every scored 200 Hz state sample. Ground truth is the existing "
             "smoothed AprilTag offline reference. A single position offset at the 8 s scoring start "
             "places the ESKF track in the original AprilTag world coordinates.", "",
             "- Axis RMSE: `sqrt(mean(e_axis^2))`.",
             "- XY vector RMSE: `sqrt(mean(e_x^2 + e_y^2))`; this is `sqrt(2)` times the "
             "legacy component-averaged XY RMSE field in the replay report.",
             "- Final position error: Euclidean XY error at the last scored sample.",
             "- 5 s relative position error: RMSE of the 2D relative-displacement residual "
             "between samples separated by 5 s.",
             "- Box plots: standard Tukey (1.5 IQR) whiskers with outliers; computed from all "
             "scored samples. X and Y boxes are absolute component errors, and XY is the 2D norm.",
             "- Velocity and trajectory lines are display-decimated only; no plotted sample is "
             "used to replace a full-sample metric.",
             "- For clearer display, the Ground Truth lines receive an additional cubic "
             f"Savitzky-Golay smoothing window of {TRUTH_VELOCITY_DISPLAY_SMOOTH_S:g} s "
             f"(world velocity) or {TRUTH_XY_DISPLAY_SMOOTH_S:g} s (XY trajectory). "
             "This display processing does not affect scores, box plots, or endpoint markers.", "",
             "| Test | Duration [s] | Velocity XY RMSE [m/s] | Position XY RMSE [m] | "
             "Final XY error [m] | 5 s relative position RMSE [m] |",
             "|---|---:|---:|---:|---:|---:|",
        ]
    for row in rows:
        lines.append(f"| {row['test']} | {row['duration_s']:.2f} | "
                     f"{row['velocity_xy_vector_rmse_mps']:.4f} | "
                     f"{row['position_xy_vector_rmse_m']:.4f} | "
                     f"{row['final_position_error_m']:.4f} | "
                     f"{row['rpe_5s_xy_vector_rmse_m']:.4f} |")
    (directory / "metrics_readme.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True, help="Completed four-test ESKF run directory")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--dpi", type=int, default=1200)
    parser.add_argument("--velocity-only", action="store_true",
                        help="Update only the four velocity curves, velocity boxplot, and separate legend")
    args = parser.parse_args()
    if args.dpi < 300:
        parser.error("--dpi must be at least 300 for paper figures")
    configure_style()
    trials, provenance = load_trials(args.run.resolve())
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    for trial in trials:
        plot_velocity(trial, output / f"test{trial.number}_world_velocity.png", args.dpi)
        if not args.velocity_only:
            plot_xy(trial, output / f"test{trial.number}_xy_trajectory.png", args.dpi)
        print(f"Test {trial.number}: {trial.metrics['sample_count']} samples, "
              f"velocity XY RMSE={trial.metrics['velocity_xy_vector_rmse_mps']:.4f} m/s, "
              f"position XY RMSE={trial.metrics['position_xy_vector_rmse_m']:.4f} m")
    plot_error_boxes(trials, "velocity", output / "velocity_error_boxplot.png", args.dpi)
    plot_velocity_eval_legend(output / "velocity_eval_legend.png", args.dpi)
    if not args.velocity_only:
        plot_error_boxes(trials, "position", output / "position_error_boxplot.png", args.dpi)
        plot_legend(output / "shared_legend.png", args.dpi)
        write_metrics(trials, provenance, output)
    print(f"Saved figures to {output}")


if __name__ == "__main__":
    main()
