"""Compare the paper full and no-RPM networks after identical ESKF fusion.

python plot_paper_no_rpm_comparison.py --no-rpm-run runs/<no-rpm-replay>
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import yaml
from matplotlib.legend_handler import HandlerTuple
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.text import Text

import plot_paper_odometry_test1_4 as base
from audit_paper_no_rpm import NO_RPM_CHECKPOINT

DEFAULT_FULL_RUN = base.ROOT / "runs" / "20260930-164118_paper_test1_4_current_model_8s"
DEFAULT_OUTPUT = base.DEFAULT_OUTPUT / "no_rpm_comparison"
FULL_LABEL = "DAIO-Net + ESKF"
NO_RPM_LABEL = "Without RPM + ESKF"
FULL_COLOR = base.COLORS["odometry"]
NO_RPM_COLOR = "#D55E00"
FONT_SCALE = 2.0


def save_figure(figure, path: Path, dpi: int) -> None:
    """Reject clipped text before exporting a fixed-aspect publication canvas."""
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    width, height = figure.canvas.get_width_height()
    unused_tick_labels = set()
    for axes in figure.axes:
        for axis in (axes.xaxis, axes.yaxis):
            lower, upper = sorted(axis.get_view_interval())
            for tick in axis.get_major_ticks() + axis.get_minor_ticks():
                if tick.get_loc() < lower - 1e-9 or tick.get_loc() > upper + 1e-9:
                    unused_tick_labels.update((id(tick.label1), id(tick.label2)))
    for item in figure.findobj(match=Text):
        if not item.get_visible() or not item.get_text() or id(item) in unused_tick_labels:
            continue
        bounds = item.get_window_extent(renderer)
        if bounds.x0 < -1 or bounds.y0 < -1 or bounds.x1 > width + 1 or bounds.y1 > height + 1:
            raise ValueError(f"{path.name}: text outside canvas: {item.get_text()!r}")
    figure.savefig(path, dpi=dpi, facecolor="white")


def validate_comparison(full_run: Path, no_rpm_run: Path,
                        full: list[base.Trial], no_rpm: list[base.Trial]) -> None:
    configs = [yaml.safe_load((path / "resolved_config.yaml").read_text(encoding="utf-8"))
               for path in (full_run, no_rpm_run)]
    for config in configs:
        config["model"].pop("checkpoint")
        config.pop("output")
    if configs[0] != configs[1]:
        raise ValueError("ESKF protocols differ beyond network checkpoint and output directory")
    for left, right in zip(full, no_rpm):
        if left.number != right.number:
            raise ValueError("Test order differs")
        np.testing.assert_array_equal(left.time, right.time)
        np.testing.assert_array_equal(left.truth_xy, right.truth_xy)
        np.testing.assert_array_equal(left.truth_velocity, right.truth_velocity)
        with np.load(full_run / f"test{left.number}" / "state.npz") as a:
            with np.load(no_rpm_run / f"test{right.number}" / "state.npz") as b:
                np.testing.assert_array_equal(a["ts"], b["ts"])


def plot_velocity(full: base.Trial, no_rpm: base.Trial, path: Path, dpi: int) -> None:
    figure, axes = base.plt.subplots(2, 1, figsize=(8, 4), sharex=True, facecolor="white")
    indices = base.display_indices(len(full.time))
    truth = base.smooth_truth_for_display(full.time, full.truth_velocity,
                                         base.TRUTH_VELOCITY_DISPLAY_SMOOTH_S)
    for component, axis in enumerate(axes):
        axis.plot(full.time[indices], truth[indices, component],
                  color=base.COLORS["truth"], lw=1.65)
        axis.plot(no_rpm.time[indices], no_rpm.odometry_velocity[indices, component],
                  color=NO_RPM_COLOR, lw=1.65, linestyle=(0, (4, 2)))
        axis.plot(full.time[indices], full.odometry_velocity[indices, component],
                  color=FULL_COLOR, lw=1.85)
        axis.set_ylabel(rf"${{}}^{{W}}v_{{{'xy'[component]}}}$" + "\n[m/s]",
                        labelpad=8)
        axis.grid(axis="y", color="#DCE3E8", lw=0.65)
        axis.set_xlim(0, float(full.time[-1]))
        values = np.r_[full.truth_velocity[:, component], full.odometry_velocity[:, component],
                       no_rpm.odometry_velocity[:, component]]
        spread = max(float(np.ptp(values)), 0.05)
        axis.set_ylim(float(values.min() - 0.08 * spread), float(values.max() + 0.08 * spread))
    axes[-1].set_xlabel("Time(s)")
    figure.text(0.29, 0.985, f"Test {full.number}", fontsize=15 * FONT_SCALE,
                fontweight="bold", va="top")
    figure.subplots_adjust(left=0.29, right=0.985, top=0.80, bottom=0.25, hspace=0.50)
    save_figure(figure, path, dpi)
    base.plt.close(figure)


def plot_boxes(full: list[base.Trial], no_rpm: list[base.Trial], kind: str,
               path: Path, dpi: int) -> None:
    figure, axis = base.plt.subplots(figsize=(13.5, 4.5), facecolor="white")
    values, positions, colors = [], [], []
    ticks, labels = [], []
    for test_index, (left, right) in enumerate(zip(full, no_rpm)):
        errors = [trial.velocity_error if kind == "velocity" else trial.position_error
                  for trial in (left, right)]
        components = [(np.abs(error[:, 0]), np.abs(error[:, 1]), np.linalg.norm(error, axis=1))
                      for error in errors]
        for component in range(3):
            offset = 6 * test_index + 2 * component + 1
            values.extend([components[0][component], components[1][component]])
            positions.extend([offset, offset + 1])
            colors.extend([FULL_COLOR, NO_RPM_COLOR])
            ticks.append(offset + 0.5)
            labels.append((r"$|e_x|$", r"$|e_y|$", r"$\|e_{xy}\|_2$")[component])
        axis.text(6 * test_index + 3.5, -0.30, f"Test {left.number}",
                  transform=axis.get_xaxis_transform(), ha="center", va="top",
                  fontsize=12 * FONT_SCALE)
    boxes = axis.boxplot(
        values, positions=positions, widths=0.65, patch_artist=True, showfliers=True,
        medianprops={"color": base.COLORS["truth"], "linewidth": 1.2},
        whiskerprops={"color": "#4A5965", "linewidth": 0.8},
        capprops={"color": "#4A5965", "linewidth": 0.8},
        flierprops={"marker": ".", "markersize": 0.8,
                    "markeredgecolor": "#4A5965", "alpha": 0.12},
    )
    for box, color in zip(boxes["boxes"], colors):
        box.set_facecolor(color)
        box.set_alpha(0.75)
        box.set_edgecolor("#4A5965")
    axis.set_xticks(ticks, labels, fontsize=9.5 * FONT_SCALE)
    axis.set_xlim(0.2, 24.8)
    for boundary in (6.5, 12.5, 18.5):
        axis.axvline(boundary, color="#DCE3E8", lw=0.65, zorder=0)
    if kind == "velocity":
        axis.set_ylim(0, 0.12)
        axis.set_yticks(np.linspace(0, 0.12, 5))
    else:
        axis.set_ylim(bottom=0)
    axis.set_ylabel("Velocity error\n[m/s]" if kind == "velocity" else "Position error\n[m]")
    axis.grid(axis="y", color="#DCE3E8", lw=0.65)
    figure.subplots_adjust(left=0.16, right=0.985, top=0.94, bottom=0.32)
    save_figure(figure, path, dpi)
    base.plt.close(figure)


def plot_legend(path: Path, dpi: int) -> None:
    figure = base.plt.figure(figsize=(8, 4), facecolor="white")
    handles = [Line2D([], [], color=base.COLORS["truth"], lw=3),
               (Line2D([], [], color=FULL_COLOR, lw=3),
                Patch(facecolor=FULL_COLOR, edgecolor="#4A5965", alpha=0.75)),
               (Line2D([], [], color=NO_RPM_COLOR, lw=3, linestyle=(0, (4, 2))),
                Patch(facecolor=NO_RPM_COLOR, edgecolor="#4A5965", alpha=0.75))]
    figure.legend(handles, ("Ground Truth (AprilTag)", FULL_LABEL, NO_RPM_LABEL),
                  handler_map={tuple: HandlerTuple(ndivide=None, pad=0.5)},
                  loc="center", frameon=False, fontsize=18 * FONT_SCALE,
                  handlelength=3.5, labelspacing=1.2)
    save_figure(figure, path, dpi)
    base.plt.close(figure)


def write_report(full: list[base.Trial], no_rpm: list[base.Trial], full_run: Path,
                 no_rpm_run: Path, provenance: dict, output: Path) -> None:
    rows = []
    for label, trials, run in ((FULL_LABEL, full, full_run), (NO_RPM_LABEL, no_rpm, no_rpm_run)):
        for trial in trials:
            replay_metrics = json.loads((run / f"test{trial.number}" / "metrics.json").read_text(encoding="utf-8"))
            rows.append({"model": label, **trial.metrics,
                         "yaw_rmse_rad": replay_metrics["yaw_rmse_rad"],
                         "network_xy_acceptance_rate": replay_metrics["updates"]["network_xy"]["acceptance_rate"],
                         "network_yaw_rate_acceptance_rate": replay_metrics["updates"]["network_yaw_rate"]["acceptance_rate"]})
    with (output / "metrics_comparison.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = ["# Test 1--4: full network versus without RPM, after ESKF", "",
             "The only replay setting changed is the network checkpoint. Both use the same "
             "0--8 s warmup, one AprilTag yaw initialization at 8 s, 11-state ESKF, IMU/depth "
             "inputs, covariance scaling, NIS gates, scoring timestamps, and AprilTag reference. "
             "Both network XY velocity and network yaw-rate observations are replaced by the "
             "respective model's predictions and learned uncertainty.", "",
             "Metrics use every 200 Hz scored state sample. Each odometry path receives one "
             "XY translation at the scoring start, retaining the AprilTag world origin. "
             "Ground-truth display smoothing (1.5 s) and line decimation do not enter scores.", "",
             "- Component RMSE: sqrt(mean(e_axis^2)).",
             "- 2D vector RMSE: sqrt(mean(e_x^2 + e_y^2)); not the component-averaged legacy RMSE.",
             "- Final position error: norm of the last XY residual.",
             "- 5 s RPE: sqrt(mean(||e_p(t+5)-e_p(t)||^2)).",
             "- Box plots: all samples, absolute X/Y residuals and XY norm; Tukey 1.5 IQR whiskers. "
             "The velocity display is clipped at 0.12 m/s, including clipping higher outliers. "
             "Clipped values still enter every score.", "",
             "| Test | Model | Velocity 2D RMSE [m/s] | Position 2D RMSE [m] | Final XY [m] | 5 s RPE [m] |",
             "|---|---|---:|---:|---:|---:|"]
    for number in range(1, 5):
        for row in rows:
            if row["test"] == f"Test {number}":
                lines.append(f"| {row['test']} | {row['model']} | {row['velocity_xy_vector_rmse_mps']:.4f} "
                             f"| {row['position_xy_vector_rmse_m']:.4f} | {row['final_position_error_m']:.4f} "
                             f"| {row['rpe_5s_xy_vector_rmse_m']:.4f} |")
    lines.extend(["", "## Aggregation", "",
                  "Macro mean is the unweighted mean of the four per-trajectory RMSE values. "
                  "Pooled RMSE uses every scored sample across the four trajectories.", ""])
    aggregates = {}
    for label, trials in ((FULL_LABEL, full), (NO_RPM_LABEL, no_rpm)):
        values = {"macro_velocity_xy_rmse_mps": float(np.mean([
                      trial.metrics["velocity_xy_vector_rmse_mps"] for trial in trials])),
                  "macro_position_xy_rmse_m": float(np.mean([
                      trial.metrics["position_xy_vector_rmse_m"] for trial in trials])),
                  "pooled_velocity_xy_rmse_mps": base._vector_rmse(np.concatenate([
                      trial.velocity_error for trial in trials])),
                  "pooled_position_xy_rmse_m": base._vector_rmse(np.concatenate([
                      trial.position_error for trial in trials]))}
        aggregates[label] = values
        lines.append(f"- {label}: {json.dumps(values)}")
    (output / "comparison_readme.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (output / "aggregate_metrics.json").write_text(json.dumps(aggregates, indent=2), encoding="utf-8")
    provenance["plot_script_sha256"] = base.sha256(Path(__file__))
    provenance["figure_style"] = {"font_scale_vs_original": FONT_SCALE,
                                  "velocity_and_legend_size_inches": [8, 4],
                                  "boxplot_size_inches": [13.5, 4.5]}
    provenance["runtime_source_sha256"] = {
        str(path.relative_to(base.ROOT)): base.sha256(path) for path in (
            base.ROOT / "src" / "rov_eskf" / "model.py",
            base.ROOT / "src" / "rov_eskf" / "windows.py")}
    (output / "provenance.json").write_text(json.dumps(provenance, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(aggregates, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full-run", type=Path, default=DEFAULT_FULL_RUN)
    parser.add_argument("--no-rpm-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--dpi", type=int, default=1200)
    args = parser.parse_args()
    if args.dpi < 300:
        parser.error("dpi must be at least 300")
    full_run, no_rpm_run = args.full_run.resolve(), args.no_rpm_run.resolve()
    full, full_provenance = base.load_trials(full_run)
    no_rpm, no_rpm_provenance = base.load_trials(no_rpm_run, expected_checkpoint=NO_RPM_CHECKPOINT)
    validate_comparison(full_run, no_rpm_run, full, no_rpm)
    audit_path = base.ROOT / "reports" / "paper_no_rpm_inference_audit.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if audit["models"]["no_rpm"]["sha256"] != no_rpm_provenance["checkpoint_sha256"]:
        raise ValueError("inference audit belongs to another no-RPM checkpoint")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    base.configure_style()
    for key in ("font.size", "axes.labelsize", "xtick.labelsize", "ytick.labelsize"):
        base.plt.rcParams[key] = float(base.plt.rcParams[key]) * FONT_SCALE
    for left, right in zip(full, no_rpm):
        plot_velocity(left, right, output / f"test{left.number}_world_velocity_comparison.png", args.dpi)
    plot_boxes(full, no_rpm, "velocity", output / "velocity_error_boxplot_comparison.png", args.dpi)
    plot_boxes(full, no_rpm, "position", output / "position_error_boxplot_comparison.png", args.dpi)
    plot_legend(output / "comparison_legend.png", args.dpi)
    write_report(full, no_rpm, full_run, no_rpm_run,
                 {"full": full_provenance, "no_rpm": no_rpm_provenance,
                  "identical_scoring_timestamps_and_reference": True,
                  "filter_config_diff": "network checkpoint only",
                  "inference_audit": audit}, output)
    print(f"Saved comparison figures and full-sample metrics: {output}")


if __name__ == "__main__":
    main()
