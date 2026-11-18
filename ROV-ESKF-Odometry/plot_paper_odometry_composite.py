"""Compose the four Test 1--4 velocity evaluations into one paper figure.

Layout follows the manuscript mockup: Test 2 / Test 4 at the top, Test 1
below Test 2, Test 3 at the lower left, and the full-sample velocity-error
boxplot with a shared legend at the lower right. XY tracks are omitted.

Example:
    python plot_paper_odometry_composite.py --run runs/<paper-test-run>
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

import plot_paper_odometry_test1_4 as source


def draw_velocity_pair(figure, slot, trial: source.Trial) -> None:
    grid = slot.subgridspec(2, 1, hspace=0.15)
    axes = [figure.add_subplot(grid[0]), figure.add_subplot(grid[1])]
    indices = source.display_indices(len(trial.time))
    truth_display = source.smooth_truth_for_display(
        trial.time, trial.truth_velocity,
        source.TRUTH_VELOCITY_DISPLAY_SMOOTH_S,
    )
    for component, axis in enumerate(axes):
        axis.plot(trial.time[indices], truth_display[indices, component],
                  color=source.COLORS["truth"], linewidth=1.45)
        axis.plot(trial.time[indices], trial.odometry_velocity[indices, component],
                  color=source.COLORS["odometry"], linewidth=1.55)
        axis.set_ylabel(rf"${{}}^{{W}}v_{{{'xy'[component]}}}$ [m/s]", fontsize=12.5)
        axis.grid(axis="y", color="#DCE3E8", linewidth=0.6)
        axis.set_xlim(0, float(trial.time[-1]))
        values = np.r_[trial.truth_velocity[:, component],
                       trial.odometry_velocity[:, component]]
        spread = max(float(np.ptp(values)), 0.05)
        axis.set_ylim(float(values.min() - 0.08 * spread),
                      float(values.max() + 0.08 * spread))
        axis.tick_params(labelsize=11, pad=2)
    axes[0].set_title(f"Test {trial.number}", loc="left", fontsize=14,
                      fontweight="bold", pad=6)
    axes[0].tick_params(labelbottom=False)
    axes[1].set_xlabel("Time(s)", fontsize=12.5,
                       labelpad=2)


def draw_velocity_boxplot(axis, trials: list[source.Trial]) -> None:
    positions: list[int] = []
    distributions: list[np.ndarray] = []
    colors: list[str] = []
    for index, trial in enumerate(trials):
        values = trial.velocity_error
        distributions.extend((np.abs(values[:, 0]), np.abs(values[:, 1]),
                              np.linalg.norm(values, axis=1)))
        positions.extend((4 * index + 1, 4 * index + 2, 4 * index + 3))
        colors.extend((source.COLORS["x"], source.COLORS["y"],
                       source.COLORS["xy"]))
    boxes = axis.boxplot(
        distributions, positions=positions, widths=0.66, patch_artist=True,
        showfliers=True,
        medianprops={"color": source.COLORS["truth"], "linewidth": 1.3},
        whiskerprops={"color": "#4A5965", "linewidth": 0.85},
        capprops={"color": "#4A5965", "linewidth": 0.85},
        flierprops={"marker": ".", "markersize": 0.8,
                    "markeredgecolor": "#4A5965", "alpha": 0.16},
    )
    for patch, color in zip(boxes["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.8)
        patch.set_edgecolor("#4A5965")
    axis.set_xticks((2, 6, 10, 14), [f"Test {i}" for i in range(1, 5)])
    axis.set_xlim(0.2, 15.8)
    axis.set_ylim(0, 0.12)
    axis.set_yticks(np.linspace(0, 0.12, 5))
    axis.set_ylabel("Velocity error [m/s]", fontsize=12.5)
    axis.tick_params(labelsize=11, pad=2)
    axis.grid(axis="y", color="#DCE3E8", linewidth=0.6)


def draw_legend(axis) -> None:
    handles = [
        Line2D([], [], color=source.COLORS["truth"], linewidth=2.2),
        Line2D([], [], color=source.COLORS["odometry"], linewidth=2.3),
        Patch(facecolor=source.COLORS["x"], edgecolor="#4A5965"),
        Patch(facecolor=source.COLORS["y"], edgecolor="#4A5965"),
        Patch(facecolor=source.COLORS["xy"], edgecolor="#4A5965"),
    ]
    labels = ("Ground Truth (AprilTag)", "Odometry (DAIO + ESKF)",
              r"$|e_x|$", r"$|e_y|$", r"$\|e_{xy}\|_2$")
    axis.axis("off")
    axis.legend(handles, labels, ncol=2, loc="center", frameon=False,
                fontsize=12.2, handlelength=1.7, columnspacing=0.7,
                labelspacing=0.6, borderaxespad=0.0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path,
                        default=source.DEFAULT_OUTPUT / "test1_4_velocity_composite.png")
    parser.add_argument("--dpi", type=int, default=1200)
    args = parser.parse_args()
    if args.dpi < 300:
        parser.error("--dpi must be at least 300 for paper figures")

    source.configure_style()
    trials, _ = source.load_trials(args.run.resolve())
    by_number = {trial.number: trial for trial in trials}
    figure = source.plt.figure(figsize=(10.5, 8.35), facecolor="white")
    grid = figure.add_gridspec(
        3, 2, left=0.095, right=0.985, top=0.965, bottom=0.06,
        wspace=0.27, hspace=0.34,
    )
    draw_velocity_pair(figure, grid[0, 0], by_number[2])
    draw_velocity_pair(figure, grid[1, 0], by_number[1])
    draw_velocity_pair(figure, grid[2, 0], by_number[3])
    draw_velocity_pair(figure, grid[0, 1], by_number[4])
    lower_right = grid[1:, 1].subgridspec(2, 1, height_ratios=(1.5, 0.5),
                                          hspace=0.08)
    draw_velocity_boxplot(figure.add_subplot(lower_right[0]), trials)
    draw_legend(figure.add_subplot(lower_right[1]))

    destination = args.output.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=args.dpi, facecolor="white")
    source.plt.close(figure)
    print(f"Saved {destination}")


if __name__ == "__main__":
    main()
