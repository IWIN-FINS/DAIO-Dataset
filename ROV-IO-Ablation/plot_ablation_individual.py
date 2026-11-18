"""Export four high-resolution ablation panels and a separate shared legend.

Run with the rov-io Python environment::

    python plot_ablation_individual.py

Each panel keeps the original one-second-mean error curve and raw-sample box
plot. The vertical plotting area is two-thirds of a 3-inch reference height;
the numerical Y range is unchanged so that no peaks are clipped.
"""

from __future__ import annotations

import plot_ablation_figure as ablation
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator


OUTPUT_DIR = ablation.PAPER_FIGURES / "ablation_individual"
FIGURE_WIDTH_IN = 12.0
FIGURE_HEIGHT_IN = 3.4
REFERENCE_AXIS_HEIGHT_IN = 3.0
AXIS_HEIGHT_IN = REFERENCE_AXIS_HEIGHT_IN * (2.0 / 3.0)
AXIS_BOTTOM_IN = 0.60
EXPORT_DPI = 1000


def style_axis(axis) -> None:
    axis.spines[["top", "right"]].set_visible(False)
    axis.spines[["left", "bottom"]].set_linewidth(1.25)
    axis.grid(True, axis="y", linewidth=0.7, color="#DCE3EA")
    axis.set_axisbelow(True)
    axis.tick_params(labelsize=15, length=5, width=1.1)
    axis.xaxis.set_major_locator(MaxNLocator(nbins=6))
    axis.yaxis.set_major_locator(MaxNLocator(nbins=5))


def draw_panel(demo, records):
    ablation.configure_style()
    figure = ablation.plt.figure(figsize=(FIGURE_WIDTH_IN, FIGURE_HEIGHT_IN), facecolor="white")
    bottom = AXIS_BOTTOM_IN / FIGURE_HEIGHT_IN
    height = AXIS_HEIGHT_IN / FIGURE_HEIGHT_IN
    line_axis = figure.add_axes((0.085, bottom, 0.735, height))
    box_axis = figure.add_axes((0.845, bottom, 0.13, height))
    style_axis(line_axis)
    style_axis(box_axis)

    raw_errors = []
    shown_peak = 0.0
    for model in ablation.MODELS:
        payload = records[model.name][demo]
        errors = ablation.instantaneous_error(payload)
        raw_errors.append(errors)
        times, shown_errors = ablation.mean_downsample(payload["time"], errors, factor=50)
        shown_peak = max(shown_peak, float(ablation.np.max(shown_errors)))
        line_axis.plot(times, shown_errors, color=ablation.PALETTE[model.name],
                       linewidth=2.0, alpha=0.95)

    line_axis.set_ylim(0.0, shown_peak if shown_peak > 0 else 0.01)
    line_axis.set_xlabel("Time [s]", fontsize=17, labelpad=6)
    line_axis.set_ylabel(r"$\Vert e_v^B\Vert_2$ [m/s]", fontsize=17, labelpad=8)
    line_axis.text(0.98, 0.94, ablation.TEST_TITLES[demo],
                   transform=line_axis.transAxes, ha="right", va="top",
                   fontsize=20, fontweight="bold",
                   bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85, "pad": 1.0})

    box = box_axis.boxplot(
        raw_errors, patch_artist=True, showfliers=False, widths=0.4,
        medianprops={"color": "#202020", "linewidth": 1.8},
        whiskerprops={"color": "#4D4D4D", "linewidth": 1.3},
        capprops={"color": "#4D4D4D", "linewidth": 1.3},
    )
    whisker_peak = max(float(ablation.np.max(item.get_ydata())) for item in box["whiskers"])
    box_axis.set_ylim(0.0, max(shown_peak, whisker_peak))
    for patch, model in zip(box["boxes"], ablation.MODELS):
        patch.set(facecolor=ablation.PALETTE[model.name], edgecolor="#303030",
                  linewidth=1.0, alpha=0.82)
    box_axis.set_xticks([])
    box_axis.tick_params(axis="y", left=False, labelleft=False)
    return figure


def draw_legend():
    ablation.configure_style()
    figure = ablation.plt.figure(figsize=(FIGURE_WIDTH_IN, 0.9), facecolor="white")
    handles = [Line2D([], [], color=ablation.PALETTE[model.name], linewidth=3.0)
               for model in ablation.MODELS]
    figure.legend(handles, [model.name for model in ablation.MODELS],
                  loc="center", bbox_to_anchor=(0.5, 0.5), ncol=5,
                  frameon=False, fontsize=18, handlelength=1.8, columnspacing=1.3)
    return figure


def main() -> None:
    records = {
        model.name: {
            demo: ablation.load_prediction(model.directory / f"{demo}_predictions.npz")
            for demo in ablation.DEMOS
        }
        for model in ablation.MODELS
    }
    ablation.validate_common_reference(records)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for demo in ablation.DEMOS:
        figure = draw_panel(demo, records)
        target = OUTPUT_DIR / f"{ablation.TEST_NAMES[demo]}_ablation.png"
        figure.savefig(target, dpi=EXPORT_DPI, facecolor="white")
        ablation.plt.close(figure)
        print(target)
    figure = draw_legend()
    target = OUTPUT_DIR / "ablation_legend.png"
    figure.savefig(target, dpi=EXPORT_DPI, facecolor="white")
    ablation.plt.close(figure)
    print(target)


if __name__ == "__main__":
    main()
