# -*- coding: utf-8 -*-
"""Plot Test 1--4 odometry: edit the user configuration section below, then run this file.

Dependencies: numpy, matplotlib, Pillow; scipy is required only for additional smoothing.
By default, read test1_odometry.csv through test4_odometry.csv from this file's directory.
Read existing odometry results without training models, rerunning filters, or rewriting source CSV files.
"""

from pathlib import Path
import csv
import hashlib
import json
import os
import sys
import tempfile
import warnings

import numpy as np
import matplotlib

matplotlib.use("Agg")  # Save figures without opening windows, suitable for terminal execution.
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
from PIL import Image

# Use UTF-8 consistently in Windows terminals to avoid garbled text.
for _stream in (sys.stdout, sys.stderr):
    if _stream is not None and hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")


# ======================== 1. USER CONFIGURATION: normally edit only this section ========================
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR                       # CSV directory; may be changed to another absolute path.
OUTPUT_DIR = BASE_DIR.parents[1] / "outputs" / "odometry_figures"
TESTS = (1, 2, 3, 4)                      # For example, use (1,) to redraw Test 1 only.
DPI = 1200                               # Publication PNG resolution; use 150 temporarily for debugging.
SAVE_PDF = False                         # Set True to also export vector PDFs.

# Keep the canvas aspect ratio fixed; do not use bbox_inches='tight', which changes exported dimensions.
VELOCITY_SIZE = (8.0, 4.0)                # Width and height in inches; default aspect ratio 2:1.
BOXPLOT_SIZE = (12.0, 3.0)                # Default aspect ratio 4:1.
LEGEND_SIZE = (19.0, 0.9)                 # Standalone horizontal legend; expands if more width is needed.
BACKGROUND = "white"                     # Figure background color.
LEGEND_TRANSPARENT = False               # Whether the legend uses a transparent background.

# Use DejaVu Sans for English publications; Microsoft YaHei supports Chinese labels if needed.
FONT_FAMILY = "DejaVu Sans"
FONT_SIZE = {
    "base": 20,
    "axis_label": 24,
    "tick": 20,
    "title": 25,
    "legend": 30,                       # 1.5 times the original font size of 20.
}

# Keys must match CSV column prefixes; label, color, linestyle, etc. are customizable.
# Draw Ground Truth on top with slightly thinner lines; methods use solid lines distinguished by color.
# Curve widths below are twice those in the previous version.
SERIES = {
    "ground_truth": {
        "label": "Ground Truth", "color": "#263746",
        "linestyle": "-", "linewidth": 3.0, "alpha": 0.90, "zorder": 5,
    },
    "DAIO": {
        "label": "DAIO", "color": "#009E87",
        "linestyle": "-", "linewidth": 4.2, "alpha": 1.0, "zorder": 4,
    },
    "DEEPVL": {
        "label": "DeepVL", "color": "#AA4499",
        "linestyle": "-", "linewidth": 3.8, "alpha": 0.95, "zorder": 3,
    },
    "AirIO": {
        # Current AirIO results are from the qualified 3D model; see source_provenance.json.
        "label": "AirIO", "color": "#0072B2",
        "linestyle": "-", "linewidth": 3.6, "alpha": 0.95, "zorder": 2,
    },
    "DAIO_withoutRPM": {
        "label": "DAIO without RPM", "color": "#D55E00",
        "linestyle": "-", "linewidth": 4.2, "alpha": 0.95, "zorder": 3,
    },
}

# Controls curve, boxplot, and legend order; remove a key here to hide a method.
METHODS = ("DAIO", "DEEPVL", "AirIO", "DAIO_withoutRPM")
LEGEND_ORDER = ("ground_truth", *METHODS)
LEGEND_HANDLE_LENGTH = 2.5                # Legend line-segment length.
LEGEND_COLUMN_SPACING = 1.8               # Horizontal spacing between legend entries.
LEGEND_LABEL_SPACING = 0.8                # Spacing between line segments and text.
LEGEND_MIN_LINE_WIDTH = 5.0              # Legend line width: twice the previous 2.5.

# Curve time axis: subtract the 8 s initialization by default so scoring starts at zero.
TIME_ZERO_AFTER_INITIALIZATION = True
# Per-figure data ranges in seconds after initialization, including both endpoints.
# Test 1 at 0--52 s corresponds to original record time 8--60 s; unlisted tests use the full scoring interval.
# Cropping applies to both curves and boxplots for the test; source CSV files are unchanged.
TEST_TIME_RANGES = {1: (0.0, 52.0)}
X_LABEL = "Time(s)"
VELOCITY_Y_LABELS = (r"$v_x$" + "\n[m/s]", r"$v_y$" + "\n[m/s]")
TITLE_FORMAT = "Test {test}"
SHOW_TITLES = True
TITLE_LOCATION = "left"                  # Options: left / center / right.

# None selects automatic padding from all raw samples; alternatively specify a tuple such as (-0.4, 0.6).
# To adjust one figure only: VELOCITY_Y_LIMITS = {1: ((-0.4, 0.6), (-0.25, 0.25))}
VELOCITY_Y_LIMITS = {}
VELOCITY_Y_MARGIN = 0.08
MAX_Y_TICKS = 4
SHOW_GRID = True
GRID_COLOR = "#DCE4E9"
GRID_WIDTH = 0.7
AXIS_WIDTH = 1.1
HIDE_TOP_RIGHT_SPINES = True

# Subsample display curves only, at approximately 10 Hz by default, retaining extrema and endpoints per block.
# None plots all 200 Hz samples. Subsampling does not affect boxplots or metrics.
DISPLAY_HZ = 10.0
PRESERVE_DISPLAY_EXTREMA = True

# Additional display smoothing for velocity curves: 0.75 s, third-order Savitzky-Golay.
# Set 0.0 to disable or increase to 1.0 for more smoothing. Display only; does not change CSV data or boxplot errors.
# Smooth after cropping so Test 1 data beyond 52 s cannot affect the current figure.
DISPLAY_SMOOTH_SECONDS = 0.75
DISPLAY_SMOOTH_POLYORDER = 3
# Ground truth may be configured separately; None uses the shared duration, 0.0 disables additional smoothing.
GT_DISPLAY_SMOOTH_SECONDS = None

# One boxplot includes four methods per test; the default error is the 2D velocity-error norm.
# Options: xy_norm / vx_abs / vy_abs; all use m/s.
BOX_ERROR = "xy_norm"
BOX_Y_LIMITS = None                       # For a fixed range, use e.g. (0, 0.15).
BOX_SHOW_OUTLIERS = True                  # Show all outliers by default without hiding failures.
BOX_WHISKERS = 1.5                        # Whisker length in IQR; (0,100) extends to min/max.
BOX_FILL_ALPHA = 0.65
BOX_WIDTH = 0.15
BOX_METHOD_SPACING = 0.20
BOX_OUTLIER_SIZE = 0.9
BOX_OUTLIER_ALPHA = 0.15
BOX_MEDIAN_COLOR = "#263746"
BOX_MEDIAN_WIDTH = 3.4                    # Doubled boxplot median-line width.
BOX_EDGE_WIDTH = 2.0                     # Doubled box, whisker, and cap widths.
BOX_Y_LABELS = {
    "xy_norm": r"$\|e_v^{XY}\|_2$ [m/s]",
    "vx_abs": r"$|e_{v_x}|$ [m/s]",
    "vy_abs": r"$|e_{v_y}|$ [m/s]",
}

MAKE_PREVIEW = True                       # Also generate a low-resolution overview for quick inspection.
# ======================== END USER CONFIGURATION; plotting implementation below =======================


def file_sha256(path):
    """Record input/output hashes to identify the exact data version used for each figure."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_trial(test):
    """Read by column name rather than position; validate 200 Hz sampling and numeric values."""
    path = DATA_DIR / f"test{test}_odometry.csv"
    with path.open(encoding="utf-8-sig", newline="") as handle:
        names = next(csv.reader(handle))
    values = np.loadtxt(path, delimiter=",", skiprows=1, encoding="utf-8-sig", ndmin=2)
    if values.shape[1] != len(names) or not np.isfinite(values).all():
        raise ValueError(f"Test {test} CSV has inconsistent column counts or contains NaN/Inf.")
    columns = dict(zip(names, values.T))
    required = ["time_s", "time_after_initialization_s"]
    for key in ("ground_truth", *METHODS):
        required.extend(f"{key}_{axis}_mps" for axis in ("vx", "vy"))
    missing = [name for name in required if name not in columns]
    if missing:
        raise ValueError(f"Test {test} is missing required columns: {missing}")
    np.testing.assert_allclose(np.diff(columns["time_s"]), 0.005, rtol=0, atol=1e-9)
    np.testing.assert_allclose(columns["time_after_initialization_s"], columns["time_s"] - 8.0,
                               rtol=0, atol=1e-9)
    source_samples = len(values)
    selected = np.ones(source_samples, dtype=bool)
    if test in TEST_TIME_RANGES:
        start, end = TEST_TIME_RANGES[test]
        if start > end:
            raise ValueError(f"Test {test} crop start must not exceed its end.")
        relative_time = columns["time_after_initialization_s"]
        selected = (relative_time >= start - 1e-9) & (relative_time <= end + 1e-9)
    if selected.sum() < 5:
        raise ValueError(f"Test {test} has too few samples after cropping; check TEST_TIME_RANGES.")
    velocity = {
        key: np.column_stack([columns[f"{key}_{axis}_mps"][selected] for axis in ("vx", "vy")])
        for key in ("ground_truth", *METHODS)
    }
    time_key = "time_after_initialization_s" if TIME_ZERO_AFTER_INITIALIZATION else "time_s"
    return {"test": test, "path": path, "time": columns[time_key][selected],
            "velocity": velocity, "samples": int(selected.sum()), "source_samples": source_samples,
            "time_after_initialization": columns["time_after_initialization_s"][selected]}


def display_indices(time, values):
    """Subsample uniformly while retaining per-block peaks to preserve short spikes."""
    if DISPLAY_HZ is None:
        return np.arange(len(time))
    if DISPLAY_HZ <= 0:
        raise ValueError("DISPLAY_HZ must be positive or None.")
    raw_dt = float(np.median(np.diff(time)))
    step = max(1, int(round(1 / (DISPLAY_HZ * raw_dt))))
    selected = [*range(0, len(time), step), len(time) - 1]
    if PRESERVE_DISPLAY_EXTREMA:
        for start in range(0, len(time), step):
            chunk = values[start:start + step]
            selected.extend((start + int(np.argmin(chunk)), start + int(np.argmax(chunk))))
    return np.unique(selected)


def smooth_for_display(time, velocity, seconds):
    """Smooth only a new display array; boxplots always use velocity before this smoothing."""
    if seconds <= 0:
        return velocity
    from scipy.signal import savgol_filter
    width = int(round(seconds / np.median(np.diff(time))))
    minimum = DISPLAY_SMOOTH_POLYORDER + 2
    width = max(minimum, width)
    width = width if width % 2 else width + 1
    width = min(width, len(time) if len(time) % 2 else len(time) - 1)
    if width <= DISPLAY_SMOOTH_POLYORDER:
        raise ValueError("Too few samples in the smoothing window for the requested polynomial order.")
    return savgol_filter(velocity, width, DISPLAY_SMOOTH_POLYORDER, axis=0, mode="interp")


def configure_style():
    plt.rcParams.update({
        "font.family": FONT_FAMILY, "font.size": FONT_SIZE["base"],
        "axes.labelsize": FONT_SIZE["axis_label"], "axes.titlesize": FONT_SIZE["title"],
        "xtick.labelsize": FONT_SIZE["tick"], "ytick.labelsize": FONT_SIZE["tick"],
        "axes.spines.top": not HIDE_TOP_RIGHT_SPINES,
        "axes.spines.right": not HIDE_TOP_RIGHT_SPINES,
        "axes.linewidth": AXIS_WIDTH, "figure.facecolor": BACKGROUND,
        "axes.facecolor": BACKGROUND, "savefig.facecolor": BACKGROUND,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })


def save_figure(fig, name, transparent=False):
    """Save at fixed dimensions; check text bounds and replace through a temporary file to avoid partial writes."""
    # Check layout at lower canvas DPI, then save at the configured publication resolution (default 1200 DPI).
    fig.canvas.draw()
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    bounds = fig.get_tightbbox(renderer)
    width, height = fig.get_size_inches()
    if bounds is not None and (bounds.x0 < -0.015 or bounds.y0 < -0.015 or
                               bounds.x1 > width + 0.015 or bounds.y1 > height + 0.015):
        plt.close(fig)
        raise ValueError(f"Text in {name} extends beyond the canvas; reduce font size or enlarge the figure.")
    path = OUTPUT_DIR / name
    handle, temporary = tempfile.mkstemp(prefix=path.stem + "_", suffix=".tmp", dir=OUTPUT_DIR)
    os.close(handle)
    try:
        fig.savefig(temporary, format="png", dpi=DPI, transparent=transparent,
                    facecolor="none" if transparent else BACKGROUND)
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)
    if SAVE_PDF:
        fig.savefig(path.with_suffix(".pdf"), format="pdf", transparent=transparent,
                    facecolor="none" if transparent else BACKGROUND)
    plt.close(fig)
    print(f"Saved: {path.name}", flush=True)
    return path


def plot_velocity(trial):
    fig, axes = plt.subplots(2, 1, figsize=VELOCITY_SIZE, sharex=True, layout="constrained")
    fig.set_constrained_layout_pads(w_pad=0.10, h_pad=0.055, hspace=0.08)
    time = trial["time"]
    display_velocity = {}
    for key, raw in trial["velocity"].items():
        seconds = DISPLAY_SMOOTH_SECONDS
        if key == "ground_truth" and GT_DISPLAY_SMOOTH_SECONDS is not None:
            seconds = GT_DISPLAY_SMOOTH_SECONDS
        display_velocity[key] = smooth_for_display(time, raw, seconds)
    for component, ax in enumerate(axes):
        for key in ("ground_truth", *METHODS):
            style = SERIES[key]
            value = display_velocity[key][:, component]
            index = display_indices(time, value)
            ax.plot(time[index], value[index], color=style["color"], linestyle=style["linestyle"],
                    linewidth=style["linewidth"], alpha=style["alpha"], zorder=style["zorder"])
        ax.set_ylabel(VELOCITY_Y_LABELS[component])
        ax.set_xlim(float(time[0]), float(time[-1]))
        limits = VELOCITY_Y_LIMITS.get(trial["test"], (None, None))[component]
        if limits is None:
            # Automatic axes use all raw samples, not subsampled values, so peaks are not silently hidden.
            # Include both raw and displayed ranges to contain smoothing overshoot near the endpoints.
            values = np.concatenate([v[:, component] for v in
                                     [*trial["velocity"].values(), *display_velocity.values()]])
            span = max(float(np.ptp(values)), 0.05)
            ax.set_ylim(float(values.min() - span * VELOCITY_Y_MARGIN),
                        float(values.max() + span * VELOCITY_Y_MARGIN))
        else:
            ax.set_ylim(*limits)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=MAX_Y_TICKS))
        if SHOW_GRID:
            ax.grid(axis="y", color=GRID_COLOR, linewidth=GRID_WIDTH)
        ax.set_axisbelow(True)
    axes[-1].set_xlabel(X_LABEL)
    if SHOW_TITLES:
        axes[0].set_title(TITLE_FORMAT.format(test=trial["test"]), loc=TITLE_LOCATION,
                          fontweight="bold", fontsize=FONT_SIZE["title"])
    return save_figure(fig, f"test{trial['test']}_velocity.png")


def error_values(trial, method):
    """Boxplots always use all errors, without display smoothing or subsampling."""
    error = trial["velocity"][method] - trial["velocity"]["ground_truth"]
    if BOX_ERROR == "xy_norm":
        return np.linalg.norm(error, axis=1)
    if BOX_ERROR == "vx_abs":
        return np.abs(error[:, 0])
    if BOX_ERROR == "vy_abs":
        return np.abs(error[:, 1])
    raise ValueError("BOX_ERROR must be xy_norm / vx_abs / vy_abs.")


def plot_boxplot(trials):
    fig, ax = plt.subplots(figsize=BOXPLOT_SIZE, layout="constrained")
    fig.set_constrained_layout_pads(w_pad=0.10, h_pad=0.06)
    max_value = 0.0
    statistics = []
    for group, trial in enumerate(trials, start=1):
        for order, method in enumerate(METHODS):
            error = error_values(trial, method)
            max_value = max(max_value, float(error.max()))
            x = group + (order - (len(METHODS) - 1) / 2) * BOX_METHOD_SPACING
            color = SERIES[method]["color"]
            result = ax.boxplot(
                error, positions=[x], widths=BOX_WIDTH, patch_artist=True, manage_ticks=False,
                whis=BOX_WHISKERS, showfliers=BOX_SHOW_OUTLIERS,
                boxprops={"edgecolor": color, "linewidth": BOX_EDGE_WIDTH},
                whiskerprops={"color": color, "linewidth": BOX_EDGE_WIDTH},
                capprops={"color": color, "linewidth": BOX_EDGE_WIDTH},
                medianprops={"color": BOX_MEDIAN_COLOR, "linewidth": BOX_MEDIAN_WIDTH},
                flierprops={"marker": "o", "markersize": BOX_OUTLIER_SIZE,
                            "markeredgecolor": color, "markerfacecolor": "none", "alpha": BOX_OUTLIER_ALPHA},
            )
            result["boxes"][0].set(facecolor=color, alpha=BOX_FILL_ALPHA)
            statistics.append({"test": trial["test"], "method": method, "samples": len(error),
                               "median_mps": float(np.median(error)),
                               "q1_mps": float(np.percentile(error, 25)), "q3_mps": float(np.percentile(error, 75)),
                               "maximum_mps": float(error.max()),
                               "outside_display_range": 0 if BOX_Y_LIMITS is None else
                               int(np.count_nonzero((error < BOX_Y_LIMITS[0]) | (error > BOX_Y_LIMITS[1])))})
    ax.set_xticks(range(1, len(trials) + 1), [TITLE_FORMAT.format(test=t["test"]) for t in trials])
    ax.set_xlim(0.45, len(trials) + 0.55)
    ax.set_ylabel(BOX_Y_LABELS[BOX_ERROR])
    ax.set_ylim(0, max(0.02, max_value * 1.04)) if BOX_Y_LIMITS is None else ax.set_ylim(*BOX_Y_LIMITS)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
    if SHOW_GRID:
        ax.grid(axis="y", color=GRID_COLOR, linewidth=GRID_WIDTH)
    ax.set_axisbelow(True)
    return save_figure(fig, "velocity_error_boxplot.png"), statistics


def plot_legend():
    """Arrange one row in configured order; expand the canvas for longer text or larger fonts to prevent clipping."""
    handles = [Line2D([], [], color=SERIES[key]["color"], linestyle=SERIES[key]["linestyle"],
                      linewidth=max(LEGEND_MIN_LINE_WIDTH, SERIES[key]["linewidth"]), label=SERIES[key]["label"])
               for key in LEGEND_ORDER if key == "ground_truth" or key in METHODS]
    fig = plt.figure(figsize=LEGEND_SIZE)
    legend = fig.legend(handles=handles, loc="center", ncol=len(handles), frameon=False,
                        fontsize=FONT_SIZE["legend"], handlelength=LEGEND_HANDLE_LENGTH,
                        columnspacing=LEGEND_COLUMN_SPACING, handletextpad=LEGEND_LABEL_SPACING)
    fig.canvas.draw()
    bounds = legend.get_window_extent(fig.canvas.get_renderer()).transformed(fig.dpi_scale_trans.inverted())
    fig.set_size_inches(max(LEGEND_SIZE[0], bounds.width + 0.4),
                        max(LEGEND_SIZE[1], bounds.height + 0.25))
    return save_figure(fig, "legend_horizontal.png", transparent=LEGEND_TRANSPARENT)


def make_preview(files):
    """Downscale publication figures for preview; use the original 1200 DPI images for paper layout."""
    Image.MAX_IMAGE_PIXELS = None
    width = 1200
    # Arrange four curve figures in a 2x2 grid, followed by the boxplot and horizontal legend.
    rows = int(np.ceil(len(TESTS) / 2))
    preview = Image.new("RGB", (width * 2, rows * 600 + 600 + 130), "white")
    for i, path in enumerate(files[:len(TESTS)]):
        with Image.open(path) as image:
            image = image.convert("RGB").resize((width, 600), Image.Resampling.LANCZOS)
            preview.paste(image, ((i % 2) * width, (i // 2) * 600))
    with Image.open(files[-2]) as image:
        preview.paste(image.convert("RGB").resize((2400, 600), Image.Resampling.LANCZOS), (0, rows * 600))
    with Image.open(files[-1]) as image:
        image = image.convert("RGB")
        image.thumbnail((2350, 120), Image.Resampling.LANCZOS)
        preview.paste(image, ((2400 - image.width) // 2, rows * 600 + 605))
    # Write a temporary file first, then replace to handle previews open in an image viewer.
    handle, temporary = tempfile.mkstemp(prefix="preview_", suffix=".tmp", dir=OUTPUT_DIR)
    os.close(handle)
    try:
        preview.save(temporary, format="PNG")
        os.replace(temporary, OUTPUT_DIR / "preview_all.png")
    finally:
        Path(temporary).unlink(missing_ok=True)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if not METHODS or len(set(METHODS)) != len(METHODS):
        raise ValueError("METHODS must be nonempty and contain no duplicates.")
    # Read the version from the local provenance file so replaced data does not retain stale status text.
    provenance_path = DATA_DIR / "source_provenance.json"
    provenance = json.loads(provenance_path.read_text(encoding="utf-8-sig")) if provenance_path.exists() else {}
    airio_info = provenance.get("models", {}).get("AirIO", {})
    if "AirIO" in METHODS and airio_info.get("diagnostic_only", False):
        warnings.warn("Current AirIO provenance is marked diagnostic; see source_provenance.json for details.", stacklevel=1)
    configure_style()
    trials = [read_trial(test) for test in TESTS]
    files = [plot_velocity(trial) for trial in trials]
    boxfile, stats = plot_boxplot(trials)
    files.extend((boxfile, plot_legend()))
    with (OUTPUT_DIR / "boxplot_statistics.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(stats[0]))
        writer.writeheader()
        writer.writerows(stats)
    # Verify exported pixel dimensions and DPI; PNG dimensions follow size in inches × DPI.
    Image.MAX_IMAGE_PIXELS = None
    inventory = []
    for path in files:
        with Image.open(path) as image:
            dpi = image.info.get("dpi")
            if dpi is None or abs(dpi[0] - DPI) > 0.1:
                raise ValueError(f"Exported DPI is incorrect for {path.name}.")
            if path.name.startswith("test"):
                expected = tuple(round(x * DPI) for x in VELOCITY_SIZE)
                assert image.size == expected, (path.name, image.size, expected)
            if path.name == "velocity_error_boxplot.png":
                expected = tuple(round(x * DPI) for x in BOXPLOT_SIZE)
                assert image.size == expected, (image.size, expected)
            inventory.append({"file": path.name, "pixels": list(image.size), "dpi": list(dpi),
                              "sha256": file_sha256(path)})
    report = {
        "script_sha256": file_sha256(__file__), "methods": list(METHODS),
        "inputs": [{"file": t["path"].name, "sha256": file_sha256(t["path"]),
                    "samples": t["samples"], "source_samples": t["source_samples"],
                    "selected_time_after_initialization_s": [float(t["time_after_initialization"][0]),
                                                               float(t["time_after_initialization"][-1])]} for t in trials],
        "coordinate_frame": "world XY", "sample_hz": 200,
        "box_error": BOX_ERROR, "box_uses_all_samples": True,
        "box_sample_scope": "all selected unsmoothed 200 Hz samples; TEST_TIME_RANGES applied",
        "test_time_ranges_after_initialization_s": TEST_TIME_RANGES,
        "display_hz": DISPLAY_HZ, "display_extrema_preserved": PRESERVE_DISPLAY_EXTREMA,
        "extra_ground_truth_display_smoothing_seconds": GT_DISPLAY_SMOOTH_SECONDS,
        "all_velocity_display_smoothing_seconds": DISPLAY_SMOOTH_SECONDS,
        "display_smoothing_polyorder": DISPLAY_SMOOTH_POLYORDER,
        "airio_status": airio_info.get("status", "unverified; inspect source_provenance.json"),
        "airio_model": airio_info,
        "source_provenance_sha256": file_sha256(provenance_path) if provenance_path.exists() else None,
        "figures": inventory,
    }
    (OUTPUT_DIR / "plot_validation.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    if MAKE_PREVIEW:
        make_preview(files)
    print(f"Completed: {len(files)} publication PNGs saved to {OUTPUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
