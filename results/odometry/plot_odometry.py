# -*- coding: utf-8 -*-
"""新 Test 1–4 里程计绘图：修改下面的“用户配置区”，再运行本文件。

依赖：numpy、matplotlib、Pillow；仅在启用额外平滑时需要 scipy。
默认从本文件所在目录读取 test1_odometry.csv ～ test4_odometry.csv。
本脚本只读取已有里程计结果，不训练模型、不重新运行滤波、不改写原始 CSV。
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

matplotlib.use("Agg")  # 无需弹窗即可保存图片，适合直接在终端运行。
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
from PIL import Image

# Windows 终端统一使用 UTF-8，避免中文提示出现乱码。
for _stream in (sys.stdout, sys.stderr):
    if _stream is not None and hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")


# ======================== 1. 用户配置区：通常只需修改这里 ========================
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR                       # CSV 所在目录，可改为其他绝对路径。
OUTPUT_DIR = BASE_DIR.parents[1] / "outputs" / "odometry_figures"
TESTS = (1, 2, 3, 4)                      # 例如改成 (1,) 只重画 Test 1。
DPI = 1200                               # 论文 PNG 清晰度；调试时可暂改为 150。
SAVE_PDF = False                         # 改为 True 时额外输出矢量 PDF。

# 固定画布比例：不使用 bbox_inches='tight'，避免导出后比例发生改变。
VELOCITY_SIZE = (8.0, 4.0)                # 宽、高，单位英寸；默认宽高比 2:1。
BOXPLOT_SIZE = (12.0, 3.0)                # 默认宽高比 4:1。
LEGEND_SIZE = (19.0, 0.9)                 # 独立横向图例，宽度不足时自动扩展。
BACKGROUND = "white"                     # 图片背景色。
LEGEND_TRANSPARENT = False               # 图例是否使用透明背景。

# 字体建议：英文论文用 DejaVu Sans；中文标签可改为 Microsoft YaHei。
FONT_FAMILY = "DejaVu Sans"
FONT_SIZE = {
    "base": 20,
    "axis_label": 24,
    "tick": 20,
    "title": 25,
    "legend": 30,                       # 原字号 20 的 1.5 倍。
}

# 键名必须与 CSV 的列名前缀一致；label、color、linestyle 等可自由修改。
# Ground Truth 在最上层绘制，线稍细；各方法均为实线，通过颜色区分。
# 以下曲线线宽已在上一版基础上加倍。
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
        # 当前为通过验证资格检查的三维 AirIO，来源见 source_provenance.json。
        "label": "AirIO", "color": "#0072B2",
        "linestyle": "-", "linewidth": 3.6, "alpha": 0.95, "zorder": 2,
    },
    "DAIO_withoutRPM": {
        "label": "DAIO without RPM", "color": "#D55E00",
        "linestyle": "-", "linewidth": 4.2, "alpha": 0.95, "zorder": 3,
    },
}

# 同时控制折线、箱线与图例的显示顺序；从这里移除某键即可隐藏该方法。
METHODS = ("DAIO", "DEEPVL", "AirIO", "DAIO_withoutRPM")
LEGEND_ORDER = ("ground_truth", *METHODS)
LEGEND_HANDLE_LENGTH = 2.5                # 图例中线段长度。
LEGEND_COLUMN_SPACING = 1.8               # 相邻图例项的水平间距。
LEGEND_LABEL_SPACING = 0.8                # 线段与文字之间的距离。
LEGEND_MIN_LINE_WIDTH = 5.0              # 图例线宽：上一版 2.5 的两倍。

# 折线的横轴：默认减去初始化的 8 秒，即评分起点显示为 0。
TIME_ZERO_AFTER_INITIALIZATION = True
# 每张图使用的数据范围，单位是“初始化结束后的秒数”；含起止端点。
# Test 1 的 0～52 s 对应原记录的 8～60 s；未列出的测试使用完整评分段。
# 此裁剪同时作用于该 Test 的折线图和箱线图，原始 CSV 不改写。
TEST_TIME_RANGES = {1: (0.0, 52.0)}
X_LABEL = "Time(s)"
VELOCITY_Y_LABELS = (r"$v_x$" + "\n[m/s]", r"$v_y$" + "\n[m/s]")
TITLE_FORMAT = "Test {test}"
SHOW_TITLES = True
TITLE_LOCATION = "left"                  # 可选 left / center / right。

# None 表示依据完整原始数据自动留白；也可手动指定元组，例如 (-0.4, 0.6)。
# 如只想调整某张图：VELOCITY_Y_LIMITS = {1: ((-0.4, 0.6), (-0.25, 0.25))}
VELOCITY_Y_LIMITS = {}
VELOCITY_Y_MARGIN = 0.08
MAX_Y_TICKS = 4
SHOW_GRID = True
GRID_COLOR = "#DCE4E9"
GRID_WIDTH = 0.7
AXIS_WIDTH = 1.1
HIDE_TOP_RIGHT_SPINES = True

# 仅对显示折线取点；默认约 10 Hz，并保留每段的极值及首末点。
# 设置为 None 表示绘制所有 200 Hz 样本。抽样不影响箱线图及指标。
DISPLAY_HZ = 10.0
PRESERVE_DISPLAY_EXTREMA = True

# 所有速度折线的额外显示平滑：0.75 秒、三阶 Savitzky–Golay。
# 设为 0.0 关闭；适当增大到 1.0 可进一步平滑。仅用于显示，不改 CSV/箱线误差。
# 裁剪完成后再平滑，所以 Test 1 的 52 秒之后数据不会影响当前图。
DISPLAY_SMOOTH_SECONDS = 0.75
DISPLAY_SMOOTH_POLYORDER = 3
# 真值可单独设置；None 表示沿用上面的统一时长，0.0 表示真值不额外平滑。
GT_DISPLAY_SMOOTH_SECONDS = None

# 一个箱线图：每个 Test 包含四个方法。默认误差为二维速度误差范数。
# 可选 xy_norm / vx_abs / vy_abs；三种误差单位均为 m/s。
BOX_ERROR = "xy_norm"
BOX_Y_LIMITS = None                       # 如需固定范围可改为 (0, 0.15)。
BOX_SHOW_OUTLIERS = True                  # 默认显示所有离群点，不截掉失败数据。
BOX_WHISKERS = 1.5                        # 须长，单位 IQR；设为 (0,100) 则须到最小/最大。
BOX_FILL_ALPHA = 0.65
BOX_WIDTH = 0.15
BOX_METHOD_SPACING = 0.20
BOX_OUTLIER_SIZE = 0.9
BOX_OUTLIER_ALPHA = 0.15
BOX_MEDIAN_COLOR = "#263746"
BOX_MEDIAN_WIDTH = 3.4                    # 箱线图中位数线宽加倍。
BOX_EDGE_WIDTH = 2.0                     # 箱体、须和端帽线宽加倍。
BOX_Y_LABELS = {
    "xy_norm": r"$\|e_v^{XY}\|_2$ [m/s]",
    "vx_abs": r"$|e_{v_x}|$ [m/s]",
    "vy_abs": r"$|e_{v_y}|$ [m/s]",
}

MAKE_PREVIEW = True                       # 另生成低分辨率总览，方便快速检查。
# ======================== 用户配置区结束，以下为绘图实现 =======================


def file_sha256(path):
    """记录输入与输出文件哈希，便于确认图片用了哪一版数据。"""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_trial(test):
    """按列名读取，避免依赖列的位置；检查 200 Hz 和数值有效性。"""
    path = DATA_DIR / f"test{test}_odometry.csv"
    with path.open(encoding="utf-8-sig", newline="") as handle:
        names = next(csv.reader(handle))
    values = np.loadtxt(path, delimiter=",", skiprows=1, encoding="utf-8-sig", ndmin=2)
    if values.shape[1] != len(names) or not np.isfinite(values).all():
        raise ValueError(f"Test {test} 的 CSV 列数不一致，或包含 NaN/Inf。")
    columns = dict(zip(names, values.T))
    required = ["time_s", "time_after_initialization_s"]
    for key in ("ground_truth", *METHODS):
        required.extend(f"{key}_{axis}_mps" for axis in ("vx", "vy"))
    missing = [name for name in required if name not in columns]
    if missing:
        raise ValueError(f"Test {test} 缺少必要列：{missing}")
    np.testing.assert_allclose(np.diff(columns["time_s"]), 0.005, rtol=0, atol=1e-9)
    np.testing.assert_allclose(columns["time_after_initialization_s"], columns["time_s"] - 8.0,
                               rtol=0, atol=1e-9)
    source_samples = len(values)
    selected = np.ones(source_samples, dtype=bool)
    if test in TEST_TIME_RANGES:
        start, end = TEST_TIME_RANGES[test]
        if start > end:
            raise ValueError(f"Test {test} 的裁剪起点不能大于终点。")
        relative_time = columns["time_after_initialization_s"]
        selected = (relative_time >= start - 1e-9) & (relative_time <= end + 1e-9)
    if selected.sum() < 5:
        raise ValueError(f"Test {test} 裁剪后样本不足，请检查 TEST_TIME_RANGES。")
    velocity = {
        key: np.column_stack([columns[f"{key}_{axis}_mps"][selected] for axis in ("vx", "vy")])
        for key in ("ground_truth", *METHODS)
    }
    time_key = "time_after_initialization_s" if TIME_ZERO_AFTER_INITIALIZATION else "time_s"
    return {"test": test, "path": path, "time": columns[time_key][selected],
            "velocity": velocity, "samples": int(selected.sum()), "source_samples": source_samples,
            "time_after_initialization": columns["time_after_initialization_s"][selected]}


def display_indices(time, values):
    """均匀抽样，并保留每个显示块内的峰值，避免漏画短时尖峰。"""
    if DISPLAY_HZ is None:
        return np.arange(len(time))
    if DISPLAY_HZ <= 0:
        raise ValueError("DISPLAY_HZ 必须为正数或 None。")
    raw_dt = float(np.median(np.diff(time)))
    step = max(1, int(round(1 / (DISPLAY_HZ * raw_dt))))
    selected = [*range(0, len(time), step), len(time) - 1]
    if PRESERVE_DISPLAY_EXTREMA:
        for start in range(0, len(time), step):
            chunk = values[start:start + step]
            selected.extend((start + int(np.argmin(chunk)), start + int(np.argmax(chunk))))
    return np.unique(selected)


def smooth_for_display(time, velocity, seconds):
    """仅平滑用于显示的新数组；箱线图始终读取未经该平滑处理的 velocity。"""
    if seconds <= 0:
        return velocity
    from scipy.signal import savgol_filter
    width = int(round(seconds / np.median(np.diff(time))))
    minimum = DISPLAY_SMOOTH_POLYORDER + 2
    width = max(minimum, width)
    width = width if width % 2 else width + 1
    width = min(width, len(time) if len(time) % 2 else len(time) - 1)
    if width <= DISPLAY_SMOOTH_POLYORDER:
        raise ValueError("平滑窗口内样本不足以拟合指定阶数。")
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
    """固定尺寸保存；先检查文字边界，再用临时文件替换，防止图片写到一半。"""
    # 以较低的画布 DPI 进行排版检查，正式保存时使用用户指定的 1200 DPI。
    fig.canvas.draw()
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    bounds = fig.get_tightbbox(renderer)
    width, height = fig.get_size_inches()
    if bounds is not None and (bounds.x0 < -0.015 or bounds.y0 < -0.015 or
                               bounds.x1 > width + 0.015 or bounds.y1 > height + 0.015):
        plt.close(fig)
        raise ValueError(f"{name} 的文字超出画布，请适度减小字号或增加图尺寸。")
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
    print(f"已保存：{path.name}", flush=True)
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
            # 自动坐标使用所有原始样本，而不是抽样值，默认不会悄悄隐藏峰值。
            # 同时包含原始与显示数据的范围，避免平滑端部拟合超出坐标范围。
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
    """箱线图始终使用全量误差，不使用显示平滑或显示抽样结果。"""
    error = trial["velocity"][method] - trial["velocity"]["ground_truth"]
    if BOX_ERROR == "xy_norm":
        return np.linalg.norm(error, axis=1)
    if BOX_ERROR == "vx_abs":
        return np.abs(error[:, 0])
    if BOX_ERROR == "vy_abs":
        return np.abs(error[:, 1])
    raise ValueError("BOX_ERROR 只能为 xy_norm / vx_abs / vy_abs。")


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
    """单行排列，顺序同配置区；用户增加文字/字号时自动扩展画布以防裁切。"""
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
    """将正式图片缩小成预览，不能代替 1200 dpi 原图用于论文排版。"""
    Image.MAX_IMAGE_PIXELS = None
    width = 1200
    # 四张折线图按 2×2 排列，下面依次放箱线图和横向图例。
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
    # 先写临时文件再替换，避免预览被图片查看器占用时直接写入失败。
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
        raise ValueError("METHODS 不能为空或重复。")
    # 从本目录的来源文件读取版本，避免替换数据后仍显示旧版状态。
    provenance_path = DATA_DIR / "source_provenance.json"
    provenance = json.loads(provenance_path.read_text(encoding="utf-8-sig")) if provenance_path.exists() else {}
    airio_info = provenance.get("models", {}).get("AirIO", {})
    if "AirIO" in METHODS and airio_info.get("diagnostic_only", False):
        warnings.warn("当前 AirIO 来源标为诊断结果，具体状态见 source_provenance.json。", stacklevel=1)
    configure_style()
    trials = [read_trial(test) for test in TESTS]
    files = [plot_velocity(trial) for trial in trials]
    boxfile, stats = plot_boxplot(trials)
    files.extend((boxfile, plot_legend()))
    with (OUTPUT_DIR / "boxplot_statistics.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(stats[0]))
        writer.writeheader()
        writer.writerows(stats)
    # 在导出后核对实际像素和 DPI；PNG 画幅以英寸尺寸×DPI 为准。
    Image.MAX_IMAGE_PIXELS = None
    inventory = []
    for path in files:
        with Image.open(path) as image:
            dpi = image.info.get("dpi")
            if dpi is None or abs(dpi[0] - DPI) > 0.1:
                raise ValueError(f"{path.name} 的导出 DPI 不正确。")
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
    print(f"完成：{len(files)} 张正式 PNG 已保存至 {OUTPUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
