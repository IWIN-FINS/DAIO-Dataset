"""生成 DAIO-Net 与四组消融模型的论文级速度误差对比图。

本脚本只读取已经导出的 ``*_predictions.npz``，绝不重新运行网络。每个
NPZ 必须包含 prediction、target、time 和 sequence；五个实验的标签和时间轴
必须完全一致，否则脚本会立即报错，以避免把不同样本集的指标并列比较。

运行：
    python plot_ablation_figure.py
"""
from __future__ import annotations

import csv
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# 服务器/Windows 环境中用户目录有时不可写；将 Matplotlib 缓存放入临时目录。
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "rovio_ablation_figure_mpl"))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator


ROOT = Path(__file__).resolve().parent
PAPER_FIGURES = ROOT.parent / "RA-L Manuscript" / "ieeeconf" / "figures"
REPORT_DIR = ROOT / "reports" / "ablation_velocity_figure"
DEMOS = ("025_demo__segment_001", "028_demo__segment_001", "029_demo__segment_001", "030_demo__segment_001")
TEST_NAMES = {demo: f"test{index}" for index, demo in enumerate(DEMOS, start=1)}
TEST_TITLES = {demo: f"Test {index}" for index, demo in enumerate(DEMOS, start=1)}

# 颜色在表格下方的共享图例中解释；DAIO-Net 使用深蓝突出完整方案。
PALETTE = {
    "DAIO-Net": "#0F4D92",
    "Without RPM": "#B64342",
    "Without Depth": "#42949E",
    "Flat MLP": "#9A4D8E",
    "CNN--GRU": "#8BCF8B",
}

# 下方折线、箱线图和顶部表格均完整呈现五种模型。
LINE_MODELS = ("DAIO-Net", "Without RPM", "Without Depth", "Flat MLP", "CNN--GRU")


@dataclass(frozen=True)
class ModelSource:
    """一个模型名称及其已完成 demo 预测所在目录。"""

    name: str
    directory: Path


MODELS = (
    ModelSource("DAIO-Net", ROOT.parent / "ROV-IO-Training" / "runs" / "v24_full_val26_27_seed42" /
                "20260915-144055_seed42" / "evaluation" / "demo"),
    ModelSource("Without RPM", ROOT / "experiments" / "no_rpm" / "runs" /
                "20260915-222643_seed42" / "evaluation" / "demo"),
    ModelSource("Without Depth", ROOT / "experiments" / "no_depth" / "runs" /
                "20260915-192215_seed42" / "evaluation" / "demo"),
    ModelSource("Flat MLP", ROOT / "experiments" / "flat_mlp" / "runs" /
                "20260916-124655_seed42" / "evaluation" / "demo"),
    ModelSource("CNN--GRU", ROOT / "experiments" / "cnn_gru" / "runs" /
                "20260916-155834_seed42" / "evaluation" / "demo"),
)


def load_prediction(path: Path) -> dict[str, np.ndarray]:
    """读取并验证单条已导出预测；本图只使用 body-frame XY 两个速度分量。"""
    if not path.is_file():
        raise FileNotFoundError(f"Missing prediction file: {path}")
    with np.load(path) as archive:
        needed = {"prediction", "target", "time", "sequence"}
        missing = needed.difference(archive.files)
        if missing:
            raise ValueError(f"{path} lacks keys: {sorted(missing)}")
        values = {key: np.asarray(archive[key]) for key in needed}
    n = values["time"].shape[0]
    if values["prediction"].shape != (n, 3) or values["target"].shape != (n, 3):
        raise ValueError(f"Unexpected prediction/target shape in {path}")
    if values["sequence"].shape != (n,):
        raise ValueError(f"Unexpected sequence shape in {path}")
    if not all(np.isfinite(value).all() for value in values.values() if value.dtype.kind in "f"):
        raise ValueError(f"Non-finite values in {path}")
    if not np.all(np.diff(values["time"]) > 0.0):
        raise ValueError(f"Time must be strictly increasing in {path}")
    return values


def validate_common_reference(records: dict[str, dict[str, dict[str, np.ndarray]]]) -> None:
    """确保五种模型在每条 demo 上面对的是逐样本相同的离线参考标签。"""
    for demo in DEMOS:
        reference = records[MODELS[0].name][demo]
        for source in MODELS[1:]:
            candidate = records[source.name][demo]
            for field in ("time", "sequence", "target"):
                if not np.array_equal(reference[field], candidate[field]):
                    raise ValueError(
                        f"{demo}: {field} differs between {MODELS[0].name} and {source.name}; "
                        "the ablation comparison would be invalid."
                    )


def instantaneous_error(payload: dict[str, np.ndarray]) -> np.ndarray:
    """返回每个 50 Hz 输出时刻的二维 body-frame 速度误差范数 [m/s]。"""
    return np.linalg.norm(payload["prediction"][:, :2] - payload["target"][:, :2], axis=1)


def mean_downsample(time: np.ndarray, values: np.ndarray, factor: int = 10) -> tuple[np.ndarray, np.ndarray]:
    """仅为可读性对绘图折线做非重叠块均值；指标和箱线图仍使用全部 50 Hz 样本。"""
    if factor <= 0:
        raise ValueError("downsample factor must be positive")
    complete = (len(values) // factor) * factor
    sampled_time = time[:complete].reshape(-1, factor).mean(axis=1)
    sampled_values = values[:complete].reshape(-1, factor).mean(axis=1)
    # 保留最后不足 factor 的少量样本，避免曲线在轨迹结束前无故截断。
    if complete < len(values):
        sampled_time = np.append(sampled_time, np.mean(time[complete:]))
        sampled_values = np.append(sampled_values, np.mean(values[complete:]))
    return sampled_time, sampled_values


def rve_5s(payload: dict[str, np.ndarray]) -> float:
    """计算同一连续段内 5 s 相对速度增量的二维 RMSE。"""
    time = payload["time"]
    sequence = payload["sequence"]
    median_dt = float(np.median(np.diff(time)))
    lag = int(round(5.0 / median_dt))
    if lag <= 0 or abs(lag * median_dt - 5.0) > 1e-6:
        raise ValueError(f"Cannot represent 5 s RVE with median dt={median_dt:.12g}")
    valid = sequence[:-lag] == sequence[lag:]
    if not np.any(valid):
        raise ValueError("No within-sequence sample pairs available for 5 s RVE")
    predicted_delta = payload["prediction"][lag:, :2] - payload["prediction"][:-lag, :2]
    target_delta = payload["target"][lag:, :2] - payload["target"][:-lag, :2]
    residual = predicted_delta[valid] - target_delta[valid]
    return float(np.sqrt(np.mean(np.sum(residual * residual, axis=1))))


def ave(payload: dict[str, np.ndarray]) -> float:
    """二维速度向量 RMSE，即 AVE [m/s]。"""
    error = payload["prediction"][:, :2] - payload["target"][:, :2]
    return float(np.sqrt(np.mean(np.sum(error * error, axis=1))))


def compute_metrics(records: dict[str, dict[str, dict[str, np.ndarray]]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for source in MODELS:
        row: dict[str, Any] = {"model": source.name}
        for demo in DEMOS:
            payload = records[source.name][demo]
            row[f"{demo}_ave_mps"] = ave(payload)
            row[f"{demo}_rve_5s_mps"] = rve_5s(payload)
        rows.append(row)
    return rows


def configure_style() -> None:
    """统一论文图字体、嵌入式 PDF 字体及简洁坐标轴风格。"""
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 5.2,
        "axes.labelsize": 5.2,
        "axes.titlesize": 5.5,
        "xtick.labelsize": 4.7,
        "ytick.labelsize": 4.7,
        "legend.fontsize": 5.2,
        "axes.linewidth": 0.8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    })


def plot_metric_table(metrics: list[dict[str, Any]]) -> plt.Figure:
    """生成独立的 AVE/RVE 跨双栏表格图。"""
    configure_style()
    figure = plt.figure(figsize=(7.16, 1.24), constrained_layout=False)
    table_axis = figure.add_axes((0.03, 0.06, 0.94, 0.88))
    table_axis.axis("off")
    headers = ["Model"]
    for demo in DEMOS:
        headers.extend([f"{TEST_TITLES[demo]}\nAVE", f"{TEST_TITLES[demo]}\nRVE$_{{5s}}$"])
    cells = []
    for row in metrics:
        values = [row["model"]]
        for demo in DEMOS:
            values.extend([f"{row[f'{demo}_ave_mps']:.3f}", f"{row[f'{demo}_rve_5s_mps']:.3f}"])
        cells.append(values)
    table = table_axis.table(cellText=cells, colLabels=headers, cellLoc="center", colLoc="center",
                             loc="center", colWidths=[0.16] + [0.105] * 8)
    table.auto_set_font_size(False)
    table.set_fontsize(6.1)
    table.scale(1.0, 1.10)
    for (row_idx, col_idx), cell in table.get_celld().items():
        cell.set_edgecolor("#6A6A6A")
        cell.set_linewidth(0.35)
        if row_idx == 0:
            cell.set_facecolor("#E8EEF6")
            cell.set_text_props(weight="bold")
        elif col_idx == 0:
            cell.set_facecolor("#F5F5F5")
            cell.set_text_props(weight="bold")
    return figure


def plot_error_panels(records: dict[str, dict[str, dict[str, np.ndarray]]]) -> plt.Figure:
    """Arrange four test error trends and distributions for one paper column."""
    configure_style()
    figure = plt.figure(figsize=(3.45, 3.45), constrained_layout=False)
    grid = figure.add_gridspec(
        nrows=2, ncols=2,
        hspace=0.48, wspace=0.28,
        left=0.145, right=0.995, top=0.985, bottom=0.235,
    )

    plotted_sources = MODELS
    legend_handles = [Line2D([], [], color=PALETTE[source.name], linewidth=1.0) for source in plotted_sources]
    # Each test cell contains a velocity-error trend and a compact box plot.
    for index, demo in enumerate(DEMOS):
        pair_grid = grid[index // 2, index % 2].subgridspec(
            1, 2, width_ratios=[3.8, 1.0], wspace=0.03
        )
        line_axis = figure.add_subplot(pair_grid[0, 0])
        box_axis = figure.add_subplot(pair_grid[0, 1])
        error_series = []
        shown_peak = 0.0
        for source in MODELS:
            payload = records[source.name][demo]
            errors = instantaneous_error(payload)
            # 50 个 50 Hz 样本约为 1 s：总览图优先展示模型误差趋势。
            if source.name in LINE_MODELS:
                error_series.append(errors)
                plot_time, plot_errors = mean_downsample(payload["time"], errors, factor=50)
                shown_peak = max(shown_peak, float(np.max(plot_errors)))
                line_axis.plot(plot_time, plot_errors, linewidth=0.75, alpha=0.92,
                               color=PALETTE[source.name], label=source.name)

        # Fit the line axis exactly to the highest value actually drawn.
        upper = shown_peak if shown_peak > 0.0 else 0.01
        line_axis.set_ylim(0.0, upper)
        # Position each label in the open upper-right portion of its line plot.
        line_axis.text(0.97, 0.94, TEST_TITLES[demo], transform=line_axis.transAxes,
                       ha="right", va="top", fontsize=5.6, fontweight="bold",
                       bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.8, "pad": 0.6})
        if index % 2 == 0:
            line_axis.set_ylabel(r"$\Vert e_v^B\Vert_2$ [m/s]", labelpad=1.0)
        line_axis.grid(True, axis="y", linewidth=0.35, alpha=0.35)
        line_axis.spines[["top", "right"]].set_visible(False)
        line_axis.xaxis.set_major_locator(MaxNLocator(nbins=3))
        line_axis.set_xlabel("Time [s]", labelpad=1.0)

        box = box_axis.boxplot(error_series, patch_artist=True, showfliers=False, widths=0.36,
                               medianprops={"color": "#202020", "linewidth": 0.9},
                               whiskerprops={"color": "#4D4D4D", "linewidth": 0.7},
                               capprops={"color": "#4D4D4D", "linewidth": 0.7})
        # Raw-sample whiskers can exceed the one-second-mean line peak.
        whisker_peak = max(float(np.max(whisker.get_ydata())) for whisker in box["whiskers"])
        box_axis.set_ylim(0.0, max(upper, whisker_peak))
        for patch, source in zip(box["boxes"], plotted_sources):
            patch.set(facecolor=PALETTE[source.name], edgecolor="#303030", linewidth=0.55, alpha=0.82)
        box_axis.set_xticks(range(1, len(plotted_sources) + 1))
        # 横向长条中箱线图宽度有限，类别由底部共享颜色图例唯一标识，避免 F/R/D/M/G 重叠。
        box_axis.set_xticklabels([])
        box_axis.tick_params(axis="y", left=False, labelleft=False)
        box_axis.spines[["top", "right"]].set_visible(False)

    # Keep the shared legend below both rows.
    figure.legend(legend_handles, [item.name for item in plotted_sources], ncol=3, loc="lower center",
                  bbox_to_anchor=(0.5, 0.015), frameon=False, handlelength=1.1,
                  columnspacing=0.6, fontsize=4.8)
    return figure


def write_reports(metrics: list[dict[str, Any]]) -> None:
    """保存可用于论文表格复核的机器可读和人工可读指标。"""
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    published = []
    for row in metrics:
        named = {"model": row["model"]}
        for demo in DEMOS:
            name = TEST_NAMES[demo]
            named[f"{name}_ave_mps"] = row[f"{demo}_ave_mps"]
            named[f"{name}_rve_5s_mps"] = row[f"{demo}_rve_5s_mps"]
        published.append(named)
    fields = ["model"] + [field for demo in DEMOS for field in
                            (f"{TEST_NAMES[demo]}_ave_mps", f"{TEST_NAMES[demo]}_rve_5s_mps")]
    with (REPORT_DIR / "ablation_velocity_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(published)
    payload = {"metric_definition": {"ave": "body-frame XY vector RMSE [m/s]",
                                       "rve_5s": "5 s relative body-frame XY velocity-increment RMSE [m/s]"},
               "tests": [{"name": TEST_NAMES[demo], "source_prediction_stem": demo} for demo in DEMOS],
               "models": published}
    (REPORT_DIR / "ablation_velocity_metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    lines = ["# 五模型消融速度误差", "", "单位：m/s；AVE 为二维机体系速度向量 RMSE，RVE 为 5 s 相对速度增量 RMSE。", "",
             "| Model | " + " | ".join(f"{TEST_TITLES[demo]} AVE | {TEST_TITLES[demo]} RVE" for demo in DEMOS) + " |",
             "|---|" + "---:|" * (2 * len(DEMOS))]
    for row in published:
        values = [row["model"]]
        for demo in DEMOS:
            name = TEST_NAMES[demo]
            values.extend([f"{row[f'{name}_ave_mps']:.6f}", f"{row[f'{name}_rve_5s_mps']:.6f}"])
        lines.append("| " + " | ".join(values) + " |")
    (REPORT_DIR / "ablation_velocity_metrics.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    records = {
        source.name: {demo: load_prediction(source.directory / f"{demo}_predictions.npz") for demo in DEMOS}
        for source in MODELS
    }
    validate_common_reference(records)
    metrics = compute_metrics(records)
    write_reports(metrics)
    PAPER_FIGURES.mkdir(parents=True, exist_ok=True)
    table_figure = plot_metric_table(metrics)
    panel_figure = plot_error_panels(records)
    for figure, stem in ((table_figure, "ablation_velocity_table"),
                         (panel_figure, "ablation_velocity_error_panels")):
        base = PAPER_FIGURES / stem
        figure.savefig(base.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.02)
        figure.savefig(base.with_suffix(".png"), dpi=600, bbox_inches="tight", pad_inches=0.02)
        plt.close(figure)
        print(f"Generated: {base.with_suffix('.pdf')}")
        print(f"Generated: {base.with_suffix('.png')}")
    print(f"Metrics:   {REPORT_DIR / 'ablation_velocity_metrics.md'}")


if __name__ == "__main__":
    main()
