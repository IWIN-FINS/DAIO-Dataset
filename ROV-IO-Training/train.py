"""ROV 机体系速度网络的唯一启动入口。

本工程默认复现 V20 的训练数据处理与网络结构；V21--V23 通过独立 YAML 启用。最常用的命令为：

    python train.py train

脚本默认读取主目录下的 ``dataset/processed`` CSV 数据。如数据集不在默认
位置，则通过 ``--data-root`` 指定其根目录（即包含 ``processed`` 的目录）。所有命令和参数均可用
``python train.py --help`` 或 ``python train.py train --help`` 查看。

配置优先级：命令行参数 > YAML 配置文件 > 本脚本内置默认路径。训练开始后，
最终生效的配置会写入本次 runs 目录，之后即使修改 YAML 也不会改变旧实验。
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Any


# -----------------------------------------------------------------------------
# 初学者通常只需修改或在命令行覆盖以下三个路径；不要把 raw 数据目录传进来。
# -----------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_ROOT = PROJECT_ROOT.parent / "dataset"
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "v20_csv_demo6.yaml"
DEFAULT_MANIFEST = PROJECT_ROOT / "manifests" / "v24_val26_27_demo4.yaml"
DEFAULT_RUNS_ROOT = PROJECT_ROOT / "runs"


def _add_data_arguments(parser: argparse.ArgumentParser, *, include_config: bool = True) -> None:
    """添加所有读取数据所需的公共参数，避免各命令的语义不一致。"""
    parser.add_argument(
        "--data-root",
        type=Path,
        default=DEFAULT_DATA_ROOT,
        help="数据集根目录，内部必须包含 processed/、processed/manifest.csv 以及 v1/v2 CSV。",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST,
        help="数据划分 YAML；默认使用论文固定划分，不建议随意修改。",
    )
    if include_config:
        parser.add_argument(
            "--config",
            type=Path,
            default=DEFAULT_CONFIG,
            help="实验 YAML 配置；未覆盖的参数均从此文件读取。",
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="ROV-IO-Training：V20--V23 ROV 速度网络训练与评估。",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    commands = parser.add_subparsers(dest="command", required=True, title="可用命令")

    show = commands.add_parser("show-config", help="打印默认 YAML 配置，方便修改前查阅。")
    show.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="要显示的 YAML 配置文件。")

    validate = commands.add_parser("validate", help="检查 processed CSV、200 Hz 时间轴、字段和划分。")
    _add_data_arguments(validate)
    validate.add_argument("--write-report", type=Path, default=None, help="可选：将 JSON 检查报告写入此文件。")

    train = commands.add_parser("train", help="从随机初始化开始一个新的不可覆盖训练实验。")
    _add_data_arguments(train)
    train.add_argument("--runs-root", type=Path, default=DEFAULT_RUNS_ROOT, help="所有实验运行目录的父目录。")
    train.add_argument("--name", type=str, default=None, help="本次实验名称；会体现在 runs 路径中。")
    train.add_argument("--seed", type=int, default=None, help="随机种子；相同环境下有助于复现。")
    train.add_argument("--device", choices=("auto", "cuda", "cpu"), default=None, help="训练设备。")
    train.add_argument("--epochs", type=int, default=None, help="总训练轮数。")
    train.add_argument("--phase1-epochs", type=int, default=None, help="仅 Huber 损失的前期轮数。")
    train.add_argument("--batch-size", type=int, default=None, help="每个优化步的窗口数量；显存不足时调小。")
    train.add_argument("--num-workers", type=int, default=None, help="DataLoader 子进程数；Windows 建议先保持 0。")
    train.add_argument("--learning-rate", type=float, default=None, help="AdamW 初始学习率。")
    train.add_argument("--weight-decay", type=float, default=None, help="AdamW 权重衰减。")
    train.add_argument("--grad-clip-norm", type=float, default=None, help="梯度范数裁剪阈值。")
    train.add_argument("--nll-weight", type=float, default=None, help="第二阶段 Gaussian NLL 的权重。")
    train.add_argument("--log-every-batches", type=int, default=None, help="大于 0 时额外打印每 N 个 batch 的训练损失。")
    train.add_argument("--plot-dpi", type=int, default=None, help="训练结束后速度图的分辨率。")
    train.add_argument("--deterministic", action="store_true", help="强制确定性 CUDA 算法；速度可能下降。")
    train.add_argument("--enable-tf32", action="store_true", help="允许 RTX 卡使用 TF32；仅吞吐实验建议开启。")
    train.add_argument("--skip-demo", action="store_true", help="训练结束后不生成共享验证/展示图，仅保留验证集结果。")
    train.add_argument("--print-resolved-config", action="store_true", help="打印命令行覆盖后的配置并退出，不启动训练。")

    resume = commands.add_parser("resume", help="从某个 runs 目录内的 last.pt 继续中断的训练。")
    resume.add_argument("--run-dir", type=Path, required=True, help="待恢复的具体 runs/<实验>/<时间戳> 目录。")
    resume.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT, help="同一版本 processed 数据集根目录。")

    evaluate = commands.add_parser("evaluate", help="使用已保存模型重新生成验证或 demo 的图表和指标。")
    evaluate.add_argument("--run-dir", type=Path, required=True, help="包含 best.pt、resolved_config.yaml 的训练目录。")
    evaluate.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT, help="同一版本 processed 数据集根目录。")
    evaluate.add_argument("--split", choices=("val", "demo"), default="demo", help="需要评估的数据划分。")
    evaluate.add_argument("--checkpoint", choices=("best", "last"), default="best", help="评估最佳或最后一轮模型。")
    evaluate.add_argument("--output", type=Path, default=None, help="结果目录；不填则写入该 run 的 evaluation/manual_<split>。")

    inspect = commands.add_parser("inspect", help="查看 checkpoint 的模型配置、输入契约和最佳指标。")
    inspect.add_argument("--run-dir", type=Path, required=True, help="训练运行目录。")
    inspect.add_argument("--checkpoint", choices=("best", "last"), default="best", help="查看的 checkpoint。")
    return parser


def _apply_train_overrides(cfg: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    """只覆盖用户显式传入的参数，防止命令行默认值悄悄改变 YAML。"""
    cfg = copy.deepcopy(cfg)
    mapping = {
        "name": ("experiment", "name"), "seed": ("experiment", "seed"), "device": ("experiment", "device"),
        "epochs": ("training", "epochs"), "phase1_epochs": ("training", "phase1_epochs"),
        "batch_size": ("training", "batch_size"), "num_workers": ("training", "num_workers"),
        "learning_rate": ("training", "lr"), "weight_decay": ("training", "weight_decay"),
        "grad_clip_norm": ("training", "grad_clip_norm"), "nll_weight": ("training", "nll_weight"),
        "log_every_batches": ("training", "log_every_batches"), "plot_dpi": ("evaluation", "plot_dpi"),
    }
    for argument, (section, key) in mapping.items():
        value = getattr(args, argument)
        if value is not None:
            cfg[section][key] = value
    if args.deterministic:
        cfg["experiment"]["deterministic"] = True
    if args.enable_tf32:
        cfg["experiment"]["tf32"] = True
    if args.skip_demo:
        cfg["evaluation"]["evaluate_demo_after_training"] = False
    return cfg


def _require_directory(path: Path, label: str) -> None:
    if not path.is_dir():
        raise FileNotFoundError(f"{label} 不存在：{path}")


def main() -> None:
    # 直接运行本文件即可，无需安装本工程为 Python package。
    sys.path.insert(0, str(PROJECT_ROOT / "src"))
    from rovio_training.config import load_config, validate_config
    from rovio_training.data import validate_dataset
    from rovio_training.engine import evaluate_existing_run, inspect_existing_run, resume_run, train_new
    from rovio_training.utils import atomic_json

    args = build_parser().parse_args()
    if args.command == "show-config":
        print(args.config.read_text(encoding="utf-8"), end="")
        return
    if args.command == "validate":
        _require_directory(args.data_root, "数据集根目录")
        cfg = load_config(args.config)
        report = validate_dataset(args.manifest, args.data_root, cfg)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        if args.write_report is not None:
            atomic_json(args.write_report, report)
            print(f"数据检查报告已写入：{args.write_report}")
        return
    if args.command == "train":
        _require_directory(args.data_root, "数据集根目录")
        cfg = validate_config(_apply_train_overrides(load_config(args.config), args))
        if args.print_resolved_config:
            print(json.dumps(cfg, ensure_ascii=False, indent=2))
            return
        run_dir = train_new(cfg, args.config, args.manifest, args.data_root, args.runs_root, PROJECT_ROOT, sys.argv)
        print(f"\n训练和自动评估已完成。实验目录：\n{run_dir}")
        return
    if args.command == "resume":
        _require_directory(args.data_root, "数据集根目录")
        run_dir = resume_run(args.run_dir, args.data_root, PROJECT_ROOT, sys.argv)
        print(f"\n恢复训练完成。实验目录：\n{run_dir}")
        return
    if args.command == "evaluate":
        _require_directory(args.data_root, "数据集根目录")
        summary = evaluate_existing_run(args.run_dir, args.data_root, args.split, args.checkpoint, args.output, PROJECT_ROOT)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return
    if args.command == "inspect":
        print(json.dumps(inspect_existing_run(args.run_dir, args.checkpoint), ensure_ascii=False, indent=2))
        return
    raise RuntimeError(f"未知命令：{args.command}")


if __name__ == "__main__":
    main()
