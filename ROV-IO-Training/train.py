"""Main entry point for the ROV body-frame velocity network.

Defaults reproduce V20 preprocessing and architecture; separate YAML files enable V21--V23. Common commands:

    python train.py train

By default, read CSV data from ``dataset/processed`` under the main directory. If the dataset is elsewhere,
set ``--data-root`` to its root (the directory containing ``processed``). List all commands and arguments with
``python train.py --help`` or ``python train.py train --help``.

Configuration priority: command-line arguments > YAML > built-in default paths. At training start,
save the resolved configuration in the run directory; later YAML edits do not change previous experiments.
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Any


# -----------------------------------------------------------------------------
# Usually only these three paths need editing or CLI overrides; do not pass a raw-data directory.
# -----------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_ROOT = PROJECT_ROOT.parent / "dataset"
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "v20_csv_demo6.yaml"
DEFAULT_MANIFEST = PROJECT_ROOT / "manifests" / "v24_val26_27_demo4.yaml"
DEFAULT_RUNS_ROOT = PROJECT_ROOT / "runs"


def _add_data_arguments(parser: argparse.ArgumentParser, *, include_config: bool = True) -> None:
    """Add shared data-reading arguments with consistent semantics across commands."""
    parser.add_argument(
        "--data-root",
        type=Path,
        default=DEFAULT_DATA_ROOT,
        help="Dataset root containing processed/, processed/manifest.csv, and v1/v2 CSV files.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST,
        help="Data-split YAML; defaults to the fixed paper split, which should normally be retained.",
    )
    if include_config:
        parser.add_argument(
            "--config",
            type=Path,
            default=DEFAULT_CONFIG,
            help="Experiment YAML configuration; supplies all parameters not explicitly overridden.",
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="ROV-IO-Training: train and evaluate V20--V23 ROV velocity networks.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    commands = parser.add_subparsers(dest="command", required=True, title="Available commands")

    show = commands.add_parser("show-config", help="Print the default YAML configuration before editing.")
    show.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="YAML configuration file to display.")

    validate = commands.add_parser("validate", help="Check processed CSV files, 200 Hz timelines, fields, and splits.")
    _add_data_arguments(validate)
    validate.add_argument("--write-report", type=Path, default=None, help="Optionally write the JSON validation report to this file.")

    train = commands.add_parser("train", help="Start a new training run from random initialization without overwriting existing runs.")
    _add_data_arguments(train)
    train.add_argument("--runs-root", type=Path, default=DEFAULT_RUNS_ROOT, help="Parent directory for all experiment runs.")
    train.add_argument("--name", type=str, default=None, help="Experiment name included in the runs path.")
    train.add_argument("--seed", type=int, default=None, help="Random seed to aid reproducibility in the same environment.")
    train.add_argument("--device", choices=("auto", "cuda", "cpu"), default=None, help="Training device.")
    train.add_argument("--epochs", type=int, default=None, help="Total number of training epochs.")
    train.add_argument("--phase1-epochs", type=int, default=None, help="Number of initial Huber-only epochs.")
    train.add_argument("--batch-size", type=int, default=None, help="Windows per optimization step; reduce if GPU memory is insufficient.")
    train.add_argument("--num-workers", type=int, default=None, help="DataLoader worker processes; start with 0 on Windows.")
    train.add_argument("--learning-rate", type=float, default=None, help="Initial AdamW learning rate.")
    train.add_argument("--weight-decay", type=float, default=None, help="AdamW weight decay.")
    train.add_argument("--grad-clip-norm", type=float, default=None, help="Gradient-norm clipping threshold.")
    train.add_argument("--nll-weight", type=float, default=None, help="Gaussian NLL weight in the second phase.")
    train.add_argument("--log-every-batches", type=int, default=None, help="If positive, also print training loss every N batches.")
    train.add_argument("--plot-dpi", type=int, default=None, help="Resolution of post-training velocity figures.")
    train.add_argument("--deterministic", action="store_true", help="Require deterministic CUDA algorithms; may reduce speed.")
    train.add_argument("--enable-tf32", action="store_true", help="Allow TF32 on RTX GPUs; recommended only for throughput experiments.")
    train.add_argument("--skip-demo", action="store_true", help="Skip shared validation/demonstration plots after training and retain validation results only.")
    train.add_argument("--print-resolved-config", action="store_true", help="Print the configuration after CLI overrides and exit without training.")

    resume = commands.add_parser("resume", help="Resume interrupted training from last.pt in a run directory.")
    resume.add_argument("--run-dir", type=Path, required=True, help="Specific runs/<experiment>/<timestamp> directory to resume.")
    resume.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT, help="Root of the same processed dataset version.")

    evaluate = commands.add_parser("evaluate", help="Regenerate validation or demonstration plots and metrics using a saved model.")
    evaluate.add_argument("--run-dir", type=Path, required=True, help="Training directory containing best.pt and resolved_config.yaml.")
    evaluate.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT, help="Root of the same processed dataset version.")
    evaluate.add_argument("--split", choices=("val", "demo"), default="demo", help="Data split to evaluate.")
    evaluate.add_argument("--checkpoint", choices=("best", "last"), default="best", help="Evaluate the best or final-epoch model.")
    evaluate.add_argument("--output", type=Path, default=None, help="Output directory; defaults to evaluation/manual_<split> under the run.")

    inspect = commands.add_parser("inspect", help="Inspect checkpoint model configuration, input contract, and best metrics.")
    inspect.add_argument("--run-dir", type=Path, required=True, help="Training run directory.")
    inspect.add_argument("--checkpoint", choices=("best", "last"), default="best", help="Checkpoint to inspect.")
    return parser


def _apply_train_overrides(cfg: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    """Override only explicitly supplied arguments so CLI defaults do not silently change YAML settings."""
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
        raise FileNotFoundError(f"{label} does not exist: {path}")


def main() -> None:
    # Run this file directly without installing the project as a Python package.
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
        _require_directory(args.data_root, "Dataset root")
        cfg = load_config(args.config)
        report = validate_dataset(args.manifest, args.data_root, cfg)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        if args.write_report is not None:
            atomic_json(args.write_report, report)
            print(f"Data validation report written to: {args.write_report}")
        return
    if args.command == "train":
        _require_directory(args.data_root, "Dataset root")
        cfg = validate_config(_apply_train_overrides(load_config(args.config), args))
        if args.print_resolved_config:
            print(json.dumps(cfg, ensure_ascii=False, indent=2))
            return
        run_dir = train_new(cfg, args.config, args.manifest, args.data_root, args.runs_root, PROJECT_ROOT, sys.argv)
        print(f"\nTraining and automatic evaluation completed. Experiment directory:\n{run_dir}")
        return
    if args.command == "resume":
        _require_directory(args.data_root, "Dataset root")
        run_dir = resume_run(args.run_dir, args.data_root, PROJECT_ROOT, sys.argv)
        print(f"\nResumed training completed. Experiment directory:\n{run_dir}")
        return
    if args.command == "evaluate":
        _require_directory(args.data_root, "Dataset root")
        summary = evaluate_existing_run(args.run_dir, args.data_root, args.split, args.checkpoint, args.output, PROJECT_ROOT)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return
    if args.command == "inspect":
        print(json.dumps(inspect_existing_run(args.run_dir, args.checkpoint), ensure_ascii=False, indent=2))
        return
    raise RuntimeError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
