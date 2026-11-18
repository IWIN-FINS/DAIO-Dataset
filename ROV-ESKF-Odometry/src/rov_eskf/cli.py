"""唯一命令行入口：加载配置、创建不可覆盖运行目录并执行回放。"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from datetime import datetime
from pathlib import Path

import torch
import yaml

from .config import config_hash, load_config
from .evaluate import evaluate_record, evaluate_split
from .model import load_checkpoint


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="11 状态姿态辅助 ESKF 里程计")
    parser.add_argument("--config", default="configs/benchmark_apriltag_init.yaml", help="YAML 配置路径")
    parser.add_argument("--split", choices=["train", "val", "demo"], help="覆盖配置中的数据 split")
    parser.add_argument("--record", help="单条 HDF5；仍会加载配置指定的网络 checkpoint")
    parser.add_argument("--device", choices=["cpu", "cuda", "auto"], help="覆盖 model.device")
    parser.add_argument("--out", help="输出根目录；每次仍自动创建独立 run 子目录")
    parser.add_argument("--run-name", help="本次运行名称")
    parser.add_argument("--check-config", action="store_true", help="仅校验配置、路径与依赖，不运行回放")
    return parser


def _device(name: str) -> str:
    if name == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("配置要求 CUDA，但当前 PyTorch 未检测到可用 CUDA")
    return name


def _sha256(path: Path) -> str | None:
    if not path.is_file(): return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _run_dir(config: dict, override_root: str | None, override_name: str | None) -> Path:
    root = Path(override_root or config["output"]["root"])
    name = override_name or str(config["output"]["run_name"])
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    directory = root / f"{stamp}_{name}"
    suffix = 1
    while directory.exists():
        suffix += 1; directory = root / f"{stamp}_{name}_{suffix:02d}"
    directory.mkdir(parents=True)
    return directory


def _write_run_metadata(directory: Path, config: dict, config_path: str, device: str) -> None:
    (directory / "resolved_config.yaml").write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8")
    metadata = {"config_path": str(Path(config_path).resolve()), "config_sha256": config_hash(config), "device": device,
                "python": sys.version, "platform": platform.platform(), "torch": torch.__version__,
                "cuda_available": torch.cuda.is_available(), "checkpoint": config["model"].get("checkpoint"),
                "checkpoint_sha256": _sha256(Path(config["model"]["checkpoint"])) if config["model"].get("checkpoint") else None}
    (directory / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    overrides: dict = {}
    if args.split: overrides["data"] = {"split": args.split}
    if args.device: overrides["model"] = {"device": args.device}
    config = load_config(args.config, overrides)
    device = _device(str(config["model"]["device"]))
    if args.check_config:
        checkpoint = Path(config["model"]["checkpoint"])
        print(f"配置有效：mode={config['evaluation']['mode']}，device={device}")
        print(f"checkpoint: {checkpoint} ({'存在' if checkpoint.is_file() else '缺失'})")
        print(f"data root: {config['data']['root']} ({'存在' if Path(config['data']['root']).is_dir() else '缺失'})")
        return 0
    directory = _run_dir(config, args.out, args.run_name); _write_run_metadata(directory, config, args.config, device)
    if args.record:
        model = checkpoint = None
        if config["model"].get("checkpoint"):
            model, checkpoint = load_checkpoint(config["model"]["checkpoint"], device)
        metrics = evaluate_record(args.record, config, model, checkpoint, device, directory)
        summary = {"split": "record", "records": [{"record": Path(args.record).stem, "metrics": metrics}]}
    else:
        summary = evaluate_split(config["data"]["split"], config, directory, device)
    (directory / "console_summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(f"完成：{directory}")
    for row in summary["records"]:
        metrics = row["metrics"]; print(f"{row['record']}: " + ", ".join(f"{k}={v:.4f}" for k, v in metrics.items() if k.endswith("rmse_m") or k.endswith("rmse_rad") or k.endswith("rmse_mps")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
