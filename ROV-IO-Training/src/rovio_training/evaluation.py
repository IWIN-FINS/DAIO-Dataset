"""Checkpoint 评估、预测导出和机体系速度图。

AprilTag 在本模块中只作为离线标签和评分参考；数据集类不会把它提供给网络输入。
"""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

# The managed Windows account may not permit a user-profile matplotlib cache.
# A writable temp cache avoids warning noise without changing experiment files.
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "rovio_training_matplotlib"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from .data import V20WindowDataset, resolve_manifest
from .metrics import AXES, network_metrics
from .utils import atomic_json


@torch.no_grad()
def predict_dataset(model: torch.nn.Module, dataset: V20WindowDataset, device: torch.device,
                    batch_size: int, num_workers: int, pin_memory: bool) -> dict[str, np.ndarray]:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                        pin_memory=pin_memory and device.type == "cuda")
    values: dict[str, list[np.ndarray]] = {"prediction": [], "std": [], "target": [], "time": [], "sequence": []}
    model.eval()
    for batch in loader:
        inputs = [batch[key].to(device, non_blocking=device.type == "cuda") for key in model.input_keys]
        mean, log_std = model(*inputs)
        values["prediction"].append(mean.cpu().numpy()); values["std"].append(np.exp(log_std.cpu().numpy()))
        values["target"].append(batch["target"].numpy()); values["time"].append(batch["time"].numpy()); values["sequence"].append(batch["sequence"].numpy())
    return {key: np.concatenate(value) for key, value in values.items()}


def _plot(record: str, payload: dict[str, np.ndarray], output: Path, dpi: int) -> None:
    figure, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True, constrained_layout=True)
    units = ("m/s", "m/s", "rad/s")
    for axis, name, unit, index in zip(axes, AXES, units, range(3)):
        axis.plot(payload["time"], payload["target"][:, index], color="black", linewidth=1.0, label="AprilTag offline label")
        axis.plot(payload["time"], payload["prediction"][:, index], color="#1565C0", linewidth=1.0, label="V20 prediction")
        axis.fill_between(payload["time"], payload["prediction"][:, index] - payload["std"][:, index],
                          payload["prediction"][:, index] + payload["std"][:, index], color="#1565C0", alpha=0.2, label="prediction ±1σ")
        axis.set_ylabel(f"{name} [{unit}]"); axis.grid(True, alpha=.25)
    axes[0].legend(loc="best", fontsize=8); axes[-1].set_xlabel("Time [s]")
    figure.suptitle(f"{record}: body-frame FLU velocity")
    figure.savefig(output, dpi=dpi); plt.close(figure)


def _save_predictions(payload: dict[str, np.ndarray], output: Path) -> None:
    """保存逐窗口预测，便于 MATLAB/Python 做二次误差分析而无需再次推理。"""
    np.savez_compressed(output, **payload)


def evaluate_split(model: torch.nn.Module, checkpoint: dict[str, Any], cfg: dict[str, Any], manifest_path: str | Path,
                   data_root: str | Path, split: str, output_dir: str | Path, device: torch.device,
                   report_split: str | None = None) -> dict[str, Any]:
    """Evaluate one split. This is the only location where labels enter evaluation."""
    if split not in {"val", "demo"}: raise ValueError("evaluation split must be val or demo")
    output_dir = Path(output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    paths = resolve_manifest(manifest_path, data_root)[split]
    normalizer = {key: np.asarray(value) for key, value in checkpoint["normalizer"].items()}
    training = cfg["training"]; combined: list[dict[str, np.ndarray]] = []; records: list[dict[str, Any]] = []
    for path in paths:
        dataset = V20WindowDataset([path], cfg, normalizer)
        payload = predict_dataset(model, dataset, device, int(training["batch_size"]), int(training["num_workers"]), bool(training["pin_memory"]))
        metrics = network_metrics(payload["prediction"], payload["std"], payload["target"])
        stem = path.stem
        _plot(stem, payload, output_dir / f"{stem}_network.png", int(cfg["evaluation"]["plot_dpi"]))
        _save_predictions(payload, output_dir / f"{stem}_predictions.npz")
        atomic_json(output_dir / f"{stem}_network.json", {"record": stem, "metrics": metrics,
                                                             "plot": f"{stem}_network.png", "predictions": f"{stem}_predictions.npz",
                                                             "samples": int(len(payload["time"]))})
        records.append({"record": stem, "metrics": metrics, "plot": f"{stem}_network.png", "predictions": f"{stem}_predictions.npz"})
        combined.append(payload)
    merged = {key: np.concatenate([item[key] for item in combined]) for key in combined[0]}
    summary = {"split": report_split or split, "checkpoint_epoch": int(checkpoint["epoch"]), "records": records,
               "combined": network_metrics(merged["prediction"], merged["std"], merged["target"])}
    atomic_json(output_dir / "network_summary.json", summary)
    return summary
