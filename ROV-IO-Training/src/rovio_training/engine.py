"""Training, resumption, checkpoint management, and live per-epoch experiment reports."""
from __future__ import annotations

import csv
import json
import os
import shutil
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from .config import config_hash, v20_input_contract
from .data import V20WindowDataset, manifest_has_shared_validation_demo, resolve_manifest
from .evaluation import evaluate_split, predict_dataset
from .metrics import AXES, hybrid_loss, network_metrics
from .model import build_model, parameter_count
from .utils import (atomic_json, atomic_yaml, canonical_json, choose_device, environment_snapshot,
                    restore_rng_state, rng_state, set_seed, sha256_file)


CSV_COLUMNS = ("epoch", "phase", "train_loss", "train_huber", "train_nll", "learning_rate", "grad_norm",
               "epoch_seconds", "gpu_peak_memory_bytes", "val_body_xy_rmse", "val_all_axis_rmse",
               "val_vx_rmse", "val_vy_rmse", "val_r_rmse", "val_vx_mae", "val_vy_mae", "val_r_mae",
               "val_vx_bias", "val_vy_bias", "val_r_bias", "val_vx_nll", "val_vy_nll", "val_r_nll",
               "val_vx_calibration", "val_vy_calibration", "val_r_calibration",
               "is_best")


def atomic_torch_save(path: str | Path, value: dict[str, Any]) -> None:
    path = Path(path); temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary); os.replace(temporary, path)


def load_checkpoint(path: str | Path, device: torch.device) -> tuple[torch.nn.Module, dict[str, Any]]:
    payload = torch.load(Path(path), map_location=device, weights_only=False)
    if payload.get("format_version") != 1: raise ValueError("unsupported checkpoint format")
    model = build_model(payload["config"]).to(device); model.load_state_dict(payload["model"], strict=True); model.eval()
    return model, payload


def _run_payload(model: torch.nn.Module, optimizer: AdamW, scheduler: CosineAnnealingLR, epoch: int,
                 best_metric: float, best_epoch: int | None, normalizer: dict[str, np.ndarray], cfg: dict[str, Any],
                 manifest_hash: str, input_contract: dict[str, Any]) -> dict[str, Any]:
    return {"format_version": 1, "epoch": int(epoch), "model": model.state_dict(), "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(), "best_metric": float(best_metric), "best_epoch": best_epoch,
            "normalizer": normalizer, "config": cfg, "config_sha256": config_hash(cfg), "manifest_sha256": manifest_hash,
            "input_contract": input_contract, "rng_state": rng_state()}


def _metrics_row(epoch: int, phase: int, train: dict[str, float], metrics: dict[str, Any], lr: float,
                 seconds: float, gpu_peak: int, is_best: bool) -> dict[str, Any]:
    axes = metrics["axes"]
    return {"epoch": epoch, "phase": phase, "train_loss": train["loss"], "train_huber": train["huber"], "train_nll": train["nll"],
            "learning_rate": lr, "grad_norm": train["grad_norm"], "epoch_seconds": seconds, "gpu_peak_memory_bytes": gpu_peak,
            "val_body_xy_rmse": metrics["body_xy_rmse"], "val_all_axis_rmse": metrics["all_axis_rmse"],
            "val_vx_rmse": axes[AXES[0]]["rmse"], "val_vy_rmse": axes[AXES[1]]["rmse"], "val_r_rmse": axes[AXES[2]]["rmse"],
            "val_vx_mae": axes[AXES[0]]["mae"], "val_vy_mae": axes[AXES[1]]["mae"], "val_r_mae": axes[AXES[2]]["mae"],
            "val_vx_bias": axes[AXES[0]]["mean_error"], "val_vy_bias": axes[AXES[1]]["mean_error"], "val_r_bias": axes[AXES[2]]["mean_error"],
            "val_vx_nll": axes[AXES[0]]["gaussian_nll"], "val_vy_nll": axes[AXES[1]]["gaussian_nll"], "val_r_nll": axes[AXES[2]]["gaussian_nll"],
            "val_vx_calibration": axes[AXES[0]]["calibration_ratio"], "val_vy_calibration": axes[AXES[1]]["calibration_ratio"],
            "val_r_calibration": axes[AXES[2]]["calibration_ratio"], "is_best": bool(is_best)}


def _append_metrics(run_dir: Path, row: dict[str, Any]) -> None:
    jsonl = run_dir / "metrics.jsonl"
    with jsonl.open("a", encoding="utf-8") as handle: handle.write(canonical_json(row) + "\n")
    csv_path = run_dir / "metrics.csv"; new = not csv_path.exists()
    with csv_path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        if new: writer.writeheader()
        writer.writerow(row)


def _print_epoch(row: dict[str, Any], best_epoch: int | None, best_metric: float) -> None:
    tag = " *BEST*" if row["is_best"] else ""
    print(f"\n[Epoch {row['epoch']:03d} | Phase {row['phase']}] Training: total_loss={row['train_loss']:.6f}, "
          f"Huber={row['train_huber']:.6f}, NLL={row['train_nll']:.6f}, learning_rate={row['learning_rate']:.3e}, "
          f"grad_norm={row['grad_norm']:.3f}, elapsed={row['epoch_seconds']:.1f}s", flush=True)
    print(f"  Validation: joint XY RMSE={row['val_body_xy_rmse']:.6f} m/s, "
          f"all-axis RMSE={row['val_all_axis_rmse']:.6f}{tag}", flush=True)
    for axis, unit in (("vx", "m/s"), ("vy", "m/s"), ("r", "rad/s")):
        print(f"  {axis:>2}: RMSE={row[f'val_{axis}_rmse']:.6f} {unit}, "
              f"MAE={row[f'val_{axis}_mae']:.6f}, Bias={row[f'val_{axis}_bias']:.6f}, "
              f"NLL={row[f'val_{axis}_nll']:.6f}, calibration_ratio={row[f'val_{axis}_calibration']:.3f}", flush=True)
    print(f"  Best so far: validation XY RMSE={best_metric:.6f} m/s (epoch {best_epoch})", flush=True)


def _tensorboard(writer: SummaryWriter, row: dict[str, Any]) -> None:
    epoch = int(row["epoch"])
    for key in ("train_loss", "train_huber", "train_nll", "learning_rate", "grad_norm", "val_body_xy_rmse", "val_all_axis_rmse"):
        writer.add_scalar(key, float(row[key]), epoch)
    for axis in ("vx", "vy", "r"):
        writer.add_scalar(f"validation/{axis}_rmse", float(row[f"val_{axis}_rmse"]), epoch)
        writer.add_scalar(f"validation/{axis}_mae", float(row[f"val_{axis}_mae"]), epoch)
        writer.add_scalar(f"validation/{axis}_nll", float(row[f"val_{axis}_nll"]), epoch)
        writer.add_scalar(f"validation/{axis}_calibration_ratio", float(row[f"val_{axis}_calibration"]), epoch)


def _build_datasets(cfg: dict[str, Any], manifest_path: str | Path, data_root: str | Path,
                    normalizer: dict[str, np.ndarray] | None = None) -> tuple[V20WindowDataset, V20WindowDataset, dict[str, list[Path]]]:
    # Fit normalization statistics on train only; val/demo always reuse them to prevent statistical leakage.
    splits = resolve_manifest(manifest_path, data_root)
    train = V20WindowDataset(splits["train"], cfg, normalizer)
    if normalizer is None: normalizer = train.fit_normalizer()
    validation = V20WindowDataset(splits["val"], cfg, normalizer)
    return train, validation, splits


def _training_loader(dataset: V20WindowDataset, cfg: dict[str, Any], device: torch.device) -> DataLoader:
    options = cfg["training"]
    return DataLoader(dataset, batch_size=int(options["batch_size"]), shuffle=True, num_workers=int(options["num_workers"]),
                      pin_memory=bool(options["pin_memory"]) and device.type == "cuda")


def _train_epoch(model: torch.nn.Module, loader: DataLoader, optimizer: AdamW, cfg: dict[str, Any], device: torch.device,
                 phase: int, epoch: int) -> dict[str, float]:
    model.train(); values = {"loss": [], "huber": [], "nll": [], "grad_norm": []}; options = cfg["training"]
    log_every = int(options.get("log_every_batches", 0))
    for batch_index, batch in enumerate(loader, start=1):
        target = batch["target"].to(device, non_blocking=device.type == "cuda")
        inputs = [batch[key].to(device, non_blocking=device.type == "cuda") for key in model.input_keys]
        mean, log_std = model(*inputs)
        loss, details = hybrid_loss(mean, log_std, target, phase, float(options["huber_delta"]), float(options["nll_weight"]))
        # set_to_none reduces gradient-tensor writes; gradient clipping limits occasional large updates.
        optimizer.zero_grad(set_to_none=True); loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float(options["grad_clip_norm"]))
        optimizer.step()
        values["loss"].append(float(loss.detach())); values["huber"].append(details["huber"]); values["nll"].append(details["nll"]); values["grad_norm"].append(float(norm))
        if log_every and batch_index % log_every == 0:
            print(f"epoch={epoch:03d} batch={batch_index:04d}/{len(loader):04d} loss={values['loss'][-1]:.6f}", flush=True)
    return {key: float(np.mean(value)) for key, value in values.items()}


def _validate(model: torch.nn.Module, validation: V20WindowDataset, cfg: dict[str, Any], device: torch.device) -> dict[str, Any]:
    options = cfg["training"]
    payload = predict_dataset(model, validation, device, int(options["batch_size"]), int(options["num_workers"]), bool(options["pin_memory"]))
    return network_metrics(payload["prediction"], payload["std"], payload["target"])


def create_run_dir(output_root: str | Path, experiment: str, seed: int) -> Path:
    import re
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", experiment).strip("_")
    stamp = time.strftime("%Y%m%d-%H%M%S")
    path = Path(output_root) / safe / f"{stamp}_seed{seed}"
    if path.exists(): raise FileExistsError(f"refusing to overwrite existing run: {path}")
    path.mkdir(parents=True)
    return path


def _dataset_inventory(splits: dict[str, list[Path]], data_root: str | Path) -> dict[str, Any]:
    """Generate a strong integrity manifest for training data.

    Record SHA-256 for every processed file. If content is overwritten at the same path, resuming training
    or reevaluating fails explicitly instead of silently mixing dataset versions.
    """
    root = Path(data_root)
    records: dict[str, list[dict[str, Any]]] = {}
    for split, paths in splits.items():
        records[split] = [
            {"relative_path": str(path.relative_to(root)), "size_bytes": path.stat().st_size, "sha256": sha256_file(path)}
            for path in paths
        ]
    snapshot = canonical_json(records)
    import hashlib
    return {"records": records, "snapshot_sha256": hashlib.sha256(snapshot.encode("utf-8")).hexdigest()}


def _assert_dataset_snapshot(run_dir: Path, manifest_path: str | Path, data_root: str | Path, expected_hash: str) -> None:
    """Prevent hidden reproducibility failures from changed content at unchanged paths before resumption or evaluation."""
    current = _dataset_inventory(resolve_manifest(manifest_path, data_root), data_root)
    if current["snapshot_sha256"] != expected_hash:
        atomic_json(run_dir / "data_snapshot_mismatch.json", {"expected": expected_hash, "actual": current["snapshot_sha256"]})
        raise RuntimeError("The dataset differs from the SHA-256 manifest saved at training start; use the original version or start a new experiment")


def _append_resume_event(run_dir: Path, invocation: list[str]) -> None:
    """Append resumption history without modifying the original configuration, model, or previous metrics."""
    path = run_dir / "resume_history.jsonl"
    event = {"time_unix": time.time(), "invocation": invocation}
    with path.open("a", encoding="utf-8") as handle:
        handle.write(canonical_json(event) + "\n")


def train_new(cfg: dict[str, Any], config_source: str | Path, manifest_path: str | Path, data_root: str | Path,
              output_root: str | Path, project_root: str | Path, invocation: list[str] | None = None) -> Path:
    """Start a new experiment.

    Each call creates a new directory to avoid overwriting an already trained model.
    """
    run_dir = create_run_dir(output_root, str(cfg["experiment"]["name"]), int(cfg["experiment"]["seed"]))
    _initialize_new_run(run_dir, cfg, config_source, manifest_path, data_root, project_root, invocation or [])
    _train(run_dir, cfg, manifest_path, data_root, project_root, resume=False)
    return run_dir


def _initialize_new_run(run_dir: Path, cfg: dict[str, Any], config_source: str | Path, manifest_path: str | Path,
                        data_root: str | Path, project_root: str | Path, invocation: list[str]) -> None:
    """Freeze training dependencies so the provenance of best.pt remains traceable."""
    device = choose_device(cfg["experiment"]["device"])
    atomic_yaml(run_dir / "resolved_config.yaml", cfg)
    paths = resolve_manifest(manifest_path, data_root)
    # Copy the manifest byte-for-byte instead of reserializing YAML so its SHA-256 matches the source.
    manifest_snapshot = run_dir / "manifest_snapshot.yaml"
    temporary_manifest = manifest_snapshot.with_suffix(".yaml.tmp")
    shutil.copyfile(manifest_path, temporary_manifest)
    os.replace(temporary_manifest, manifest_snapshot)
    inventory = _dataset_inventory(paths, data_root)
    atomic_json(run_dir / "dataset_inventory.json", inventory)
    atomic_json(run_dir / "run_metadata.json", {"config_source": str(Path(config_source).resolve()), "config_sha256": config_hash(cfg),
                                                  "manifest_source": str(Path(manifest_path).resolve()), "manifest_sha256": sha256_file(manifest_path),
                                                  "manifest_snapshot": manifest_snapshot.name,
                                                  "shared_validation_demo": manifest_has_shared_validation_demo(manifest_path),
                                                  "data_snapshot_sha256": inventory["snapshot_sha256"],
                                                  "data_root_at_start": str(Path(data_root).resolve()), "input_contract": v20_input_contract(cfg),
                                                  "invocation": list(invocation), "environment": environment_snapshot(device, project_root)})
    atomic_json(run_dir / "resolved_manifest.json", {split: [str(path.relative_to(Path(data_root))) for path in records]
                                                       for split, records in paths.items()})


def resume_run(run_dir: str | Path, data_root: str | Path, project_root: str | Path, invocation: list[str] | None = None) -> Path:
    run_dir = Path(run_dir); cfg_path = run_dir / "resolved_config.yaml"; metadata_path = run_dir / "run_metadata.json"
    manifest_snapshot = run_dir / "manifest_snapshot.yaml"
    if not cfg_path.is_file() or not metadata_path.is_file() or not manifest_snapshot.is_file() or not (run_dir / "last.pt").is_file():
        raise FileNotFoundError("resume requires a run directory containing resolved_config.yaml, manifest_snapshot.yaml, run_metadata.json, and last.pt")
    from .config import load_config
    cfg = load_config(cfg_path); metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if sha256_file(manifest_snapshot) != metadata["manifest_sha256"]:
        raise RuntimeError("manifest_snapshot.yaml was modified; refusing to resume with inconsistent data splits")
    _assert_dataset_snapshot(run_dir, manifest_snapshot, data_root, metadata["data_snapshot_sha256"])
    _append_resume_event(run_dir, invocation or [])
    _train(run_dir, cfg, manifest_snapshot, data_root, project_root, resume=True)
    return run_dir


def _train(run_dir: Path, cfg: dict[str, Any], manifest_path: str | Path, data_root: str | Path,
           project_root: str | Path, resume: bool) -> None:
    device = choose_device(cfg["experiment"]["device"]); set_seed(int(cfg["experiment"]["seed"]), bool(cfg["experiment"]["deterministic"]))
    if device.type == "cuda": torch.backends.cuda.matmul.allow_tf32 = bool(cfg["experiment"]["tf32"])
    resume_payload: dict[str, Any] | None = None
    if resume:
        resume_payload = torch.load(run_dir / "last.pt", map_location=device, weights_only=False)
        if resume_payload.get("config_sha256") != config_hash(cfg): raise RuntimeError("resolved config does not match last checkpoint")
    resume_normalizer = None if resume_payload is None else {key: np.asarray(value) for key, value in resume_payload["normalizer"].items()}
    train_data, validation, _ = _build_datasets(cfg, manifest_path, data_root, resume_normalizer)
    loader = _training_loader(train_data, cfg, device); model = build_model(cfg).to(device)
    options = cfg["training"]; optimizer = AdamW(model.parameters(), lr=float(options["lr"]), weight_decay=float(options["weight_decay"]))
    scheduler = CosineAnnealingLR(optimizer, T_max=int(options["epochs"]))
    best_metric, best_epoch, start_epoch = float("inf"), None, 1
    if resume:
        assert resume_payload is not None
        model.load_state_dict(resume_payload["model"], strict=True); optimizer.load_state_dict(resume_payload["optimizer"]); scheduler.load_state_dict(resume_payload["scheduler"])
        restore_rng_state(resume_payload["rng_state"]); best_metric = float(resume_payload["best_metric"]); best_epoch = resume_payload.get("best_epoch"); start_epoch = int(resume_payload["epoch"]) + 1
        if start_epoch > int(options["epochs"]): raise RuntimeError("run is already complete; use evaluate instead of resume")
    normalizer = train_data.normalizer
    assert normalizer is not None
    input_contract = v20_input_contract(cfg); manifest_hash = sha256_file(manifest_path)
    with SummaryWriter(log_dir=str(run_dir / "tensorboard"), purge_step=start_epoch if resume else None) as writer:
        for epoch in range(start_epoch, int(options["epochs"]) + 1):
            if device.type == "cuda": torch.cuda.reset_peak_memory_stats(device)
            started = time.perf_counter(); phase = 1 if epoch <= int(options["phase1_epochs"]) else 2
            try:
                train_metrics = _train_epoch(model, loader, optimizer, cfg, device, phase, epoch)
                scheduler.step(); validation_metrics = _validate(model, validation, cfg, device)
            except RuntimeError as error:
                if "out of memory" in str(error).lower():
                    atomic_json(run_dir / "failure.json", {"epoch": epoch, "error": str(error), "action": "stopped; batch size was not changed automatically"})
                raise
            seconds = time.perf_counter() - started; gpu_peak = int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
            # Update best.pt only in phase two (Huber + NLL), since the covariance head is unconstrained in phase one.
            current = float(validation_metrics["body_xy_rmse"]); is_best = phase == 2 and current < best_metric
            if is_best: best_metric, best_epoch = current, epoch
            checkpoint = _run_payload(model, optimizer, scheduler, epoch, best_metric, best_epoch, normalizer, cfg, manifest_hash, input_contract)
            atomic_torch_save(run_dir / "last.pt", checkpoint)
            if is_best: atomic_torch_save(run_dir / "best.pt", checkpoint)
            row = _metrics_row(epoch, phase, train_metrics, validation_metrics, float(optimizer.param_groups[0]["lr"]), seconds, gpu_peak, is_best)
            _append_metrics(run_dir, row); _tensorboard(writer, row); writer.flush(); _print_epoch(row, best_epoch, best_metric)
    if not (run_dir / "best.pt").is_file(): raise RuntimeError("training ended without phase-2 best.pt")
    best_model, best_checkpoint = load_checkpoint(run_dir / "best.pt", device)
    shared_validation_demo = manifest_has_shared_validation_demo(manifest_path)
    final_output = run_dir / "evaluation" / ("shared_validation_demo" if shared_validation_demo else "val")
    val_summary = evaluate_split(best_model, best_checkpoint, cfg, manifest_path, data_root, "val", final_output, device,
                                 report_split="shared_validation_demo" if shared_validation_demo else None)
    final = {"best_checkpoint": "best.pt", "best_epoch": best_checkpoint["epoch"],
             "best_validation_body_xy_rmse": best_checkpoint["best_metric"], "validation": val_summary,
             "shared_validation_demo": shared_validation_demo}
    if bool(cfg["evaluation"]["evaluate_demo_after_training"]) and not shared_validation_demo:
        final["demo"] = evaluate_split(best_model, best_checkpoint, cfg, manifest_path, data_root, "demo", run_dir / "evaluation" / "demo", device)
    elif shared_validation_demo:
        final["demo"] = {"shared_with": "val", "blind_test": False, "results": val_summary}
    atomic_json(run_dir / "final_summary.json", final)
    print(f"training_complete run={run_dir} best_epoch={best_checkpoint['epoch']} best_val_xy_rmse={best_checkpoint['best_metric']:.6f}", flush=True)


def evaluate_existing_run(run_dir: str | Path, data_root: str | Path, split: str, checkpoint_name: str,
                          output_dir: str | Path | None, project_root: str | Path) -> dict[str, Any]:
    """Reevaluate an existing experiment without rewriting training logs or affecting model selection."""
    run_dir = Path(run_dir)
    if split not in {"val", "demo"}:
        raise ValueError("split must be val or demo")
    checkpoint_path = run_dir / f"{checkpoint_name}.pt"
    config_path = run_dir / "resolved_config.yaml"
    manifest_path = run_dir / "manifest_snapshot.yaml"
    metadata_path = run_dir / "run_metadata.json"
    if not checkpoint_path.is_file() or not config_path.is_file() or not manifest_path.is_file() or not metadata_path.is_file():
        raise FileNotFoundError("The run directory is missing a checkpoint, resolved_config.yaml, manifest_snapshot.yaml, or run_metadata.json")
    from .config import load_config
    cfg = load_config(config_path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    _assert_dataset_snapshot(run_dir, manifest_path, data_root, metadata["data_snapshot_sha256"])
    device = choose_device(cfg["experiment"]["device"])
    model, payload = load_checkpoint(checkpoint_path, device)
    if payload["config_sha256"] != config_hash(cfg):
        raise RuntimeError("checkpoint does not match resolved_config.yaml; refusing to use it")
    destination = Path(output_dir) if output_dir is not None else run_dir / "evaluation" / f"manual_{split}_{checkpoint_name}"
    result = evaluate_split(model, payload, cfg, manifest_path, data_root, split, destination, device)
    atomic_json(destination / "re_evaluation_context.json", {"checkpoint": checkpoint_path.name, "data_snapshot_sha256": metadata["data_snapshot_sha256"],
                                                               "project_root": str(Path(project_root).resolve())})
    return result


def inspect_existing_run(run_dir: str | Path, checkpoint_name: str) -> dict[str, Any]:
    """Return commonly needed checkpoint metadata without loading or running the dataset."""
    run_dir = Path(run_dir); checkpoint_path = run_dir / f"{checkpoint_name}.pt"
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    return {"checkpoint": str(checkpoint_path), "epoch": payload.get("epoch"), "best_epoch": payload.get("best_epoch"),
            "best_validation_body_xy_rmse": payload.get("best_metric"), "input_contract": payload.get("input_contract"),
            "config": payload.get("config"), "normalizer": payload.get("normalizer")}
