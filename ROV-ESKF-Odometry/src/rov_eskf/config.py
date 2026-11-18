"""Load YAML configuration, resolve paths, and validate filter parameters."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import yaml


def read_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"{path}: the YAML top level must be a mapping")
    return value


def _deep_merge(base: dict[str, Any], extra: dict[str, Any]) -> dict[str, Any]:
    result = dict(base)
    for key, value in extra.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        elif value is not None:
            result[key] = value
    return result


def _resolve_path(value: str, base: Path) -> str:
    path = Path(value)
    return str((base / path).resolve()) if not path.is_absolute() else str(path)


def _resolve_records(records: object, data_root: Path) -> list[dict[str, object]]:
    """Normalize per-trajectory replay windows while supporting legacy string paths."""
    if not isinstance(records, list):
        raise ValueError("data.records must be a list")
    resolved: list[dict[str, object]] = []
    for index, item in enumerate(records):
        if isinstance(item, str):
            raw_path, start_s, end_s = item, None, None
        elif isinstance(item, dict):
            unknown = set(item) - {"path", "start_s", "end_s"}
            if unknown or not isinstance(item.get("path"), str):
                raise ValueError(f"data.records[{index}] may contain only path/start_s/end_s, and path must be a string")
            raw_path, start_s, end_s = str(item["path"]), item.get("start_s"), item.get("end_s")
        else:
            raise ValueError(f"data.records[{index}] must be a path string or a mapping")
        path = Path(raw_path)
        absolute = str((data_root / path).resolve()) if not path.is_absolute() else str(path)
        resolved.append({"path": absolute, "start_s": start_s, "end_s": end_s})
    return resolved


def load_config(path: str | Path, overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    path = Path(path).resolve()
    config = read_yaml(path)
    if overrides:
        config = _deep_merge(config, overrides)
    # Resolve configuration paths relative to configs/, independently of the current PowerShell directory.
    base = path.parent
    config["data"]["root"] = _resolve_path(str(config["data"]["root"]), base)
    config["data"]["manifest"] = _resolve_path(str(config["data"]["manifest"]), base)
    config["data"]["records"] = _resolve_records(config["data"].get("records", []),
                                                     Path(config["data"]["root"]))
    if config["model"].get("checkpoint"):
        config["model"]["checkpoint"] = _resolve_path(str(config["model"]["checkpoint"]), base)
    config["output"]["root"] = _resolve_path(str(config["output"]["root"]), base)
    return validate_config(config)


def _matrix(section: dict[str, Any], key: str, shape: tuple[int, int]) -> None:
    value = np.asarray(section[key], dtype=float)
    if value.shape != shape or not np.all(np.isfinite(value)):
        raise ValueError(f"{key} must be a finite {shape[0]}x{shape[1]} matrix")
    value = .5 * (value + value.T)
    if np.min(np.linalg.eigvalsh(value)) < -1e-12:
        raise ValueError(f"{key} must be positive semidefinite")


def _positive(section: dict[str, Any], *keys: str) -> None:
    for key in keys:
        if not np.isfinite(float(section[key])) or float(section[key]) <= 0:
            raise ValueError(f"{key} must be finite and positive")


def validate_config(config: dict[str, Any]) -> dict[str, Any]:
    required = ("data", "model", "initialization", "filter", "benchmark", "evaluation", "debug", "output")
    for key in required:
        if not isinstance(config.get(key), dict):
            raise ValueError(f"Missing configuration section: {key}")
    data, model, init, fcfg = config["data"], config["model"], config["initialization"], config["filter"]
    if float(data["imu_hz"]) != 200.0 or float(data["network_hz"]) != 50.0 or int(data["window_samples"]) != 256:
        raise ValueError("The current V20 contract requires IMU=200 Hz, network=50 Hz, and window=256")
    _positive(init, "startup_s", "position_std_m", "velocity_std_mps", "accel_bias_std_mps2", "yaw_std_rad", "yaw_rate_correction_std_radps")
    for index, item in enumerate(data["records"]):
        start_s = item.get("start_s")
        end_s = item.get("end_s")
        if start_s is not None and (not np.isfinite(float(start_s)) or float(start_s) < float(init["startup_s"])):
            raise ValueError(f"data.records[{index}].start_s must be at least initialization.startup_s")
        if end_s is not None and (not np.isfinite(float(end_s)) or float(end_s) <= 0):
            raise ValueError(f"data.records[{index}].end_s must be finite and positive, or null")
        if start_s is not None and end_s is not None and float(end_s) <= float(start_s):
            raise ValueError(f"data.records[{index}].end_s must be greater than start_s")
    _matrix(fcfg, "accel_sample_covariance", (3, 3)); _matrix(fcfg, "accel_bias_rw_psd", (3, 3))
    _positive(fcfg, "height_variance_m2", "depth_nis_gate_1d", "velocity_nis_gate_2d", "yaw_rate_nis_gate_1d",
              "yaw_rate_source_psd_rad2_per_s", "yaw_rate_correction_rw_psd_rad2_per_s3")
    yaw_source = fcfg.get("yaw_rate_source")
    if not isinstance(yaw_source, dict):
        raise ValueError("filter.yaw_rate_source is required")
    yaw_source_type = str(yaw_source.get("type", "solved_yaw_backward_difference"))
    if yaw_source_type not in {"solved_yaw_backward_difference", "raw_gyro_z"}:
        raise ValueError("filter.yaw_rate_source.type must be solved_yaw_backward_difference or raw_gyro_z")
    if yaw_source_type == "solved_yaw_backward_difference":
        _positive(yaw_source, "backward_window_s", "max_abs_rate_radps")
        median_samples = int(yaw_source.get("median_filter_samples", 0))
        if median_samples < 1 or median_samples % 2 != 1:
            raise ValueError("filter.yaw_rate_source.median_filter_samples must be a positive odd integer")
    network = fcfg.get("network")
    if not isinstance(network, dict) or not isinstance(network.get("xy"), dict) or not isinstance(network.get("yaw_rate"), dict):
        raise ValueError("filter.network.xy and filter.network.yaw_rate are required")
    _positive(network["xy"], "min_std", "max_std", "covariance_scale", "max_step_mps")
    _positive(network["yaw_rate"], "min_std", "max_std", "variance_scale")
    if float(network["xy"]["min_std"]) >= float(network["xy"]["max_std"]) or float(network["yaw_rate"]["min_std"]) >= float(network["yaw_rate"]["max_std"]):
        raise ValueError("The network std lower bound must be smaller than the upper bound")
    if str(fcfg.get("yaw_mode", "estimated_network_r")) not in {"estimated_network_r", "direct_imu"}:
        raise ValueError("filter.yaw_mode must be estimated_network_r or direct_imu")
    if str(config["evaluation"]["mode"]) not in {"benchmark_apriltag_init", "deployment_relative_yaw"}:
        raise ValueError("evaluation.mode must be benchmark_apriltag_init or deployment_relative_yaw")
    _positive(config["benchmark"], "yaw_reanchor_std_rad")
    if str(model["device"]) not in {"cpu", "cuda", "auto"}:
        raise ValueError("model.device must be cpu/cuda/auto")
    _positive(model, "batch_size")
    _positive(config["debug"], "max_imu_gap_s")
    return config


def config_hash(config: dict[str, Any]) -> str:
    canonical = json.dumps(config, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
