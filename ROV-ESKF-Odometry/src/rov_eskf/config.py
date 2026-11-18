"""YAML 配置读取、路径解析和滤波参数校验。"""
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
        raise ValueError(f"{path}: YAML 顶层必须为 mapping")
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
    """规范化逐轨迹回放窗口，同时兼容旧版字符串路径配置。"""
    if not isinstance(records, list):
        raise ValueError("data.records 必须是列表")
    resolved: list[dict[str, object]] = []
    for index, item in enumerate(records):
        if isinstance(item, str):
            raw_path, start_s, end_s = item, None, None
        elif isinstance(item, dict):
            unknown = set(item) - {"path", "start_s", "end_s"}
            if unknown or not isinstance(item.get("path"), str):
                raise ValueError(f"data.records[{index}] 只能包含 path/start_s/end_s，且 path 必须是字符串")
            raw_path, start_s, end_s = str(item["path"]), item.get("start_s"), item.get("end_s")
        else:
            raise ValueError(f"data.records[{index}] 必须是路径字符串或 mapping")
        path = Path(raw_path)
        absolute = str((data_root / path).resolve()) if not path.is_absolute() else str(path)
        resolved.append({"path": absolute, "start_s": start_s, "end_s": end_s})
    return resolved


def load_config(path: str | Path, overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    path = Path(path).resolve()
    config = read_yaml(path)
    if overrides:
        config = _deep_merge(config, overrides)
    # 配置内路径相对于 configs/ 目录，因此不依赖当前 PowerShell 工作目录。
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
        raise ValueError(f"{key} 必须是有限 {shape[0]}x{shape[1]} 矩阵")
    value = .5 * (value + value.T)
    if np.min(np.linalg.eigvalsh(value)) < -1e-12:
        raise ValueError(f"{key} 必须半正定")


def _positive(section: dict[str, Any], *keys: str) -> None:
    for key in keys:
        if not np.isfinite(float(section[key])) or float(section[key]) <= 0:
            raise ValueError(f"{key} 必须为有限正数")


def validate_config(config: dict[str, Any]) -> dict[str, Any]:
    required = ("data", "model", "initialization", "filter", "benchmark", "evaluation", "debug", "output")
    for key in required:
        if not isinstance(config.get(key), dict):
            raise ValueError(f"配置缺少 section: {key}")
    data, model, init, fcfg = config["data"], config["model"], config["initialization"], config["filter"]
    if float(data["imu_hz"]) != 200.0 or float(data["network_hz"]) != 50.0 or int(data["window_samples"]) != 256:
        raise ValueError("当前 V20 契约固定 IMU=200 Hz、网络=50 Hz、窗口=256")
    _positive(init, "startup_s", "position_std_m", "velocity_std_mps", "accel_bias_std_mps2", "yaw_std_rad", "yaw_rate_correction_std_radps")
    for index, item in enumerate(data["records"]):
        start_s = item.get("start_s")
        end_s = item.get("end_s")
        if start_s is not None and (not np.isfinite(float(start_s)) or float(start_s) < float(init["startup_s"])):
            raise ValueError(f"data.records[{index}].start_s 必须不小于 initialization.startup_s")
        if end_s is not None and (not np.isfinite(float(end_s)) or float(end_s) <= 0):
            raise ValueError(f"data.records[{index}].end_s 必须为有限正数或 null")
        if start_s is not None and end_s is not None and float(end_s) <= float(start_s):
            raise ValueError(f"data.records[{index}].end_s 必须大于 start_s")
    _matrix(fcfg, "accel_sample_covariance", (3, 3)); _matrix(fcfg, "accel_bias_rw_psd", (3, 3))
    _positive(fcfg, "height_variance_m2", "depth_nis_gate_1d", "velocity_nis_gate_2d", "yaw_rate_nis_gate_1d",
              "yaw_rate_source_psd_rad2_per_s", "yaw_rate_correction_rw_psd_rad2_per_s3")
    yaw_source = fcfg.get("yaw_rate_source")
    if not isinstance(yaw_source, dict):
        raise ValueError("filter.yaw_rate_source 必须存在")
    yaw_source_type = str(yaw_source.get("type", "solved_yaw_backward_difference"))
    if yaw_source_type not in {"solved_yaw_backward_difference", "raw_gyro_z"}:
        raise ValueError("filter.yaw_rate_source.type 只能是 solved_yaw_backward_difference 或 raw_gyro_z")
    if yaw_source_type == "solved_yaw_backward_difference":
        _positive(yaw_source, "backward_window_s", "max_abs_rate_radps")
        median_samples = int(yaw_source.get("median_filter_samples", 0))
        if median_samples < 1 or median_samples % 2 != 1:
            raise ValueError("filter.yaw_rate_source.median_filter_samples 必须是正奇数")
    network = fcfg.get("network")
    if not isinstance(network, dict) or not isinstance(network.get("xy"), dict) or not isinstance(network.get("yaw_rate"), dict):
        raise ValueError("filter.network.xy 与 filter.network.yaw_rate 必须存在")
    _positive(network["xy"], "min_std", "max_std", "covariance_scale", "max_step_mps")
    _positive(network["yaw_rate"], "min_std", "max_std", "variance_scale")
    if float(network["xy"]["min_std"]) >= float(network["xy"]["max_std"]) or float(network["yaw_rate"]["min_std"]) >= float(network["yaw_rate"]["max_std"]):
        raise ValueError("网络 std 下限必须小于上限")
    if str(fcfg.get("yaw_mode", "estimated_network_r")) not in {"estimated_network_r", "direct_imu"}:
        raise ValueError("filter.yaw_mode 只能是 estimated_network_r 或 direct_imu")
    if str(config["evaluation"]["mode"]) not in {"benchmark_apriltag_init", "deployment_relative_yaw"}:
        raise ValueError("evaluation.mode 只能是 benchmark_apriltag_init 或 deployment_relative_yaw")
    _positive(config["benchmark"], "yaw_reanchor_std_rad")
    if str(model["device"]) not in {"cpu", "cuda", "auto"}:
        raise ValueError("model.device 必须为 cpu/cuda/auto")
    _positive(model, "batch_size")
    _positive(config["debug"], "max_imu_gap_s")
    return config


def config_hash(config: dict[str, Any]) -> str:
    canonical = json.dumps(config, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
