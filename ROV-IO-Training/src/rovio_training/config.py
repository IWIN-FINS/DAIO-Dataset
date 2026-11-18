"""Configuration validation. All tuning knobs live in YAML, never in code paths."""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

from .utils import canonical_json, read_yaml


REQUIRED_TOP_LEVEL = {"experiment", "data", "model", "training", "evaluation"}


def load_config(path: str | Path) -> dict[str, Any]:
    cfg = read_yaml(path)
    return validate_config(cfg)


def validate_config(cfg: dict[str, Any]) -> dict[str, Any]:
    """Validate an in-memory configuration, including ``train.py`` settings."""
    if not isinstance(cfg, dict): raise ValueError("configuration must be a mapping")
    missing = REQUIRED_TOP_LEVEL - set(cfg)
    if missing: raise ValueError(f"config missing sections: {sorted(missing)}")
    cfg = deepcopy(cfg)
    data, model, training = cfg["data"], cfg["model"], cfg["training"]
    source_hz = float(data.get("source_sample_hz", data["imu_hz"]))
    input_hz = float(data.get("network_input_hz", data["imu_hz"]))
    output_hz = float(data["network_hz"])
    if int(data["window_samples"]) != 256 or source_hz != 200.0 or output_hz != 50.0:
        raise ValueError("V20--V25 require 256 samples, a 200 Hz processed source, and 50 Hz output")
    ratio = source_hz / input_hz
    if input_hz <= 0 or ratio < 1 or abs(ratio - round(ratio)) > 1e-9:
        raise ValueError("network_input_hz must divide source_sample_hz exactly")
    if str(model["name"]) not in {"v20_four_branch_cnn_transformer", "v21_four_branch_cnn_transformer", "v22_four_branch_cnn_transformer", "v23_four_branch_cnn_transformer", "v24_four_branch_cnn_transformer", "v25_four_branch_cnn_transformer"}:
        raise ValueError("this project exposes V20--V25 four-branch CNN--Transformer models only")
    if int(model["d_model"]) % int(model["transformer_heads"]) != 0:
        raise ValueError("d_model must be divisible by transformer_heads")
    if int(training["phase1_epochs"]) < 1 or int(training["phase1_epochs"]) >= int(training["epochs"]):
        raise ValueError("phase1_epochs must be in [1, epochs-1]")
    if int(training["batch_size"]) < 1: raise ValueError("batch_size must be positive")
    if float(training["nll_weight"]) < 0: raise ValueError("nll_weight must be non-negative")
    if int(training.get("log_every_batches", 0)) < 0: raise ValueError("log_every_batches must be non-negative")
    if cfg["evaluation"].get("selection_metric") != "body_xy_rmse":
        raise ValueError("the V20 baseline selection_metric is fixed to body_xy_rmse")
    source_format = str(data.get("source_format", "legacy_hdf5"))
    if source_format not in {"processed_csv_200hz", "legacy_hdf5"}:
        raise ValueError("data.source_format must be processed_csv_200hz or legacy_hdf5")
    if source_format == "processed_csv_200hz":
        if float(data.get("label_savgol_window_s", 0.0)) <= 0:
            raise ValueError("processed_csv_200hz requires positive data.label_savgol_window_s")
    elif int(data.get("label_savgol_window", 0)) < 5:
        raise ValueError("legacy_hdf5 requires data.label_savgol_window >= 5")
    cfg["data"]["source_sample_hz"] = source_hz
    cfg["data"]["network_input_hz"] = input_hz
    return cfg


def config_hash(cfg: dict[str, Any]) -> str:
    import hashlib
    return hashlib.sha256(canonical_json(cfg).encode("utf-8")).hexdigest()


def v20_input_contract(cfg: dict[str, Any]) -> dict[str, Any]:
    data = cfg["data"]
    source_format = str(data.get("source_format", "legacy_hdf5"))
    depth_contract: dict[str, Any]
    if source_format == "processed_csv_200hz":
        depth_contract = {
            "source": "CSV pressure (already mean four-sensor physical depth [m])",
            "transform": "direct_depth_train_split_zscore_no_pressure_reconversion",
        }
    else:
        pressure = data["pressure"]
        depth_contract = {
            "source": "depth/depth4_raw + depth_calibration",
            "transform": "causal_physical_mean_depth_train_split_zscore",
            "atmosphere_pa": float(pressure["atmosphere_pa"]),
            "water_density_kg_m3": float(pressure["water_density_kg_m3"]),
            "gravity_mps2": float(pressure["gravity_mps2"]),
        }
    source_hz = float(data.get("source_sample_hz", data["imu_hz"]))
    input_hz = float(data.get("network_input_hz", data["imu_hz"]))
    downsample = int(round(source_hz / input_hz))
    mode = {200.0: "v20_200hz_1p28s", 100.0: "v21_100hz_2p56s", 50.0: "v22_50hz_5p12s", 40.0: "v23_40hz_6p4s", 200.0 / 6.0: "v24_33p33hz_7p68s", 25.0: "v25_25hz_10p24s"}.get(
        input_hz, f"custom_{input_hz:g}hz"
    )
    return {"mode": mode, "coordinate_frame": "FLU",
            "source_format": source_format,
            "rpm": {"channels": 8, "transform": "raw_rpm / 1000", "mean": 0.0, "std": 1000.0},
            "accel_xy": {"source": "raw acc_x, acc_y" if source_format == "processed_csv_200hz" else "imu/accel[:,0:2]",
                         "transform": "global_train_split_zscore_no_debias"},
            "gyro_z": {"source": "raw gyro_z" if source_format == "processed_csv_200hz" else "imu/gyro[:,2]",
                       "transform": "global_train_split_zscore_no_debias"},
            "depth": depth_contract,
            "window": {"samples": 256, "source_hz": source_hz, "input_hz": input_hz,
                       "duration_s_nominal": 256.0 / input_hz, "network_hz": float(data["network_hz"]),
                       "downsample": f"causal_{downsample}_sample_mean" if downsample > 1 else "none",
                       "downsample_factor": downsample},
            "target": "offline_raw_apriltag_body_flu_vx_vy_r", "online_apriltag_input": False}
