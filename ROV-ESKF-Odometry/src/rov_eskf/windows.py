"""冻结 checkpoint 驱动的因果窗口构建与 50 Hz 网络观测推理。

窗口契约由 checkpoint 的 ``config.data`` 决定，并与训练工程完全一致：
- 固定 256 个 token；原始 200 Hz 采样按 checkpoint 指定的因果块均值降采样；
- 输入分支 ``rpm [B,8,256]``、``accel_xy [B,2,256]``、``gyro_z [B,1,256]``、
  ``depth [B,1,256]``（channels × time，与模型一致）；
- 归一化使用 checkpoint 内保存的训练统计（键 ``rpm_* / accel_xy_* / gyro_z_* / depth_*``，
  RPM 固定 mean=0/std=1000，即除以 1000）；
- depth 特征 = CSV 中已换算的物理深度，或旧 HDF5 的四压力平均物理深度。
V24 ablations construct and normalize only config.model.modalities; a no_rpm
checkpoint needs no RPM normalizer and never supplies RPM to the model.

与训练唯一的差异：运行时不做 AprilTag 跨度过滤（网络是因果预测器，应全程可推理），
仅保留窗口首末深度均有限的有效窗口；评估自然只统计 AprilTag 覆盖时段。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from .data import Sequence, physical_mean_depth

_DEFAULT_PRESSURE = {"atmosphere_pa": 101325.0, "water_density_kg_m3": 1000.0, "gravity_mps2": 9.80665}


@dataclass(frozen=True)
class NetworkObservations:
    """50 Hz 网络观测流：时间、均值与标准差（机体系 FLU [vx, vy, r]）。"""
    ts: np.ndarray      # [K]
    mean: np.ndarray    # [K,3]
    std: np.ndarray     # [K,3]  exp(log_std)


def _window_view(values: np.ndarray, size: int) -> np.ndarray:
    """对 axis=0 做滑动窗口：``[N, C] -> [N-size+1, C, size]``（窗口维度放最后）。"""
    return np.lib.stride_tricks.sliding_window_view(values, size, axis=0)


class V20WindowBuilder:
    """一次构建整条轨迹的全部因果窗口（不按 AprilTag 过滤）。"""

    def __init__(self, seq: Sequence, checkpoint: dict[str, Any], pressure_cfg: dict[str, Any] | None = None,
                 window: int | None = None, stride: int | None = None) -> None:
        data_cfg = checkpoint.get("config", {}).get("data", {})
        self.input_keys = tuple(checkpoint.get("config", {}).get("model", {}).get(
            "modalities", ("rpm", "accel_xy", "gyro_z", "depth")))
        allowed = {"rpm", "throttle", "accel_xy", "accel_xyz", "gyro_z", "gyro_xyz", "depth"}
        if (not self.input_keys or len(set(self.input_keys)) != len(self.input_keys)
                or any(key not in allowed for key in self.input_keys)):
            raise ValueError("invalid checkpoint input modalities")
        source_hz = float(data_cfg.get("source_sample_hz", data_cfg.get("imu_hz", 200.0)))
        input_hz = float(data_cfg.get("network_input_hz", source_hz))
        output_hz = float(data_cfg.get("network_hz", 50.0))
        self.window = int(data_cfg.get("window_samples", 256) if window is None else window)
        self.downsample_factor = int(round(source_hz / input_hz))
        self.stride = int(round(source_hz / output_hz) if stride is None else stride)
        if self.window != 256 or not np.isclose(source_hz / input_hz, self.downsample_factor) or self.downsample_factor < 1:
            raise ValueError("checkpoint 输入契约要求 256 token 且 source_hz / input_hz 为正整数")
        if not np.isclose(source_hz / output_hz, self.stride) or self.stride < 1:
            raise ValueError("checkpoint 输入契约要求 source_hz / network_hz 为正整数")
        self.raw_window_samples = self.window * self.downsample_factor
        normalizer = checkpoint["normalizer"]
        self._check_normalizer(normalizer, self.input_keys)
        self.normalizer = {key: np.asarray(value) for key, value in normalizer.items()}
        pressure = dict(_DEFAULT_PRESSURE) if pressure_cfg is None else dict(pressure_cfg)
        depth = physical_mean_depth(seq, seq.imu_ts, float(pressure["atmosphere_pa"]),
                                    float(pressure["water_density_kg_m3"]), float(pressure["gravity_mps2"]))[:, None]
        # [N-raw_window+1, C, raw_window]；之后按训练时的因果块均值降采样。
        values = {"rpm": seq.rpm, "accel_xy": seq.accel[:, :2],
                  "gyro_z": seq.gyro[:, 2:3], "depth": depth}
        values.update(accel_xyz=seq.accel,gyro_xyz=seq.gyro)
        if 'gyro_xyz' in self.input_keys and not seq.full_gyro_available:
            raise ValueError('Full gyro model requires measured gyro_x/y/z')
        if 'throttle' in self.input_keys:
            if seq.throttle is None:raise ValueError('Command model requires measured throttle_1..8')
            values['throttle']=seq.throttle
        self.windows = {key: _window_view(values[key], self.raw_window_samples)
                        for key in self.input_keys}
        end_idx = np.arange(self.raw_window_samples - 1, len(seq.imu_ts), self.stride)
        # 窗口首末采样点的深度均需有限（首=end-raw_window+1，末=end）
        valid = np.isfinite(depth[end_idx, 0]) & np.isfinite(depth[end_idx - self.raw_window_samples + 1, 0])
        # Online quality gates may only inspect active inputs. A command-only
        # model must remain runnable without any RPM feedback, even for gating.
        if 'rpm' in self.input_keys and np.any(np.abs(seq.rpm)>=32700):
            saturated=np.flatnonzero(np.any(np.abs(seq.rpm)>=32700,axis=1))
            for index in saturated:
                valid &= ~((end_idx>=index)&(end_idx<index+self.raw_window_samples))
        self.ends = end_idx[valid]
        if len(self.ends) == 0:
            raise ValueError("轨迹没有有效的因果窗口（深度覆盖不足？）")
        self.ts = seq.imu_ts[self.ends]

    @staticmethod
    def _check_normalizer(normalizer: dict[str, np.ndarray],
                          input_keys: tuple[str, ...] = ("rpm", "accel_xy", "gyro_z", "depth")) -> None:
        shapes = {"rpm": (8,), "throttle": (8,), "accel_xy": (2,), "accel_xyz": (3,), "gyro_z": (1,), "gyro_xyz": (3,), "depth": (1,)}
        for key in input_keys:
            shape = shapes[key]
            for suffix in ("mean", "std"):
                value = np.asarray(normalizer[f"{key}_{suffix}"])
                if (value.shape != shape or not np.isfinite(value).all()
                        or (suffix == "std" and np.any(value <= 0))):
                    raise ValueError(f"normalizer {key}_{suffix} 形状错误或 std 非正")
        if "rpm" in input_keys and (not np.allclose(normalizer["rpm_mean"], 0.0)
                                    or not np.allclose(normalizer["rpm_std"], 1000.0)):
            raise ValueError("V20 RPM 归一化必须是 mean=0/std=1000")

    def feature_arrays(self) -> tuple[np.ndarray, ...]:
        """Return only checkpoint modalities, in model.input_keys order."""
        starts = self.ends - self.raw_window_samples + 1
        arrays = []
        for key in self.input_keys:
            raw = self.windows[key][starts]  # [K, C, raw_window]
            if self.downsample_factor > 1:
                raw = raw.reshape(len(raw), raw.shape[1], self.window, self.downsample_factor).mean(axis=3)
            mean = self.normalizer[f"{key}_mean"][None, :, None]
            std = self.normalizer[f"{key}_std"][None, :, None]
            arrays.append(np.asarray((raw - mean) / std, dtype=np.float32))
        return tuple(arrays)

    def window_times(self) -> np.ndarray:
        """窗口末端时间戳（50 Hz）。"""
        return self.ts


@torch.no_grad()
def network_observations(seq: Sequence, model: torch.nn.Module, checkpoint: dict[str, Any],
                         device: str | torch.device = "cpu", batch_size: int = 64,
                         pressure_cfg: dict[str, Any] | None = None) -> NetworkObservations:
    """对整条轨迹推理 V20 网络，返回 50 Hz 的 ``[vx, vy, r]`` 均值与标准差。"""
    device = torch.device(device)
    builder = V20WindowBuilder(seq, checkpoint, pressure_cfg=pressure_cfg)
    if tuple(model.input_keys) != builder.input_keys:
        raise ValueError("model inputs differ from checkpoint window modalities")
    arrays = builder.feature_arrays()
    times = builder.window_times()
    model.eval()
    means: list[np.ndarray] = []
    stds: list[np.ndarray] = []
    for begin in range(0, len(times), int(batch_size)):
        inputs = [
            torch.from_numpy(np.asarray(value[begin:begin + int(batch_size)], dtype=np.float32)).to(device)
            for value in arrays
        ]
        mean, log_std = model(*inputs)
        if not torch.isfinite(mean).all() or not torch.isfinite(log_std).all():
            raise RuntimeError("non-finite network prediction")
        means.append(mean.cpu().numpy())
        stds.append(np.exp(log_std.cpu().numpy()))
    return NetworkObservations(times, np.concatenate(means), np.concatenate(stds))
