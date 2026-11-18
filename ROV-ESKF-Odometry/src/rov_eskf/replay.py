"""异步回放：200 Hz IMU 传播、原始深度事件与 50 Hz 网络独立更新。"""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.spatial.transform import Rotation

from .data import Sequence, height_measurement
from .eskf import AttitudeAidedESKF11, DirectYawEKF9, Update, rotation_matrix_from_tilt_yaw
from .windows import NetworkObservations, network_observations
from .yaw_source import CausalYawRateSource, causal_tilt, causal_yaw_rate_source, raw_gyro_z_source


@dataclass(frozen=True)
class Replay11Output:
    ts: np.ndarray
    state: np.ndarray                 # [p,v,q,ba,yaw,yaw_rate_correction]
    covariance_diag: np.ndarray
    update_ts: np.ndarray
    update_kind: np.ndarray
    update_accepted: np.ndarray
    update_nis: np.ndarray
    update_residual: np.ndarray
    event_meta: list[dict[str, Any]]
    startup_source: str
    yaw_source_age_s: np.ndarray
    reanchor_time: float | None = None
    reanchor_index: int | None = None


def _event_intervals(event_ts: np.ndarray, imu_ts: np.ndarray) -> dict[int, list[int]]:
    """将事件放入唯一 IMU 区间 ``(t_i, t_{i+1}]``。"""
    indices = np.clip(np.searchsorted(imu_ts, event_ts, side="left") - 1, 0, len(imu_ts) - 2)
    groups: dict[int, list[int]] = defaultdict(list)
    for event_index, interval_index in enumerate(indices):
        groups[int(interval_index)].append(event_index)
    return groups


def _startup(sequence: Sequence, config: dict, tilt_all: np.ndarray) -> tuple[np.ndarray, np.ndarray, float, str]:
    """按用户协议，以完整 0--startup_s 窗口中位数完成初始化。"""
    startup_s = float(config["initialization"]["startup_s"])
    stop = float(sequence.imu_ts[0] + startup_s)
    depth_mask = sequence.depth_ts <= stop
    if sequence.mean_depth_m is None:
        depth = sequence.depth4[depth_mask]
        height = float(np.median(-np.mean(depth, axis=1)))
    else:
        source_index = np.searchsorted(sequence.imu_ts, sequence.depth_ts[depth_mask], side="right") - 1
        depth = sequence.mean_depth_m[source_index]
        height = float(np.median(-depth))
    mask = sequence.imu_ts <= stop
    if len(depth) == 0 or np.count_nonzero(mask) < 20:
        raise ValueError("初始化窗口内需要足够的深度与 IMU 样本")
    g = np.asarray(config["filter"]["gravity_flu"], dtype=float)
    rotations = np.stack([rotation_matrix_from_tilt_yaw(roll, pitch, 0.0)
                          for roll, pitch in tilt_all[mask]])
    expected_specific_force = -np.einsum("nij,j->ni", rotations.transpose(0, 2, 1), g)
    accel_bias = np.median(sequence.accel[mask] - expected_specific_force, axis=0)
    correction_std = float(config["initialization"]["yaw_rate_correction_std_radps"])
    return np.array([0.0, 0.0, height]), accel_bias, correction_std, "median_first_startup_window"


def _initial_covariance(config: dict, correction_std: float) -> np.ndarray:
    cfg = config["initialization"]
    return np.diag(np.r_[np.full(3, float(cfg["position_std_m"]) ** 2),
                         np.full(3, float(cfg["velocity_std_mps"]) ** 2),
                         np.full(3, float(cfg["accel_bias_std_mps2"]) ** 2),
                         float(cfg["yaw_std_rad"]) ** 2, correction_std ** 2])


def _pack(filter_: AttitudeAidedESKF11, roll: float, pitch: float) -> np.ndarray:
    quat = Rotation.from_matrix(rotation_matrix_from_tilt_yaw(roll, pitch, filter_.yaw)).as_quat()
    return np.r_[filter_.p, filter_.v, quat, filter_.ba, filter_.yaw, filter_.yaw_rate_correction]


def _network_covariances(network_std: np.ndarray, config: dict) -> tuple[np.ndarray, float, dict[str, Any]]:
    net_cfg = config["filter"]["network"]
    xy_cfg, r_cfg = net_cfg["xy"], net_cfg["yaw_rate"]
    raw = np.asarray(network_std, dtype=float).reshape(3)
    xy_std = np.clip(raw[:2], float(xy_cfg["min_std"]), float(xy_cfg["max_std"]))
    r_std = float(np.clip(raw[2], float(r_cfg["min_std"]), float(r_cfg["max_std"])))
    Rxy = float(xy_cfg["covariance_scale"]) * np.diag(xy_std * xy_std)
    Rr = float(r_cfg["variance_scale"]) * r_std * r_std
    meta = {"raw_std_vx": float(raw[0]), "raw_std_vy": float(raw[1]), "raw_std_yaw_rate": float(raw[2]),
            "effective_std_vx": float(np.sqrt(Rxy[0, 0])), "effective_std_vy": float(np.sqrt(Rxy[1, 1])),
            "effective_std_yaw_rate": float(np.sqrt(Rr)), "xy_covariance_scale": float(xy_cfg["covariance_scale"]),
            "yaw_rate_variance_scale": float(r_cfg["variance_scale"])}
    return Rxy, Rr, meta


def replay_attitude_aided_11(sequence: Sequence, config: dict, model: torch.nn.Module | None = None,
                             checkpoint: dict[str, Any] | None = None, device: str | torch.device = "cpu",
                             reanchor_yaw: float | None = None,
                             network: NetworkObservations | None = None,
                             reanchor_time: float | None = None,
                             end_time: float | None = None) -> Replay11Output:
    """运行时不读取 AprilTag；``reanchor_yaw`` 只能由 benchmark 评估层传入。"""
    if (model is None) != (checkpoint is None):
        raise ValueError("model 与 checkpoint 必须同时提供或同时省略")
    imu_ts = sequence.imu_ts
    if end_time is None:
        last_imu_index = len(imu_ts) - 1
    else:
        last_imu_index = int(np.searchsorted(imu_ts, float(end_time), side="right") - 1)
        if last_imu_index < 1:
            raise ValueError("回放结束时间早于第二个 IMU 样本")
    max_gap = float(config["debug"]["max_imu_gap_s"])
    if np.any(np.diff(imu_ts) > max_gap):
        raise ValueError(f"{sequence.path}: IMU 间隔超过 {max_gap:g} s")
    tilt_all = causal_tilt(sequence, imu_ts)
    position, accel_bias, correction_std, startup_source = _startup(sequence, config, tilt_all)
    filter_ = AttitudeAidedESKF11(config["filter"], float(np.median(np.diff(imu_ts))))
    filter_.initialize(position, accel_bias, _initial_covariance(config, correction_std), yaw=0.0)
    yaw_source_cfg = config["filter"].get("yaw_rate_source", {})
    yaw_source_type = str(yaw_source_cfg.get("type", "solved_yaw_backward_difference"))
    if yaw_source_type == "raw_gyro_z":
        # 此分支不读取 IMU 解算 yaw，也不对其差分；仅以原始 gyro_z 传播 yaw。
        yaw_source = raw_gyro_z_source(sequence)
        startup_source += "; yaw_rate_source=raw_gyro_z"
    elif yaw_source_type == "solved_yaw_backward_difference":
        yaw_source = causal_yaw_rate_source(sequence, yaw_source_cfg)
        startup_source += "; yaw_rate_source=solved_yaw_backward_difference"
    else:
        raise ValueError(f"不支持的 yaw-rate 来源: {yaw_source_type}")

    if network is None and model is not None:
        network = network_observations(sequence, model, checkpoint, device,
                                       batch_size=int(config["model"]["batch_size"]),
                                       pressure_cfg=config["data"]["pressure"])
    network_groups: dict[int, int] = {}
    if network is not None:
        for index, time in enumerate(network.ts):
            imu_index = int(np.searchsorted(imu_ts, time, side="left"))
            if imu_index >= len(imu_ts) or abs(float(imu_ts[imu_index]) - float(time)) > 1e-7:
                raise ValueError("网络时间戳必须对齐到 IMU 时间轴")
            network_groups[imu_index] = index
    depth_groups = _event_intervals(sequence.depth_ts, imu_ts)
    attitude_groups = _event_intervals(sequence.attitude_ts[1:], imu_ts)

    startup_time = float(imu_ts[0] + config["initialization"]["startup_s"])
    if reanchor_yaw is None:
        reanchor_index = None
    else:
        requested_reanchor = startup_time if reanchor_time is None else float(reanchor_time)
        reanchor_index = int(np.searchsorted(imu_ts, requested_reanchor, side="left"))
        if reanchor_index < 1 or reanchor_index > last_imu_index:
            raise ValueError("yaw 重锚时间必须位于本次回放的 IMU 时间范围内")
    state = np.empty((last_imu_index + 1, 15)); covariance = np.empty((last_imu_index + 1, 11))
    state[0] = _pack(filter_, *tilt_all[0]); covariance[0] = np.diag(filter_.P)
    update_ts: list[float] = []; update_kind: list[str] = []; update_ok: list[bool] = []
    update_nis: list[float] = []; update_residual: list[np.ndarray] = []; event_meta: list[dict[str, Any]] = []
    last_accepted_xy: np.ndarray | None = None

    def emit(time: float, kind: str, result: Update, **meta: Any) -> None:
        update_ts.append(float(time)); update_kind.append(kind); update_ok.append(result.accepted)
        update_nis.append(float(result.nis)); update_residual.append(np.asarray(result.residual, dtype=float))
        event_meta.append({"time": float(time), "kind": kind, "accepted": bool(result.accepted),
                           "nis": float(result.nis), **meta})

    for interval_index in range(last_imu_index):
        left, right = float(imu_ts[interval_index]), float(imu_ts[interval_index + 1])
        interval, cursor = right - left, left

        def propagate(target: float) -> None:
            nonlocal cursor
            if target <= cursor + 1e-12:
                return
            a0, a1 = (cursor - left) / interval, (target - left) / interval
            acc0 = (1 - a0) * sequence.accel[interval_index] + a0 * sequence.accel[interval_index + 1]
            acc1 = (1 - a1) * sequence.accel[interval_index] + a1 * sequence.accel[interval_index + 1]
            roll_mid, pitch_mid = causal_tilt(sequence, np.array([(cursor + target) * .5]))[0]
            filter_.predict_interval(acc0, acc1, yaw_source.rate_at(cursor), roll_mid, pitch_mid, target - cursor)
            cursor = target

        events: list[tuple[float, int, str, int]] = []
        events.extend((float(sequence.attitude_ts[index + 1]), -1, "attitude", index)
                      for index in attitude_groups[interval_index])
        events.extend((float(sequence.depth_ts[index]), 0, "height", index)
                      for index in depth_groups[interval_index])
        if interval_index + 1 in network_groups:
            events.append((right, 1, "network", network_groups[interval_index + 1]))
        for event_time, _, kind, index in sorted(events):
            propagate(event_time)
            if kind == "attitude":
                continue  # 下一段 propagate 会通过 yaw_source.rate_at 使用新后向差分率。
            if kind == "height":
                result = filter_.update_height(height_measurement(sequence, index),
                                                float(config["filter"]["height_variance_m2"]),
                                                float(config["filter"]["depth_nis_gate_1d"]))
                emit(event_time, "height", result, measurement_variance=float(config["filter"]["height_variance_m2"]))
                continue
            assert network is not None
            roll, pitch = causal_tilt(sequence, np.array([event_time]))[0]
            filter_.set_external_tilt(roll, pitch)
            Rxy, Rr, meta = _network_covariances(network.std[index], config)
            step_limit = float(config["filter"]["network"]["xy"]["max_step_mps"])
            temporal_ok = last_accepted_xy is None or np.linalg.norm(network.mean[index, :2] - last_accepted_xy) <= step_limit
            if temporal_ok:
                xy_result = filter_.update_body_velocity(network.mean[index, :2], Rxy,
                                                          float(config["filter"]["velocity_nis_gate_2d"]))
                if xy_result.accepted:
                    last_accepted_xy = network.mean[index, :2].copy()
            else:
                xy_result = Update(False, float("nan"), np.zeros(2))
            emit(event_time, "network_xy", xy_result, temporal_gate=bool(temporal_ok),
                 r00=float(Rxy[0, 0]), r11=float(Rxy[1, 1]), **meta)
            rate_result = filter_.update_yaw_rate(float(network.mean[index, 2]), yaw_source.rate_at(event_time), Rr,
                                                   float(config["filter"]["yaw_rate_nis_gate_1d"]))
            emit(event_time, "network_yaw_rate", rate_result, imu_yaw_rate=float(yaw_source.rate_at(event_time)),
                 measurement_variance=float(Rr), **meta)
        propagate(right)
        if reanchor_index is not None and interval_index + 1 == reanchor_index:
            filter_.reanchor_yaw(float(reanchor_yaw), float(config["benchmark"]["yaw_reanchor_std_rad"]))
            startup_source += "; benchmark_apriltag_yaw_reanchor"
        state[interval_index + 1] = _pack(filter_, *tilt_all[interval_index + 1])
        covariance[interval_index + 1] = np.diag(filter_.P)

    width = max((len(value) for value in update_residual), default=3)
    residual = np.full((len(update_residual), width), np.nan)
    for index, value in enumerate(update_residual):
        residual[index, :len(value)] = value
    output_ts = imu_ts[:last_imu_index + 1]
    ages = np.array([yaw_source.age_at(float(t)) for t in output_ts])
    actual_reanchor_time = None if reanchor_index is None else float(output_ts[reanchor_index])
    return Replay11Output(output_ts, state, covariance, np.asarray(update_ts), np.asarray(update_kind, dtype="U24"),
                          np.asarray(update_ok, dtype=bool), np.asarray(update_nis), residual, event_meta,
                          startup_source, ages, actual_reanchor_time, reanchor_index)


def _direct_yaw_at(sequence: Sequence, time: float, offset: float) -> float:
    """严格因果零阶保持的 IMU 解算 yaw；offset 仅表示离线 benchmark 世界系重锚。"""
    index = int(np.searchsorted(sequence.attitude_ts, float(time), side="right") - 1)
    if index < 0:
        raise ValueError("当前时刻之前没有可用 IMU yaw")
    return float(sequence.attitude_rpy[index, 2] + offset)


def _pack_direct(filter_: DirectYawEKF9, yaw: float) -> np.ndarray:
    quat = Rotation.from_matrix(rotation_matrix_from_tilt_yaw(0.0, 0.0, yaw)).as_quat()
    return np.r_[filter_.p, filter_.v, quat, filter_.ba, float(yaw), 0.0]


def replay_direct_imu_yaw_9(sequence: Sequence, config: dict, model: torch.nn.Module | None = None,
                            checkpoint: dict[str, Any] | None = None, device: str | torch.device = "cpu",
                            reanchor_yaw: float | None = None, network: NetworkObservations | None = None,
                            reanchor_time: float | None = None, end_time: float | None = None) -> Replay11Output:
    """9 状态直接 yaw 回放：只融合平均深度和网络 XY 速度，不估计/更新 yaw。"""
    if (model is None) != (checkpoint is None):
        raise ValueError("model 与 checkpoint 必须同时提供或同时省略")
    imu_ts = sequence.imu_ts
    last = len(imu_ts) - 1 if end_time is None else int(np.searchsorted(imu_ts, float(end_time), side="right") - 1)
    if last < 1: raise ValueError("回放结束时间早于第二个 IMU 样本")
    if np.any(np.diff(imu_ts) > float(config["debug"]["max_imu_gap_s"])):
        raise ValueError(f"{sequence.path}: IMU 间隔超过允许上限")
    tilt = causal_tilt(sequence, imu_ts); position, accel_bias, _, startup_source = _startup(sequence, config, tilt)
    filter_ = DirectYawEKF9(config["filter"], float(np.median(np.diff(imu_ts))))
    filter_.initialize(position, accel_bias, _initial_covariance(config, 0.0)[:9, :9])
    if network is None and model is not None:
        network = network_observations(sequence, model, checkpoint, device, batch_size=int(config["model"]["batch_size"]),
                                       pressure_cfg=config["data"]["pressure"])
    network_groups = {int(np.searchsorted(imu_ts, t, side="left")): i for i, t in enumerate(network.ts)} if network is not None else {}
    depth_groups = _event_intervals(sequence.depth_ts, imu_ts)
    startup_time = float(imu_ts[0] + config["initialization"]["startup_s"])
    target_reanchor = startup_time if reanchor_time is None else float(reanchor_time)
    reanchor_index = None if reanchor_yaw is None else int(np.searchsorted(imu_ts, target_reanchor, side="left"))
    yaw_offset = 0.0
    state = np.empty((last + 1, 15)); covariance = np.empty((last + 1, 11))
    state[0] = _pack_direct(filter_, _direct_yaw_at(sequence, imu_ts[0], yaw_offset)); covariance[0] = np.r_[np.diag(filter_.P), 0., 0.]
    update_ts: list[float] = []; update_kind: list[str] = []; update_ok: list[bool] = []; update_nis: list[float] = []
    update_residual: list[np.ndarray] = []; event_meta: list[dict[str, Any]] = []; last_accepted_xy: np.ndarray | None = None

    def emit(time: float, kind: str, result: Update, **meta: Any) -> None:
        update_ts.append(float(time)); update_kind.append(kind); update_ok.append(result.accepted); update_nis.append(float(result.nis))
        update_residual.append(np.asarray(result.residual, dtype=float)); event_meta.append({"time": float(time), "kind": kind, "accepted": bool(result.accepted), "nis": float(result.nis), **meta})

    for interval_index in range(last):
        left, right, cursor = float(imu_ts[interval_index]), float(imu_ts[interval_index + 1]), float(imu_ts[interval_index])
        interval = right - left
        def propagate(target: float) -> None:
            nonlocal cursor
            if target <= cursor + 1e-12: return
            a0, a1 = (cursor - left) / interval, (target - left) / interval
            acc0 = (1 - a0) * sequence.accel[interval_index] + a0 * sequence.accel[interval_index + 1]
            acc1 = (1 - a1) * sequence.accel[interval_index] + a1 * sequence.accel[interval_index + 1]
            filter_.predict_interval(acc0, acc1, _direct_yaw_at(sequence, .5 * (cursor + target), yaw_offset), target - cursor)
            cursor = target
        events = [(float(sequence.depth_ts[index]), 0, "height", index) for index in depth_groups[interval_index]]
        if interval_index + 1 in network_groups: events.append((right, 1, "network", network_groups[interval_index + 1]))
        for event_time, _, kind, index in sorted(events):
            propagate(event_time)
            if kind == "height":
                result = filter_.update_height(height_measurement(sequence, index), float(config["filter"]["height_variance_m2"]), float(config["filter"]["depth_nis_gate_1d"]))
                emit(event_time, "height", result, measurement_variance=float(config["filter"]["height_variance_m2"])); continue
            assert network is not None
            Rxy, _, meta = _network_covariances(network.std[index], config)
            temporal_ok = last_accepted_xy is None or np.linalg.norm(network.mean[index, :2] - last_accepted_xy) <= float(config["filter"]["network"]["xy"]["max_step_mps"])
            yaw = _direct_yaw_at(sequence, event_time, yaw_offset)
            xy_result = filter_.update_body_velocity(network.mean[index, :2], yaw, Rxy, float(config["filter"]["velocity_nis_gate_2d"])) if temporal_ok else Update(False, float("nan"), np.zeros(2))
            if xy_result.accepted: last_accepted_xy = network.mean[index, :2].copy()
            emit(event_time, "network_xy", xy_result, temporal_gate=bool(temporal_ok), r00=float(Rxy[0, 0]), r11=float(Rxy[1, 1]), **meta)
        propagate(right)
        if reanchor_index is not None and interval_index + 1 == reanchor_index:
            yaw_before = _direct_yaw_at(sequence, right, yaw_offset); yaw_offset += float(reanchor_yaw) - yaw_before
            filter_.reanchor(float(reanchor_yaw) - yaw_before)
            startup_source += "; benchmark_apriltag_yaw_reanchor; direct_imu_yaw"
        yaw = _direct_yaw_at(sequence, right, yaw_offset)
        state[interval_index + 1] = _pack_direct(filter_, yaw); covariance[interval_index + 1] = np.r_[np.diag(filter_.P), 0., 0.]
    width = max((len(value) for value in update_residual), default=2); residual = np.full((len(update_residual), width), np.nan)
    for index, value in enumerate(update_residual): residual[index, :len(value)] = value
    output_ts = imu_ts[:last + 1]
    return Replay11Output(output_ts, state, covariance, np.asarray(update_ts), np.asarray(update_kind, dtype="U24"), np.asarray(update_ok, dtype=bool),
                           np.asarray(update_nis), residual, event_meta, startup_source or "direct_imu_yaw", np.zeros(len(output_ts)),
                           None if reanchor_index is None else float(imu_ts[reanchor_index]), reanchor_index)


def save_replay11(output: Replay11Output, directory: str | Path, *, save_covariance: bool = True,
                  save_event_trace: bool = True, suffix: str = "") -> None:
    """保存可调试状态、协方差、完整观测日志和 yaw 源健康度。"""
    directory = Path(directory); directory.mkdir(parents=True, exist_ok=True)
    header = "time,p_x,p_y,p_z,v_x,v_y,v_z,q_x,q_y,q_z,q_w,b_a_x,b_a_y,b_a_z,yaw,yaw_rate_correction"
    np.savetxt(directory / f"state{suffix}.csv", np.c_[output.ts, output.state], delimiter=",", header=header, comments="")
    if save_covariance:
        np.savez_compressed(directory / f"state{suffix}.npz", ts=output.ts, state=output.state,
                            covariance_diag=output.covariance_diag, yaw_source_age_s=output.yaw_source_age_s,
                            update_ts=output.update_ts, update_kind=output.update_kind,
                            update_accepted=output.update_accepted, update_nis=output.update_nis,
                            update_residual=output.update_residual)
    if save_event_trace:
        fields = sorted({key for row in output.event_meta for key in row})
        with (directory / f"updates{suffix}.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(output.event_meta)
    diagnostics = {"startup_source": output.startup_source, "yaw_source_age_median_s": float(np.median(output.yaw_source_age_s[np.isfinite(output.yaw_source_age_s)])),
                   "yaw_source_age_max_s": float(np.max(output.yaw_source_age_s[np.isfinite(output.yaw_source_age_s)])),
                   "reanchor_time": output.reanchor_time}
    (directory / f"diagnostics{suffix}.json").write_text(json.dumps(diagnostics, indent=2), encoding="utf-8")
