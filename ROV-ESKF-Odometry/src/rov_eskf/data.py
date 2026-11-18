"""数据契约：运行时传感器流与离线 AprilTag 参考严格分离。

同时支持历史 HDF5 和最终 ``dataset/processed`` 的 200 Hz CSV。CSV 的 ``pressure``
已经是四压力计均值换算后的物理深度[m]，而 ``yaw`` 是 IMU 解算的多圈角度[rad]。
"""
from __future__ import annotations

import json
import csv
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import yaml
from scipy.signal import savgol_filter


@dataclass(frozen=True)
class Sequence:
    """单条记录的运行时传感器数据。

    AprilTag 字段可选：部署回放不要求 HDF5 含这些字段；只有离线
    标签、benchmark 初始化和指标计算会显式要求它们。
    """

    path: Path
    imu_ts: np.ndarray
    accel: np.ndarray
    gyro: np.ndarray
    rpm_ts: np.ndarray
    rpm: np.ndarray
    depth_ts: np.ndarray
    depth4: np.ndarray
    depth4_raw: np.ndarray
    attitude_ts: np.ndarray
    attitude_rpy: np.ndarray
    pressure_zero_pa: np.ndarray
    pressure_to_depth_m_per_pa: float
    tag_ts: np.ndarray | None = None
    tag_xy: np.ndarray | None = None
    tag_yaw: np.ndarray | None = None
    tag_yaw_rate: np.ndarray | None = None
    # 新 CSV 只有一个已换算的平均深度；保留为独立字段，避免伪造四压力阵列语义。
    mean_depth_m: np.ndarray | None = None
    source_format: str = "legacy_hdf5"
    throttle: np.ndarray | None = None
    full_gyro_available: bool = False


def _increasing(name: str, ts: np.ndarray, value: np.ndarray) -> None:
    if ts.ndim != 1 or len(ts) < 2 or len(ts) != len(value) or not np.all(np.diff(ts) > 0):
        raise ValueError(f"{name}: 时间戳必须严格递增，且长度应与数据一致")


def _optional_reference(handle: h5py.File) -> tuple[np.ndarray | None, ...]:
    required = ("apriltag/ts", "apriltag/xy", "apriltag/yaw", "apriltag/yaw_rate")
    present = [name in handle for name in required]
    if not any(present):
        return None, None, None, None
    if not all(present):
        missing = [name for name, ok in zip(required, present) if not ok]
        raise ValueError(f"{handle.filename}: AprilTag 字段不完整: {missing}")
    ts = np.asarray(handle["apriltag/ts"], dtype=np.float64)
    return (ts, np.asarray(handle["apriltag/xy"], dtype=np.float64),
            np.unwrap(np.asarray(handle["apriltag/yaw"], dtype=np.float64)),
            np.asarray(handle["apriltag/yaw_rate"], dtype=np.float64))


def load_sequence(path: str | Path, *, require_reference: bool = False) -> Sequence:
    """读取可部署运行的传感器数据；``require_reference`` 仅用于离线评估。"""
    path = Path(path)
    if path.suffix.lower() == ".csv":
        seq = _load_processed_csv(path)
        if require_reference:
            require_apriltag(seq)
        return seq
    with h5py.File(path, "r") as handle:
        required = ("imu/ts", "imu/accel", "imu/gyro", "rpm/ts", "rpm/rpm",
                    "depth/ts", "depth/depth4", "aux/native_ts", "aux/rpy_logged_rad")
        missing = [name for name in required if name not in handle]
        if missing:
            raise ValueError(f"{path}: 缺少运行时必需字段 {missing}")
        raw = handle.attrs.get("depth_calibration")
        if raw is None:
            raise ValueError(f"{path}: 缺少 depth_calibration 属性")
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8")
        calibration = json.loads(str(raw))
        depth4 = np.asarray(handle["depth/depth4"], dtype=np.float64)
        depth4_raw = (np.asarray(handle["depth/depth4_raw"], dtype=np.float64)
                      if "depth4_raw" in handle["depth"] else depth4.copy())
        tag_ts, tag_xy, tag_yaw, tag_yaw_rate = _optional_reference(handle)
        seq = Sequence(
            path=path,
            imu_ts=np.asarray(handle["imu/ts"], dtype=np.float64),
            accel=np.asarray(handle["imu/accel"], dtype=np.float64),
            gyro=np.asarray(handle["imu/gyro"], dtype=np.float64),
            rpm_ts=np.asarray(handle["rpm/ts"], dtype=np.float64),
            rpm=np.asarray(handle["rpm/rpm"], dtype=np.float64),
            depth_ts=np.asarray(handle["depth/ts"], dtype=np.float64),
            depth4=depth4,
            depth4_raw=depth4_raw,
            attitude_ts=np.asarray(handle["aux/native_ts"], dtype=np.float64),
            attitude_rpy=np.asarray(handle["aux/rpy_logged_rad"], dtype=np.float64),
            pressure_zero_pa=np.asarray(calibration["sensor_zero_pressure_pa"], dtype=np.float64),
            pressure_to_depth_m_per_pa=float(calibration["pressure_to_depth_m_per_pa"]),
            tag_ts=tag_ts, tag_xy=tag_xy, tag_yaw=tag_yaw, tag_yaw_rate=tag_yaw_rate,
        )
    _validate_sequence(seq)
    if require_reference:
        require_apriltag(seq)
    return seq


_CSV_REQUIRED = (
    "sample_time_s", "pressure", "acc_x", "acc_y", "acc_z", "gyro_z", "yaw",
    "int_rpm_1", "int_rpm_2", "int_rpm_3", "int_rpm_4", "int_rpm_5", "int_rpm_6",
    "int_rpm_7", "int_rpm_8", "tag_x_m", "tag_y_m", "tag_yaw_rad",
)


def _load_processed_csv(path: Path) -> Sequence:
    """读取最终 200 Hz CSV；只把 Tag 写入离线参考字段，运行时不使用它。"""
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"{path}: CSV 缺少表头")
        missing = sorted(set(_CSV_REQUIRED) - set(reader.fieldnames))
        if missing:
            raise ValueError(f"{path}: CSV 缺少字段 {missing}")
        rows = list(reader)
    if len(rows) < 2:
        raise ValueError(f"{path}: CSV 样本不足")

    def column(name: str) -> np.ndarray:
        try:
            values = np.asarray([float(row[name]) for row in rows], dtype=np.float64)
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"{path}: {name} 存在缺失或非数值值") from error
        if not np.all(np.isfinite(values)):
            raise ValueError(f"{path}: {name} 包含 NaN/Inf")
        return values

    ts = column("sample_time_s")
    accel = np.column_stack((column("acc_x"), column("acc_y"), column("acc_z")))
    gyro_z = column("gyro_z")
    full_gyro = all(c in reader.fieldnames for c in ('gyro_x','gyro_y'))
    gyro = np.column_stack(tuple(column(f'gyro_{a}') for a in 'xyz')) if full_gyro else np.column_stack((np.zeros_like(gyro_z), np.zeros_like(gyro_z), gyro_z))
    rpm = np.column_stack([column(f"int_rpm_{index}") for index in range(1, 9)])
    # 新数据的 yaw 已是多圈 rad；只 unwrap 防御可能的日志包装，不做单位缩放。
    # roll/pitch 依照近水平假设固定为零。
    yaw_rad = np.unwrap(column("yaw"))
    attitude = np.column_stack((np.zeros_like(yaw_rad), np.zeros_like(yaw_rad), yaw_rad))
    depth = column("pressure")
    # 深度与网络在同一 50 Hz 时刻更新：选择每个 20 ms 区间的末端 200 Hz 样本。
    depth_indices = np.arange(3, len(ts), 4, dtype=int)
    seq = Sequence(
        path=path, imu_ts=ts, accel=accel, gyro=gyro, rpm_ts=ts.copy(), rpm=rpm,
        depth_ts=ts[depth_indices], depth4=np.repeat(depth[depth_indices, None], 4, axis=1),
        depth4_raw=np.repeat(depth[depth_indices, None], 4, axis=1), attitude_ts=ts.copy(),
        attitude_rpy=attitude, pressure_zero_pa=np.zeros(4), pressure_to_depth_m_per_pa=1.0,
        tag_ts=ts.copy(), tag_xy=np.column_stack((column("tag_x_m"), column("tag_y_m"))),
        tag_yaw=np.unwrap(column("tag_yaw_rad")), tag_yaw_rate=np.zeros_like(ts),
        mean_depth_m=depth, source_format="processed_csv_200hz", full_gyro_available=full_gyro,
        throttle=np.column_stack(tuple(column(f'throttle_{i}') for i in range(1,9))) if all(f'throttle_{i}' in reader.fieldnames for i in range(1,9)) else None,
    )
    _validate_sequence(seq)
    return seq


def require_apriltag(seq: Sequence) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """仅供离线层调用，避免 AprilTag 误进入运行时融合。"""
    if any(value is None for value in (seq.tag_ts, seq.tag_xy, seq.tag_yaw, seq.tag_yaw_rate)):
        raise ValueError(f"{seq.path}: 此操作需要离线 AprilTag 参考")
    return seq.tag_ts, seq.tag_xy, seq.tag_yaw, seq.tag_yaw_rate  # type: ignore[return-value]


def _validate_sequence(seq: Sequence) -> None:
    _increasing(f"{seq.path}: imu", seq.imu_ts, seq.accel)
    _increasing(f"{seq.path}: rpm", seq.rpm_ts, seq.rpm)
    _increasing(f"{seq.path}: depth", seq.depth_ts, seq.depth4)
    _increasing(f"{seq.path}: attitude", seq.attitude_ts, seq.attitude_rpy)
    if seq.accel.shape != seq.gyro.shape or seq.accel.ndim != 2 or seq.accel.shape[1] != 3:
        raise ValueError(f"{seq.path}: imu/accel 与 imu/gyro 必须为 [N,3]")
    if seq.rpm.shape != (len(seq.imu_ts), 8) or not np.allclose(seq.rpm_ts, seq.imu_ts, atol=2e-4, rtol=0):
        raise ValueError(f"{seq.path}: RPM 必须为与 200 Hz IMU 对齐的 [N,8]")
    if seq.depth4.ndim != 2 or seq.depth4.shape[1] != 4 or seq.depth4_raw.shape != seq.depth4.shape:
        raise ValueError(f"{seq.path}: depth4/depth4_raw 必须为同形 [N,4]")
    if seq.attitude_rpy.ndim != 2 or seq.attitude_rpy.shape[1] != 3:
        raise ValueError(f"{seq.path}: aux/rpy_logged_rad 必须为 [N,3]")
    if not np.allclose(np.diff(seq.imu_ts), 1 / 200.0, atol=2e-4, rtol=0):
        raise ValueError(f"{seq.path}: 当前 V20/ESKF 契约要求规则 200 Hz IMU")
    if seq.pressure_zero_pa.shape != (4,) or seq.pressure_to_depth_m_per_pa <= 0:
        raise ValueError(f"{seq.path}: 压力标定元数据无效")
    if seq.mean_depth_m is not None and (seq.mean_depth_m.shape != seq.imu_ts.shape or
                                         not np.all(np.isfinite(seq.mean_depth_m))):
        raise ValueError(f"{seq.path}: mean_depth_m 必须是有限 [N]")
    runtime_arrays = (seq.accel, seq.gyro, seq.rpm, seq.depth4, seq.depth4_raw, seq.attitude_rpy)
    if not all(np.all(np.isfinite(value)) for value in runtime_arrays):
        raise ValueError(f"{seq.path}: 运行时传感器字段含 NaN/Inf")
    if seq.tag_ts is not None:
        assert seq.tag_xy is not None and seq.tag_yaw is not None and seq.tag_yaw_rate is not None
        _increasing(f"{seq.path}: apriltag", seq.tag_ts, seq.tag_xy)
        if seq.tag_xy.shape != (len(seq.tag_ts), 2) or seq.tag_yaw.shape != seq.tag_ts.shape \
                or seq.tag_yaw_rate.shape != seq.tag_ts.shape:
            raise ValueError(f"{seq.path}: AprilTag 形状无效")


def resolve_manifest(manifest_path: str | Path, data_root: str | Path) -> dict[str, list[Path]]:
    manifest = yaml.safe_load(Path(manifest_path).read_text(encoding="utf-8"))
    if manifest.get("offline_apriltag_only") is not True:
        raise ValueError("manifest 必须声明 offline_apriltag_only: true")
    root = Path(data_root) / str(manifest.get("processed_root", "."))
    output: dict[str, list[Path]] = {}
    for split in ("train", "val", "demo"):
        entry = manifest.get("splits", {}).get(split)
        if not isinstance(entry, dict):
            raise ValueError(f"manifest 缺少 split={split}")
        paths = sorted(root.glob(str(entry["glob"])))
        if len(paths) != int(entry["expected_count"]):
            raise ValueError(f"{split}: 期望 {entry['expected_count']} 条，实际 {len(paths)} 条")
        output[split] = paths
    flat = [path.resolve() for records in output.values() for path in records]
    if len(flat) != len(set(flat)):
        raise ValueError("manifest split 泄漏：同一记录出现在多个 split")
    return output


def physical_mean_depth(seq: Sequence, query_ts: np.ndarray, atmosphere_pa: float,
                        water_density_kg_m3: float, gravity_mps2: float) -> np.ndarray:
    """网络深度特征：CSV 直用物理深度，旧 HDF5 保持原始恢复规则。"""
    if seq.mean_depth_m is not None:
        index = np.searchsorted(seq.imu_ts, np.asarray(query_ts, dtype=np.float64), side="right") - 1
        output = np.full(len(index), np.nan, dtype=np.float64)
        valid = index >= 0
        output[valid] = seq.mean_depth_m[index[valid]]
        return output
    ix = np.searchsorted(seq.depth_ts, np.asarray(query_ts, dtype=np.float64), side="right") - 1
    output = np.full(len(ix), np.nan, dtype=np.float64)
    valid = ix >= 0
    absolute = seq.depth4_raw[ix[valid]] / seq.pressure_to_depth_m_per_pa + seq.pressure_zero_pa
    output[valid] = (absolute.mean(axis=1) - atmosphere_pa) / (water_density_kg_m3 * gravity_mps2)
    return output


def height_from_depth4(depth4_row: np.ndarray) -> float:
    """滤波观测：校准相对深度平均值转换为 FLU 向上高度。"""
    return -float(np.mean(np.asarray(depth4_row, dtype=float)))


def height_measurement(seq: Sequence, index: int) -> float:
    """返回 FLU 向上高度；CSV 的单路深度与历史四阵列分别处理。"""
    if seq.mean_depth_m is not None:
        imu_index = int(np.searchsorted(seq.imu_ts, seq.depth_ts[index], side="right") - 1)
        return -float(seq.mean_depth_m[imu_index])
    return height_from_depth4(seq.depth4[index])


def offline_labels(seq: Sequence, query_ts: np.ndarray, window: int, polyorder: int) -> np.ndarray:
    """离线训练标签 [body vx, body vy, Euler yaw-rate]。"""
    tag_ts, tag_xy, tag_yaw, _ = require_apriltag(seq)
    count = len(tag_ts)
    window = min(int(window), count if count % 2 else count - 1)
    if window < 5 or window <= polyorder:
        raise ValueError(f"{seq.path}: AprilTag 样本不足以构造平滑标签")
    xy = savgol_filter(tag_xy, window, polyorder, axis=0, mode="interp")
    yaw = savgol_filter(tag_yaw, window, polyorder, mode="interp")
    world_velocity = np.gradient(xy, tag_ts, axis=0, edge_order=1)
    yaw_rate = np.gradient(yaw, tag_ts, edge_order=1)
    t = np.asarray(query_ts, dtype=np.float64)
    vx = np.interp(t, tag_ts, world_velocity[:, 0])
    vy = np.interp(t, tag_ts, world_velocity[:, 1])
    psi = np.interp(t, tag_ts, yaw)
    return np.column_stack((np.cos(psi) * vx + np.sin(psi) * vy,
                            -np.sin(psi) * vx + np.cos(psi) * vy,
                            np.interp(t, tag_ts, yaw_rate)))
