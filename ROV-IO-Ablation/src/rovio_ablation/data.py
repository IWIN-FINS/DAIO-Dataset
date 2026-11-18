"""消融实验统一数据契约、特征构造和划分解析。

默认路径是 ``dataset/processed`` 下的 200 Hz CSV segment。旧 HDF5 路径仍然
可以读取，以便历史 checkpoint 排查；但它不再是默认训练来源。AprilTag 仅在
``offline_labels`` 中用于离线标签，任何网络特征函数都不接收 AprilTag 数据。
"""
from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable

import h5py
import numpy as np
import torch
from scipy.signal import savgol_filter
from torch.utils.data import Dataset

from .utils import read_yaml


CSV_REQUIRED_COLUMNS = (
    "sample_time_s", "pressure", "acc_x", "acc_y", "acc_z", "gyro_z",
    "int_rpm_1", "int_rpm_2", "int_rpm_3", "int_rpm_4", "int_rpm_5",
    "int_rpm_6", "int_rpm_7", "int_rpm_8", "tag_x_m", "tag_y_m", "tag_yaw_rad",
)


@dataclass(frozen=True)
class Sequence:
    """单个连续数据段的统一内存表示。"""

    path: Path
    source_format: str
    imu_ts: np.ndarray
    accel: np.ndarray
    gyro: np.ndarray
    rpm_ts: np.ndarray
    rpm: np.ndarray
    tag_ts: np.ndarray
    tag_xy: np.ndarray
    tag_yaw: np.ndarray
    # CSV 的 depth_m 已是物理深度。旧 HDF5 则保留原压力信息，供历史兼容用。
    depth_m: np.ndarray | None = None
    depth_ts: np.ndarray | None = None
    depth4_raw: np.ndarray | None = None
    pressure_zero_pa: np.ndarray | None = None
    pressure_to_depth_m_per_pa: float | None = None
    network_window_exclusion_end_intervals_s: np.ndarray | None = None
    # 从 processed/manifest.csv 的原始 RPM 饱和值行换算而来；各实验据此按实际
    # 输入窗动态扩展禁用区间，不能复用旧 1.28 s 的预计算区间。
    rpm_saturation_times_s: np.ndarray | None = None
    throttle: np.ndarray | None = None
    full_gyro_available: bool = False


def _validate_time(name: str, ts: np.ndarray, values: np.ndarray) -> None:
    if (ts.ndim != 1 or len(ts) < 2 or len(ts) != len(values) or
            not np.all(np.isfinite(ts)) or not np.all(np.diff(ts) > 0)):
        raise ValueError(f"{name}: timestamps must be finite, strictly increasing, and match values")


def _find_processed_root(path: Path) -> Path:
    """由 ``processed/v1/foo.csv`` 定位到 ``processed`` 根目录。"""
    for candidate in (path.parent, *path.parents):
        if candidate.name == "processed" and (candidate / "manifest.csv").is_file():
            return candidate
    raise ValueError(f"{path}: CSV must be located below a processed directory containing manifest.csv")


@dataclass(frozen=True)
class _ProcessedCsvMetadata:
    source_row_start: int | None
    rpm_saturation_source_rows: np.ndarray
    legacy_exclusion_end_intervals_s: np.ndarray


@lru_cache(maxsize=8)
def _load_processed_manifest(manifest_path: str) -> dict[str, _ProcessedCsvMetadata]:
    """读取处理工具写出的 RPM 异常元数据，并缓存避免重复扫描。"""
    output: dict[str, _ProcessedCsvMetadata] = {}
    with Path(manifest_path).open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            relative = row.get("processed_relative_path", "")
            raw = row.get("network_window_exclusion_end_intervals_s", "[]") or "[]"
            try:
                intervals = np.asarray(json.loads(raw), dtype=np.float64)
            except json.JSONDecodeError as error:
                raise ValueError(f"{manifest_path}: invalid exclusion intervals for {relative}") from error
            if intervals.size == 0:
                intervals = np.empty((0, 2), dtype=np.float64)
            if intervals.ndim != 2 or intervals.shape[1] != 2 or not np.all(np.isfinite(intervals)):
                raise ValueError(f"{manifest_path}: invalid exclusion interval shape for {relative}")
            if np.any(intervals[:, 1] < intervals[:, 0]):
                raise ValueError(f"{manifest_path}: interval end precedes start for {relative}")
            raw_saturation_rows = row.get("rpm_saturation_source_rows", "[]") or "[]"
            try:
                saturation_rows = np.asarray(json.loads(raw_saturation_rows), dtype=np.int64)
            except json.JSONDecodeError as error:
                raise ValueError(f"{manifest_path}: invalid RPM saturation rows for {relative}") from error
            if saturation_rows.ndim != 1:
                raise ValueError(f"{manifest_path}: RPM saturation rows must be a flat list for {relative}")
            source_row_start_raw = row.get("source_row_start", "")
            source_row_start = int(source_row_start_raw) if source_row_start_raw not in {None, ""} else None
            if len(saturation_rows) and source_row_start is None:
                raise ValueError(f"{manifest_path}: saturation rows require source_row_start for {relative}")
            output[relative.replace("\\", "/")] = _ProcessedCsvMetadata(
                source_row_start=source_row_start,
                rpm_saturation_source_rows=saturation_rows,
                legacy_exclusion_end_intervals_s=intervals,
            )
    return output


def _csv_rpm_metadata(path: Path, sample_time_s: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """返回旧版区间和精确异常时刻；异常时刻优先用于窗口长度相关的屏蔽。"""
    processed = _find_processed_root(path)
    relative = path.resolve().relative_to(processed.resolve()).as_posix()
    metadata = _load_processed_manifest(str((processed / "manifest.csv").resolve()))
    if relative not in metadata:
        raise ValueError(f"{path}: record is missing from {processed / 'manifest.csv'}")
    record = metadata[relative]
    # Inserted slots make source_row - start_row inaccurate. Prefer explicit times.
    with (processed / "manifest.csv").open(encoding="utf-8", newline="") as f:
        extended = next(r for r in csv.DictReader(f) if r['processed_relative_path'] == relative)
    if "rpm_saturation_times_s" in extended:
        return record.legacy_exclusion_end_intervals_s.copy(), np.asarray(json.loads(extended['rpm_saturation_times_s']), dtype=float)
    if not len(record.rpm_saturation_source_rows):
        return record.legacy_exclusion_end_intervals_s.copy(), np.empty(0, dtype=np.float64)
    assert record.source_row_start is not None
    frame_indices = record.rpm_saturation_source_rows - record.source_row_start
    if np.any(frame_indices < 0) or np.any(frame_indices >= len(sample_time_s)):
        raise ValueError(f"{path}: RPM saturation row lies outside this processed segment")
    return record.legacy_exclusion_end_intervals_s.copy(), sample_time_s[frame_indices].astype(np.float64, copy=True)


def _load_csv_sequence(path: Path) -> Sequence:
    """读取新 ``processed_csv_200hz`` 契约。

    ``sample_time_s`` 是唯一训练时间轴。pressure 已是四压力平均表压转换后的
    深度[m]，因此这里绝不再减大气压、除以 rho*g 或插值。
    """
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"{path}: CSV header is missing")
        missing = sorted(set(CSV_REQUIRED_COLUMNS) - set(reader.fieldnames))
        if missing:
            raise ValueError(f"{path}: missing required CSV columns {missing}")
        rows = list(reader)
    if len(rows) < 2:
        raise ValueError(f"{path}: fewer than two samples")

    def column(name: str) -> np.ndarray:
        try:
            values = np.asarray([float(row[name]) for row in rows], dtype=np.float64)
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"{path}: {name} contains missing or non-numeric values") from error
        if not np.all(np.isfinite(values)):
            raise ValueError(f"{path}: {name} contains NaN or Inf")
        return values

    imu_ts = column("sample_time_s")
    accel = np.column_stack((column("acc_x"), column("acc_y"), column("acc_z")))
    # 新数据只有 gyro_z；V20 只使用这一维，补零仅用于统一 [N,3] 的内部接口。
    full_gyro = all(c in reader.fieldnames for c in ("gyro_x", "gyro_y"))
    gyro = np.column_stack(tuple(column(f"gyro_{a}") for a in 'xyz')) if full_gyro else np.column_stack((np.zeros_like(imu_ts), np.zeros_like(imu_ts), column("gyro_z")))
    rpm = np.column_stack(tuple(column(f"int_rpm_{index}") for index in range(1, 9)))
    tag_xy = np.column_stack((column("tag_x_m"), column("tag_y_m")))
    sequence = Sequence(
        path=path,
        source_format="processed_csv_200hz",
        imu_ts=imu_ts,
        accel=accel,
        gyro=gyro,
        rpm_ts=imu_ts.copy(),
        rpm=rpm,
        tag_ts=imu_ts.copy(),
        tag_xy=tag_xy,
        tag_yaw=np.unwrap(column("tag_yaw_rad")),
        depth_m=column("pressure"),
        throttle=np.column_stack(tuple(column(f"throttle_{i}") for i in range(1,9))) if all(f'throttle_{i}' in reader.fieldnames for i in range(1,9)) else None,
        full_gyro_available=full_gyro,
        network_window_exclusion_end_intervals_s=np.empty((0, 2), dtype=np.float64),
    )
    legacy_intervals, saturation_times = _csv_rpm_metadata(path, imu_ts)
    sequence = Sequence(
        **{**sequence.__dict__, "network_window_exclusion_end_intervals_s": legacy_intervals,
           "rpm_saturation_times_s": saturation_times}
    )
    _validate_sequence(sequence)
    return sequence


def _load_hdf5_sequence(path: Path) -> Sequence:
    """兼容旧 processed HDF5，仅供历史 checkpoint 排查，非默认训练路径。"""
    with h5py.File(path, "r") as handle:
        required = (
            "imu/ts", "imu/accel", "imu/gyro", "rpm/ts", "rpm/rpm", "depth/ts",
            "depth/depth4_raw", "apriltag/ts", "apriltag/xy", "apriltag/yaw",
        )
        missing = [name for name in required if name not in handle]
        if missing:
            raise ValueError(f"{path}: missing required fields {missing}")
        raw_calibration = handle.attrs.get("depth_calibration")
        if raw_calibration is None:
            raise ValueError(f"{path}: missing depth_calibration HDF5 attribute")
        if isinstance(raw_calibration, bytes):
            raw_calibration = raw_calibration.decode("utf-8")
        calibration = json.loads(str(raw_calibration))
        sequence = Sequence(
            path=path,
            source_format="legacy_hdf5",
            imu_ts=np.asarray(handle["imu/ts"], dtype=np.float64),
            accel=np.asarray(handle["imu/accel"], dtype=np.float64),
            gyro=np.asarray(handle["imu/gyro"], dtype=np.float64),
            rpm_ts=np.asarray(handle["rpm/ts"], dtype=np.float64),
            rpm=np.asarray(handle["rpm/rpm"], dtype=np.float64),
            depth_ts=np.asarray(handle["depth/ts"], dtype=np.float64),
            depth4_raw=np.asarray(handle["depth/depth4_raw"], dtype=np.float64),
            tag_ts=np.asarray(handle["apriltag/ts"], dtype=np.float64),
            tag_xy=np.asarray(handle["apriltag/xy"], dtype=np.float64),
            tag_yaw=np.unwrap(np.asarray(handle["apriltag/yaw"], dtype=np.float64)),
            pressure_zero_pa=np.asarray(calibration["sensor_zero_pressure_pa"], dtype=np.float64),
            pressure_to_depth_m_per_pa=float(calibration["pressure_to_depth_m_per_pa"]),
        )
    _validate_sequence(sequence)
    return sequence


def load_sequence(path: str | Path) -> Sequence:
    """按文件后缀选择新 CSV 或历史 HDF5 读取器。"""
    path = Path(path)
    if path.suffix.lower() == ".csv":
        return _load_csv_sequence(path)
    if path.suffix.lower() in {".h5", ".hdf5"}:
        return _load_hdf5_sequence(path)
    raise ValueError(f"{path}: unsupported record type; expected .csv, .h5, or .hdf5")


def _validate_sequence(seq: Sequence) -> None:
    _validate_time(f"{seq.path}: imu", seq.imu_ts, seq.accel)
    _validate_time(f"{seq.path}: rpm", seq.rpm_ts, seq.rpm)
    _validate_time(f"{seq.path}: apriltag", seq.tag_ts, seq.tag_xy)
    if seq.accel.shape != seq.gyro.shape or seq.accel.ndim != 2 or seq.accel.shape[1] != 3:
        raise ValueError(f"{seq.path}: accel and gyro must be [N,3]")
    if seq.rpm.ndim != 2 or seq.rpm.shape[1] != 8:
        raise ValueError(f"{seq.path}: RPM must be [N,8]")
    if seq.tag_xy.ndim != 2 or seq.tag_xy.shape[1] != 2 or seq.tag_yaw.shape != seq.tag_ts.shape:
        raise ValueError(f"{seq.path}: AprilTag requires xy [N,2] and yaw [N]")
    if not np.allclose(np.diff(seq.imu_ts), 1.0 / 200.0, atol=2e-6, rtol=0):
        raise ValueError(f"{seq.path}: sample_time_s must be a regular 200 Hz processed stream")
    if len(seq.rpm) != len(seq.imu_ts) or not np.allclose(seq.rpm_ts, seq.imu_ts, atol=2e-6, rtol=0):
        raise ValueError(f"{seq.path}: RPM must align with the 200 Hz IMU stream")
    if not all(np.all(np.isfinite(value)) for value in (seq.accel, seq.gyro, seq.rpm, seq.tag_xy, seq.tag_yaw)):
        raise ValueError(f"{seq.path}: source contains NaN or Inf")
    if seq.depth_m is not None:
        if seq.depth_m.shape != seq.imu_ts.shape or not np.all(np.isfinite(seq.depth_m)):
            raise ValueError(f"{seq.path}: CSV physical depth must be finite [N]")
    else:
        if seq.depth_ts is None or seq.depth4_raw is None or seq.pressure_zero_pa is None or seq.pressure_to_depth_m_per_pa is None:
            raise ValueError(f"{seq.path}: legacy HDF5 depth fields are incomplete")
        _validate_time(f"{seq.path}: depth", seq.depth_ts, seq.depth4_raw)
        if seq.depth4_raw.ndim != 2 or seq.depth4_raw.shape[1] != 4:
            raise ValueError(f"{seq.path}: HDF5 depth4_raw must be [N,4]")
        if seq.pressure_zero_pa.shape != (4,) or seq.pressure_to_depth_m_per_pa <= 0:
            raise ValueError(f"{seq.path}: invalid legacy depth calibration")
    intervals = seq.network_window_exclusion_end_intervals_s
    if intervals is not None and (intervals.ndim != 2 or intervals.shape[1] != 2):
        raise ValueError(f"{seq.path}: RPM exclusion intervals must be [K,2]")
    if seq.rpm_saturation_times_s is not None and (seq.rpm_saturation_times_s.ndim != 1 or
                                                   not np.all(np.isfinite(seq.rpm_saturation_times_s))):
        raise ValueError(f"{seq.path}: RPM saturation times must be finite [K]")


def manifest_has_shared_validation_demo(manifest_path: str | Path) -> bool:
    """返回 demo 是否被明确声明为 validation 的共享展示集。"""
    manifest = read_yaml(manifest_path)
    demo = manifest.get("splits", {}).get("demo")
    return isinstance(demo, dict) and demo.get("alias_of") == "val"


def _allowed_suffixes(manifest: dict[str, Any]) -> set[str]:
    data_format = manifest.get("data_format", "legacy_hdf5")
    if data_format == "processed_csv_200hz":
        return {".csv"}
    if data_format == "legacy_hdf5":
        return {".h5", ".hdf5"}
    raise ValueError(f"unsupported manifest data_format: {data_format}")


def _resolve_manifest_entry(entry: dict[str, Any], processed: Path, split: str,
                            allowed_suffixes: set[str]) -> list[Path]:
    """解析非别名划分，支持 ``glob`` + ``exclude_files`` 或显式 ``files``。"""
    has_glob, has_files = "glob" in entry, "files" in entry
    if has_glob == has_files:
        raise ValueError(f"{split}: exactly one of glob or files is required")
    processed_root = processed.resolve()
    if has_glob:
        paths = sorted(path.resolve() for path in processed.glob(str(entry["glob"])) if path.is_file())
        excluded = entry.get("exclude_files", [])
        if not isinstance(excluded, list) or not all(isinstance(item, str) for item in excluded):
            raise ValueError(f"{split}: exclude_files must be a list of relative paths")
        excluded_paths = {(processed / item).resolve() for item in excluded}
        if any(processed_root not in path.parents or path.suffix.lower() not in allowed_suffixes for path in excluded_paths):
            raise ValueError(f"{split}: exclude_files contains an invalid processed record")
        missing_exclusions = [path for path in excluded_paths if not path.is_file()]
        if missing_exclusions:
            raise FileNotFoundError(f"{split}: excluded record does not exist: {missing_exclusions[0]}")
        paths = [path for path in paths if path not in excluded_paths]
    else:
        files = entry["files"]
        if not isinstance(files, list) or not files or not all(isinstance(item, str) for item in files):
            raise ValueError(f"{split}: files must be a non-empty list of relative paths")
        paths = []
        for relative in files:
            candidate = (processed / relative).resolve()
            if processed_root not in candidate.parents or candidate.suffix.lower() not in allowed_suffixes:
                raise ValueError(f"{split}: file must be a relative allowed record under processed: {relative}")
            if not candidate.is_file():
                raise FileNotFoundError(f"{split}: configured record does not exist: {relative}")
            paths.append(candidate)
    paths = [path for path in paths if path.suffix.lower() in allowed_suffixes]
    if not paths:
        raise ValueError(f"{split}: no records resolved")
    if len(paths) != len(set(paths)):
        raise ValueError(f"{split}: resolved duplicate records")
    expected = int(entry["expected_count"])
    if len(paths) != expected:
        raise ValueError(f"{split}: expected {expected} records, found {len(paths)}")
    return paths


def resolve_manifest(manifest_path: str | Path, data_root: str | Path) -> dict[str, list[Path]]:
    """解析数据划分，并禁止 train 与验证/展示集重叠。"""
    manifest = read_yaml(manifest_path)
    root = Path(data_root)
    if manifest.get("offline_apriltag_only") is not True:
        raise ValueError("manifest must declare offline_apriltag_only: true")
    processed = root / str(manifest.get("processed_root", "processed"))
    if not processed.is_dir():
        raise FileNotFoundError(f"processed data directory does not exist: {processed}")
    allowed_suffixes = _allowed_suffixes(manifest)
    output: dict[str, list[Path]] = {}
    for split in ("train", "val"):
        entry = manifest.get("splits", {}).get(split)
        if not isinstance(entry, dict):
            raise ValueError(f"manifest missing split {split}")
        if "alias_of" in entry:
            raise ValueError(f"{split}: alias_of is only supported for demo")
        output[split] = _resolve_manifest_entry(entry, processed, split, allowed_suffixes)
    demo = manifest.get("splits", {}).get("demo")
    if not isinstance(demo, dict):
        raise ValueError("manifest missing split demo")
    if "alias_of" in demo:
        if demo.get("alias_of") != "val" or set(demo) - {"alias_of", "expected_count"}:
            raise ValueError("demo alias only supports alias_of: val and expected_count")
        output["demo"] = output["val"]
        if len(output["demo"]) != int(demo["expected_count"]):
            raise ValueError("demo: expected_count does not match aliased val records")
    else:
        output["demo"] = _resolve_manifest_entry(demo, processed, "demo", allowed_suffixes)

    train, val, demo_paths = (set(output[name]) for name in ("train", "val", "demo"))
    if train & val or train & demo_paths:
        raise ValueError("manifest split leakage: train overlaps with a holdout split")
    if not manifest_has_shared_validation_demo(manifest_path) and val & demo_paths:
        raise ValueError("manifest split leakage: val overlaps with demo")
    return output


def physical_mean_depth(seq: Sequence, query_ts: np.ndarray, atmosphere_pa: float,
                        water_density_kg_m3: float, gravity_mps2: float) -> np.ndarray:
    """返回与 ``query_ts`` 对齐的深度：CSV 直用，HDF5 按历史规则恢复。"""
    query_ts = np.asarray(query_ts, dtype=np.float64)
    if seq.depth_m is not None:
        return np.interp(query_ts, seq.imu_ts, seq.depth_m)
    assert seq.depth_ts is not None and seq.depth4_raw is not None
    assert seq.pressure_zero_pa is not None and seq.pressure_to_depth_m_per_pa is not None
    ix = np.searchsorted(seq.depth_ts, query_ts, side="right") - 1
    output = np.full(len(ix), np.nan, dtype=np.float64)
    valid = ix >= 0
    absolute_pressure = seq.depth4_raw[ix[valid]] / seq.pressure_to_depth_m_per_pa + seq.pressure_zero_pa
    output[valid] = (absolute_pressure.mean(axis=1) - atmosphere_pa) / (water_density_kg_m3 * gravity_mps2)
    return output


def offline_labels(seq: Sequence, query_ts: np.ndarray, window: int, polyorder: int) -> np.ndarray:
    """用原始 Tag XY/yaw 构造离线机体系 ``[vx, vy, r]`` 标签。"""
    count = len(seq.tag_ts)
    window = min(int(window), count if count % 2 else count - 1)
    if window < 5 or window <= polyorder:
        raise ValueError(f"{seq.path}: insufficient AprilTag samples for Savitzky-Golay label")
    xy = savgol_filter(seq.tag_xy, window, polyorder, axis=0, mode="interp")
    yaw = savgol_filter(seq.tag_yaw, window, polyorder, mode="interp")
    world_velocity = np.gradient(xy, seq.tag_ts, axis=0, edge_order=1)
    yaw_rate = np.gradient(yaw, seq.tag_ts, edge_order=1)
    query_ts = np.asarray(query_ts, dtype=np.float64)
    qx = np.interp(query_ts, seq.tag_ts, world_velocity[:, 0])
    qy = np.interp(query_ts, seq.tag_ts, world_velocity[:, 1])
    qyaw = np.interp(query_ts, seq.tag_ts, yaw)
    c, s = np.cos(qyaw), np.sin(qyaw)
    return np.column_stack((
        c * qx + s * qy,
        -s * qx + c * qy,
        np.interp(query_ts, seq.tag_ts, yaw_rate),
    ))


def _window_view(values: np.ndarray, size: int) -> np.ndarray:
    return np.lib.stride_tricks.sliding_window_view(values, size, axis=0)


def _label_window_samples(cfg: dict[str, Any]) -> int:
    data = cfg["data"]
    if "label_savgol_window_s" in data:
        source_hz = float(data.get("source_sample_hz", data["imu_hz"]))
        samples = int(round(float(data["label_savgol_window_s"]) * source_hz))
        return samples + 1 if samples % 2 == 0 else samples
    return int(data["label_savgol_window"])


class V20WindowDataset(Dataset):
    """沿用原数据集类名的共享因果窗口实现，训练和评估共用特征构造。

    本实验固定 200 Hz 原始数据的六点均值。每个窗口截止于当前输出时刻，
    不访问未来传感器；离线 Tag 标签的平滑允许使用未来 Tag，但不进入输入。
    """

    input_keys = ("rpm", "accel_xy", "gyro_z", "depth")

    def __init__(self, paths: Iterable[str | Path], cfg: dict[str, Any],
                 normalizer: dict[str, np.ndarray] | None = None) -> None:
        self.cfg = cfg
        self.input_keys = tuple(cfg['model']['modalities'])
        data_cfg = cfg["data"]
        self.window = int(data_cfg["window_samples"])
        self.source_hz = float(data_cfg.get("source_sample_hz", data_cfg["imu_hz"]))
        self.input_hz = float(data_cfg.get("network_input_hz", data_cfg["imu_hz"]))
        self.output_hz = float(data_cfg["network_hz"])
        self.downsample_factor = int(round(self.source_hz / self.input_hz))
        self.stride = int(round(self.source_hz / self.output_hz))
        if not np.isclose(self.source_hz / self.input_hz, self.downsample_factor) or self.downsample_factor < 1:
            raise ValueError("source_sample_hz / network_input_hz must be a positive integer")
        if not np.isclose(self.source_hz / self.output_hz, self.stride) or self.stride < 1:
            raise ValueError("source_sample_hz / network_hz must be a positive integer")
        # V21/V22/V23 分别需要 256*2 / 256*4 / 256*5 个原始样本；V20 系数为 1 时退化回 256 点。
        self.raw_window_samples = self.window * self.downsample_factor
        self.sequences = [load_sequence(path) for path in paths]
        self.windows: list[dict[str, np.ndarray]] = []
        self.entries: list[tuple[int, int, float, np.ndarray]] = []
        pressure_cfg = cfg["data"].get("pressure", {})
        for sequence_index, seq in enumerate(self.sequences):
            if len(seq.imu_ts) < self.raw_window_samples:
                self.windows.append({})
                continue
            if "gyro_xyz" in self.input_keys and not seq.full_gyro_available:
                raise ValueError(f"{seq.path}: full gyro input requires actual gyro_x/y/z")
            if "throttle" in self.input_keys and seq.throttle is None:
                raise ValueError(f"{seq.path}: throttle input requires actual eight commands")
            depth = physical_mean_depth(
                seq, seq.imu_ts,
                float(pressure_cfg.get("atmosphere_pa", 101325.0)),
                float(pressure_cfg.get("water_density_kg_m3", 1000.0)),
                float(pressure_cfg.get("gravity_mps2", 9.80665)),
            )[:, None]
            values = {
                "rpm": _window_view(seq.rpm, self.raw_window_samples),
                "accel_xy": _window_view(seq.accel[:, :2], self.raw_window_samples),
                "gyro_z": _window_view(seq.gyro[:, 2:3], self.raw_window_samples),
                "depth": _window_view(depth, self.raw_window_samples),
                "accel_xyz": _window_view(seq.accel, self.raw_window_samples),
                "gyro_xyz": _window_view(seq.gyro, self.raw_window_samples),
            }
            if seq.throttle is not None:
                values['throttle'] = _window_view(seq.throttle, self.raw_window_samples)
            end_idx = np.arange(self.raw_window_samples - 1, len(seq.imu_ts), self.stride)
            valid = (
                (seq.imu_ts[end_idx] >= seq.tag_ts[0]) &
                (seq.imu_ts[end_idx] <= seq.tag_ts[-1]) &
                np.isfinite(depth[end_idx, 0]) &
                np.isfinite(depth[end_idx - self.raw_window_samples + 1, 0])
            )
            # CSV 记录可定位到原始 RPM 饱和帧。针对当前真实窗口长度动态扩展：
            # 只要饱和样本落入当前原始历史（V20--V23）就拒绝该窗口。
            if seq.rpm_saturation_times_s is not None and len(seq.rpm_saturation_times_s):
                history_s = (self.raw_window_samples - 1) / self.source_hz
                intervals = np.column_stack((seq.rpm_saturation_times_s,
                                             seq.rpm_saturation_times_s + history_s))
            else:
                # 旧 HDF5 或旧 fixture 没有饱和样本行时，向后兼容已有窗口端点区间。
                intervals = seq.network_window_exclusion_end_intervals_s
            if intervals is not None and len(intervals):
                endpoint_time = seq.imu_ts[end_idx]
                excluded = np.zeros(len(end_idx), dtype=bool)
                for start_s, end_s in intervals:
                    excluded |= (endpoint_time >= start_s) & (endpoint_time <= end_s)
                valid &= ~excluded
            ends = end_idx[valid]
            labels = offline_labels(
                seq, seq.imu_ts[ends], _label_window_samples(cfg),
                int(cfg["data"]["label_savgol_polyorder"]),
            )
            self.windows.append(values)
            self.entries.extend(
                (sequence_index, int(end - self.raw_window_samples + 1), float(seq.imu_ts[end]), labels[index])
                for index, end in enumerate(ends)
            )
        if not self.entries:
            raise ValueError("dataset has no valid causal feature/label windows")
        self.normalizer = normalizer
        if normalizer is not None:
            self._check_normalizer(normalizer)

    def _check_normalizer(self, normalizer: dict[str, np.ndarray]) -> None:
        shapes = {"rpm": (8,), "throttle": (8,), "accel_xy": (2,), "accel_xyz": (3,), "gyro_z": (1,), "gyro_xyz": (3,), "depth": (1,)}
        for key in self.input_keys:
            shape = shapes[key]
            for suffix in ("mean", "std"):
                value = np.asarray(normalizer[f"{key}_{suffix}"])
                if value.shape != shape or not np.all(np.isfinite(value)) or (suffix == "std" and np.any(value <= 0)):
                    raise ValueError(f"invalid normalizer {key}_{suffix}")
        if "rpm" in self.input_keys and (not np.allclose(normalizer["rpm_mean"], 0.0) or not np.allclose(normalizer["rpm_std"], 1000.0)):
            raise ValueError("V20 RPM normalizer must be exactly mean=0, std=1000 for every channel")
        if 'throttle' in self.input_keys and (not np.allclose(normalizer['throttle_mean'],0) or not np.allclose(normalizer['throttle_std'],1)):
            raise ValueError('Throttle normalization must be identity')

    def _feature_window(self, sequence_index: int, start: int, key: str) -> np.ndarray:
        """返回单个网络输入窗口；降采样实验在此执行因果块均值，避免复制全量窗口。"""
        raw = self.windows[sequence_index][key][start]
        if self.downsample_factor == 1:
            return raw
        return raw.reshape(raw.shape[0], self.window, self.downsample_factor).mean(axis=2)

    def fit_normalizer(self, maximum_windows: int = 4096) -> dict[str, np.ndarray]:
        """以训练集的统一 z-score 统计量拟合非 RPM 通道。"""
        selection = np.linspace(0, len(self.entries) - 1, min(len(self.entries), maximum_windows), dtype=int)
        output: dict[str, np.ndarray] = {
            "rpm_mean": np.zeros(8, dtype=np.float32),
            "rpm_std": np.full(8, 1000.0, dtype=np.float32),
        }
        if "rpm" not in self.input_keys:
            output.clear()
        if "throttle" in self.input_keys:
            output.update(throttle_mean=np.zeros(8,dtype=np.float32),throttle_std=np.ones(8,dtype=np.float32))
        for key in (k for k in self.input_keys if k not in ("rpm", "throttle")):
            blocks = [self._feature_window(self.entries[index][0], self.entries[index][1], key).T for index in selection]
            values = np.concatenate(blocks, axis=0)
            output[f"{key}_mean"] = values.mean(axis=0).astype(np.float32)
            output[f"{key}_std"] = np.maximum(values.std(axis=0), 1e-3).astype(np.float32)
        self.normalizer = output
        self._check_normalizer(output)
        return {key: value.copy() for key, value in output.items()}

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        if self.normalizer is None:
            raise RuntimeError("call fit_normalizer or provide training normalizer first")
        sequence_index, start, timestamp, target = self.entries[index]
        output: dict[str, torch.Tensor] = {
            "target": torch.as_tensor(target, dtype=torch.float32),
            "time": torch.tensor(timestamp, dtype=torch.float64),
            "sequence": torch.tensor(sequence_index, dtype=torch.int64),
        }
        for key in self.input_keys:
            raw = self._feature_window(sequence_index, start, key)
            normalized = (raw - self.normalizer[f"{key}_mean"][:, None]) / self.normalizer[f"{key}_std"][:, None]
            output[key] = torch.from_numpy(np.asarray(normalized, dtype=np.float32))
        return output


def validate_dataset(manifest_path: str | Path, data_root: str | Path,
                     cfg: dict[str, Any] | None = None) -> dict[str, Any]:
    """检查全部记录、CSV 契约、数据划分及可用因果窗口，返回 JSON 友好报告。"""
    splits = resolve_manifest(manifest_path, data_root)
    report: dict[str, Any] = {"splits": {}, "ok": True}
    for split, paths in splits.items():
        durations: list[float] = []
        usable_windows = 0
        short_records = []
        source_formats: set[str] = set()
        for path in paths:
            # 有配置时由 Dataset 读取一次即可；避免 validate 为同一大型 CSV
            # 先读一次、构造窗口时又读第二次，导致训练前检查无谓地翻倍耗时。
            # Short terminal segments remain in the snapshot but cannot form history.
            if cfg is not None:
                seq = load_sequence(path)
                required = int(cfg['data']['window_samples'])*int(round(float(cfg['data']['source_sample_hz'])/float(cfg['data']['network_input_hz'])))
                if len(seq.imu_ts)<required:
                    short_records.append(str(path.relative_to(Path(data_root).resolve())))
                    durations.append(float(seq.imu_ts[-1]-seq.imu_ts[0]))
                    source_formats.add(seq.source_format)
                    continue
            record_dataset = V20WindowDataset([path], cfg) if cfg is not None else None
            seq = record_dataset.sequences[0] if record_dataset is not None else load_sequence(path)
            durations.append(float(seq.imu_ts[-1] - seq.imu_ts[0]))
            source_formats.add(seq.source_format)
            if record_dataset is not None:
                usable_windows += len(record_dataset)
        report["splits"][split] = {
            "records": len(paths),
            "duration_s_total": float(sum(durations)),
            "source_formats": sorted(source_formats),
            "usable_50hz_windows": usable_windows if cfg is not None else None,
            "short_records_without_windows": short_records,
            "files": [str(path.relative_to(Path(data_root).resolve())) for path in paths],
        }
    report["shared_validation_demo"] = manifest_has_shared_validation_demo(manifest_path)
    return report
