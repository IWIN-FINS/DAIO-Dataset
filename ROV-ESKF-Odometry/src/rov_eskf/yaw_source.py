"""严格因果、抗时间戳抖动的 IMU 内解航向角速度源。"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .data import Sequence


@dataclass(frozen=True)
class CausalYawRateSource:
    """由固定后向时间窗差分形成的 Euler yaw-rate 流。

    姿态包的时间戳存在短间隔抖动。若直接对相邻两包求差分，一个仅数十
    微秒的间隔会把很小的 yaw 量化误差放大为非物理角速度。这里第 j 包到达
    后，只与 ``backward_window_s`` 之前最近的已到达包求差分，再对过去若干
    个候选作因果中值滤波并限幅。结果从第 j 包时刻起零阶保持，因此从不读取
    未来姿态包。
    """

    change_ts: np.ndarray
    rates: np.ndarray
    source_packet_ts: np.ndarray

    def rate_at(self, query_time: float) -> float:
        index = int(np.searchsorted(self.change_ts, float(query_time), side="right") - 1)
        return 0.0 if index < 0 else float(self.rates[index])

    def age_at(self, query_time: float) -> float:
        index = int(np.searchsorted(self.change_ts, float(query_time), side="right") - 1)
        return float("inf") if index < 0 else float(query_time - self.source_packet_ts[index])


def causal_yaw_rate_source(seq: Sequence, config: dict | None = None) -> CausalYawRateSource:
    """构建绝不使用未来姿态包的稳健 yaw-rate 源。

    ``config`` 来自 ``filter.yaw_rate_source``。保留默认值使该函数也能在
    独立单元测试中使用；正式回放必须由 YAML 显式冻结这些参数。
    """
    cfg = config or {}
    window_s = float(cfg.get("backward_window_s", 0.05))
    median_samples = int(cfg.get("median_filter_samples", 5))
    max_abs_rate = float(cfg.get("max_abs_rate_radps", 1.0))
    if not np.isfinite(window_s) or window_s <= 0:
        raise ValueError("yaw-rate 后向差分窗口必须为有限正数")
    if median_samples < 1 or median_samples % 2 != 1:
        raise ValueError("yaw-rate 因果中值滤波长度必须是正奇数")
    if not np.isfinite(max_abs_rate) or max_abs_rate <= 0:
        raise ValueError("yaw-rate 物理限幅必须为有限正数")
    ts = np.asarray(seq.attitude_ts, dtype=float)
    yaw = np.unwrap(np.asarray(seq.attitude_rpy[:, 2], dtype=float))
    if len(ts) < 2:
        raise ValueError(f"{seq.path}: 至少需要两包 IMU 解算姿态才能微分 yaw")
    # 对第 j 包，从不晚于 t_j-window 的最近历史包差分。这样分母始终接近
    # 固定窗口，而不是受异常密集的姿态包影响。
    previous = np.searchsorted(ts, ts - window_s, side="right") - 1
    valid = previous >= 0
    current = np.flatnonzero(valid)
    if len(current) == 0:
        return CausalYawRateSource(np.empty(0), np.empty(0), np.empty(0))
    dt = ts[current] - ts[previous[current]]
    raw = (yaw[current] - yaw[previous[current]]) / dt
    clipped = np.clip(raw, -max_abs_rate, max_abs_rate)
    rates = np.empty_like(clipped)
    for index in range(len(clipped)):
        begin = max(0, index - median_samples + 1)
        rates[index] = np.median(clipped[begin:index + 1])
    return CausalYawRateSource(change_ts=ts[current].copy(), rates=rates,
                               source_packet_ts=ts[current].copy())


def raw_gyro_z_source(seq: Sequence) -> CausalYawRateSource:
    """将原始 IMU ``gyro_z`` 直接作为严格因果的 yaw-rate 传播源。

    该路径刻意**不读取** ``attitude_rpy`` 或记录中的 IMU 解算 yaw：每个
    陀螺仪样本到达后，其 z 轴角速度从该时间戳起零阶保持至下一 IMU 样本。
    网络 yaw-rate 观测仍由 11 状态滤波器估计的 ``yaw_rate_correction``
    校正该原始传播率。
    """
    ts = np.asarray(seq.imu_ts, dtype=float)
    gyro_z = np.asarray(seq.gyro[:, 2], dtype=float)
    if len(ts) != len(gyro_z) or len(ts) == 0:
        raise ValueError(f"{seq.path}: 原始 gyro_z 时间轴或长度无效")
    if not np.all(np.isfinite(gyro_z)):
        raise ValueError(f"{seq.path}: 原始 gyro_z 包含非有限值")
    return CausalYawRateSource(change_ts=ts.copy(), rates=gyro_z.copy(), source_packet_ts=ts.copy())


def causal_tilt(seq: Sequence, query_ts: np.ndarray) -> np.ndarray:
    """最近过去姿态包的 roll/pitch 零阶保持，不插值也不使用未来数据。"""
    query = np.asarray(query_ts, dtype=float)
    indices = np.searchsorted(seq.attitude_ts, query, side="right") - 1
    if np.any(indices < 0):
        raise ValueError(f"{seq.path}: 查询时刻之前没有可用 IMU 姿态包")
    return np.asarray(seq.attitude_rpy[indices, :2], dtype=float)
