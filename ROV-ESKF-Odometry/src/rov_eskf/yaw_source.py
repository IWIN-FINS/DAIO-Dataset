"""Strictly causal IMU-solved yaw-rate source robust to timestamp jitter."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .data import Sequence


@dataclass(frozen=True)
class CausalYawRateSource:
    """Euler yaw-rate stream computed from fixed backward time windows.

    Attitude timestamps can contain very short intervals. Differencing adjacent packets can amplify
    small yaw quantization errors into unphysical angular rates when an interval is only tens of microseconds.
    When packet j arrives, difference it against the latest available packet at or before
    ``backward_window_s`` earlier, then apply a causal median filter and clipping to recent candidates.
    Hold the result from packet j onward, without ever reading future attitude packets.
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
    """Build a robust yaw-rate source without using future attitude packets.

    ``config`` comes from ``filter.yaw_rate_source``. Defaults support standalone unit tests;
    production replay must explicitly freeze these parameters in YAML.
    """
    cfg = config or {}
    window_s = float(cfg.get("backward_window_s", 0.05))
    median_samples = int(cfg.get("median_filter_samples", 5))
    max_abs_rate = float(cfg.get("max_abs_rate_radps", 1.0))
    if not np.isfinite(window_s) or window_s <= 0:
        raise ValueError("The yaw-rate backward-difference window must be finite and positive")
    if median_samples < 1 or median_samples % 2 != 1:
        raise ValueError("The causal yaw-rate median filter length must be a positive odd integer")
    if not np.isfinite(max_abs_rate) or max_abs_rate <= 0:
        raise ValueError("The yaw-rate physical limit must be finite and positive")
    ts = np.asarray(seq.attitude_ts, dtype=float)
    yaw = np.unwrap(np.asarray(seq.attitude_rpy[:, 2], dtype=float))
    if len(ts) < 2:
        raise ValueError(f"{seq.path}: at least two IMU attitude packets are required to differentiate yaw")
    # For packet j, difference against the latest historical packet at or before t_j-window.
    # This keeps the denominator near the fixed window rather than an abnormally short packet interval.
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
    """Use raw IMU ``gyro_z`` directly as a strictly causal yaw-rate propagation source.

    This path deliberately does not read ``attitude_rpy`` or IMU-solved yaw. When each gyro sample
    arrives, hold its z-axis angular velocity from that timestamp until the next IMU sample.
    Network yaw-rate observations still correct the raw propagation rate through
    ``yaw_rate_correction`` estimated by the 11-state filter.
    """
    ts = np.asarray(seq.imu_ts, dtype=float)
    gyro_z = np.asarray(seq.gyro[:, 2], dtype=float)
    if len(ts) != len(gyro_z) or len(ts) == 0:
        raise ValueError(f"{seq.path}: invalid raw gyro_z timeline or length")
    if not np.all(np.isfinite(gyro_z)):
        raise ValueError(f"{seq.path}: raw gyro_z contains nonfinite values")
    return CausalYawRateSource(change_ts=ts.copy(), rates=gyro_z.copy(), source_packet_ts=ts.copy())


def causal_tilt(seq: Sequence, query_ts: np.ndarray) -> np.ndarray:
    """Hold roll/pitch from the latest past attitude packet without interpolation or future data."""
    query = np.asarray(query_ts, dtype=float)
    indices = np.searchsorted(seq.attitude_ts, query, side="right") - 1
    if np.any(indices < 0):
        raise ValueError(f"{seq.path}: no IMU attitude packet is available at or before the query time")
    return np.asarray(seq.attitude_rpy[indices, :2], dtype=float)
