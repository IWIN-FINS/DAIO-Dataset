"""姿态包因果性与网络协方差语义测试。"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from rov_eskf.data import Sequence
from rov_eskf.replay import _network_covariances
from rov_eskf.yaw_source import causal_yaw_rate_source, raw_gyro_z_source


def _sequence(yaw: np.ndarray) -> Sequence:
    imu_ts = np.arange(8, dtype=float) * .005
    attitude_ts = np.array([0., .01, .02, .03])
    return Sequence(Path("synthetic.h5"), imu_ts, np.zeros((8, 3)), np.zeros((8, 3)), imu_ts,
                    np.zeros((8, 8)), np.array([0., .02]), np.zeros((2, 4)), np.zeros((2, 4)),
                    attitude_ts, np.c_[np.zeros(4), np.zeros(4), yaw], np.zeros(4), 1.)


def test_causal_rate_does_not_read_future_attitude_packet() -> None:
    cfg = {"backward_window_s": .01, "median_filter_samples": 1, "max_abs_rate_radps": 100.}
    source_a = causal_yaw_rate_source(_sequence(np.array([0., .1, .2, .3])), cfg)
    source_b = causal_yaw_rate_source(_sequence(np.array([0., .1, .2, 99.])), cfg)
    # t=0.025 时只能知道 t=0.02 包，不应受 t=0.03 的未来改动影响。
    assert np.isclose(source_a.rate_at(.025), source_b.rate_at(.025))
    assert np.isclose(source_a.rate_at(.025), 10.)
    assert np.isclose(source_a.age_at(.025), .005)


def test_fixed_window_rate_rejects_irregular_short_timestamp_spike() -> None:
    sequence = _sequence(np.array([0., .01, .0101, .02]))
    sequence = Sequence(sequence.path, sequence.imu_ts, sequence.accel, sequence.gyro, sequence.rpm_ts,
                        sequence.rpm, sequence.depth_ts, sequence.depth4, sequence.depth4_raw,
                        np.array([0., .01, .0101, .06]), sequence.attitude_rpy,
                        sequence.pressure_zero_pa, sequence.pressure_to_depth_m_per_pa)
    source = causal_yaw_rate_source(sequence, {"backward_window_s": .05, "median_filter_samples": 3,
                                                "max_abs_rate_radps": 1.})
    # 相邻包法会在 0.0101 s 产生 100 rad/s；固定窗法只在 0.06 s 后产生有限值。
    assert source.rate_at(.02) == 0.0
    assert abs(source.rate_at(.061)) <= 1.0


def test_raw_gyro_z_source_does_not_read_solved_yaw() -> None:
    sequence_a = _sequence(np.array([0., .1, .2, .3]))
    sequence_b = _sequence(np.array([0., .1, .2, 99.]))
    sequence_a = Sequence(sequence_a.path, sequence_a.imu_ts, sequence_a.accel,
                          np.c_[np.zeros(8), np.zeros(8), np.arange(8, dtype=float)], sequence_a.rpm_ts,
                          sequence_a.rpm, sequence_a.depth_ts, sequence_a.depth4, sequence_a.depth4_raw,
                          sequence_a.attitude_ts, sequence_a.attitude_rpy,
                          sequence_a.pressure_zero_pa, sequence_a.pressure_to_depth_m_per_pa)
    sequence_b = Sequence(sequence_b.path, sequence_b.imu_ts, sequence_b.accel, sequence_a.gyro,
                          sequence_b.rpm_ts, sequence_b.rpm, sequence_b.depth_ts, sequence_b.depth4,
                          sequence_b.depth4_raw, sequence_b.attitude_ts, sequence_b.attitude_rpy,
                          sequence_b.pressure_zero_pa, sequence_b.pressure_to_depth_m_per_pa)
    source_a, source_b = raw_gyro_z_source(sequence_a), raw_gyro_z_source(sequence_b)
    assert source_a.rate_at(.025) == source_b.rate_at(.025) == 5.0
    assert source_a.age_at(.025) == 0.0


def test_network_covariance_scale_is_applied_to_R_directly() -> None:
    config = {"filter": {"network": {"xy": {"min_std": .001, "max_std": 1., "covariance_scale": 16., "max_step_mps": .1},
                                        "yaw_rate": {"min_std": .001, "max_std": 1., "variance_scale": 9.}}}}
    Rxy, Rr, _ = _network_covariances(np.array([.1, .2, .3]), config)
    assert np.allclose(np.diag(Rxy), [.16, .64])
    assert np.isclose(Rr, .81)
