"""11 状态滤波器的纯数值回归测试。"""
from __future__ import annotations

import numpy as np

from rov_eskf.eskf import AttitudeAidedESKF11, rotation_matrix_from_tilt_yaw


def _cfg() -> dict:
    return {"gravity_flu": [0., 0., -9.80665], "accel_sample_covariance": np.eye(3) * 1e-4,
            "accel_bias_rw_psd": np.eye(3) * 1e-6, "yaw_rate_source_psd_rad2_per_s": 1e-6,
            "yaw_rate_correction_rw_psd_rad2_per_s3": 1e-7}


def _filter() -> AttitudeAidedESKF11:
    f = AttitudeAidedESKF11(_cfg())
    f.initialize(np.zeros(3), np.zeros(3), np.eye(11), yaw=0.)
    return f


def test_rotation_is_zyx() -> None:
    R = rotation_matrix_from_tilt_yaw(.1, -.2, .3)
    assert np.allclose(R.T @ R, np.eye(3), atol=1e-12)
    assert np.isclose(np.linalg.det(R), 1.0, atol=1e-12)


def test_yaw_rate_correction_enters_nominal_propagation() -> None:
    f = _filter(); f.yaw_rate_correction = .05
    f.predict_interval(np.zeros(3), np.zeros(3), .2, 0., 0., .5)
    assert np.isclose(f.yaw, .125, atol=1e-12)


def test_network_yaw_rate_update_changes_later_yaw() -> None:
    f = _filter()
    update = f.update_yaw_rate(.3, .2, 1e-8, 1e6)
    assert update.accepted and f.yaw_rate_correction > .09
    f.predict_interval(np.zeros(3), np.zeros(3), .2, 0., 0., 1.)
    assert f.yaw > .29


def test_xy_and_yaw_rate_updates_are_independent() -> None:
    f = _filter(); f.P = np.eye(11) * 1e-6; f.P[10, 10] = 1.0
    xy = f.update_body_velocity(np.array([10., 10.]), np.eye(2) * 1e-8, 9.21)
    rate = f.update_yaw_rate(.1, 0., 1e-8, 6.63)
    assert not xy.accepted
    assert rate.accepted
    assert f.yaw_rate_correction > .09


def test_reanchor_rotates_velocity_and_keeps_covariance_psd() -> None:
    f = _filter(); f.v = np.array([1., 0., .2]); f.p = np.array([2., 3., 1.])
    f.reanchor_yaw(np.pi / 2, .1)
    assert np.allclose(f.v, [0., 1., .2], atol=1e-12)
    assert np.allclose(f.p, [0., 0., 1.])
    assert np.min(np.linalg.eigvalsh(f.P)) >= -1e-12


def test_velocity_jacobian_has_yaw_coupling() -> None:
    f = _filter(); f.v = np.array([1., .2, 0.]); f.yaw = .3
    _, H = f.velocity_prediction()
    assert H.shape == (2, 11)
    assert np.linalg.norm(H[:, 9]) > 0
