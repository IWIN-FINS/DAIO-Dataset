"""11-state attitude-aided ESKF (FLU).

State: ``[p_w(3), v_w(3), b_a^b(3), yaw, yaw_rate_correction]``.
Roll/pitch come from the IMU attitude solution; yaw propagates using causal IMU yaw-rate plus a slow correction.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

_EZ = np.array([0.0, 0.0, 1.0])


@dataclass(frozen=True)
class Update:
    accepted: bool
    nis: float
    residual: np.ndarray


def _rx(angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[1., 0., 0.], [0., c, -s], [0., s, c]])


def _ry(angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, 0., s], [0., 1., 0.], [-s, 0., c]])


def _rz(angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]])


def _skew(vector: np.ndarray) -> np.ndarray:
    x, y, z = np.asarray(vector, dtype=float).reshape(3)
    return np.array([[0., -z, y], [z, 0., -x], [-y, x, 0.]])


def rotation_matrix_from_tilt_yaw(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """ZYX rotation from the FLU body frame to the world frame."""
    return _rz(float(yaw))
    # @ _ry(float(pitch)) @ _rx(float(roll))


class AttitudeAidedESKF11:
    """11-state ESKF driven by external roll/pitch and correctable yaw-rate."""

    def __init__(self, cfg: dict, nominal_imu_dt: float = 0.005) -> None:
        self.g = np.asarray(cfg["gravity_flu"], dtype=float).reshape(3)
        sample_cov = self._psd(np.asarray(cfg["accel_sample_covariance"], dtype=float))
        self.accel_noise_density = sample_cov * float(nominal_imu_dt)
        self.accel_bias_rw_psd = self._psd(np.asarray(cfg["accel_bias_rw_psd"], dtype=float))
        self.yaw_rate_source_psd = float(cfg["yaw_rate_source_psd_rad2_per_s"])
        self.yaw_rate_correction_rw_psd = float(cfg["yaw_rate_correction_rw_psd_rad2_per_s3"])
        self.p = np.zeros(3)
        self.v = np.zeros(3)
        self.ba = np.zeros(3)
        self.yaw = 0.0
        self.yaw_rate_correction = 0.0
        self.roll_imu = 0.0
        self.pitch_imu = 0.0
        self.P = np.eye(11)

    @staticmethod
    def _psd(matrix: np.ndarray) -> np.ndarray:
        matrix = .5 * (np.asarray(matrix, dtype=float) + np.asarray(matrix, dtype=float).T)
        if not np.all(np.isfinite(matrix)) or np.min(np.linalg.eigvalsh(matrix)) < -1e-12:
            raise ValueError("Covariance must be finite and positive semidefinite")
        return matrix

    @staticmethod
    def _project_psd(matrix: np.ndarray) -> np.ndarray:
        matrix = .5 * (matrix + matrix.T)
        values, vectors = np.linalg.eigh(matrix)
        return (vectors * np.maximum(values, 1e-12)) @ vectors.T

    def set_external_tilt(self, roll: float, pitch: float) -> None:
        self.roll_imu, self.pitch_imu = float(roll), float(pitch)

    def rotation_matrix(self) -> np.ndarray:
        return rotation_matrix_from_tilt_yaw(self.roll_imu, self.pitch_imu, self.yaw)

    def initialize(self, position: np.ndarray, accel_bias: np.ndarray, covariance: np.ndarray,
                   *, velocity: np.ndarray | None = None, yaw: float = 0.0,
                   yaw_rate_correction: float = 0.0) -> None:
        covariance = self._psd(np.asarray(covariance, dtype=float))
        if covariance.shape != (11, 11):
            raise ValueError("Initial covariance must be 11x11")
        self.p = np.asarray(position, dtype=float).reshape(3)
        self.v = np.zeros(3) if velocity is None else np.asarray(velocity, dtype=float).reshape(3)
        self.ba = np.asarray(accel_bias, dtype=float).reshape(3)
        self.yaw = float(yaw)
        self.yaw_rate_correction = float(yaw_rate_correction)
        self.P = covariance

    def reanchor_yaw(self, yaw_absolute: float, yaw_std: float) -> None:
        """One-time world-frame rebase for offline benchmarking, not a deployment observation."""
        if not np.isfinite(yaw_std) or yaw_std <= 0:
            raise ValueError("Yaw re-anchor standard deviation must be positive")
        delta = float(yaw_absolute) - self.yaw
        transform = np.eye(11)
        transform[:3, :3] = _rz(delta)
        transform[3:6, 3:6] = _rz(delta)
        self.v = _rz(delta) @ self.v
        self.p[:2] = 0.0  # Define the local world origin at the vehicle position when re-anchoring.
        self.yaw = float(yaw_absolute)
        self.P = transform @ self.P @ transform.T
        # The initialization protocol defines the origin and benchmark heading; clear their old correlations.
        self.P[:2, :] = 0.0
        self.P[:, :2] = 0.0
        self.P[0, 0] = self.P[1, 1] = 1e-6
        self.P[9, :] = 0.0
        self.P[:, 9] = 0.0
        self.P[9, 9] = yaw_std ** 2
        self.P = self._project_psd(self.P)

    def predict_interval(self, accel_start: np.ndarray, accel_end: np.ndarray, imu_yaw_rate: float,
                         roll_mid: float, pitch_mid: float, dt: float) -> None:
        """Midpoint integration: ``yaw_dot = imu_yaw_rate + yaw_rate_correction``."""
        dt = float(dt)
        if not np.isfinite(dt) or dt <= 0:
            raise ValueError("Prediction dt must be finite and positive")
        self.set_external_tilt(roll_mid, pitch_mid)
        f = .5 * (np.asarray(accel_start, dtype=float).reshape(3)
                  + np.asarray(accel_end, dtype=float).reshape(3)) - self.ba
        yaw_rate = float(imu_yaw_rate) + self.yaw_rate_correction
        yaw_mid = self.yaw + .5 * yaw_rate * dt
        R = rotation_matrix_from_tilt_yaw(self.roll_imu, self.pitch_imu, yaw_mid)
        a_world = R @ f + self.g
        self.p = self.p + self.v * dt + .5 * a_world * dt * dt
        self.v = self.v + a_world * dt
        self.yaw += yaw_rate * dt

        F = np.zeros((11, 11))
        G = np.zeros((11, 8))
        F[:3, 3:6] = np.eye(3)
        F[3:6, 6:9] = -R
        F[3:6, 9] = _skew(_EZ) @ (R @ f)
        F[9, 10] = 1.0
        G[3:6, :3] = -R
        G[6:9, 3:6] = np.eye(3)
        G[9, 6] = 1.0
        G[10, 7] = 1.0
        Qc = np.zeros((8, 8))
        Qc[:3, :3] = self.accel_noise_density
        Qc[3:6, 3:6] = self.accel_bias_rw_psd
        Qc[6, 6] = self.yaw_rate_source_psd
        Qc[7, 7] = self.yaw_rate_correction_rw_psd
        F2, F3 = F @ F, F @ F @ F
        Phi = np.eye(11) + F * dt + .5 * F2 * dt**2 + F3 * dt**3 / 6.0
        Q = G @ Qc @ G.T
        Qd = (Q * dt + .5 * (F @ Q + Q @ F.T) * dt**2
              + (F2 @ Q + 2 * F @ Q @ F.T + Q @ F2.T) * dt**3 / 6.0)
        self.P = self._project_psd(Phi @ self.P @ Phi.T + Qd)

    def _inject(self, dx: np.ndarray) -> None:
        self.p += dx[:3]
        self.v += dx[3:6]
        self.ba += dx[6:9]
        self.yaw += dx[9]
        self.yaw_rate_correction += dx[10]

    def update(self, residual: np.ndarray, H: np.ndarray, covariance: np.ndarray, gate: float) -> Update:
        residual, H = np.asarray(residual, dtype=float).reshape(-1), np.asarray(H, dtype=float)
        covariance = self._psd(np.asarray(covariance, dtype=float))
        if H.shape != (len(residual), 11) or covariance.shape != (len(residual), len(residual)):
            raise ValueError("Observation dimensions do not match the 11-state model")
        S = self._psd(H @ self.P @ H.T + covariance)
        nis = float(residual @ np.linalg.solve(S, residual))
        if not np.isfinite(nis) or nis > float(gate):
            return Update(False, nis, residual.copy())
        K = np.linalg.solve(S, H @ self.P).T
        identity = np.eye(11)
        self.P = self._project_psd((identity - K @ H) @ self.P @ (identity - K @ H).T + K @ covariance @ K.T)
        self._inject(K @ residual)
        return Update(True, nis, residual.copy())

    def update_height(self, height: float, variance: float, gate: float) -> Update:
        H = np.zeros((1, 11)); H[0, 2] = 1.0
        return self.update(np.array([float(height) - self.p[2]]), H, np.array([[float(variance)]]), gate)

    def velocity_prediction(self) -> tuple[np.ndarray, np.ndarray]:
        R = self.rotation_matrix()
        body_velocity = R.T @ self.v
        H = np.zeros((2, 11))
        H[:, 3:6] = R.T[:2, :]
        H[:, 9] = -(R.T @ (_skew(_EZ) @ self.v))[:2]
        return body_velocity[:2], H

    def update_body_velocity(self, velocity_xy: np.ndarray, covariance: np.ndarray, gate: float) -> Update:
        predicted, H = self.velocity_prediction()
        return self.update(np.asarray(velocity_xy, dtype=float).reshape(2) - predicted, H, covariance, gate)

    def update_yaw_rate(self, yaw_rate: float, imu_yaw_rate: float, variance: float, gate: float) -> Update:
        """Observe the difference between network Euler yaw-rate and causal IMU yaw-rate."""
        H = np.zeros((1, 11)); H[0, 10] = 1.0
        predicted = float(imu_yaw_rate) + self.yaw_rate_correction
        return self.update(np.array([float(yaw_rate) - predicted]), H, np.array([[float(variance)]]), gate)

    def vector(self) -> np.ndarray:
        return np.r_[self.p, self.v, self.ba, self.yaw, self.yaw_rate_correction]


class DirectYawEKF9:
    """9-state error-state KF using external IMU yaw directly.

    State: ``[p_w(3), v_w(3), b_a^b(3)]``. Yaw is not a state, is not integrated, and receives no
    network yaw-rate updates; it is a known input for specific-force rotation and body-XY velocity observations.
    """

    def __init__(self, cfg: dict, nominal_imu_dt: float = 0.005) -> None:
        self.g = np.asarray(cfg["gravity_flu"], dtype=float).reshape(3)
        sample_cov = AttitudeAidedESKF11._psd(np.asarray(cfg["accel_sample_covariance"], dtype=float))
        self.accel_noise_density = sample_cov * float(nominal_imu_dt)
        self.accel_bias_rw_psd = AttitudeAidedESKF11._psd(np.asarray(cfg["accel_bias_rw_psd"], dtype=float))
        self.p = np.zeros(3); self.v = np.zeros(3); self.ba = np.zeros(3); self.P = np.eye(9)

    def initialize(self, position: np.ndarray, accel_bias: np.ndarray, covariance: np.ndarray,
                   *, velocity: np.ndarray | None = None) -> None:
        covariance = AttitudeAidedESKF11._psd(np.asarray(covariance, dtype=float))
        if covariance.shape != (9, 9):
            raise ValueError("DirectYawEKF9 initial covariance must be 9x9")
        self.p = np.asarray(position, dtype=float).reshape(3)
        self.v = np.zeros(3) if velocity is None else np.asarray(velocity, dtype=float).reshape(3)
        self.ba = np.asarray(accel_bias, dtype=float).reshape(3); self.P = covariance

    def reanchor(self, delta_yaw: float) -> None:
        """Rotate current p/v and covariance into the benchmark world frame established at 8 s."""
        rotation = _rz(float(delta_yaw)); transform = np.eye(9)
        transform[:3, :3] = rotation; transform[3:6, 3:6] = rotation
        self.p = rotation @ self.p; self.v = rotation @ self.v; self.p[:2] = 0.0
        self.P = transform @ self.P @ transform.T
        self.P[:2, :] = 0.0; self.P[:, :2] = 0.0; self.P[0, 0] = self.P[1, 1] = 1e-6
        self.P = AttitudeAidedESKF11._project_psd(self.P)

    def predict_interval(self, accel_start: np.ndarray, accel_end: np.ndarray, yaw_mid: float, dt: float) -> None:
        dt = float(dt)
        if not np.isfinite(dt) or dt <= 0:
            raise ValueError("Prediction dt must be finite and positive")
        f = .5 * (np.asarray(accel_start, dtype=float).reshape(3) + np.asarray(accel_end, dtype=float).reshape(3)) - self.ba
        R = _rz(float(yaw_mid)); a_world = R @ f + self.g
        self.p = self.p + self.v * dt + .5 * a_world * dt * dt
        self.v = self.v + a_world * dt
        F = np.zeros((9, 9)); G = np.zeros((9, 6)); F[:3, 3:6] = np.eye(3); F[3:6, 6:9] = -R
        G[3:6, :3] = -R; G[6:9, 3:6] = np.eye(3)
        Qc = np.zeros((6, 6)); Qc[:3, :3] = self.accel_noise_density; Qc[3:6, 3:6] = self.accel_bias_rw_psd
        F2, F3 = F @ F, F @ F @ F
        Phi = np.eye(9) + F * dt + .5 * F2 * dt**2 + F3 * dt**3 / 6.0
        Q = G @ Qc @ G.T
        Qd = Q * dt + .5 * (F @ Q + Q @ F.T) * dt**2 + (F2 @ Q + 2 * F @ Q @ F.T + Q @ F2.T) * dt**3 / 6.0
        self.P = AttitudeAidedESKF11._project_psd(Phi @ self.P @ Phi.T + Qd)

    def update(self, residual: np.ndarray, H: np.ndarray, covariance: np.ndarray, gate: float) -> Update:
        residual, H = np.asarray(residual, dtype=float).reshape(-1), np.asarray(H, dtype=float)
        covariance = AttitudeAidedESKF11._psd(np.asarray(covariance, dtype=float))
        if H.shape != (len(residual), 9) or covariance.shape != (len(residual), len(residual)):
            raise ValueError("Observation dimensions do not match the 9-state model")
        S = AttitudeAidedESKF11._psd(H @ self.P @ H.T + covariance)
        nis = float(residual @ np.linalg.solve(S, residual))
        if not np.isfinite(nis) or nis > float(gate): return Update(False, nis, residual.copy())
        K = np.linalg.solve(S, H @ self.P).T; identity = np.eye(9)
        self.P = AttitudeAidedESKF11._project_psd((identity - K @ H) @ self.P @ (identity - K @ H).T + K @ covariance @ K.T)
        dx = K @ residual; self.p += dx[:3]; self.v += dx[3:6]; self.ba += dx[6:9]
        return Update(True, nis, residual.copy())

    def update_height(self, height: float, variance: float, gate: float) -> Update:
        H = np.zeros((1, 9)); H[0, 2] = 1.0
        return self.update(np.array([float(height) - self.p[2]]), H, np.array([[float(variance)]]), gate)

    def update_body_velocity(self, velocity_xy: np.ndarray, yaw: float, covariance: np.ndarray, gate: float) -> Update:
        R = _rz(float(yaw)); predicted = R.T @ self.v
        H = np.zeros((2, 9)); H[:, 3:6] = R.T[:2, :]
        return self.update(np.asarray(velocity_xy, dtype=float).reshape(2) - predicted[:2], H, covariance, gate)
