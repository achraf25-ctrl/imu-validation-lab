"""Orientation estimation from gyro + accelerometer: dead-reckoning, complementary filter, EKF.

All filters run on tool-frame IMU data (gyro rad/s, specific force m/s^2) and are
initialised with the true initial orientation (no magnetometer -> yaw is unobservable,
so heading is reported separately as drift).
"""
from __future__ import annotations

import numpy as np

from . import quat as Q
from .sensors import G

EZ = np.array([0.0, 0.0, 1.0])


def _up_pred(q: np.ndarray) -> np.ndarray:
    """World 'up' expressed in the body frame: R^T e_z."""
    return Q.to_matrix(q).T @ EZ


def _dyn_dev(f: np.ndarray) -> float:
    """Relative deviation of |f| from 1 g (large during dynamic motion)."""
    return abs(np.linalg.norm(f) - G) / G


def gyro_integration(q0: np.ndarray, gyro: np.ndarray, dt: float) -> np.ndarray:
    q = np.empty((len(gyro), 4))
    q[0] = q0
    for k in range(1, len(gyro)):
        q[k] = Q.integrate_body(q[k - 1], gyro[k], dt)
    return q


def complementary(q0, gyro, accel, dt, kp: float = 0.5, ki: float = 0.01, gate: float = 0.15) -> np.ndarray:
    """Mahony-style filter with accel gating (trust drops as |f| departs from 1 g)."""
    q = np.empty((len(gyro), 4))
    q[0] = q0
    integ = np.zeros(3)
    for k in range(1, len(gyro)):
        f = accel[k]
        nf = np.linalg.norm(f)
        w = 1.0 / (1.0 + (_dyn_dev(f) / gate) ** 2) if nf > 1e-6 else 0.0
        u_meas = f / nf if nf > 1e-6 else EZ
        e = np.cross(u_meas, _up_pred(q[k - 1]))
        integ += ki * w * e * dt
        q[k] = Q.integrate_body(q[k - 1], gyro[k] + kp * w * e + integ, dt)
    return q


def ekf(
    q0, gyro, accel, dt,
    gyro_noise: float = 0.005,       # rad/s per sample (white)
    bias_walk: float = 2e-5,         # rad/s per sqrt(s)
    accel_dir_noise: float = 0.3,    # rad: gravity-direction noise. Deliberately large:
                                     # horizontal tool acceleration (~0.15 g) tilts the
                                     # measured "up" by ~0.15 rad but barely changes |f|.
    dyn_gain: float = 1.5,           # extra direction noise per unit of |f| deviation / g
    p0_att: float = np.deg2rad(2.0),
    p0_bias: float = np.deg2rad(0.2),
) -> tuple[np.ndarray, np.ndarray]:
    """Error-state EKF: state = [attitude error (body, 3), gyro bias (3)].

    Returns the quaternion series and the estimated gyro-bias series.

    Tuning note: with optimistic accel noise and a loose bias prior, correlated dynamic
    acceleration leaks into the bias state and heading drifts by tens of degrees
    (observed in this project; see docs/REPORT.md). The defaults keep the bias state
    tight; loosen p0_bias / bias_walk for IMUs with large or wandering bias.
    """
    n = len(gyro)
    qs = np.empty((n, 4))
    bs = np.zeros((n, 3))
    q, b = np.array(q0, float), np.zeros(3)
    qs[0] = q
    P = np.diag([p0_att**2] * 3 + [p0_bias**2] * 3)
    Qd = np.diag([(gyro_noise**2) * dt] * 3 + [(bias_walk**2) * dt] * 3)
    I6 = np.eye(6)
    for k in range(1, n):
        w = gyro[k] - b
        q = Q.integrate_body(q, w, dt)
        F = np.eye(6)
        F[:3, :3] -= Q.skew(w) * dt
        F[:3, 3:] = -np.eye(3) * dt
        P = F @ P @ F.T + Qd

        f = accel[k]
        nf = np.linalg.norm(f)
        if nf > 1e-6:
            z = f / nf
            h = _up_pred(q)
            sig = accel_dir_noise + dyn_gain * _dyn_dev(f)
            H = np.hstack([Q.skew(h), np.zeros((3, 3))])
            S = H @ P @ H.T + sig**2 * np.eye(3)
            K = P @ H.T @ np.linalg.inv(S)
            dx = K @ (z - h)
            q = Q.normalize(Q.mul(q, Q.exp_rotvec(dx[:3])))
            b = b + dx[3:]
            IKH = I6 - K @ H
            P = IKH @ P @ IKH.T + K @ (sig**2 * np.eye(3)) @ K.T
        qs[k], bs[k] = q, b
    return qs, bs


def orientation_errors(q_true: np.ndarray, q_est: np.ndarray) -> dict:
    """Total / tilt / heading errors (degrees) over time."""
    return dict(
        total_deg=np.rad2deg(Q.angle_between(q_true, q_est)),
        tilt_deg=np.rad2deg(Q.tilt_error(q_true, q_est)),
        yaw_deg=np.rad2deg(Q.yaw_error(q_true, q_est)),
    )


def summarize_errors(err: dict, skip: int = 0) -> dict:
    t, ti, y = err["total_deg"][skip:], err["tilt_deg"][skip:], err["yaw_deg"][skip:]
    return dict(
        total_rmse_deg=float(np.sqrt(np.mean(t**2))),
        total_final_deg=float(err["total_deg"][-1]),
        tilt_rmse_deg=float(np.sqrt(np.mean(ti**2))),
        heading_drift_deg=float(np.abs(err["yaw_deg"][-1])),
        heading_rmse_deg=float(np.sqrt(np.mean(y**2))),
    )
