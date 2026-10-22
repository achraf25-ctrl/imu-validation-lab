"""Quaternion utilities. Convention: q = [w, x, y, z], Hamilton, unit norm.

All functions are vectorised over leading dimensions.
q maps body -> world:  v_world = q * v_body * q^-1.
"""
from __future__ import annotations

import numpy as np


def normalize(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=float)
    return q / np.linalg.norm(q, axis=-1, keepdims=True)


def conj(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=float)
    out = q.copy()
    out[..., 1:] *= -1.0
    return out


def mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    aw, ax, ay, az = np.moveaxis(a, -1, 0)
    bw, bx, by, bz = np.moveaxis(b, -1, 0)
    return np.stack(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ],
        axis=-1,
    )


def hemisphere(q: np.ndarray) -> np.ndarray:
    """Flip signs so w >= 0 (q and -q are the same rotation)."""
    q = np.asarray(q, dtype=float)
    return np.where(q[..., :1] < 0, -q, q)


def make_continuous(q: np.ndarray) -> np.ndarray:
    """Remove sign flips along the first axis so consecutive quats have dot >= 0."""
    q = np.array(q, dtype=float)
    for i in range(1, len(q)):
        if np.dot(q[i - 1], q[i]) < 0:
            q[i] *= -1.0
    return q


def exp_rotvec(rv: np.ndarray) -> np.ndarray:
    """Rotation vector (axis * angle, rad) -> quaternion."""
    rv = np.asarray(rv, dtype=float)
    ang = np.linalg.norm(rv, axis=-1, keepdims=True)
    half = 0.5 * ang
    small = ang < 1e-8
    safe = np.where(small, 1.0, ang)
    k = np.where(small, 0.5 - ang**2 / 48.0, np.sin(half) / safe)
    w = np.where(small, 1.0 - ang**2 / 8.0, np.cos(half))
    return np.concatenate([w, k * rv], axis=-1)


def log_rotvec(q: np.ndarray) -> np.ndarray:
    """Quaternion -> rotation vector (shortest rotation)."""
    q = hemisphere(normalize(q))
    v = q[..., 1:]
    s = np.linalg.norm(v, axis=-1, keepdims=True)
    ang = 2.0 * np.arctan2(s, q[..., :1])
    small = s < 1e-9
    scale = np.where(small, 2.0, ang / np.where(small, 1.0, s))
    return scale * v


def to_matrix(q: np.ndarray) -> np.ndarray:
    q = normalize(q)
    w, x, y, z = np.moveaxis(q, -1, 0)
    R = np.stack(
        [
            1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w),
            2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w),
            2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y),
        ],
        axis=-1,
    )
    return R.reshape(q.shape[:-1] + (3, 3))


def rotate(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate vector(s) v by quaternion(s) q (body -> world)."""
    return np.einsum("...ij,...j->...i", to_matrix(q), np.asarray(v, dtype=float))


def from_matrix(R: np.ndarray) -> np.ndarray:
    """Rotation matrix (..., 3, 3) -> quaternion (Shepperd's method)."""
    R = np.asarray(R, dtype=float)
    flat = R.reshape(-1, 3, 3)
    out = np.empty((flat.shape[0], 4))
    for i, M in enumerate(flat):
        t = np.trace(M)
        if t > 0:
            s = np.sqrt(t + 1.0) * 2
            q = [0.25 * s, (M[2, 1] - M[1, 2]) / s, (M[0, 2] - M[2, 0]) / s, (M[1, 0] - M[0, 1]) / s]
        elif M[0, 0] > M[1, 1] and M[0, 0] > M[2, 2]:
            s = np.sqrt(1.0 + M[0, 0] - M[1, 1] - M[2, 2]) * 2
            q = [(M[2, 1] - M[1, 2]) / s, 0.25 * s, (M[0, 1] + M[1, 0]) / s, (M[0, 2] + M[2, 0]) / s]
        elif M[1, 1] > M[2, 2]:
            s = np.sqrt(1.0 + M[1, 1] - M[0, 0] - M[2, 2]) * 2
            q = [(M[0, 2] - M[2, 0]) / s, (M[0, 1] + M[1, 0]) / s, 0.25 * s, (M[1, 2] + M[2, 1]) / s]
        else:
            s = np.sqrt(1.0 + M[2, 2] - M[0, 0] - M[1, 1]) * 2
            q = [(M[1, 0] - M[0, 1]) / s, (M[0, 2] + M[2, 0]) / s, (M[1, 2] + M[2, 1]) / s, 0.25 * s]
        out[i] = q
    return hemisphere(normalize(out)).reshape(R.shape[:-2] + (4,))


def skew(v: np.ndarray) -> np.ndarray:
    x, y, z = v
    return np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]], dtype=float)


def angle_between(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Geodesic angle (rad) between orientations."""
    d = np.abs(np.sum(normalize(q1) * normalize(q2), axis=-1))
    return 2.0 * np.arccos(np.clip(d, -1.0, 1.0))


def slerp(q0: np.ndarray, q1: np.ndarray, t: float) -> np.ndarray:
    q0, q1 = normalize(q0), normalize(q1)
    d = np.dot(q0, q1)
    if d < 0:
        q1, d = -q1, -d
    if d > 0.9995:
        return normalize(q0 + t * (q1 - q0))
    th = np.arccos(d)
    return (np.sin((1 - t) * th) * q0 + np.sin(t * th) * q1) / np.sin(th)


def resample(t_old: np.ndarray, q_old: np.ndarray, t_new: np.ndarray) -> np.ndarray:
    """SLERP resampling of a quaternion time series."""
    q_old = make_continuous(q_old)
    idx = np.clip(np.searchsorted(t_old, t_new) - 1, 0, len(t_old) - 2)
    out = np.empty((len(t_new), 4))
    for k, (i, tn) in enumerate(zip(idx, t_new)):
        u = np.clip((tn - t_old[i]) / (t_old[i + 1] - t_old[i]), 0.0, 1.0)
        out[k] = slerp(q_old[i], q_old[i + 1], u)
    return out


def angular_velocity_body(q: np.ndarray, dt: float) -> np.ndarray:
    """Body-frame angular velocity from an orientation series (central differences)."""
    q = make_continuous(q)
    n = len(q)
    w = np.empty((n, 3))
    w[1:-1] = log_rotvec(mul(conj(q[:-2]), q[2:])) / (2 * dt)
    w[0] = log_rotvec(mul(conj(q[0]), q[1])) / dt
    w[-1] = log_rotvec(mul(conj(q[-2]), q[-1])) / dt
    return w


def integrate_body(q: np.ndarray, omega_b: np.ndarray, dt: float) -> np.ndarray:
    """One exact step: q <- q * exp(omega_b dt)."""
    return normalize(mul(q, exp_rotvec(np.asarray(omega_b) * dt)))


def yaw_error(q_true: np.ndarray, q_est: np.ndarray) -> np.ndarray:
    """Signed heading error (rad) about world z of R_true R_est^T (swing-twist)."""
    qe = hemisphere(mul(q_true, conj(q_est)))
    return 2.0 * np.arctan2(qe[..., 3], qe[..., 0])


def tilt_error(q_true: np.ndarray, q_est: np.ndarray) -> np.ndarray:
    """Angle (rad) between the gravity directions seen in the body frame."""
    ez = np.array([0.0, 0.0, 1.0])
    ut = np.einsum("...ji,j->...i", to_matrix(q_true), ez)
    ue = np.einsum("...ji,j->...i", to_matrix(q_est), ez)
    c = np.clip(np.sum(ut * ue, axis=-1), -1.0, 1.0)
    return np.arccos(c)
