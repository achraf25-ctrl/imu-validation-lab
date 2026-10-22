"""Virtual sensors: a configurable IMU and a ground-truth tracker.

Ideal signals are derived from the true tool pose:
    gyro   omega_b                       (body frame, rad/s)
    accel  f_b = R^T (a_world - g)       (specific force, body frame, m/s^2)
The IMU then applies, in order: lever-arm effect -> time delay -> axis convention and
mounting misalignment -> per-axis scale error -> bias (+ random walk) -> white noise
-> causal low-pass filter.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict

import numpy as np
from scipy.signal import butter, lfilter, lfilter_zi

from . import quat as Q
from .trajectory import Trajectory

G = 9.80665
GRAVITY_W = np.array([0.0, 0.0, -G])


# ----------------------------------------------------------------- ground truth
def ideal_imu_signals(pose: Trajectory) -> dict:
    """Ideal gyro / specific force / angular acceleration from a pose series."""
    dt = pose.dt
    R = Q.to_matrix(pose.quat)
    omega = Q.angular_velocity_body(pose.quat, dt)
    acc_w = np.gradient(np.gradient(pose.pos, dt, axis=0), dt, axis=0)
    f_b = np.einsum("nji,nj->ni", R, acc_w - GRAVITY_W)
    alpha = np.gradient(omega, dt, axis=0)
    return dict(omega=omega, f=f_b, alpha=alpha, acc_world=acc_w)


@dataclass
class TrackerConfig:
    """Ground-truth tracker (mocap-like). Ideal by default."""
    pos_noise_m: float = 0.0
    rot_noise_deg: float = 0.0
    delay_s: float = 0.0
    seed: int = 1


def track(truth: Trajectory, cfg: TrackerConfig | None = None) -> Trajectory:
    cfg = cfg or TrackerConfig()
    t, pos, quat = truth.t, truth.pos.copy(), truth.quat.copy()
    if cfg.delay_s:
        pos = np.stack([np.interp(t - cfg.delay_s, t, pos[:, i]) for i in range(3)], axis=1)
        quat = Q.resample(t, quat, np.clip(t - cfg.delay_s, t[0], t[-1]))
    rng = np.random.default_rng(cfg.seed)
    if cfg.pos_noise_m > 0:
        pos = pos + rng.normal(0, cfg.pos_noise_m, pos.shape)
    if cfg.rot_noise_deg > 0:
        quat = Q.normalize(Q.mul(quat, Q.exp_rotvec(np.deg2rad(rng.normal(0, cfg.rot_noise_deg, (len(t), 3))))))
    return Trajectory(t=t, pos=pos, quat=Q.make_continuous(quat))


# ------------------------------------------------------------------------ IMU
def parse_axis_convention(spec: str) -> np.ndarray:
    """"-y,x,z" -> P with sensor_i = sum_j P[i,j] body_j (a signed permutation)."""
    toks = [s.strip().lower() for s in spec.split(",")]
    if len(toks) != 3:
        raise ValueError(f"axis convention needs 3 comma-separated axes, got {spec!r}")
    P = np.zeros((3, 3))
    for i, tk in enumerate(toks):
        sign = -1.0 if tk.startswith("-") else 1.0
        ax = tk.lstrip("+-")
        if ax not in ("x", "y", "z"):
            raise ValueError(f"bad axis token {tk!r}")
        P[i, "xyz".index(ax)] = sign
    if not np.allclose(np.abs(P).sum(0), 1):
        raise ValueError(f"axis convention {spec!r} is not a permutation")
    return P


def convention_string(P: np.ndarray) -> str:
    out = []
    for i in range(3):
        j = int(np.argmax(np.abs(P[i])))
        out.append(("-" if P[i, j] < 0 else "") + "xyz"[j])
    return ",".join(out)


@dataclass
class IMUConfig:
    rate: float = 120.0
    axis_convention: str = "x,y,z"
    mount_rotvec_deg: tuple = (0.0, 0.0, 0.0)       # small mounting misalignment
    gyro_bias_dps: tuple = (0.0, 0.0, 0.0)
    accel_bias: tuple = (0.0, 0.0, 0.0)             # m/s^2
    gyro_scale_err: tuple = (0.0, 0.0, 0.0)         # fractional, 0.02 = +2 %
    accel_scale_err: tuple = (0.0, 0.0, 0.0)
    gyro_noise_dps: float = 0.0                     # white noise std per sample
    accel_noise: float = 0.0                        # m/s^2 per sample
    gyro_bias_walk_dps: float = 0.0                 # random walk, deg/s per sqrt(s)
    lowpass_hz: float | None = None                 # causal 2nd-order Butterworth
    delay_s: float = 0.0                            # pure transport delay
    lever_arm_m: tuple = (0.0, 0.0, 0.0)            # IMU offset from tool frame (body frame)
    seed: int = 0

    def to_dict(self) -> dict:
        return asdict(self)

    def mapping(self) -> tuple[np.ndarray, np.ndarray]:
        """True linear maps body -> sensor for gyro and accel (without bias)."""
        P = parse_axis_convention(self.axis_convention)
        Rm = Q.to_matrix(Q.exp_rotvec(np.deg2rad(self.mount_rotvec_deg)))
        base = Rm @ P
        return np.diag(1 + np.asarray(self.gyro_scale_err)) @ base, np.diag(1 + np.asarray(self.accel_scale_err)) @ base


@dataclass
class IMUData:
    t: np.ndarray
    gyro: np.ndarray    # (N,3) rad/s, sensor axes
    accel: np.ndarray   # (N,3) m/s^2, sensor axes

    @property
    def dt(self) -> float:
        return float(self.t[1] - self.t[0])


def _delay(t: np.ndarray, x: np.ndarray, delay: float) -> np.ndarray:
    if delay == 0:
        return x
    return np.stack([np.interp(t - delay, t, x[:, i]) for i in range(x.shape[1])], axis=1)


def simulate_imu(truth: Trajectory, cfg: IMUConfig) -> IMUData:
    """Simulate IMU output from the true tool trajectory (must be sampled at cfg.rate)."""
    if abs(truth.dt - 1.0 / cfg.rate) > 1e-9:
        raise ValueError("truth sample rate must match IMUConfig.rate")
    rng = np.random.default_rng(cfg.seed)
    sig = ideal_imu_signals(truth)
    t, omega, f = truth.t, sig["omega"], sig["f"]

    r = np.asarray(cfg.lever_arm_m, float)
    if np.any(r != 0):
        f = f + np.cross(sig["alpha"], r) + np.cross(omega, np.cross(omega, r))

    omega, f = _delay(t, omega, cfg.delay_s), _delay(t, f, cfg.delay_s)

    Mg, Ma = cfg.mapping()
    gyro = omega @ Mg.T
    acc = f @ Ma.T

    n = len(t)
    bias_g = np.deg2rad(np.asarray(cfg.gyro_bias_dps, float)) + np.zeros((n, 3))
    if cfg.gyro_bias_walk_dps > 0:
        steps = rng.normal(0, np.deg2rad(cfg.gyro_bias_walk_dps) * np.sqrt(truth.dt), (n, 3))
        bias_g = bias_g + np.cumsum(steps, axis=0)
    gyro = gyro + bias_g + rng.normal(0, np.deg2rad(cfg.gyro_noise_dps), (n, 3))
    acc = acc + np.asarray(cfg.accel_bias, float) + rng.normal(0, cfg.accel_noise, (n, 3))

    if cfg.lowpass_hz:
        b, a = butter(2, cfg.lowpass_hz / (0.5 * cfg.rate))
        zi = lfilter_zi(b, a)
        gyro = np.stack([lfilter(b, a, gyro[:, i], zi=zi * gyro[0, i])[0] for i in range(3)], axis=1)
        acc = np.stack([lfilter(b, a, acc[:, i], zi=zi * acc[0, i])[0] for i in range(3)], axis=1)
    return IMUData(t=t, gyro=gyro, accel=acc)


def lowpass_group_delay_s(cfg: IMUConfig) -> float:
    """Low-frequency group delay of the on-board 2nd-order Butterworth low-pass."""
    return float(np.sqrt(2.0) / (2 * np.pi * cfg.lowpass_hz)) if cfg.lowpass_hz else 0.0
