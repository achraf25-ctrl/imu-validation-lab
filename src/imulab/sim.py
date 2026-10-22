"""Replay simulation: joint-servo model + forward kinematics -> ground-truth tool pose.

KinematicSim is the default, dependency-free backend. pybullet_backend.PyBulletSim
replays the same joint plan in PyBullet (kinematic or dynamic) for cross-checking.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import butter, filtfilt

from .arm import Arm
from .pipeline import JointPlan
from .trajectory import Trajectory


@dataclass
class ReplayConfig:
    servo_tau_s: float = 0.012          # first-order joint servo lag
    exec_noise_deg: float = 0.02        # std of smooth run-to-run joint perturbation
    exec_noise_cutoff_hz: float = 1.5   # bandwidth of that perturbation
    seed: int = 0


@dataclass
class ReplayResult:
    t: np.ndarray
    q: np.ndarray
    truth: Trajectory   # actual tool pose (what the "ground truth tracker" sees)


def _smooth_noise(n: int, dt: float, std: float, cutoff: float, rng: np.random.Generator, dof: int = 6):
    if std <= 0:
        return np.zeros((n, dof))
    b, a = butter(2, cutoff / (0.5 / dt))
    raw = rng.standard_normal((n, dof))
    sm = filtfilt(b, a, raw, axis=0)
    return sm / (sm.std(axis=0, keepdims=True) + 1e-12) * std


class KinematicSim:
    def __init__(self, arm: Arm, cfg: ReplayConfig | None = None):
        self.arm = arm
        self.cfg = cfg or ReplayConfig()

    def replay(self, plan: JointPlan) -> ReplayResult:
        cfg = self.cfg
        dt = float(plan.t[1] - plan.t[0])
        rng = np.random.default_rng(cfg.seed)
        q_cmd = plan.q + np.deg2rad(_smooth_noise(len(plan.t), dt, cfg.exec_noise_deg, cfg.exec_noise_cutoff_hz, rng))
        # exact discretisation of a first-order lag
        alpha = 1.0 - np.exp(-dt / cfg.servo_tau_s) if cfg.servo_tau_s > 0 else 1.0
        q = np.empty_like(q_cmd)
        q[0] = q_cmd[0]
        for k in range(1, len(q)):
            q[k] = q[k - 1] + alpha * (q_cmd[k] - q[k - 1])
        pos, quat = self.arm.fk(q)
        return ReplayResult(t=plan.t.copy(), q=q, truth=Trajectory(t=plan.t.copy(), pos=pos, quat=quat))
