"""Trajectory pipeline: pose trajectory (quaternions) -> limit-checked joint trajectory.

Steps: IK -> smoothing -> joint position / speed / acceleration checks -> uniform
time-scaling (slow-down) if speed or acceleration limits are exceeded -> resampling
at the original rate. The retimed *target* pose trajectory is returned too, so replay
accuracy can be measured against what was actually commanded.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.signal import savgol_filter

from . import quat as Q
from .arm import Arm
from .trajectory import Trajectory


@dataclass
class JointPlan:
    t: np.ndarray
    q: np.ndarray            # (N, 6) commanded joint angles
    qd: np.ndarray
    qdd: np.ndarray
    target: Trajectory       # target tool pose on the (retimed) time base
    report: dict = field(default_factory=dict)


def _derivs(q: np.ndarray, dt: float) -> tuple[np.ndarray, np.ndarray]:
    qd = np.gradient(q, dt, axis=0)
    qdd = np.gradient(qd, dt, axis=0)
    return qd, qdd


def plan_trajectory(arm: Arm, traj: Trajectory, smooth_window: int = 9, safety: float = 1.02) -> JointPlan:
    dt = traj.dt
    q_raw, ok = arm.ik_trajectory(traj.pos, traj.quat)
    win = smooth_window if smooth_window % 2 == 1 else smooth_window + 1
    q = savgol_filter(q_raw, win, 3, axis=0) if smooth_window > 3 else q_raw

    pos_viol = (q < arm.q_min - 1e-9) | (q > arm.q_max + 1e-9)
    qd, qdd = _derivs(q, dt)
    speed_ratio = float(np.max(np.abs(qd) / arm.qd_max))
    acc_ratio = float(np.max(np.abs(qdd) / arm.qdd_max))
    slow = max(1.0, speed_ratio * safety, np.sqrt(acc_ratio) * safety)

    target = traj
    t_out = traj.t
    if slow > 1.0:
        # time-scale: new motion q'(tau) = q(tau / slow) -> speeds / slow, accelerations / slow^2
        n_new = int(np.floor(traj.t[-1] * slow / dt)) + 1
        t_out = np.arange(n_new) * dt
        tau = t_out / slow
        q = CubicSpline(traj.t, q, axis=0)(tau)
        pos = CubicSpline(traj.t, traj.pos, axis=0)(tau)
        quat = Q.resample(traj.t, traj.quat, tau)
        target = Trajectory(t=t_out, pos=pos, quat=quat)
        qd, qdd = _derivs(q, dt)

    # tracking of the smoothed joint solution vs the requested pose
    p_ik, q_ik = arm.fk(q)
    report = dict(
        ik_converged_fraction=float(np.mean(ok)),
        joint_limit_violations=int(pos_viol.sum()),
        speed_ratio_before=speed_ratio,
        accel_ratio_before=acc_ratio,
        slowdown_factor=float(slow),
        retimed=bool(slow > 1.0),
        max_joint_speed_dps=np.rad2deg(np.abs(qd).max(0)).round(1).tolist(),
        max_joint_accel_dps2=np.rad2deg(np.abs(qdd).max(0)).round(0).tolist(),
        within_limits=bool(
            pos_viol.sum() == 0
            and np.all(np.abs(qd) <= arm.qd_max * 1.001)
            and np.all(np.abs(qdd) <= arm.qdd_max * 1.001)
        ),
        plan_pos_err_mm=float(1000 * np.max(np.linalg.norm(p_ik - target.pos, axis=1))),
        plan_rot_err_deg=float(np.rad2deg(np.max(Q.angle_between(q_ik, target.quat)))),
    )
    return JointPlan(t=t_out, q=q, qd=qd, qdd=qdd, target=target, report=report)
